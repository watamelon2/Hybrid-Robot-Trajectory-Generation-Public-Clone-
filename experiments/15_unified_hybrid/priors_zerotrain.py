"""Zero-training warm-start priors for the unified hybrid driver.

These priors answer the question *"is BC's MLP actually doing real
work, or could we just retrieve action sequences from the demo
dataset directly?"* by replacing the learned BC policy with a
non-parametric retrieval rule over the proficient-human ``can/ph``
demo bank (200 demos x ~150 steps = ~30 000 (obs, action) pairs).

Four variants, all sharing the same factory shape as exp 15's
``_load_prior``:

  * ``demo_1nn``   -- nearest-neighbour over the 46-d obs vector;
                       action window = next ``H`` actions from the
                       matched demo timestep (clipped at demo end).
  * ``demo_knn``   -- ``k``-NN, action window = mean over the ``k``
                       matched action windows.
  * ``demo_random``-- uniform-random demo, starting at t=0; controls
                       for "does state-conditioning matter?".
  * ``demo_mean``  -- dataset-wide mean action sequence; controls
                       for "does any demo information matter?".

All four reuse BC's pull-toward-warm-start score function
(``score_fn(mu, k) = a_target - mu``) so that ``--lambda-prior > 0``
turns the prior into a per-step score-bridge with the same
algorithmic shape as the BC headline.  No new score-bridge code.

Public API:

    >>> bank = load_demo_bank(Path("/.../low_dim_v141.hdf5"))
    >>> initial_mean_fn, score_fn_factory, label = make_zero_train_prior(
    ...     kind="demo_1nn", bank=bank, H=120, knn_k=5, rng_seed=0,
    ... )

The factory signatures match exp 07/15's ``_load_prior``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# Several experiments ship a module named ``dataset`` (exp 03 has the
# canonical BC dataset; exp 08 has a flow-specific one).  Plain
# ``from dataset import ...`` is order-sensitive once the unified
# driver has added exp 08 to ``sys.path``.  Load exp 03's
# ``dataset.py`` by absolute file path under a private module name to
# avoid the collision.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP03_DATASET = _REPO_ROOT / "experiments" / "03_bc_baseline" / "dataset.py"


def _load_exp03_dataset_module():
    spec = importlib.util.spec_from_file_location(
        "_exp03_dataset_for_zerotrain", str(_EXP03_DATASET)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {_EXP03_DATASET}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_exp03_dataset_for_zerotrain"] = mod
    spec.loader.exec_module(mod)
    return mod


_ds = _load_exp03_dataset_module()
OBS_KEYS = _ds.OBS_KEYS  # Can default (kept for backward compat)
_DATASET_OBJECT_SLICES = _ds._DATASET_OBJECT_SLICES  # Can default
_concat_obs_dict = _ds._concat_obs_dict
compute_obs_stats = _ds.compute_obs_stats
get_obs_keys = _ds.get_obs_keys  # task-aware lookup
get_object_slices = _ds.get_object_slices
# Synthetic-key support for the DP paper's 23-D ``can_dp`` schema
# (Phase-B / Phase-C, 2026-05-29).  ``get_synthetic_keys(task)``
# returns ``{synth_key: (src1, src2)}`` mappings used by
# ``compute_synthetic_obs_keys`` to derive e.g.
# ``Can_to_eef_pos_synth = robot0_eef_pos - Can_pos`` at runtime
# from the base obs keys robosuite exposes.  Re-exported here so
# the unified pipeline's ``_make_obs_vec_for_prior`` can apply them
# without depending on the exp 08 dataset module.
get_synthetic_keys = _ds.get_synthetic_keys
compute_synthetic_obs_keys = _ds.compute_synthetic_obs_keys


# ---------------------------------------------------------------------------
# Demo bank: flat (obs, action) arrays with per-demo boundary tracking.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoBank:
    """Per-timestep flat arrays + demo-boundary metadata.

    Attributes
    ----------
    obs:           (N, D) float32, **un-normalised** raw obs vectors.
                   ``D`` includes the action-history columns when
                   ``action_history_len > 0`` (A1, 2026-05-20).
    obs_norm:      (N, D) float32, normalised by ``stats``.
    actions:       (N, A) float32.
    demo_starts:   (M,) int64, row index of each demo's first timestep.
    demo_lengths:  (M,) int64, per-demo length in timesteps.
    demo_ids:      list[str], HDF5 group names of consumed demos.
    obs_mean:      (D,) float32.
    obs_std:       (D,) float32 (clipped >= eps).
    action_history_len:
                   ``int``, length of action history baked into the
                   last ``action_history_len * action_dim`` columns of
                   ``obs``.  ``0`` means no augmentation.
    """

    obs: np.ndarray
    obs_norm: np.ndarray
    actions: np.ndarray
    demo_starts: np.ndarray
    demo_lengths: np.ndarray
    demo_ids: list[str]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_history_len: int = 0

    @property
    def n_total(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_demos(self) -> int:
        return len(self.demo_ids)

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def normalize_obs(self, s0: np.ndarray) -> np.ndarray:
        return ((s0.astype(np.float32) - self.obs_mean) / self.obs_std).astype(
            np.float32
        )

    # ---- retrieval --------------------------------------------------------

    def nearest_index(self, s0_norm: np.ndarray) -> int:
        """Argmin L2 distance in normalised obs space."""
        d2 = ((self.obs_norm - s0_norm[None, :]) ** 2).sum(axis=-1)
        return int(np.argmin(d2))

    def k_nearest_indices(self, s0_norm: np.ndarray, k: int) -> np.ndarray:
        d2 = ((self.obs_norm - s0_norm[None, :]) ** 2).sum(axis=-1)
        k = max(1, min(k, d2.shape[0]))
        # argpartition is O(N); sort the top-k by actual distance for stability
        top = np.argpartition(d2, k - 1)[:k]
        order = np.argsort(d2[top])
        return top[order].astype(np.int64)

    def slice_actions(self, idx: int, H: int) -> np.ndarray:
        """``H`` actions starting at row ``idx``, clipped at demo end.

        If the demo runs out before ``H`` steps, the tail is held at the
        last available action (end-of-demo behaviour in robomimic
        ``ph`` data is usually "open gripper / hold pose", which is a
        safe default tail).
        """
        # Locate which demo idx falls into.
        di = int(np.searchsorted(self.demo_starts, idx, side="right") - 1)
        di = max(0, min(di, self.n_demos - 1))
        demo_end = int(self.demo_starts[di]) + int(self.demo_lengths[di])
        avail = max(0, demo_end - idx)
        n_copy = min(H, avail)
        out = np.empty((H, self.action_dim), dtype=np.float32)
        if n_copy > 0:
            out[:n_copy] = self.actions[idx : idx + n_copy]
            if n_copy < H:
                out[n_copy:] = self.actions[idx + n_copy - 1]
        else:
            # idx already past end; hold the demo's final action.
            out[:] = self.actions[demo_end - 1]
        return out

    def demo_start_actions(self, demo_idx: int, H: int) -> np.ndarray:
        """Action window starting at t=0 of demo ``demo_idx``."""
        di = int(demo_idx) % self.n_demos
        return self.slice_actions(int(self.demo_starts[di]), H)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _bank_cache_key(
    dataset_path: Path,
    n_demos: int | None,
    obs_keys: tuple[str, ...],
    action_history_len: int = 0,
) -> str:
    """Stable short hash of inputs that affect the bank arrays.

    Includes dataset path + mtime so the cache invalidates if the
    underlying HDF5 is replaced; includes ``n_demos``, the obs key
    order, and ``action_history_len`` so different bank subsets /
    schemas / A1-augmentations don't collide.
    """
    mtime = os.path.getmtime(dataset_path)
    payload = "\n".join(
        [
            str(dataset_path.resolve()),
            f"mtime={mtime:.6f}",
            f"n_demos={n_demos}",
            "obs_keys=" + ",".join(obs_keys),
            f"action_history_len={action_history_len}",
        ]
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _build_action_history(actions: np.ndarray, n: int) -> np.ndarray:
    """Return per-timestep flattened action history of shape ``(T, n*A)``.

    For row ``t``, the output is the concatenation of
    ``actions[t-n], actions[t-n+1], ..., actions[t-1]``, zero-padded
    in the front when ``t < n``.  The oldest action is at the lowest
    column index; the most recent is at the highest.  ``n=0`` returns
    a zero-width matrix (no augmentation).

    Used by ``load_demo_bank(action_history_len=n)`` to add action
    context to the retrieval obs vector (A1, added 2026-05-20).
    """
    T = int(actions.shape[0])
    A = int(actions.shape[1]) if actions.ndim > 1 else 0
    if n <= 0 or A == 0:
        return np.zeros((T, 0), dtype=np.float32)
    pad = np.zeros((n, A), dtype=np.float32)
    padded = np.concatenate([pad, actions.astype(np.float32)], axis=0)
    out = np.empty((T, n * A), dtype=np.float32)
    for t in range(T):
        out[t] = padded[t : t + n].reshape(-1)
    return out


def _bank_cache_path(dataset_path: Path, key: str) -> Path:
    """Cache file lives next to the dataset, dot-prefixed (hidden)."""
    return dataset_path.with_name(f".{dataset_path.name}.bank.{key}.npz")


def _save_bank(bank: "DemoBank", cache_path: Path) -> None:
    """Atomic save: write to a tmp file then rename.

    Avoids partial writes if the process dies mid-save and lets
    multiple Slate cells racing on the same bank file recover cleanly.

    Uses a binary file handle for ``np.savez`` so the tmp filename is
    not mangled (``np.savez`` auto-appends ``.npz`` to a string path).
    """
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            obs=bank.obs,
            obs_norm=bank.obs_norm,
            actions=bank.actions,
            demo_starts=bank.demo_starts,
            demo_lengths=bank.demo_lengths,
            demo_ids=np.array(bank.demo_ids, dtype=object),
            obs_mean=bank.obs_mean,
            obs_std=bank.obs_std,
            action_history_len=np.int64(bank.action_history_len),
        )
    os.replace(tmp, cache_path)


def _load_bank(cache_path: Path) -> "DemoBank":
    z = np.load(cache_path, allow_pickle=True)
    # ``action_history_len`` was added 2026-05-20 (A1); older cache
    # files lack the key and default to 0.
    if "action_history_len" in z.files:
        ahl = int(z["action_history_len"])
    else:
        ahl = 0
    return DemoBank(
        obs=z["obs"].astype(np.float32),
        obs_norm=z["obs_norm"].astype(np.float32),
        actions=z["actions"].astype(np.float32),
        demo_starts=z["demo_starts"].astype(np.int64),
        demo_lengths=z["demo_lengths"].astype(np.int64),
        demo_ids=[str(x) for x in z["demo_ids"]],
        obs_mean=z["obs_mean"].astype(np.float32),
        obs_std=z["obs_std"].astype(np.float32),
        action_history_len=ahl,
    )


def load_demo_bank(
    dataset_path: Path,
    n_demos: int | None = None,
    obs_keys: tuple[str, ...] | None = None,
    task: str = "can",
    use_cache: bool = True,
    verbose: bool = True,
    action_history_len: int = 0,
) -> DemoBank:
    """Read demos from the robomimic-style HDF5 and return a DemoBank.

    Honours the task-aware obs schema in
    ``experiments/03_bc_baseline/dataset.py``: pass ``task="square"``
    to get the Square nut keys, or override the whole tuple via
    ``obs_keys``.  When neither is set, the Can default applies (for
    backward compatibility with all pre-Slate-F callers).

    The obs vector dimension order matches the BC / flow training
    pipeline bit-for-bit for the corresponding task -- mandatory
    because the score-bridge depends on the live env producing the
    same obs vector at planning time.

    Caching (added 2026-05-20 / Slate G prep -- "D2"):
    --------------------------------------------------
    When ``use_cache=True`` (default), a normalized ``.npz`` of the
    bank is stored next to the dataset HDF5 under a content-keyed
    filename (``.dataset.hdf5.bank.<sha1[:12]>.npz``).  The key
    hashes dataset path + mtime + ``n_demos`` + ``obs_keys`` so
    different schemas / subsets don't collide and an HDF5 replacement
    invalidates the cache.  On hit, the loader skips the 30-60 s
    HDF5 read + normalization step.
    """
    dataset_path = Path(dataset_path)
    if obs_keys is None:
        obs_keys = get_obs_keys(task)

    cache_key = None
    cache_path = None
    if use_cache:
        try:
            cache_key = _bank_cache_key(
                dataset_path, n_demos, obs_keys,
                action_history_len=int(action_history_len),
            )
            cache_path = _bank_cache_path(dataset_path, cache_key)
            if cache_path.exists():
                t0 = time.perf_counter()
                bank = _load_bank(cache_path)
                if verbose:
                    print(
                        f"[priors_zerotrain] bank cache HIT "
                        f"{cache_path.name}  "
                        f"({time.perf_counter() - t0:.2f}s)"
                    )
                return bank
        except Exception as e:  # pragma: no cover - fall back to rebuild
            if verbose:
                print(
                    f"[priors_zerotrain] bank cache lookup failed "
                    f"({type(e).__name__}: {e}); rebuilding"
                )

    t_build = time.perf_counter()
    object_slices = get_object_slices(task)
    synth_keys = get_synthetic_keys(task)
    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    lengths: list[int] = []
    used: list[str] = []

    with h5py.File(str(dataset_path), "r") as h5:
        all_demos = sorted(
            h5["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        if n_demos is not None and n_demos > 0:
            all_demos = all_demos[:n_demos]
        for demo_id in all_demos:
            demo = h5["data"][demo_id]
            obs_dict: dict[str, np.ndarray] = {}
            for k in obs_keys:
                if k in object_slices:
                    src, sl = object_slices[k]
                    obs_dict[k] = np.asarray(demo["obs"][src][:, sl])
                elif k in synth_keys:
                    compute_synthetic_obs_keys(
                        obs_dict, {k: synth_keys[k]},
                    )
                else:
                    obs_dict[k] = np.asarray(demo["obs"][k][()])
            obs_flat = _concat_obs_dict(obs_dict, obs_keys)  # (T, D)
            actions = np.asarray(demo["actions"][()], dtype=np.float32)
            assert obs_flat.shape[0] == actions.shape[0], (
                f"obs / action length mismatch in {demo_id}: "
                f"{obs_flat.shape[0]} vs {actions.shape[0]}"
            )
            # A1: append per-timestep action history columns.
            # When ``action_history_len=0`` this is a (T, 0) array
            # and the concatenation is a no-op.
            if int(action_history_len) > 0:
                hist = _build_action_history(
                    actions, int(action_history_len)
                )
                obs_flat = np.concatenate(
                    [obs_flat.astype(np.float32), hist], axis=1
                )
            obs_chunks.append(obs_flat.astype(np.float32))
            act_chunks.append(actions.astype(np.float32))
            lengths.append(int(obs_flat.shape[0]))
            used.append(demo_id)

    obs = np.concatenate(obs_chunks, axis=0)
    actions = np.concatenate(act_chunks, axis=0)
    stats = compute_obs_stats(obs)
    obs_norm = stats.normalize(obs).astype(np.float32)

    lengths_arr = np.asarray(lengths, dtype=np.int64)
    starts_arr = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(lengths_arr)[:-1]]
    )

    bank = DemoBank(
        obs=obs,
        obs_norm=obs_norm,
        actions=actions,
        demo_starts=starts_arr,
        demo_lengths=lengths_arr,
        demo_ids=used,
        obs_mean=stats.mean.astype(np.float32),
        obs_std=stats.std.astype(np.float32),
        action_history_len=int(action_history_len),
    )

    if verbose:
        print(
            f"[priors_zerotrain] bank built from HDF5  "
            f"({time.perf_counter() - t_build:.2f}s)  "
            f"n_demos={bank.n_demos}  n_steps={bank.n_total}"
        )

    if use_cache and cache_path is not None:
        try:
            _save_bank(bank, cache_path)
            if verbose:
                print(
                    f"[priors_zerotrain] bank cached -> {cache_path.name}"
                )
        except Exception as e:  # pragma: no cover - non-fatal
            if verbose:
                print(
                    f"[priors_zerotrain] bank cache save failed "
                    f"({type(e).__name__}: {e}); continuing without cache"
                )

    return bank


# ---------------------------------------------------------------------------
# Pull-target score-bridge factory (identical shape to BC's in run_can.py)
# ---------------------------------------------------------------------------


def _make_pull_score_factory():
    """Return a ``score_fn_factory(s0, K)`` mirroring exp 07/15 BC.

    ``unified_loop.py`` sets ``score_fn.state["a_bc"] = initial_mean``
    before each chunk; the closure returns ``(a_target - mu)`` per
    step, which the denoise loop scales by ``lambda_prior`` and the
    score-bridge denominator.
    """

    def score_fn_factory(s0, K):
        del s0
        state = {"a_bc": None}

        def score_fn(mu, k):
            a_t = state["a_bc"]
            if a_t is None:
                return np.zeros_like(mu, dtype=np.float32)
            return (a_t - mu).astype(np.float32)

        score_fn.state = state  # type: ignore[attr-defined]
        return score_fn

    return score_fn_factory


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


# By construction (see ``OBS_KEYS`` / ``OBS_KEYS_SQUARE`` in exp 03's
# ``dataset.py``), the obs vector layout for every supported task is
# ``[<obj>_pos(3), <obj>_quat(4), <proprio>]``, so the first 7 dims
# are the object-state slice and the rest are robot proprio.  The
# Slate F mixture-of-features ablations (``demo_1nn_proprio`` /
# ``demo_1nn_object``) use this constant; revisit when a task with a
# richer object schema joins.
_N_OBJECT_DIMS = 7


def make_zero_train_prior(
    kind: str,
    bank: DemoBank,
    H: int,
    knn_k: int = 5,
    rng_seed: int = 0,
    retrieval_beta: float = 1.0,
    contrastive_encoder_path: "Path | str | None" = None,
):
    """Return ``(initial_mean_fn, score_fn_factory, label)``.

    Parameters
    ----------
    kind:
        One of ``demo_1nn``, ``demo_knn``, ``demo_random``,
        ``demo_mean``, ``demo_lwr``, ``demo_topk_random``,
        ``demo_1nn_proprio``, ``demo_1nn_object``, ``demo_softnn``,
        ``demo_1nn_contrastive``.
    bank:
        Pre-loaded ``DemoBank`` (call ``load_demo_bank`` once).
    H:
        Action horizon for each warm-start trajectory.
    knn_k:
        Number of neighbours for ``demo_knn`` / ``demo_lwr`` /
        ``demo_topk_random`` / ``demo_softnn``.  Ignored otherwise.
    rng_seed:
        Seed for the stochastic variants (``demo_random``,
        ``demo_topk_random``).  Reset before each call so that the
        warm-start is **deterministic per chunk** -- same seed gives
        the same demo across runs, satisfying the project's
        determinism rule.
    retrieval_beta:
        Temperature for ``demo_softnn`` (added 2026-05-20 / A3).
        Weights are ``softmax(-beta * d2)`` over the top-``knn_k``
        nearest neighbours.  ``beta = 0`` recovers a uniform mean
        over top-k (= ``demo_knn`` with k=knn_k); ``beta -> inf``
        recovers hard 1-NN (= ``demo_1nn``).  Intermediate values
        give a continuous bias-variance knob; sweeping ``beta``
        draws the bias-variance curve directly as a publishable
        figure.  Default ``1.0`` is a sensible mid-point on
        normalised obs (d2 typically O(1-10) for in-distribution
        states; beta=1 gives the closest neighbour ~e^9 weight
        more than the farthest of k=5, still effectively
        mode-selecting but with some smoothing).

    Returns
    -------
    initial_mean_fn:
        ``(env, init_state, s0) -> (H, A) float32`` warm-start trajectory.
    score_fn_factory:
        Standard pull-toward-warm-start factory (re-used from BC).
    label:
        Short string for the output JSON ``prior`` field.
    """
    if kind == "demo_1nn":
        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            s0_norm = bank.normalize_obs(np.asarray(s0))
            idx = bank.nearest_index(s0_norm)
            return bank.slice_actions(idx, H)

        return initial_mean_fn, _make_pull_score_factory(), "demo_1nn"

    if kind == "demo_knn":
        k = int(knn_k)

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            s0_norm = bank.normalize_obs(np.asarray(s0))
            idxs = bank.k_nearest_indices(s0_norm, k)
            window = np.stack(
                [bank.slice_actions(int(i), H) for i in idxs], axis=0
            )  # (k, H, A)
            return window.mean(axis=0).astype(np.float32)

        return initial_mean_fn, _make_pull_score_factory(), f"demo_knn_k{k}"

    if kind == "demo_random":
        # Per-call RNG seeded from (rng_seed, chunk-call-count) so the
        # warm-start is reproducible across runs.  Using a counter
        # in a closure ensures successive chunks see *different*
        # random demos within the same rollout, but identical
        # demos across runs with the same seed.
        counter = {"n": 0}

        def initial_mean_fn(env, init_state, s0):
            del env, init_state, s0
            rs = np.random.RandomState(int(rng_seed) * 1_000_003 + counter["n"])
            counter["n"] += 1
            demo_idx = int(rs.randint(0, bank.n_demos))
            return bank.demo_start_actions(demo_idx, H)

        return initial_mean_fn, _make_pull_score_factory(), "demo_random"

    if kind == "demo_mean":
        # One-time precompute: the **dataset-mean action trajectory**
        # of length H.  We take the mean of the first H actions across
        # all demos, holding shorter demos at their last action.
        per_demo = np.stack(
            [bank.demo_start_actions(i, H) for i in range(bank.n_demos)],
            axis=0,
        )  # (M, H, A)
        mean_traj = per_demo.mean(axis=0).astype(np.float32)

        def initial_mean_fn(env, init_state, s0):
            del env, init_state, s0
            return mean_traj.copy()

        return initial_mean_fn, _make_pull_score_factory(), "demo_mean"

    if kind == "demo_lwr":
        # Locally-weighted regression: k-NN with inverse-distance
        # weights, so distant neighbours contribute less.  Sits
        # between ``demo_1nn`` (k=1, all weight on the closest demo)
        # and ``demo_knn`` (k>=2 with uniform mean, which collapses
        # multimodal demo branches).  Hypothesis: distance weighting
        # recovers most of the 1-NN mode-selection benefit while
        # reducing the per-rollout demo-pick variance.
        k = int(knn_k)

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            s0_norm = bank.normalize_obs(np.asarray(s0))
            idxs = bank.k_nearest_indices(s0_norm, k)
            d2 = ((bank.obs_norm[idxs] - s0_norm[None, :]) ** 2).sum(axis=-1)
            # Inverse-distance weights with a small epsilon so that
            # an exact-match neighbour (d2 = 0) still gets a finite
            # but dominant weight.
            w = 1.0 / np.sqrt(d2 + 1e-6)
            w = w / w.sum()
            window = np.stack(
                [bank.slice_actions(int(i), H) for i in idxs], axis=0
            )  # (k, H, A)
            return (
                w[:, None, None] * window
            ).sum(axis=0).astype(np.float32)

        return (
            initial_mean_fn,
            _make_pull_score_factory(),
            f"demo_lwr_k{k}",
        )

    if kind == "demo_topk_random":
        # Top-k neighbours, then a *uniform random* pick among them.
        # Tests whether the 1-NN's choice is uniquely good or whether
        # any "close enough" neighbour suffices.  Adds determinism
        # via per-chunk RNG (same convention as ``demo_random``).
        k = int(knn_k)
        counter = {"n": 0}

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            s0_norm = bank.normalize_obs(np.asarray(s0))
            idxs = bank.k_nearest_indices(s0_norm, k)
            rs = np.random.RandomState(
                int(rng_seed) * 1_000_003 + counter["n"]
            )
            counter["n"] += 1
            pick = int(rs.randint(0, len(idxs)))
            return bank.slice_actions(int(idxs[pick]), H)

        return (
            initial_mean_fn,
            _make_pull_score_factory(),
            f"demo_topk_random_k{k}",
        )

    if kind == "demo_1nn_proprio":
        # 1-NN computed only on the proprio dims (drop object pos/quat).
        # Information ablation: if proprio-only retrieval works as well
        # as full-obs retrieval, the matching is really about robot
        # state, not object pose.  If it collapses, object info is
        # essential.
        d_obj = _N_OBJECT_DIMS
        proprio_obs_norm = bank.obs_norm[:, d_obj:]

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            s0_norm = bank.normalize_obs(np.asarray(s0))[d_obj:]
            d2 = (
                (proprio_obs_norm - s0_norm[None, :]) ** 2
            ).sum(axis=-1)
            idx = int(np.argmin(d2))
            return bank.slice_actions(idx, H)

        return (
            initial_mean_fn,
            _make_pull_score_factory(),
            "demo_1nn_proprio",
        )

    if kind == "demo_1nn_object":
        # 1-NN computed only on the object dims (pos + quat).  The
        # complementary ablation to ``demo_1nn_proprio``.
        d_obj = _N_OBJECT_DIMS
        object_obs_norm = bank.obs_norm[:, :d_obj]

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            s0_norm = bank.normalize_obs(np.asarray(s0))[:d_obj]
            d2 = (
                (object_obs_norm - s0_norm[None, :]) ** 2
            ).sum(axis=-1)
            idx = int(np.argmin(d2))
            return bank.slice_actions(idx, H)

        return (
            initial_mean_fn,
            _make_pull_score_factory(),
            "demo_1nn_object",
        )

    if kind == "demo_softnn":
        # Soft 1-NN with temperature beta (A3, added 2026-05-20).
        # Recovers ``demo_1nn`` as beta -> inf and a uniform mean over
        # top-k as beta -> 0.  The intermediate regime gives a
        # continuous bias-variance knob; sweeping beta at fixed k
        # produces the bias-variance tradeoff curve directly.
        k = int(knn_k)
        beta = float(retrieval_beta)

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            s0_norm = bank.normalize_obs(np.asarray(s0))
            idxs = bank.k_nearest_indices(s0_norm, k)
            d2 = (
                (bank.obs_norm[idxs] - s0_norm[None, :]) ** 2
            ).sum(axis=-1)
            # Stable softmax with shift; treat beta=0 as uniform.
            if beta <= 0.0:
                w = np.full(len(idxs), 1.0 / len(idxs), dtype=np.float32)
            else:
                logits = (-beta * d2).astype(np.float64)
                logits -= logits.max()
                w = np.exp(logits)
                w = (w / w.sum()).astype(np.float32)
            window = np.stack(
                [bank.slice_actions(int(i), H) for i in idxs], axis=0
            )  # (k, H, A)
            return (
                w[:, None, None] * window
            ).sum(axis=0).astype(np.float32)

        # Encode beta in the label using a compact, filename-safe form.
        beta_tag = f"{beta:g}".replace(".", "p")
        return (
            initial_mean_fn,
            _make_pull_score_factory(),
            f"demo_softnn_k{k}_b{beta_tag}",
        )

    if kind == "demo_1nn_contrastive":
        # C1 (Slate I prep, 2026-05-21): retrieve via L2 in a
        # learned contrastive embedding space instead of raw
        # normalised obs.  The encoder was trained by
        # ``train_contrastive_encoder.py`` with InfoNCE on
        # temporally-adjacent obs pairs inside each demo, so
        # phase-equivalent states embed close together even when
        # their instantaneous proprio/object readings are nearly
        # identical (the Slate F/G Square hypothesis).  At
        # inference we project the entire bank through the encoder
        # once (cached), then 1-NN in embedding space.
        if contrastive_encoder_path is None:
            raise ValueError(
                "demo_1nn_contrastive requires --contrastive-encoder-path; "
                "train one first with train_contrastive_encoder.py"
            )
        import torch  # local import to keep torch optional for non-C1 priors

        ckpt_path = Path(contrastive_encoder_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"contrastive encoder checkpoint not found: {ckpt_path}"
            )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        input_dim = int(ckpt["input_dim"])
        if input_dim != bank.obs.shape[1]:
            raise ValueError(
                "contrastive encoder input_dim does not match bank obs "
                f"dim ({input_dim} vs {bank.obs.shape[1]}); did you "
                "change --action-history-len since training, or use a "
                "checkpoint trained on a different task?"
            )
        ckpt_ahl = int(ckpt.get("action_history_len", 0))
        if ckpt_ahl != int(bank.action_history_len):
            raise ValueError(
                "contrastive encoder was trained with "
                f"action_history_len={ckpt_ahl}, but the runtime bank "
                f"has action_history_len={bank.action_history_len}.  "
                "Retrain the encoder or rerun with the matching "
                "--action-history-len."
            )

        # Lazy import of the encoder module: keep the heavy torch
        # path out of the import graph for the other priors.
        from train_contrastive_encoder import ContrastiveEncoder

        encoder = ContrastiveEncoder(
            input_dim=input_dim,
            hidden_dim=int(ckpt["hidden_dim"]),
            embed_dim=int(ckpt["embed_dim"]),
        )
        encoder.load_state_dict(ckpt["state_dict"])
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        # Project the full bank to embedding space once.  At
        # bank.n_total ~ 30k and embed_dim = 32, this is 4 MB --
        # negligible compared to the bank itself.
        with torch.no_grad():
            obs_norm_t = torch.as_tensor(bank.obs_norm, dtype=torch.float32)
            bank_emb = encoder(obs_norm_t).cpu().numpy().astype(np.float32)
        # Embeddings are L2-normalised by the encoder, so squared
        # L2 distance ||a - b||^2 = 2 - 2 * (a . b); argmin on L2
        # is equivalent to argmax on cosine sim, which is in turn
        # equivalent to argmax on dot product after normalisation.
        # Keep L2 form for symmetry with the other priors.

        def initial_mean_fn(env, init_state, s0):
            del env, init_state
            s0_norm = bank.normalize_obs(np.asarray(s0))
            with torch.no_grad():
                q = encoder(
                    torch.as_tensor(s0_norm, dtype=torch.float32)[None, :]
                ).cpu().numpy().astype(np.float32)[0]
            d2 = ((bank_emb - q[None, :]) ** 2).sum(axis=-1)
            idx = int(np.argmin(d2))
            return bank.slice_actions(idx, H)

        return (
            initial_mean_fn,
            _make_pull_score_factory(),
            "demo_1nn_contrastive",
        )

    raise ValueError(
        f"unknown zero-train prior kind {kind!r}; expected one of "
        "demo_1nn, demo_knn, demo_random, demo_mean, demo_lwr, "
        "demo_topk_random, demo_1nn_proprio, demo_1nn_object, "
        "demo_softnn, demo_1nn_contrastive"
    )


# ---------------------------------------------------------------------------
# Smoke entry point: ``python priors_zerotrain.py`` prints bank stats.
# ---------------------------------------------------------------------------


def _smoke_main() -> None:
    from common import default_dataset_path  # type: ignore  # noqa: E402

    path = default_dataset_path("can")
    print(f"loading {path} ...")
    bank = load_demo_bank(path, n_demos=None)
    print(
        f"loaded {bank.n_demos} demos, {bank.n_total} timesteps, "
        f"obs_dim={bank.obs.shape[1]}, action_dim={bank.action_dim}"
    )
    print(
        f"demo length stats: min={bank.demo_lengths.min()}, "
        f"max={bank.demo_lengths.max()}, "
        f"mean={bank.demo_lengths.mean():.1f}"
    )

    # Round-trip each kind
    for kind in ("demo_1nn", "demo_knn", "demo_random", "demo_mean"):
        im, sf, label = make_zero_train_prior(
            kind=kind, bank=bank, H=120, knn_k=5, rng_seed=0
        )
        # Use a fake s0 = mean obs (in-distribution).
        s0 = bank.obs_mean.copy()
        out = im(None, None, s0)
        print(
            f"{label:20s}  warm-start shape={out.shape}  "
            f"mean|a|={np.abs(out).mean():.4f}  "
            f"first_step={out[0].round(3).tolist()}"
        )


if __name__ == "__main__":
    _smoke_main()
