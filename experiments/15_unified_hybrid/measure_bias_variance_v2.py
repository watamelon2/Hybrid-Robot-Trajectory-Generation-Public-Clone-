"""Per-step action (bias, variance) decomposition -- runnable version.

Follow-up C6 from the 2026-06-01 backlog research.  Promotes the
earlier scaffold (``measure_bias_variance.py``, marked DEFERRED to
a journal version since 2026-05-31) to a runnable consumer.

What changed since the scaffold
-------------------------------
The scaffold required an action-saving hook in
``unified_plan_and_execute`` that did not exist.  That hook IS now
landed on this branch (``--save-executed-actions``) -- see
``run_can.py`` and ``unified_loop.py``.  This v2 script therefore
no longer needs to *replay* anything; it reads the executed-actions
arrays directly out of the rollout JSONs produced by
``slate_C6_bias_var_capture_2026_06_01.sh``.

Inputs
------
A directory of rollout JSONs produced by ``run_can.py`` with
``--save-executed-actions``.  Each JSON's ``per_rollout[i].chunks[j].
executed_actions`` carries the actual env.step()-executed action
sequence for chunk ``j`` of rollout ``i``.

Methodology
-----------

Let:
  - i index init-state (1..N),
  - s index stochastic seed (1..S),
  - t index env step (1..H),
  - d index action dimension (1..A),
  - m index method (``hybrid`` for T2.1, ``bc`` for T2.3).

For each method, build a tensor a[m, i, s, t, d] of executed actions.
The reference action is the canonical (s=0) hybrid action at the
canonical mass = 1.0:

  a_ref[i, t, d] := a[hybrid_canonical, i, 0, t, d]

Per (m, t, d):
  bias[m, t, d]      = mean_{i, s}( a[m, i, s, t, d] - a_ref[i, t, d] )
  variance[m, t, d]  = var_{s}( a[m, :, s, t, d] ) averaged over i

Aggregate into scalars per method:
  bias_sq[m]         = mean_{t, d}( bias[m, t, d]^2 )
  variance[m]        = mean_{t, d}( variance[m, t, d] )

The Pareto theorem (docs/theory/08_asymmetric_robustness_model.md)
predicts:
  - T2.3 (BC retrained on 5 demos) has high variance and low bias --
    fits the perturbed demos but trains a fresh head per seed.
  - T2.1 (hybrid + 5-demo retrieval) has low variance and a small
    bias O(beta) under the small-beta asymptotic.
  - Both lie on the same theoretical Pareto frontier; T2.1 strictly
    dominates by virtue of lower variance at near-equal bias.

Inputs schema
-------------
We expect the slate to drop, under ``--results-dir``:

  ``hybrid_canonical_seed{s}.json``  s in 0..S-1   -- T2.1 hybrid eval at mass=1.0
  ``hybrid_t2_1_seed{s}.json``       s in 0..S-1   -- T2.1 hybrid eval at mass=1.3
  ``bc_t2_3_seed{s}.json``           s in 0..S-1   -- T2.3 BC eval at mass=1.3

(Or any other naming; the script matches the filename prefix to a
method label via the --method-prefix flag.)

Usage
-----

::

    python experiments/15_unified_hybrid/measure_bias_variance_v2.py \\
        --results-dir experiments/15_unified_hybrid/results/slate_C6_bias_var_2026_06_01 \\
        --method-prefix hybrid_canonical=ref \\
        --method-prefix hybrid_t2_1=t2_1_hybrid \\
        --method-prefix bc_t2_3=t2_3_bc \\
        --output-dir experiments/15_unified_hybrid/results/slate_C6_bias_var_2026_06_01/pareto

Outputs in ``--output-dir``:
  - ``bias_variance.json``  -- decomposed scalars per method
  - ``bias_variance.csv``   -- per (method, t, d) cells
  - ``bias_variance.md``    -- human-readable summary
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_rollout_actions(json_path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (actions, demo_ids).

    actions[i, t, d] = executed action at step t, dim d, rollout i.
    demo_ids[i]      = the rollout's ``demo`` key (used to align across files).

    Rollouts whose ``chunks`` do not contain ``executed_actions`` are
    skipped with a warning.  Rollouts with mismatched lengths are
    right-truncated to the shortest length seen in this file.
    """
    payload = json.loads(json_path.read_text())
    rollouts = payload.get("per_rollout", [])
    per_rollout: list[np.ndarray] = []
    demo_ids: list[str] = []
    for r in rollouts:
        chunks = r.get("chunks", [])
        a_seq = []
        for c in chunks:
            ea = c.get("executed_actions")
            if ea is None:
                continue
            a_seq.extend(ea)
        if not a_seq:
            continue
        per_rollout.append(np.asarray(a_seq, dtype=np.float64))
        demo_ids.append(str(r.get("demo", "?")))

    if not per_rollout:
        return np.zeros((0, 0, 0), dtype=np.float64), []

    min_len = min(a.shape[0] for a in per_rollout)
    action_dim = per_rollout[0].shape[1]
    aligned = np.stack([a[:min_len, :] for a in per_rollout], axis=0)
    # aligned: (n_rollouts, H, A)
    return aligned, demo_ids


def match_method(filename: str, prefix_map: dict[str, str]) -> str | None:
    """Map a filename to its method label using the --method-prefix table."""
    base = filename
    for prefix, method in prefix_map.items():
        if prefix in base:
            return method
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--results-dir", type=Path, required=True,
        help="Directory containing rollout JSONs with executed_actions.",
    )
    p.add_argument(
        "--method-prefix", action="append", default=[],
        help="Pattern: 'filename_prefix=method_label'.  May be repeated. "
             "First matching prefix wins.  One method must be labelled "
             "'ref' -- it provides the reference action a_ref.",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write bias_variance.{json,csv,md}.",
    )
    p.add_argument(
        "--max-steps", type=int, default=0,
        help="Optional cap on H (default 0 = use shortest available).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse --method-prefix entries.
    prefix_map: dict[str, str] = {}
    for spec in args.method_prefix:
        if "=" not in spec:
            raise SystemExit(
                f"--method-prefix '{spec}' missing '=' separator "
                "(expected 'filename_prefix=method_label')."
            )
        k, v = spec.split("=", 1)
        prefix_map[k] = v

    if "ref" not in prefix_map.values():
        raise SystemExit(
            "Exactly one --method-prefix must map to 'ref' (the "
            "canonical reference action used to compute bias)."
        )

    # Gather data: method -> list of (demo_ids, actions) per JSON.
    by_method: dict[str, list[tuple[list[str], np.ndarray]]] = defaultdict(list)
    for js in sorted(args.results_dir.glob("*.json")):
        if "summary" in js.name:
            continue
        method = match_method(js.name, prefix_map)
        if method is None:
            print(f"[bias-var] skip (no prefix match): {js.name}")
            continue
        actions, demo_ids = load_rollout_actions(js)
        if actions.size == 0:
            print(f"[bias-var] skip (no executed_actions in JSON): {js.name}")
            continue
        by_method[method].append((demo_ids, actions))
        print(
            f"[bias-var] loaded {js.name}  method={method}  "
            f"actions.shape={actions.shape}  demos={len(demo_ids)}"
        )

    if "ref" not in by_method:
        raise SystemExit(
            "No 'ref' rollout JSONs found.  Check --method-prefix mapping "
            "and that the slate produced files matching the 'ref' prefix."
        )

    # Build aligned tensor: method -> (n_seeds, n_demos, H, A).
    # Strategy: per method, treat each loaded JSON as one seed.  Inside
    # a JSON, align rollouts by demo_id (assume same set of init states
    # across seeds).
    def align_method(method: str) -> tuple[np.ndarray, list[str]]:
        per_seed: list[tuple[list[str], np.ndarray]] = by_method[method]
        if not per_seed:
            return np.zeros((0, 0, 0, 0)), []
        # canonical demo order from the first JSON
        ref_demos = per_seed[0][0]
        stacked = []
        for demo_ids, actions in per_seed:
            # Reorder this seed's rollouts to match ref_demos.
            idx_map = {d: i for i, d in enumerate(demo_ids)}
            present = [d for d in ref_demos if d in idx_map]
            if len(present) != len(ref_demos):
                missing = set(ref_demos) - set(present)
                print(
                    f"[bias-var] method={method}: dropping {len(missing)} "
                    f"init states absent from one of the seeds."
                )
                ref_demos = present
            stacked.append(
                np.stack([actions[idx_map[d]] for d in ref_demos], axis=0)
            )
        # right-align to the shortest H seen across all seeds for the method
        min_H = min(s.shape[1] for s in stacked)
        if args.max_steps > 0:
            min_H = min(min_H, int(args.max_steps))
        stacked = [s[:, :min_H, :] for s in stacked]
        tensor = np.stack(stacked, axis=0)  # (S, N, H, A)
        return tensor, ref_demos

    method_tensors: dict[str, tuple[np.ndarray, list[str]]] = {}
    for m in by_method:
        method_tensors[m] = align_method(m)
        S, N, H, A = method_tensors[m][0].shape
        print(f"[bias-var] method={m}  tensor shape (S, N, H, A) = ({S}, {N}, {H}, {A})")

    # Build reference action a_ref[i, t, d] = mean_{s}( ref_tensor[s, i, t, d] ).
    ref_tensor, ref_demos = method_tensors["ref"]
    if ref_tensor.size == 0:
        raise SystemExit("Reference tensor is empty; cannot compute bias.")
    a_ref = ref_tensor.mean(axis=0)  # (N, H, A)
    # Truncate every other method to (N, H, A) compatible with a_ref.
    N_ref, H_ref, A_ref = a_ref.shape

    # Per-method decomposition.
    results: dict[str, Any] = {"per_method": {}}
    csv_rows: list[dict[str, Any]] = []

    for m, (tensor, demos) in method_tensors.items():
        if m == "ref":
            continue
        S, N, H, A = tensor.shape
        # Align (N, H, A) to ref by demo id.
        idx_map = {d: i for i, d in enumerate(demos)}
        keep_demos = [d for d in ref_demos if d in idx_map]
        if not keep_demos:
            print(f"[bias-var] method={m}: no demos in common with ref; skip.")
            continue
        keep_idx = np.array([idx_map[d] for d in keep_demos], dtype=int)
        tensor = tensor[:, keep_idx, :, :]
        a_ref_m = a_ref[
            np.array([ref_demos.index(d) for d in keep_demos], dtype=int)
        ]
        N_eff = len(keep_demos)
        H_eff = min(H, H_ref)
        A_eff = min(A, A_ref)
        tensor = tensor[:, :, :H_eff, :A_eff]
        a_ref_m = a_ref_m[:, :H_eff, :A_eff]

        # mean over seeds, per (i, t, d): a_mean[m, i, t, d]
        a_mean = tensor.mean(axis=0)
        # Per-cell bias and variance.
        bias_cell = a_mean - a_ref_m        # (N, H, A)
        # Sample variance over seeds (ddof=1); use 0 if S<=1.
        if S <= 1:
            var_cell = np.zeros_like(a_ref_m)
        else:
            var_cell = tensor.var(axis=0, ddof=1)
        # Aggregate.
        bias_sq_mean = float((bias_cell ** 2).mean())
        var_mean = float(var_cell.mean())
        bias_norm = float(np.linalg.norm(bias_cell) / math.sqrt(bias_cell.size))
        results["per_method"][m] = {
            "n_seeds": int(S),
            "n_init_states": int(N_eff),
            "H": int(H_eff),
            "action_dim": int(A_eff),
            "bias_sq_mean": bias_sq_mean,
            "variance_mean": var_mean,
            "bias_rms": bias_norm,
        }
        # Per-step / per-dim CSV row.
        for t in range(H_eff):
            for d in range(A_eff):
                csv_rows.append({
                    "method": m,
                    "t": t,
                    "d": d,
                    "bias_mean_over_i": float(bias_cell[:, t, d].mean()),
                    "variance_over_s_mean_over_i": float(var_cell[:, t, d].mean()),
                })
        print(
            f"[bias-var] method={m}  bias^2={bias_sq_mean:.4e}  "
            f"variance={var_mean:.4e}  bias_rms={bias_norm:.4e}"
        )

    # Pareto-dominance verdict.
    methods_sorted = list(results["per_method"].keys())
    verdicts: list[str] = []
    for i, a in enumerate(methods_sorted):
        for b in methods_sorted[i + 1:]:
            ra = results["per_method"][a]
            rb = results["per_method"][b]
            a_dom_b = (ra["bias_sq_mean"] <= rb["bias_sq_mean"]
                       and ra["variance_mean"] <= rb["variance_mean"])
            b_dom_a = (rb["bias_sq_mean"] <= ra["bias_sq_mean"]
                       and rb["variance_mean"] <= ra["variance_mean"])
            if a_dom_b and not b_dom_a:
                verdicts.append(f"{a} Pareto-dominates {b}")
            elif b_dom_a and not a_dom_b:
                verdicts.append(f"{b} Pareto-dominates {a}")
            else:
                verdicts.append(f"{a} and {b} are Pareto-incomparable")
    results["pareto_verdicts"] = verdicts

    # Write outputs.
    (args.output_dir / "bias_variance.json").write_text(
        json.dumps(results, indent=2)
    )
    csv_path = args.output_dir / "bias_variance.csv"
    if csv_rows:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    md_path = args.output_dir / "bias_variance.md"
    with md_path.open("w") as fh:
        fh.write("# Per-step (bias, variance) decomposition\n\n")
        fh.write("Generated by `measure_bias_variance_v2.py` (C6, 2026-06-01).\n\n")
        fh.write("| method | n_seeds | n_init | H | A | bias\u00b2 (mean) | variance (mean) | bias_rms |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for m, r in results["per_method"].items():
            fh.write(
                f"| {m} | {r['n_seeds']} | {r['n_init_states']} | "
                f"{r['H']} | {r['action_dim']} | "
                f"{r['bias_sq_mean']:.4e} | {r['variance_mean']:.4e} | "
                f"{r['bias_rms']:.4e} |\n"
            )
        fh.write("\n## Pareto verdicts\n\n")
        for v in verdicts:
            fh.write(f"- {v}\n")
    print(f"[bias-var] wrote {args.output_dir/'bias_variance.json'}")
    print(f"[bias-var] wrote {csv_path}")
    print(f"[bias-var] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
