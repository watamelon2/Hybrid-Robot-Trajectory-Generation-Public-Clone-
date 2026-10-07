"""Unit tests for the demo-phase pseudo-cost core (demo_phase_reward.py).

Pure-numpy; no robosuite / no dataset.  Covers the two functions that
carry the logic -- ``_phase_from_bank`` (per-row phase) and
``_nearest_phase`` (nearest-demo-step lookup) -- backing doc 09 §7.1
route 3 / follow-up #3.

Run: ``python -m pytest experiments/15_unified_hybrid/test_demo_phase_reward.py``
"""

from __future__ import annotations

import numpy as np

from demo_phase_reward import (
    _nearest_phase,
    _obs_vec_from_dict,
    _phase_from_bank,
)


def test_phase_is_zero_to_one_per_demo():
    # Two demos: lengths 3 and 4, concatenated -> 7 rows.
    starts = np.array([0, 3])
    lengths = np.array([3, 4])
    phase = _phase_from_bank(starts, lengths, 7)
    assert np.allclose(phase[:3], [0.0, 0.5, 1.0])
    assert np.allclose(phase[3:], [0.0, 1 / 3, 2 / 3, 1.0])


def test_phase_singleton_demo():
    phase = _phase_from_bank(np.array([0]), np.array([1]), 1)
    assert phase[0] == 1.0


def test_nearest_phase_picks_closest_row():
    # 1-D feature; bank rows at 0,1,2 with phases 0,0.5,1.
    bank = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    mean = np.array([0.0], dtype=np.float32)
    std = np.array([1.0], dtype=np.float32)
    phase = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    bank_norm = (bank - mean) / std
    # query near row 2 -> phase 1.0
    assert _nearest_phase(np.array([1.9]), bank_norm, phase, mean, std) == 1.0
    # query near row 0 -> phase 0.0
    assert _nearest_phase(np.array([0.1]), bank_norm, phase, mean, std) == 0.0


def test_nearest_phase_uses_normalisation():
    # Without normalisation dim-1 (huge scale) would dominate; with the
    # std normalisation both dims count equally so the second feature
    # (the one that actually separates the rows) wins the match.
    bank = np.array([[0.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    mean = np.array([0.0, 50.0], dtype=np.float32)
    std = np.array([1.0, 50.0], dtype=np.float32)
    phase = np.array([0.0, 1.0], dtype=np.float32)
    bank_norm = (bank - mean) / std
    # query closest to row 1 in feature-2 -> phase 1.0
    assert _nearest_phase(
        np.array([0.0, 90.0]), bank_norm, phase, mean, std
    ) == 1.0


def test_obs_vec_concat_order():
    obs = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}
    v = _obs_vec_from_dict(obs, ("b", "a"))
    assert np.allclose(v, [3.0, 1.0, 2.0])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
