"""V_phi state-value baseline model.

A small MLP that regresses cumulative future cost (cost-to-go) from
the BC OBS_KEYS-ordered observation. Used as a control variate inside
the MBD denoise loop: per-trajectory cost ``c_i`` becomes
``c_i - sum_t V_phi(s_{i,t})``, which reduces the variance of MBD's
softmax-over-cost weighting without changing its expectation. This
is the standard REINFORCE / A2C / PPO baseline trick applied to MBD.

Architecture mirrors the BC policy (``experiments/03_bc_baseline/policy.py``):
2-layer MLP, hidden 256, ReLU, scalar output. Inputs are normalised
with the BC policy's ``(obs_mean, obs_std)``. Sharing the normalisation
keeps the train / eval feature axes aligned with everything else in
the project. We do NOT share the ``MLPPolicy`` class because the head
is different (scalar value vs tanh-squashed 7-D action) and we want
a clean separate save/load surface for V_phi.pt.

Checkpoint format (single ``torch.save`` blob)::

    {
        "format_version": 1,
        "config": {"obs_dim": 59, "hidden_sizes": [256, 256], "obs_keys": [...]},
        "state_dict": {...},
        "obs_mean": np.ndarray (obs_dim,),
        "obs_std":  np.ndarray (obs_dim,),
        "extra":    {"dataset": ..., "epochs": ..., "seed": ..., "n_trajs": ...},
    }
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Reuse exp 03's OBS_KEYS and ObsStats so V_phi.pt and policy.pt
# carry the same observation convention. The 'dataset' module lives
# under experiments/03_bc_baseline/.
_BC_DIR = (Path(__file__).resolve().parent.parent / "03_bc_baseline")
if str(_BC_DIR) not in sys.path:
    sys.path.insert(0, str(_BC_DIR))

from dataset import OBS_KEYS, ObsStats  # noqa: E402


@dataclass(frozen=True)
class ValueConfig:
    obs_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    obs_keys: tuple[str, ...] = OBS_KEYS

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "hidden_sizes": list(self.hidden_sizes),
            "obs_keys": list(self.obs_keys),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValueConfig":
        return cls(
            obs_dim=int(d["obs_dim"]),
            hidden_sizes=tuple(int(h) for h in d["hidden_sizes"]),
            obs_keys=tuple(str(k) for k in d["obs_keys"]),
        )


class MLPValue(nn.Module):
    """Standard fully-connected scalar value head."""

    def __init__(self, config: ValueConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.obs_dim
        for h in config.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
        # Returns shape (..., ) — squeezes trailing 1-dim.
        return self.net(obs_norm).squeeze(-1)


class ValuePhi:
    """High-level wrapper bundling MLPValue + obs normalisation.

    Use ``ValuePhi.load(path)`` at planner-runtime; use ``.value(obs)``
    or ``.value_batch(obs_batch)`` to evaluate. The wrapper handles
    numpy <-> torch conversion and ObsStats normalisation so callers
    can stay in numpy.
    """

    def __init__(
        self,
        net: MLPValue,
        stats: ObsStats,
        device: str | torch.device = "cpu",
    ):
        self.net = net.to(device)
        self.net.eval()
        self.stats = stats
        self.device = torch.device(device)

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        """Single-state value query."""
        obs_norm = self.stats.normalize(obs.astype(np.float32))
        t = torch.from_numpy(np.asarray(obs_norm, dtype=np.float32)).to(self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        v = self.net(t).cpu().numpy()
        return float(v.reshape(-1)[0])

    @torch.no_grad()
    def value_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Batched value query. ``obs_batch`` shape ``(..., obs_dim)`` ->
        scalar field with leading dims preserved.

        This is the hot path during planning: called once per denoising
        iteration with shape ``(M, H+1, obs_dim)`` -> ``(M, H+1)``,
        then summed along the time axis to produce the per-trajectory
        control variate.
        """
        obs_norm = self.stats.normalize(obs_batch.astype(np.float32))
        t = torch.from_numpy(np.asarray(obs_norm, dtype=np.float32)).to(self.device)
        v = self.net(t).cpu().numpy()
        return v

    @staticmethod
    def save(
        path: Path,
        net: MLPValue,
        stats: ObsStats,
        extra: dict[str, Any] | None = None,
    ) -> None:
        blob = {
            "format_version": 1,
            "config": net.config.to_dict(),
            "state_dict": net.state_dict(),
            "obs_mean": np.asarray(stats.mean, dtype=np.float32),
            "obs_std": np.asarray(stats.std, dtype=np.float32),
            "extra": dict(extra or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)

    @classmethod
    def load(
        cls,
        path: Path,
        device: str | torch.device = "cpu",
    ) -> "ValuePhi":
        blob = torch.load(path, map_location=device, weights_only=False)
        config = ValueConfig.from_dict(blob["config"])
        net = MLPValue(config)
        net.load_state_dict(blob["state_dict"])
        stats = ObsStats(
            mean=np.asarray(blob["obs_mean"], dtype=np.float32),
            std=np.asarray(blob["obs_std"], dtype=np.float32),
        )
        return cls(net, stats, device=device)
