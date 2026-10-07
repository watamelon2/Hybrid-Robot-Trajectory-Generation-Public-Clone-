"""Tiny MLP policy + checkpoint format used by both train.py and eval.py.

The architecture is deliberately minimal: two hidden layers of 256
units, ReLU activations, and a tanh-squashed 7-dim output to match the
robosuite OSC_POSE + gripper action range of [-1, 1]. With the can/ph
dataset (~23k transitions) and an MLP this small, training fits in a
few minutes on a Mac CPU.

A checkpoint is a single ``torch.save`` blob (a plain ``dict``) so the
file is loadable without depending on any class definitions outside
this module. ``Policy.from_checkpoint`` re-hydrates the model and the
obs-normalisation stats together so eval-time pre-processing is
guaranteed to match training.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# This package's directory name starts with a digit so it cannot be
# imported via the dotted-package syntax. The entry scripts (train.py,
# eval.py) add this directory to sys.path; do the same here so policy.py
# is also importable from a plain "import policy" inside this folder.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset import OBS_KEYS, ObsStats  # noqa: E402


@dataclass(frozen=True)
class PolicyConfig:
    obs_dim: int
    action_dim: int = 7
    hidden_sizes: tuple[int, ...] = (256, 256)
    obs_keys: tuple[str, ...] = OBS_KEYS

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": list(self.hidden_sizes),
            "obs_keys": list(self.obs_keys),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyConfig:
        return cls(
            obs_dim=int(d["obs_dim"]),
            action_dim=int(d["action_dim"]),
            hidden_sizes=tuple(int(h) for h in d["hidden_sizes"]),
            obs_keys=tuple(str(k) for k in d["obs_keys"]),
        )


class MLPPolicy(nn.Module):
    """Standard fully-connected BC policy with a tanh-squashed head."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.obs_dim
        for h in config.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.action_dim))
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.trunk(x))


class Policy:
    """Eval-time wrapper bundling the network and obs normalisation."""

    def __init__(
        self,
        net: MLPPolicy,
        stats: ObsStats,
        device: str | torch.device = "cpu",
    ):
        self.net = net.to(device).eval()
        self.stats = stats
        self.device = torch.device(device)

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray) -> np.ndarray:
        """Return a single action for a single flat observation."""
        x_norm = self.stats.normalize(obs_vec.astype(np.float32))
        x = torch.from_numpy(x_norm).to(self.device).unsqueeze(0)
        a = self.net(x).cpu().numpy()[0]
        return np.asarray(a, dtype=np.float64)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def save(
        path: Path,
        net: MLPPolicy,
        stats: ObsStats,
        extra: dict[str, Any] | None = None,
    ) -> None:
        blob: dict[str, Any] = {
            "format_version": 1,
            "config": net.config.to_dict(),
            "state_dict": net.state_dict(),
            "obs_mean": stats.mean.astype(np.float32),
            "obs_std": stats.std.astype(np.float32),
        }
        if extra:
            blob["extra"] = extra
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: str | torch.device = "cpu",
    ) -> Policy:
        blob = torch.load(path, map_location=device, weights_only=False)
        config = PolicyConfig.from_dict(blob["config"])
        net = MLPPolicy(config)
        net.load_state_dict(blob["state_dict"])
        stats = ObsStats(
            mean=np.asarray(blob["obs_mean"], dtype=np.float32),
            std=np.asarray(blob["obs_std"], dtype=np.float32),
        )
        return cls(net, stats, device=device)
