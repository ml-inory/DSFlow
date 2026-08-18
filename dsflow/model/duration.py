"""Duration predictor: text hidden -> per-token log durations (in mel frames)."""

from __future__ import annotations

import torch
from torch import nn


class DurationPredictor(nn.Module):
    def __init__(self, text_dim: int, hidden: int, layers: int):
        super().__init__()
        blocks = []
        in_dim = text_dim
        for _ in range(layers):
            blocks.append(nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden)))
            in_dim = hidden
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, text_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict log durations [B, Tt]; padded positions are zeroed."""
        log_dur = self.net(text_feats).squeeze(-1)
        return torch.where(mask, log_dur, torch.zeros_like(log_dur))
