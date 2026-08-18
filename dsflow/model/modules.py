"""Shared building blocks: sinusoidal embeddings, MLPs, duration expansion."""

from __future__ import annotations

import math

import torch
from torch import nn

from dsflow.text import PAD


def sinusoidal_embedding(t: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embeddings; *t* can be scalar per row or a position grid."""
    half = dim // 2
    freqs = torch.exp(-math.log(theta) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.unsqueeze(-1).float() * freqs  # [..., half]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def positions_grid(length: int, dim: int, device) -> torch.Tensor:
    return sinusoidal_embedding(torch.arange(length, device=device), dim)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def expand_by_durations(
    hidden: torch.Tensor, durations: torch.Tensor, max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each token's hidden vector according to its duration.

    Returns (expanded [B, max_len, D], valid mask [B, max_len]).
    """
    B, Tt, D = hidden.shape
    device = hidden.device
    starts = torch.cumsum(durations, dim=1) - durations  # [B, Tt]
    grid = torch.arange(max_len, device=device).unsqueeze(0).expand(B, max_len)  # [B, L]
    token_idx = (grid.unsqueeze(-1) >= starts.unsqueeze(1)).sum(-1) - 1  # [B, L]
    token_idx = token_idx.clamp(min=0, max=Tt - 1)
    expanded = torch.gather(hidden, 1, token_idx.unsqueeze(-1).expand(B, max_len, D))
    valid = grid < durations.sum(dim=1, keepdim=True)
    return expanded, valid


def text_mask(text: torch.Tensor, text_len: torch.Tensor) -> torch.Tensor:
    """Valid-position mask [B, Tt] for token ids (padding_id == PAD)."""
    return torch.arange(text.size(1), device=text.device).unsqueeze(0) < text_len.unsqueeze(1)


def mel_mask(mel_len: torch.Tensor, max_len: int) -> torch.Tensor:
    """Valid-frame mask [B, T] for mel features."""
    return torch.arange(max_len, device=mel_len.device).unsqueeze(0) < mel_len.unsqueeze(1)
