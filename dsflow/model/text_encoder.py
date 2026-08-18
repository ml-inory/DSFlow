"""Transformer text encoder with fixed sinusoidal positions."""

from __future__ import annotations

import torch
from torch import nn

from dsflow.model.modules import positions_grid


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, text: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode token ids [B, Tt] -> [B, Tt, D]; *mask* marks valid tokens."""
        x = self.embedding(text)
        pos = positions_grid(text.size(1), self.dim, text.device)
        x = x + pos
        return self.encoder(x, src_key_padding_mask=~mask)
