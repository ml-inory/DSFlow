"""Step-aware DiT-style decoder for one-step flow matching."""

from __future__ import annotations

import torch
from torch import nn

from dsflow.model.modules import MLP, positions_grid, sinusoidal_embedding


class StepAwareBlock(nn.Module):
    """Transformer block with AdaLN time conditioning and a scalar step gate.

    The scalar step gate (sigmoid of a learned projection of the time embedding)
    is the "step-aware" residual blend: at t=1 the model learns to route a
    maximal correction, matching the one-step inference distribution.
    """

    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = MLP(dim, ff_mult, dropout=dropout)
        self.modulation = nn.Linear(dim, 6 * dim)
        self.step_gate = nn.Linear(dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        text_feats: torch.Tensor,
        text_mask: torch.Tensor,
        mel_mask: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.modulation(time_emb)  # [B, 6*D]
        g1, b1, g2, b2, g3, b3 = mod.chunk(6, dim=-1)
        gate = torch.sigmoid(self.step_gate(time_emb))  # [B, 1]

        h = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        h = self.attn(h, h, h, key_padding_mask=~mel_mask, need_weights=False)[0]
        h = x + h

        h2 = self.norm2(h) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        h2 = self.cross(h2, text_feats, text_feats, key_padding_mask=~text_mask, need_weights=False)[0]
        h2 = h + h2

        h3 = self.norm3(h2) * (1 + g3.unsqueeze(1)) + b3.unsqueeze(1)
        h3 = self.ff(h3)
        h3 = h2 + h3
        return x + gate.unsqueeze(1) * (h3 - x)


class StepAwareDecoder(nn.Module):
    """DiT-style mel decoder conditioned on time step and text.

    Dual output heads share the body: one predicts the flow velocity v
    (CFM supervision), the other predicts the clean data x0 directly
    (direct one-step supervision).
    """

    def __init__(self, n_mels: int, dim: int, layers: int, heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(n_mels, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.blocks = nn.ModuleList(
            [StepAwareBlock(dim, heads, ff_mult, dropout) for _ in range(layers)]
        )
        self.norm_out = nn.LayerNorm(dim)
        self.v_head = nn.Linear(dim, n_mels)
        self.x0_head = nn.Linear(dim, n_mels)

    def forward(
        self,
        mel: torch.Tensor,
        t: torch.Tensor,
        text_feats: torch.Tensor,
        text_mask: torch.Tensor,
        mel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, T = mel.shape
        x = self.in_proj(mel.transpose(1, 2))  # [B, T, D]
        x = x + positions_grid(T, self.dim, mel.device)
        time_emb = self.time_mlp(sinusoidal_embedding(t, self.dim))  # [B, D]
        for block in self.blocks:
            x = block(x, time_emb, text_feats, text_mask, mel_mask)
        x = self.norm_out(x)
        v = self.v_head(x).transpose(1, 2)  # [B, M, T]
        x0 = self.x0_head(x).transpose(1, 2)
        return v, x0
