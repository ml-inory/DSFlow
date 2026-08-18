"""DSFlow: dual-supervision + step-aware flow matching TTS model."""

from __future__ import annotations

import torch
from torch import nn

from dsflow.model.decoder import StepAwareDecoder
from dsflow.model.duration import DurationPredictor
from dsflow.model.modules import expand_by_durations, mel_mask, text_mask
from dsflow.model.text_encoder import TextEncoder


class DSFlow(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int,
        text_dim: int = 256,
        text_layers: int = 4,
        text_heads: int = 4,
        duration_dim: int = 256,
        duration_layers: int = 2,
        decoder_dim: int = 512,
        decoder_layers: int = 8,
        decoder_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, text_dim, text_layers, text_heads, ff_mult, dropout)
        self.duration_predictor = DurationPredictor(text_dim, duration_dim, duration_layers)
        self.text_proj = nn.Linear(text_dim, decoder_dim)
        self.decoder = StepAwareDecoder(n_mels, decoder_dim, decoder_layers, decoder_heads, ff_mult, dropout)

    def forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        mel: torch.Tensor,
        t: torch.Tensor,
        durations: torch.Tensor | None = None,
        mel_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (v_pred, x0_pred, log_dur_pred, text_mask)."""
        tmask = text_mask(text, text_len)
        h_text = self.text_encoder(text, tmask)
        log_dur = self.duration_predictor(h_text, tmask)
        if durations is None:
            # Inference: synthesize with predicted durations directly.
            durations = log_dur.exp().round().long().clamp(min=1)
        text_feats = self.text_proj(h_text)
        text_feats, text_feats_mask = expand_by_durations(text_feats, durations, mel.size(-1))
        if mel_len is None:
            mel_len = torch.full((mel.size(0),), mel.size(-1), dtype=torch.long, device=mel.device)
        mmask = mel_mask(mel_len, mel.size(-1))
        # expand_by_durations already constrains cross-attention to valid frames.
        v, x0 = self.decoder(mel, t, text_feats, text_feats_mask, mmask)
        return v, x0, log_dur, tmask
