"""PyTorch dataset over preprocessed records with proportional text-to-mel alignment."""

from __future__ import annotations

from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dsflow.text import PAD

MEL_PAD = -10.0  # log10 magnitude of near-silence used to pad mel frames


def proportional_durations(n_tokens: int, n_frames: int) -> torch.Tensor:
    """Distribute mel frames across tokens as evenly as possible."""
    if n_tokens <= 0 or n_frames <= 0:
        raise ValueError("tokens and frames must be positive")
    base, rem = divmod(n_frames, n_tokens)
    durations = torch.full((n_tokens,), base, dtype=torch.long)
    if rem:
        durations[:rem] += 1
    return durations


class MelDataset(Dataset):
    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        mel = torch.load(rec["mel_path"], weights_only=True)["mel"].float()
        tokens = torch.tensor(rec["tokens"], dtype=torch.long)
        durations = proportional_durations(tokens.numel(), mel.size(-1))
        return {"text": tokens, "durations": durations, "mel": mel}


def collate_mel(batch: List[dict]) -> dict:
    text = pad_sequence([b["text"] for b in batch], batch_first=True, padding_value=PAD)
    mel = pad_sequence([b["mel"].transpose(0, 1) for b in batch], batch_first=True, padding_value=MEL_PAD)
    mel = mel.transpose(1, 2)  # [B, M, T]
    durations = pad_sequence([b["durations"] for b in batch], batch_first=True, padding_value=0)
    text_len = torch.tensor([b["text"].numel() for b in batch])
    mel_len = torch.tensor([b["mel"].size(-1) for b in batch])
    return {
        "text": text,
        "durations": durations,
        "mel": mel,
        "text_len": text_len,
        "mel_len": mel_len,
    }
