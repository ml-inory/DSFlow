"""Audio I/O and a torch-based mel-spectrogram front end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taf

from dsflow.config import MelConfig


@torch.inference_mode()
def load_wav(path, sample_rate: int) -> torch.Tensor:
    """Load a mono float32 waveform at *sample_rate*."""
    # soundfile I/O avoids torchaudio's optional torchcodec backend.
    wave, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    wave = torch.from_numpy(wave.T)  # [C, T]
    if wave.size(0) > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wave = taf.resample(wave, sr, sample_rate)
    return wave.squeeze(0).float()


@torch.inference_mode()
def mel_spectrogram(wave: torch.Tensor, cfg: MelConfig) -> torch.Tensor:
    """Compute log10 mel power spectrogram of shape [n_mels, T]."""
    wav = wave.float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    window = torch.hann_window(cfg.win_length).to(wav.device)
    spec = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spec.abs().pow(2)  # [1, F, T]
    fbanks = taf.melscale_fbanks(
        n_freqs=cfg.n_fft // 2 + 1,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        n_mels=cfg.n_mels,
        sample_rate=cfg.sample_rate,
    ).to(wav.device)
    mel = torch.matmul(fbanks.T, power)  # [1, M, T]
    return torch.log10(torch.clamp(mel, min=1e-10)).squeeze(0)


def save_wav(path, wave: torch.Tensor, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.float().cpu().numpy(), sample_rate)
