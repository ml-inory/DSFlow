import torch

from dsflow.audio import mel_spectrogram
from dsflow.config import MelConfig


def test_mel_shape_and_values():
    cfg = MelConfig(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80)
    wave = torch.randn(22050)  # 1 second
    mel = mel_spectrogram(wave, cfg)
    assert mel.dim() == 2
    assert mel.size(0) == cfg.n_mels
    assert mel.size(1) == 87  # floor(22050/256) + 1
    assert torch.isfinite(mel).all()
    assert mel.dtype == torch.float32


def test_mel_deterministic():
    cfg = MelConfig()
    wave = torch.randn(44100)
    a = mel_spectrogram(wave, cfg)
    b = mel_spectrogram(wave, cfg)
    assert torch.equal(a, b)
