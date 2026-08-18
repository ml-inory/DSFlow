import torch

from dsflow.audio import mel_spectrogram
from dsflow.config import MelConfig
from dsflow.vocoder import GriffinLimVocoder, HiFiGANGenerator


def test_hifigan_generator_shape():
    generator = HiFiGANGenerator(in_channels=20)
    out = generator(torch.randn(1, 20, 10))
    assert out.shape == (1, 1, 10 * 512)
    assert torch.isfinite(out).all()


def test_griffinlim_vocoder_roundtrip():
    cfg = MelConfig()
    seconds = 1.5
    t = torch.linspace(0, seconds, int(22050 * seconds))
    wave = 0.3 * torch.sin(2 * 3.14159 * 220 * t) + 0.2 * torch.sin(2 * 3.14159 * 440 * t)
    mel = mel_spectrogram(wave, cfg)
    vocoder = GriffinLimVocoder(device="cpu")
    out = vocoder(mel)
    assert out.dim() == 1 and out.numel() > 0
    assert torch.isfinite(out).all()
    assert abs(out.numel() - wave.numel()) / wave.numel() < 0.05
    assert vocoder.sample_rate == 22050
