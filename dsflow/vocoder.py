"""Vocoders: cached HiFi-GAN (jaketae/hifigan-lj-v1) with a Griffin-Lim fallback.

Both vocoders consume the same log10 magnitude mel [n_mels, T] produced by
``dsflow.audio.mel_spectrogram``.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1

HFG_REPO_URL = "https://hf-mirror.com/jaketae/hifigan-lj-v1/resolve/main/pytorch_model.bin"
HFG_FILENAME = "hifigan-lj-v1.bin"


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    """Multi-receptive-field block from the official HiFi-GAN implementation."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=_get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))
                )
                for _ in dilation
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = F.leaky_relu(conv1(xt), LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in (*self.convs1, *self.convs2):
            remove_weight_norm(conv)


class HiFiGANGenerator(nn.Module):
    """Standard jik876 HiFi-GAN v1 generator (state-dict compatible)."""

    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates=(8, 8, 2, 4),
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList(
            [
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
            ]
        )
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for k, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(channels, k, dilation))
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dsflow/0.1"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(256 * 1024):
            fh.write(chunk)


class HiFiGANVocoder:
    """HiFi-GAN vocoder for LJSpeech-style 80-band log-mel.

    The checkpoint is downloaded once from HuggingFace (hf-mirror) and cached
    under *cache_dir*. Its native output rate is 44.1 kHz (upsample x512 on a
    86.1 frames/s mel), so waveforms are resampled to *sample_rate*.
    """

    def __init__(self, cache_dir: str = "data/vocoder", device: str = "cuda", sample_rate: int = 22050):
        cache_dir = Path(cache_dir)
        ckpt_path = cache_dir / HFG_FILENAME
        if not ckpt_path.exists():
            _download(HFG_REPO_URL, ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        generator = HiFiGANGenerator()
        generator.load_state_dict(state)
        generator.eval().to(device)
        generator.remove_weight_norm()
        self.generator = generator
        self.device = device
        self.sample_rate = sample_rate
        self._native_rate = 44100

    @torch.inference_mode()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize [n_mels, T] log-mel into a mono waveform at *sample_rate*."""
        wav = self.generator(mel.unsqueeze(0).to(self.device)).squeeze()
        if self._native_rate != self.sample_rate:
            wav = F.interpolate(wav.view(1, 1, -1), scale_factor=self.sample_rate / self._native_rate, mode="linear").view(-1)
        return wav.cpu()


class GriffinLimVocoder:
    """Offline Griffin-Lim vocoder (torchaudio) for the same 80-band log-mel."""

    def __init__(self, device: str = "cuda"):
        from torchaudio.pipelines import TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH

        self.vocoder = TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH.get_vocoder().to(device)
        self.device = device
        self.sample_rate = 22050

    @torch.inference_mode()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize [n_mels, T] log10-mel into a mono 22.05 kHz waveform."""
        log_mel = (mel.to(self.device) * math.log(10)).unsqueeze(0)  # natural-log for torchaudio
        wav, _ = self.vocoder(log_mel)
        return wav.squeeze(0).cpu()


def build_vocoder(cache_dir: str = "data/vocoder", device: str = "cuda"):
    """Prefer the cached HiFi-GAN, fall back to Griffin-Lim."""
    try:
        return HiFiGANVocoder(cache_dir=cache_dir, device=device)
    except Exception as exc:  # pragma: no cover - exercised only on download failures
        print(f"[vocoder] HiFi-GAN unavailable ({exc}); using Griffin-Lim")
        return GriffinLimVocoder(device=device)
