"""Central configuration dataclasses for DSFlow."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0


@dataclass
class DataConfig:
    data_root: str = "data"
    ljspeech_url: str = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    ljspeech_dir: str = "data/LJSpeech-1.1"
    cache_dir: str = "data/cache"
    mel: MelConfig = field(default_factory=MelConfig)
    min_seconds: float = 1.0
    max_seconds: float = 12.0
    use_phonemes: bool = True


@dataclass
class ModelConfig:
    vocab_size: int = 128
    text_dim: int = 256
    text_layers: int = 4
    text_heads: int = 4
    duration_dim: int = 256
    duration_layers: int = 2
    decoder_dim: int = 512
    decoder_layers: int = 8
    decoder_heads: int = 8
    ff_mult: int = 4
    max_seq_len: int = 4096
    step_dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    steps: int = 2000
    warmup_steps: int = 100
    log_every: int = 50
    ckpt_every: int = 500
    ckpt_dir: str = "checkpoints"
    seed: int = 0
    device: str = "cuda"
    # Dual-supervision weighting: lambda_direct controls the direct one-step loss.
    lambda_cfm: float = 1.0
    lambda_direct: float = 1.0


@dataclass
class InferConfig:
    ckpt_path: str = "checkpoints/last.pt"
    output_dir: str = "outputs"
    steps: int = 1
    cfg_strength: float = 0.0
    max_mel_len: int = 2048
