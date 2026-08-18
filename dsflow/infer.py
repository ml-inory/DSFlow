"""One-step / multi-step inference from text to mel and waveform."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from dsflow.audio import save_wav
from dsflow.config import MelConfig, ModelConfig
from dsflow.model import DSFlow
from dsflow.model.modules import text_mask
from dsflow.text import TextTokenizer
from dsflow.vocoder import build_vocoder


def build_model_from_config(model_cfg: ModelConfig, n_mels: int) -> DSFlow:
    return DSFlow(
        vocab_size=model_cfg.vocab_size,
        n_mels=n_mels,
        text_dim=model_cfg.text_dim,
        text_layers=model_cfg.text_layers,
        text_heads=model_cfg.text_heads,
        duration_dim=model_cfg.duration_dim,
        duration_layers=model_cfg.duration_layers,
        decoder_dim=model_cfg.decoder_dim,
        decoder_layers=model_cfg.decoder_layers,
        decoder_heads=model_cfg.decoder_heads,
        ff_mult=model_cfg.ff_mult,
        dropout=model_cfg.dropout,
    )


def load_checkpoint(ckpt_path: str, device: str) -> tuple[DSFlow, TextTokenizer, MelConfig]:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_cfg = ModelConfig(**state["model_config"])
    mel_cfg = MelConfig(**state["mel_config"])
    model = build_model_from_config(model_cfg, mel_cfg.n_mels)
    model.load_state_dict(state["model"])
    model.eval().to(device)
    return model, TextTokenizer(state["vocab"]), mel_cfg


@torch.inference_mode()
def predict_durations(
    model: DSFlow, tokenizer: TextTokenizer, text: str, device: str, duration_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode text and predict per-token durations (mel frames)."""
    ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    text_len = torch.tensor([ids.size(1)], dtype=torch.long, device=device)
    mask = text_mask(ids, text_len)
    hidden = model.text_encoder(ids, mask)
    log_dur = model.duration_predictor(hidden, mask)
    durations = (log_dur.exp() * duration_scale).round().long().clamp(min=1)
    return ids, text_len, durations


@torch.inference_mode()
def synthesize_mel(
    model: DSFlow,
    tokenizer: TextTokenizer,
    text: str,
    device: str,
    steps: int = 1,
    duration_scale: float = 1.0,
    max_mel_len: int = 4096,
) -> torch.Tensor:
    """Flow-match from noise to mel in *steps* Euler steps (1 = one-step via x0 head)."""
    ids, text_len, durations = predict_durations(model, tokenizer, text, device, duration_scale)
    n_mels = model.decoder.v_head.out_features
    total = int(durations.sum().item())
    if total > max_mel_len:
        scale = max_mel_len / total
        durations = (durations.float() * scale).floor().long().clamp(min=1)
        durations[-1] += max_mel_len - int(durations.sum().item())
        total = max_mel_len
    x = torch.randn(1, n_mels, total, device=device)
    mel_len = torch.tensor([total], dtype=torch.long, device=device)

    if steps == 1:
        _, x0, _, _ = model(ids, text_len, x, torch.ones(1, device=device), durations, mel_len)
        return x0.squeeze(0)

    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((1,), 1.0 - i * dt, device=device)
        v, _, _, _ = model(ids, text_len, x, t, durations, mel_len)
        x = x - dt * v
    return x.squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize speech from text with DSFlow")
    parser.add_argument("--ckpt", required=True, help="path to a DSFlow checkpoint")
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--out", default="outputs/out.wav")
    parser.add_argument("--steps", type=int, default=1, help="flow steps (1 = one-step)")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocoder-cache", default="data/vocoder")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_checkpoint(args.ckpt, device)
    mel = synthesize_mel(model, tokenizer, args.text, device, args.steps, args.duration_scale)
    vocoder = build_vocoder(cache_dir=args.vocoder_cache, device=device)
    wav = vocoder(mel)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(out_path, wav, vocoder.sample_rate)
    torch.save({"mel": mel}, out_path.with_suffix(".mel.pt"))
    print(f"[infer] steps={args.steps} mel_frames={mel.size(-1)} wav_samples={wav.numel()} -> {out_path}")


if __name__ == "__main__":
    main()
