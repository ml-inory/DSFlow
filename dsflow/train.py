"""Training loop for DSFlow with step dropout and dual-supervision losses."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsflow.config import DataConfig, MelConfig, ModelConfig, TrainConfig
from dsflow.data import MelDataset, collate_mel, preprocess_ljspeech
from dsflow.losses import dual_supervision_loss, duration_loss
from dsflow.model import DSFlow
from dsflow.text import TextTokenizer


def metadata_texts(data_cfg: DataConfig) -> list[str]:
    lj_dir = Path(data_cfg.ljspeech_dir)
    if not lj_dir.exists():
        return []
    lines = [ln.strip() for ln in (lj_dir / "metadata.csv").read_text().splitlines() if ln.strip()]
    return [line.split("|", 2)[2] for line in lines]


def build_tokenizer(data_cfg: DataConfig, texts: list[str]) -> TextTokenizer:
    cache = Path(data_cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    vocab_path = cache / "vocab.json"
    if vocab_path.exists():
        return TextTokenizer.load(vocab_path)
    tokenizer = TextTokenizer.from_corpus(texts, use_phonemes=data_cfg.use_phonemes)
    tokenizer.save(vocab_path)
    return tokenizer


def sample_t(batch_size: int, step_dropout: float, device) -> torch.Tensor:
    """Sample flow steps; with probability *step_dropout* force t=1 (one-step exposure)."""
    t = torch.rand(batch_size, device=device)
    if step_dropout > 0:
        drop = torch.rand(batch_size, device=device) < step_dropout
        t = torch.where(drop, torch.ones_like(t), t)
    return t


def train(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig) -> list[float]:
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    texts = metadata_texts(data_cfg)
    tokenizer = build_tokenizer(data_cfg, texts)
    records = preprocess_ljspeech(data_cfg, tokenizer, max_files=None)
    model_cfg.vocab_size = tokenizer.vocab_size

    model = DSFlow(
        vocab_size=model_cfg.vocab_size,
        n_mels=data_cfg.mel.n_mels,
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
    ).to(device)
    print(f"[train] model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    dataset = MelDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_mel,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    def lr_at(step: int) -> float:
        if step < train_cfg.warmup_steps:
            return train_cfg.lr * (step + 1) / max(1, train_cfg.warmup_steps)
        progress = (step - train_cfg.warmup_steps) / max(1, train_cfg.steps - train_cfg.warmup_steps)
        return train_cfg.lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    ckpt_dir = Path(train_cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    step = 0
    if last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        print(f"[train] resumed from step {step}")

    losses: list[float] = []
    pbar = tqdm(total=train_cfg.steps, initial=step, desc="train")
    iterator = iter(loader)
    while step < train_cfg.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        mel = batch["mel"].to(device)
        text = batch["text"].to(device)
        text_len = batch["text_len"].to(device)
        durations = batch["durations"].to(device)
        mel_len = batch["mel_len"].to(device)
        mmask = torch.arange(mel.size(-1), device=device).unsqueeze(0) < mel_len.unsqueeze(1)

        x0 = mel
        x1 = torch.randn_like(mel)
        t = sample_t(mel.size(0), model_cfg.step_dropout, device)
        x_t = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
        v_target = x1 - x0

        v_pred, x0_pred, log_dur, tmask = model(text, text_len, x_t, t, durations, mel_len)
        loss_flow, parts = dual_supervision_loss(
            v_pred, v_target, x0_pred, x0, mmask,
            lambda_cfm=train_cfg.lambda_cfm,
            lambda_direct=train_cfg.lambda_direct,
        )
        loss_dur = duration_loss(log_dur, durations, tmask)
        loss = loss_flow + loss_dur

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        losses.append(loss.item())
        step += 1
        pbar.update(1)
        if step % train_cfg.log_every == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cfm=f"{parts['cfm'].item():.4f}",
                direct=f"{parts['direct'].item():.4f}",
                dur=f"{loss_dur.item():.4f}",
            )
        if step % train_cfg.ckpt_every == 0 or step == train_cfg.steps:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(model_cfg),
                    "mel_config": asdict(data_cfg.mel),
                    "vocab": tokenizer.symbols,
                },
                last_ckpt,
            )
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(model_cfg),
                    "mel_config": asdict(data_cfg.mel),
                    "vocab": tokenizer.symbols,
                },
                ckpt_dir / f"step_{step}.pt",
            )
    pbar.close()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DSFlow")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--step-dropout", type=float, default=0.1)
    parser.add_argument("--lambda-cfm", type=float, default=1.0)
    parser.add_argument("--lambda-direct", type=float, default=1.0)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_cfg = DataConfig(data_root=args.data_root, cache_dir=args.cache_dir)
    model_cfg = ModelConfig(decoder_layers=args.decoder_layers, decoder_dim=args.decoder_dim, step_dropout=args.step_dropout)
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        seed=args.seed,
        lambda_cfm=args.lambda_cfm,
        lambda_direct=args.lambda_direct,
    )
    losses = train(data_cfg, model_cfg, train_cfg)
    print(f"[train] final loss {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
