"""Distill Chatterbox S3Gen (10-step CFM) into a one-step student.

Training recipe (DSFlow-style):
- teacher 10-step Euler endpoint x1_teach for the same noise z;
- reflow targets on the straight path z -> x1_teach:
    velocity target v = x1_teach - z, direct target x0 = x1_teach;
- dual supervision: CFM MSE on the velocity head + L1 on the x0 head;
- step dropout: with probability *step_dropout* force t=1 so the student is
  exposed to the exact one-step inference input distribution.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dsflow.chatterbox.data import ChatterboxDataConfig, load_record, prepare_records
from dsflow.chatterbox.model import OneStepS3Gen, flow_conditions, load_teacher, teacher_euler, teacher_euler_plain


class RecordDataset(Dataset):
    def __init__(self, records, max_mel_len=None):
        self.records = records
        if max_mel_len is not None:
            self.records = [r for r in records if r["mel_len"] <= max_mel_len]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return load_record(self.records[i], "cpu")


class PairDataset(Dataset):
    """Precomputed (z, x1_teacher) distillation pairs."""

    def __init__(self, records, pairs_dir: str):
        self.pairs_dir = Path(pairs_dir)
        self.records = [
            r for r in records if (self.pairs_dir / "z" / f"{r['id']}.pt").exists()
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        z = torch.load(self.pairs_dir / "z" / f"{rec['id']}.pt", weights_only=True)["z"]
        x1 = torch.load(self.pairs_dir / "x1" / f"{rec['id']}.pt", weights_only=True)["x1"]
        tok = torch.load(rec["tok_path"], weights_only=True)
        emb = torch.load(rec["emb_path"], weights_only=True)["embedding"]
        return {"z": z, "x1": x1, "tokens": tok["tokens"], "token_len": tok["token_len"], "embedding": emb}


def collate(batch):
    max_tok = max(b["tokens"].size(-1) for b in batch)
    max_mel = max(b["mel"].size(-1) for b in batch)
    tokens = torch.zeros(len(batch), max_tok, dtype=torch.long)
    token_len = torch.zeros(len(batch), dtype=torch.long)
    mel = torch.zeros(len(batch), 80, max_mel)
    emb = torch.zeros(len(batch), batch[0]["embedding"].size(-1))
    for i, b in enumerate(batch):
        t, m = b["tokens"].size(-1), b["mel"].size(-1)
        tokens[i, :t] = b["tokens"].long()
        token_len[i] = t
        mel[i, :, :m] = b["mel"]
        emb[i] = b["embedding"]
    return {"tokens": tokens, "token_len": token_len, "mel": mel, "embedding": emb}


def collate_pairs(batch):
    max_tok = max(b["tokens"].size(-1) for b in batch)
    max_mel = max(b["z"].size(-1) for b in batch)
    tokens = torch.zeros(len(batch), max_tok, dtype=torch.long)
    token_len = torch.zeros(len(batch), dtype=torch.long)
    z = torch.zeros(len(batch), 80, max_mel)
    x1 = torch.zeros(len(batch), 80, max_mel)
    emb = torch.zeros(len(batch), batch[0]["embedding"].size(-1))
    for i, b in enumerate(batch):
        t, m = b["tokens"].size(-1), b["z"].size(-1)
        tokens[i, :t] = b["tokens"].long()
        token_len[i] = t
        z[i, :, :m] = b["z"]
        x1[i, :, :m] = b["x1"]
        emb[i] = b["embedding"]
    return {"tokens": tokens, "token_len": token_len, "z": z, "x1": x1, "embedding": emb}


def sample_t(batch_size, step_dropout, device):
    t = torch.rand(batch_size, device=device)
    if step_dropout > 0:
        drop = torch.rand(batch_size, device=device) < step_dropout
        # S3Gen CFM convention: t=0 is pure noise = the one-step inference input.
        t = torch.where(drop, torch.zeros_like(t), t)
    return t


def masked_mse(pred, target, mask):
    denom = (mask.sum() * pred.size(1)).clamp(min=1)
    return ((pred - target).pow(2) * mask).sum() / denom


def masked_l1(pred, target, mask):
    denom = (mask.sum() * pred.size(1)).clamp(min=1)
    return ((pred - target).abs() * mask).sum() / denom


def distill(args) -> list[float]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    teacher = load_teacher(args.ckpt_dir, device)
    student = OneStepS3Gen(teacher).to(device)
    print(f"[distill] student params: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")

    records = prepare_records(ChatterboxDataConfig(out_dir=args.records_dir), teacher, max_files=args.max_files, device=device)
    if args.max_files:
        records = records[: args.max_files]
    use_pairs = (not args.fresh_z) and Path(args.pairs_dir).exists() and Path(args.pairs_dir, "index.json").exists()
    if use_pairs:
        dataset = PairDataset(records, args.pairs_dir)
        collate_fn = collate_pairs
    else:
        dataset = RecordDataset(records, max_mel_len=args.max_mel_len)
        collate_fn = collate
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    optimizer = torch.optim.AdamW(student.flow.parameters(), lr=args.lr, weight_decay=args.wd)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = out_dir / "last.pt"
    step = 0
    if last_ckpt.exists() and args.resume:
        state = torch.load(last_ckpt, map_location=device, weights_only=True)
        student.load_state_dict(state["student"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        print(f"[distill] resumed from step {step}")

    losses = []
    pbar = tqdm(total=args.steps, initial=step, desc="distill")
    iterator = iter(loader)
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        tokens = batch["tokens"].to(device)
        token_len = batch["token_len"].to(device)
        ref = {"embedding": batch["embedding"].to(device)}
        mu, mask, spks, conds, mel_len1, mel_len2 = flow_conditions(student.flow, tokens, token_len, ref)

        if use_pairs:
            z = batch["z"].to(device)
            x1_teach = batch["x1"].to(device)
        else:
            mu_t, mask_t, spks_t, conds_t, _, _ = flow_conditions(teacher.flow, tokens, token_len, ref)
            z = torch.randn_like(mu)
            with torch.no_grad():
                if args.teacher_cfg:
                    x1_teach = teacher_euler(teacher.flow.decoder, mu_t, mask_t, spks_t, conds_t, z, args.teacher_steps)
                else:
                    x1_teach = teacher_euler_plain(teacher.flow.decoder, mu_t, mask_t, spks_t, conds_t, z, args.teacher_steps)

        t = sample_t(tokens.size(0), args.step_dropout, device)
        x_t = (1 - t.view(-1, 1, 1)) * z + t.view(-1, 1, 1) * x1_teach
        v_pred, x0_pred = student.flow.decoder.estimator(x_t, mask, mu, t, spks, conds)

        gen_mask = mask.clone()
        if mel_len1 > 0:
            gen_mask[:, :, :mel_len1] = False
        loss_cfm = masked_mse(v_pred[:, :, mel_len1:], (x1_teach - z)[:, :, mel_len1:], gen_mask[:, :, mel_len1:])
        loss_direct = masked_l1(x0_pred[:, :, mel_len1:], x1_teach[:, :, mel_len1:], gen_mask[:, :, mel_len1:])
        loss = loss_cfm + args.lambda_direct * loss_direct

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.flow.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        step += 1
        pbar.update(1)
        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", cfm=f"{loss_cfm.item():.4f}", direct=f"{loss_direct.item():.4f}")
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save(
                {"step": step, "student": student.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)},
                last_ckpt,
            )
    pbar.close()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill Chatterbox S3Gen to one step")
    parser.add_argument("--ckpt-dir", default="data/chatterbox")
    parser.add_argument("--records-dir", default="data/chatterbox/records")
    parser.add_argument("--pairs-dir", default="data/chatterbox/pairs")
    parser.add_argument("--fresh-z", action="store_true", help="sample fresh noise per step with online teacher")
    parser.add_argument("--teacher-cfg", action="store_true", help="use CFG-blended teacher endpoints")
    parser.add_argument("--max-mel-len", type=int, default=800, help="cap training clip length (mel frames)")
    parser.add_argument("--out-dir", default="checkpoints/chatterbox")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--teacher-steps", type=int, default=10)
    parser.add_argument("--step-dropout", type=float, default=0.2)
    parser.add_argument("--lambda-direct", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    distill(args)


if __name__ == "__main__":
    main()
