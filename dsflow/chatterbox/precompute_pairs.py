"""Precompute teacher 10-step endpoints (z, x1) for distillation pairs."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path

import torch
from tqdm import tqdm

from dsflow.chatterbox.data import ChatterboxDataConfig, load_record, prepare_records
from dsflow.chatterbox.model import flow_conditions, load_teacher, teacher_euler


def worker(rank: int, records, out_dir: str, ckpt_dir: str, device: str, seed: int) -> int:
    torch.manual_seed(seed + rank)
    teacher = load_teacher(ckpt_dir, device)
    out = Path(out_dir)
    (out / "z").mkdir(parents=True, exist_ok=True)
    (out / "x1").mkdir(parents=True, exist_ok=True)
    done = 0
    for rec in tqdm(records, desc=f"worker{rank}", position=rank):
        z_path = out / "z" / f"{rec['id']}.pt"
        if z_path.exists():
            done += 1
            continue
        data = load_record(rec, device)
        mu, mask, spks, conds, mel_len1, mel_len2 = flow_conditions(
            teacher.flow, data["tokens"], data["token_len"], {"embedding": data["embedding"]}
        )
        z = torch.randn_like(mu)
        x1 = teacher_euler(teacher.flow.decoder, mu, mask, spks, conds, z, 10)
        torch.save({"z": z.cpu()}, z_path)
        torch.save({"x1": x1[:, :, mel_len1:].cpu()}, out / "x1" / f"{rec['id']}.pt")
        done += 1
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="data/chatterbox")
    parser.add_argument("--records-dir", default="data/chatterbox/records")
    parser.add_argument("--pairs-dir", default="data/chatterbox/pairs")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    teacher = load_teacher(args.ckpt_dir, args.devices.split(",")[0])
    records = prepare_records(ChatterboxDataConfig(out_dir=args.records_dir), teacher, max_files=args.max_files, device=args.devices.split(",")[0])
    if args.max_files:
        records = records[: args.max_files]

    devices = args.devices.split(",")
    n_workers = min(len(devices), 4)
    chunks = [records[i::n_workers] for i in range(n_workers)]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        results = [
            pool.apply_async(worker, (rank, chunks[rank], args.pairs_dir, args.ckpt_dir, devices[rank % len(devices)], args.seed))
            for rank in range(n_workers)
        ]
        total = sum(r.get() for r in results)
    index = {"n": total, "pairs_dir": args.pairs_dir}
    Path(args.pairs_dir, "index.json").write_text(json.dumps(index))
    print(f"[precompute] {total} pairs -> {args.pairs_dir}")


if __name__ == "__main__":
    main()
