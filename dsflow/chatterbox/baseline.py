"""Baseline: teacher 10-step vs direct 1-step Euler on held-out clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dsflow.chatterbox.data import ChatterboxDataConfig, load_record, prepare_records
from dsflow.chatterbox.model import flow_conditions, load_teacher, teacher_endpoint
from dsflow.config import DataConfig
from dsflow.data.ljspeech import ensure_ljspeech


def _masked_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.size(-1), b.size(-1))
    a, b = a[..., :n], b[..., :n]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def _masked_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.size(-1), b.size(-1))
    return float((a[..., :n] - b[..., :n]).abs().mean())


def run_baseline(ckpt_dir, records_dir, max_clips=40, device="cuda", seed=0):
    torch.manual_seed(seed)
    teacher = load_teacher(ckpt_dir, device)
    cfm = teacher.flow.decoder
    records = prepare_records(ChatterboxDataConfig(out_dir=records_dir), teacher, device=device)
    records = records[:max_clips]

    metrics = []
    for i, rec in enumerate(records):
        data = load_record(rec, device)
        mel = data["mel"].unsqueeze(0).to(device)
        tokens = data["tokens"].to(device)
        token_len = data["token_len"].to(device)
        embedding = data["embedding"].to(device)
        ref = {"embedding": embedding}
        mu, mask, spks, conds, mel_len1, _ = flow_conditions(teacher.flow, tokens, token_len, ref)
        z = torch.randn_like(mu)

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        mel10 = teacher_endpoint(cfm, mu, mask, spks, conds, z, n_timesteps=10)[:, :, mel_len1:]
        t1.record()
        torch.cuda.synchronize()
        time10 = t0.elapsed_time(t1) / 1000.0

        # one-step Euler with the teacher velocity at t=0 (same z)
        t_span = torch.linspace(0, 1, 2, device=device, dtype=z.dtype)
        if cfm.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        v0 = cfm.estimator(z, mask, mu, t_span[0].reshape(1), spks, conds)
        mel1 = (z + (t_span[1] - t_span[0]) * v0)[:, :, mel_len1:]

        metrics.append(
            {
                "id": rec["id"],
                "corr_10": _masked_corr(mel10, mel),
                "corr_1": _masked_corr(mel1, mel),
                "l1_10": _masked_l1(mel10, mel),
                "l1_1": _masked_l1(mel1, mel),
                "time_10s": time10,
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="data/chatterbox")
    parser.add_argument("--records-dir", default="data/chatterbox/records")
    parser.add_argument("--max-clips", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="outputs/chatterbox_baseline.json")
    args = parser.parse_args()

    # make sure LJSpeech exists (for records)
    ensure_ljspeech(DataConfig(data_root="data"))
    metrics = run_baseline(args.ckpt_dir, args.records_dir, args.max_clips, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    import statistics

    print(f"corr10={statistics.mean(m['corr_10'] for m in metrics):.4f} "
          f"corr1={statistics.mean(m['corr_1'] for m in metrics):.4f} "
          f"l1_10={statistics.mean(m['l1_10'] for m in metrics):.4f} "
          f"l1_1={statistics.mean(m['l1_1'] for m in metrics):.4f}")


if __name__ == "__main__":
    main()
