"""Evaluate teacher (10-step / 1-step Euler) vs distilled student (1-step)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from dsflow.audio import save_wav
from dsflow.chatterbox.data import ChatterboxDataConfig, load_record, prepare_records
from dsflow.chatterbox.model import OneStepS3Gen, flow_conditions, load_teacher, teacher_endpoint
from chatterbox.models.s3gen import S3GEN_SR


def corr(a, b):
    n = min(a.size(-1), b.size(-1))
    a, b = a[..., :n], b[..., :n]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def l1(a, b):
    n = min(a.size(-1), b.size(-1))
    return float((a[..., :n] - b[..., :n]).abs().mean())


def run_eval(args) -> dict:
    torch.manual_seed(args.seed)
    device = args.device
    teacher = load_teacher(args.ckpt_dir, device)
    student = OneStepS3Gen(teacher).to(device)
    state = torch.load(Path(args.student_ckpt), map_location=device, weights_only=True)
    student.load_state_dict(state["student"])
    student.eval()

    records = prepare_records(ChatterboxDataConfig(out_dir=args.records_dir), teacher, max_files=None, device=device)
    records = records[-args.max_clips :]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for rec in records:
        data = load_record(rec, device)
        mel = data["mel"].unsqueeze(0).to(device)
        tokens = data["tokens"].to(device)
        token_len = data["token_len"].to(device)
        ref = {"embedding": data["embedding"].to(device)}
        mu_t, mask_t, spks_t, conds_t, mel_len1, _ = flow_conditions(teacher.flow, tokens, token_len, ref)
        z = torch.randn_like(mu_t)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        mel10 = teacher_endpoint(teacher.flow.decoder, mu_t, mask_t, spks_t, conds_t, z, 10)[:, :, mel_len1:]
        t1.record()
        torch.cuda.synchronize()
        time10 = t0.elapsed_time(t1) / 1000.0

        t0.record()
        mel1_student = student.one_step(tokens, token_len, ref, z, t=1.0)
        t1.record()
        torch.cuda.synchronize()
        time1 = t0.elapsed_time(t1) / 1000.0

        # teacher direct 1-step Euler (velocity at t=0)
        t_span = torch.linspace(0, 1, 2, device=device, dtype=z.dtype)
        if teacher.flow.decoder.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        v0 = teacher.flow.decoder.estimator(z, mask_t, mu_t, t_span[0].reshape(1), spks_t, conds_t)
        mel1_teacher = (z + (t_span[1] - t_span[0]) * v0)[:, :, mel_len1:]

        results.append(
            {
                "id": rec["id"],
                "teacher10_corr": corr(mel10, mel),
                "teacher1_corr": corr(mel1_teacher, mel),
                "student1_corr": corr(mel1_student, mel),
                "teacher10_l1": l1(mel10, mel),
                "teacher1_l1": l1(mel1_teacher, mel),
                "student1_l1": l1(mel1_student, mel),
                "time_teacher10_s": time10,
                "time_student1_s": time1,
            }
        )

        if len(results) <= args.save_wavs:
            for name, m in [("t10", mel10), ("t1", mel1_teacher), ("s1", mel1_student)]:
                wav, *_ = teacher.mel2wav.inference(
                    speech_feat=m.to(device), cache_source=torch.zeros(1, 1, 0, device=device)
                )
                save_wav(out_dir / f"{rec['id']}_{name}.wav", wav.squeeze(0).cpu(), S3GEN_SR)

    summary = {
        "n": len(results),
        "teacher10_corr": statistics.mean(r["teacher10_corr"] for r in results),
        "teacher1_corr": statistics.mean(r["teacher1_corr"] for r in results),
        "student1_corr": statistics.mean(r["student1_corr"] for r in results),
        "teacher10_l1": statistics.mean(r["teacher10_l1"] for r in results),
        "teacher1_l1": statistics.mean(r["teacher1_l1"] for r in results),
        "student1_l1": statistics.mean(r["student1_l1"] for r in results),
        "time_teacher10_s": statistics.mean(r["time_teacher10_s"] for r in results),
        "time_student1_s": statistics.mean(r["time_student1_s"] for r in results),
        "speedup": statistics.mean(r["time_teacher10_s"] for r in results)
        / max(1e-6, statistics.mean(r["time_student1_s"] for r in results)),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="data/chatterbox")
    parser.add_argument("--records-dir", default="data/chatterbox/records")
    parser.add_argument("--student-ckpt", default="checkpoints/chatterbox/last.pt")
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--save-wavs", type=int, default=3)
    parser.add_argument("--out-dir", default="outputs/chatterbox_eval")
    parser.add_argument("--out", default="outputs/chatterbox_eval.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
