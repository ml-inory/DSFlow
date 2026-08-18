"""End-to-end Chatterbox demo: official 10-step vs distilled one-step S3Gen.

Loads the official Chatterbox components (T3 token generator, VoiceEncoder,
tokenizer, built-in voice conditionals), runs T3 autoregressive token
generation once, then decodes with either the official 10-step S3Gen or the
distilled one-step student, and saves both WAVs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.t3 import T3
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.tts import Conditionals, punc_norm

from dsflow.audio import save_wav
from dsflow.chatterbox.model import OneStepS3Gen, flow_conditions, load_teacher


def load_tts(ckpt_dir: str, device: str):
    ckpt_dir = Path(ckpt_dir)
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    ve.to(device).eval()

    t3 = T3()
    state = load_file(ckpt_dir / "t3_cfg.safetensors")
    if "model" in state:
        state = state["model"][0]
    t3.load_state_dict(state)
    t3.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen.to(device).eval()

    tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))
    conds = Conditionals.load(ckpt_dir / "conds.pt", map_location="cpu").to(device)
    return ve, t3, s3gen, tokenizer, conds


def generate_tokens(t3, ve, tokenizer, conds, text, device, cfg_weight=0.5):
    text = punc_norm(text)
    text_tokens = tokenizer.text_to_tokens(text).to(device)
    if cfg_weight > 0.0:
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    sot, eot = t3.hp.start_text_token, t3.hp.stop_text_token
    text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=sot)
    text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=eot)
    speech_tokens = t3.inference(
        t3_cond=conds.t3,
        text_tokens=text_tokens,
        max_new_tokens=1000,
        temperature=0.8,
        cfg_weight=cfg_weight,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    )[0]
    speech_tokens = drop_invalid_tokens(speech_tokens)
    speech_tokens = speech_tokens[speech_tokens < 6561].unsqueeze(0).to(device)
    return speech_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="data/chatterbox")
    parser.add_argument("--student-ckpt", default="checkpoints/chatterbox/last.pt")
    parser.add_argument("--text", default="Ezreal and Jinx teamed up with Ahri to take down the enemy Nexus.")
    parser.add_argument("--out-dir", default="outputs/chatterbox_tts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    ve, t3, s3gen, tokenizer, conds = load_tts(args.ckpt_dir, device)
    teacher = load_teacher(args.ckpt_dir, device)
    student = OneStepS3Gen(teacher).to(device)
    state = torch.load(Path(args.student_ckpt), map_location=device, weights_only=True)
    student.load_state_dict(state["student"])
    student.eval()

    tokens = generate_tokens(t3, ve, tokenizer, conds, args.text, device)
    print(f"[demo] T3 tokens: {tokens.size(-1)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Official 10-step
    t0 = time.time()
    wav10, _ = s3gen.inference(speech_tokens=tokens, ref_dict=conds.gen)
    t10 = time.time() - t0
    save_wav(out_dir / "teacher_10step.wav", wav10.squeeze(0).cpu(), S3GEN_SR)

    # Student one-step
    mu, mask, spks, conds_flow, mel_len1, _ = flow_conditions(student.flow, tokens, torch.tensor([tokens.size(-1)], device=device), conds.gen)
    z = torch.randn_like(mu)
    t0 = time.time()
    mel1 = student.one_step(tokens, torch.tensor([tokens.size(-1)], device=device), conds.gen, z)
    t1 = time.time() - t0
    wav1, *_ = teacher.mel2wav.inference(speech_feat=mel1, cache_source=torch.zeros(1, 1, 0, device=device))
    # Loudness-normalize the one-step output to the teacher's level for a fair A/B listen.
    rms1, rms10 = wav1.pow(2).mean().sqrt(), wav10.pow(2).mean().sqrt()
    if rms1 > 1e-6 and rms10 > 1e-6:
        wav1 = wav1 * (rms10 / rms1)
    save_wav(out_dir / "student_1step.wav", wav1.squeeze(0).cpu(), S3GEN_SR)

    print(f"[demo] teacher10: {t10:.2f}s -> {out_dir / 'teacher_10step.wav'}")
    print(f"[demo] student1 : {t1:.2f}s (x{rms10 / max(rms1, 1e-6):.1f} loudness-normalized) -> {out_dir / 'student_1step.wav'}")


if __name__ == "__main__":
    main()
