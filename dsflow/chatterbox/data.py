"""Prepare S3Gen distillation records from speech audio.

For each clip we cache: S3 speech tokens (16 kHz), the target log-mel
(24 kHz, 80 bands), and the x-vector speaker embedding — everything the
teacher/student CFM needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio.functional as taf
from tqdm import tqdm

from chatterbox.models.s3gen import S3GEN_SR
from chatterbox.models.s3gen.utils.mel import mel_spectrogram
from chatterbox.models.s3tokenizer import S3_SR

from dsflow.audio import load_wav


@dataclass
class ChatterboxDataConfig:
    ljspeech_dir: str = "data/LJSpeech-1.1"
    out_dir: str = "data/chatterbox/records"
    sample_rate: int = S3GEN_SR  # 24000 mel
    token_sr: int = S3_SR  # 16000 tokens
    min_seconds: float = 2.5


def prepare_records(cfg: ChatterboxDataConfig, teacher, max_files=None, device="cuda") -> list[dict]:
    out_dir = Path(cfg.out_dir)
    mel_dir = out_dir / "mel"
    tok_dir = out_dir / "tok"
    emb_dir = out_dir / "emb"
    for d in (mel_dir, tok_dir, emb_dir):
        d.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())

    teacher = teacher.to(device).eval()
    tokenizer = teacher.tokenizer.to(device)
    speaker_encoder = teacher.speaker_encoder.to(device)

    lines = [
        ln.strip()
        for ln in (Path(cfg.ljspeech_dir) / "metadata.csv").read_text().splitlines()
        if ln.strip()
    ]
    if max_files is not None:
        lines = lines[:max_files]

    records = []
    for line in tqdm(lines, desc="Preparing chatterbox records"):
        fid, _, _ = line.split("|", 2)
        wav = load_wav(Path(cfg.ljspeech_dir) / "wavs" / f"{fid}.wav", cfg.sample_rate)
        if wav.numel() / cfg.sample_rate < cfg.min_seconds:
            continue
        wav_24 = wav.to(device)
        mel = mel_spectrogram(wav_24.unsqueeze(0))[0].cpu()  # [80, T]
        wav_16 = taf.resample(wav_24.view(1, -1), cfg.sample_rate, cfg.token_sr).view(-1)
        tokens, token_len = tokenizer(wav_16.unsqueeze(0).float())
        embedding = speaker_encoder.inference(wav_16.unsqueeze(0).float().to(device))

        tok_path = tok_dir / f"{fid}.pt"
        torch.save({"tokens": tokens.cpu(), "token_len": token_len.cpu()}, tok_path)
        torch.save({"mel": mel}, mel_dir / f"{fid}.pt")
        torch.save({"embedding": embedding.cpu()}, emb_dir / f"{fid}.pt")
        records.append(
            {
                "id": fid,
                "mel_len": int(mel.size(-1)),
                "token_len": int(token_len.item()),
                "mel_path": str(mel_dir / f"{fid}.pt"),
                "tok_path": str(tok_dir / f"{fid}.pt"),
                "emb_path": str(emb_dir / f"{fid}.pt"),
            }
        )
    index_path.write_text(json.dumps(records))
    return records


def load_record(record: dict, device="cpu") -> dict:
    mel = torch.load(record["mel_path"], weights_only=True)["mel"].to(device)
    tok = torch.load(record["tok_path"], weights_only=True)
    emb = torch.load(record["emb_path"], weights_only=True)["embedding"].to(device)
    return {
        "mel": mel,
        "tokens": tok["tokens"].to(device),
        "token_len": tok["token_len"].to(device),
        "embedding": emb,
        "mel_len": mel.size(-1),
    }
