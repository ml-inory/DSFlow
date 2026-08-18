"""LJSpeech download/extract and offline preprocessing into cached mel tensors + records."""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from dsflow.audio import load_wav, mel_spectrogram
from dsflow.config import DataConfig
from dsflow.text import TextTokenizer


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "dsflow/0.1"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as bar:
            while chunk := resp.read(256 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))


def ensure_ljspeech(cfg: DataConfig) -> Path:
    """Return the LJSpeech root directory, downloading and extracting it if needed."""
    lj_dir = Path(cfg.ljspeech_dir)
    if lj_dir.exists():
        return lj_dir
    root = Path(cfg.data_root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "LJSpeech-1.1.tar.bz2"
    if not archive.exists():
        _download(cfg.ljspeech_url, archive)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(root)
    return lj_dir


def preprocess_ljspeech(
    cfg: DataConfig,
    tokenizer: TextTokenizer,
    force: bool = False,
    max_files: Optional[int] = None,
) -> List[dict]:
    """Preprocess LJSpeech into cached per-utterance mel files plus metadata.json."""
    lj_dir = ensure_ljspeech(cfg)
    cache_dir = Path(cfg.cache_dir)
    mel_dir = cache_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    lines = [ln.strip() for ln in (lj_dir / "metadata.csv").read_text().splitlines() if ln.strip()]
    if max_files is not None:
        lines = lines[:max_files]

    records = []
    for line in tqdm(lines, desc="Preprocessing LJSpeech"):
        fid, _, text = line.split("|", 2)
        wav_path = lj_dir / "wavs" / f"{fid}.wav"
        wave = load_wav(wav_path, cfg.mel.sample_rate)
        seconds = wave.numel() / cfg.mel.sample_rate
        if not (cfg.min_seconds <= seconds <= cfg.max_seconds):
            continue
        mel = mel_spectrogram(wave, cfg.mel)
        tokens = tokenizer.encode(text)
        torch_save_path = mel_dir / f"{fid}.pt"
        torch_save({"mel": mel}, torch_save_path)
        records.append(
            {
                "id": fid,
                "text": text,
                "tokens": tokens,
                "text_len": len(tokens),
                "mel_len": mel.size(-1),
                "mel_path": str(torch_save_path),
            }
        )
    meta_path.write_text(json.dumps(records))
    return records


def torch_save(obj, path: Path) -> None:
    import torch

    torch.save(obj, path)
