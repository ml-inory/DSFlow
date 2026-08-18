import json

import torch

from dsflow.audio import save_wav
from dsflow.config import DataConfig
from dsflow.data import MelDataset, collate_mel, preprocess_ljspeech, proportional_durations
from dsflow.text import TextTokenizer


def make_fake_ljspeech(root, n=4):
    wavs = root / "LJSpeech-1.1" / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        fid = f"LJ00{i+1}_001"
        seconds = 1.0 + 0.5 * i
        wave = torch.randn(int(22050 * seconds)) * 0.1 + 0.2 * torch.sin(
            torch.linspace(0, 2 * 3.14159 * 220 * seconds, int(22050 * seconds))
        )
        save_wav(wavs / f"{fid}.wav", wave, 22050)
        rows.append(f"{fid}|fake|sentence number {i+1}.")
    (root / "LJSpeech-1.1" / "metadata.csv").write_text("\n".join(rows) + "\n")


def test_preprocess_and_dataset(tmp_path):
    make_fake_ljspeech(tmp_path)
    cfg = DataConfig(
        data_root=str(tmp_path),
        ljspeech_dir=str(tmp_path / "LJSpeech-1.1"),
        cache_dir=str(tmp_path / "cache"),
    )
    tokenizer = TextTokenizer.from_corpus(["sentence number one"], use_phonemes=False)
    records = preprocess_ljspeech(cfg, tokenizer)
    assert len(records) == 4
    assert (tmp_path / "cache" / "metadata.json").exists()

    ds = MelDataset(records)
    batch = collate_mel([ds[i] for i in range(len(ds))])
    B = len(ds)
    assert batch["text"].dim() == 2 and batch["text"].size(0) == B
    assert batch["mel"].dim() == 3 and batch["mel"].size(1) == cfg.mel.n_mels
    assert batch["durations"].size(0) == B
    assert batch["durations"].sum(dim=1).eq(batch["mel_len"]).all()
    assert batch["text_len"].size(0) == B
    assert batch["mel_len"].size(0) == B
    assert batch["mel"].isfinite().all()


def test_proportional_durations():
    out = proportional_durations(3, 10)
    assert out.sum() == 10 and out.numel() == 3
    assert sorted(out.tolist()) == [3, 3, 4]
