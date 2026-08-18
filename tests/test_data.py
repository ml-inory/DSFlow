from dsflow.config import DataConfig
from dsflow.data import MelDataset, collate_mel, preprocess_ljspeech, proportional_durations
from dsflow.text import TextTokenizer


def test_preprocess_and_dataset(fake_ljspeech):
    tmp_path = fake_ljspeech
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
