import torch

from dsflow.config import DataConfig, ModelConfig, TrainConfig
from dsflow.infer import load_checkpoint, synthesize_mel
from dsflow.train import train


def test_synthesize_from_checkpoint(fake_ljspeech):
    data_cfg = DataConfig(
        data_root=str(fake_ljspeech),
        ljspeech_dir=str(fake_ljspeech / "LJSpeech-1.1"),
        cache_dir=str(fake_ljspeech / "cache"),
        use_phonemes=False,
    )
    model_cfg = ModelConfig(
        text_dim=32,
        text_layers=2,
        text_heads=2,
        duration_dim=32,
        duration_layers=1,
        decoder_dim=64,
        decoder_layers=2,
        decoder_heads=4,
        ff_mult=2,
        step_dropout=0.2,
        dropout=0.0,
    )
    train_cfg = TrainConfig(
        batch_size=4,
        lr=1e-3,
        steps=10,
        warmup_steps=2,
        log_every=5,
        ckpt_every=10,
        ckpt_dir=str(fake_ljspeech / "ckpt"),
        device="cpu",
        seed=1,
    )
    train(data_cfg, model_cfg, train_cfg)

    model, tokenizer, mel_cfg = load_checkpoint(str(fake_ljspeech / "ckpt" / "last.pt"), "cpu")
    one_step = synthesize_mel(model, tokenizer, "sentence number one", "cpu", steps=1)
    assert one_step.dim() == 2 and one_step.size(0) == mel_cfg.n_mels
    assert torch.isfinite(one_step).all()
    multi_step = synthesize_mel(model, tokenizer, "sentence number one", "cpu", steps=4)
    assert multi_step.shape == one_step.shape
