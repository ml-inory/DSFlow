from dsflow.config import DataConfig, ModelConfig, TrainConfig
from dsflow.train import sample_t, train


def test_sample_t_step_dropout():
    t_all_one = sample_t(500, 1.0, "cpu")
    assert (t_all_one == 1.0).all()
    t_no_drop = sample_t(500, 0.0, "cpu")
    assert (t_no_drop >= 0.0).all() and (t_no_drop < 1.0).all()


def test_train_small(fake_ljspeech):
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
        steps=20,
        warmup_steps=2,
        log_every=5,
        ckpt_every=10,
        ckpt_dir=str(fake_ljspeech / "ckpt"),
        device="cpu",
        seed=0,
    )
    losses = train(data_cfg, model_cfg, train_cfg)
    assert len(losses) == 20
    assert losses[-1] < losses[0]
    assert (fake_ljspeech / "ckpt" / "last.pt").exists()
    assert (fake_ljspeech / "ckpt" / "step_10.pt").exists()
