import torch

from dsflow.losses import cfm_loss, direct_loss, duration_loss, dual_supervision_loss
from dsflow.model import DSFlow
from dsflow.model.modules import expand_by_durations


def make_model(**kwargs) -> DSFlow:
    params = dict(
        vocab_size=24,
        n_mels=20,
        text_dim=32,
        text_layers=2,
        text_heads=4,
        duration_dim=32,
        duration_layers=2,
        decoder_dim=64,
        decoder_layers=2,
        decoder_heads=4,
        ff_mult=2,
        dropout=0.0,
    )
    params.update(kwargs)
    return DSFlow(**params)


def test_forward_shapes():
    model = make_model()
    B, Tt, M, T = 2, 8, 20, 40
    text = torch.randint(4, 24, (B, Tt))
    text_len = torch.tensor([8, 6])
    mel = torch.randn(B, M, T)
    t = torch.tensor([0.3, 1.0])
    durations = torch.tensor([[5, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7, 7, 0, 0]])
    v, x0, log_d, tmask = model(text, text_len, mel, t, durations, torch.tensor([40, 38]))
    assert v.shape == (B, M, T)
    assert x0.shape == (B, M, T)
    assert log_d.shape == (B, Tt)
    assert tmask.shape == (B, Tt)
    assert torch.isfinite(v).all() and torch.isfinite(x0).all()


def test_inference_duration_path():
    """durations=None must still produce a valid forward pass (predicted durations)."""
    model = make_model()
    B, Tt, M, T = 1, 5, 20, 60
    text = torch.randint(4, 24, (B, Tt))
    text_len = torch.tensor([5])
    mel = torch.randn(B, M, T)
    t = torch.ones(B)
    v, x0, _, _ = model(text, text_len, mel, t)
    assert v.shape == (B, M, T)
    assert x0.shape == (B, M, T)


def test_expand_by_durations():
    hidden = torch.arange(3 * 4, dtype=torch.float32).reshape(1, 3, 4)
    durations = torch.tensor([[2, 0, 3]])
    out, mask = expand_by_durations(hidden, durations, 5)
    assert out[0, 0].tolist() == hidden[0, 0].tolist()
    assert out[0, 1].tolist() == hidden[0, 0].tolist()
    assert out[0, 2].tolist() == hidden[0, 2].tolist()
    assert out[0, 4].tolist() == hidden[0, 2].tolist()
    assert mask.all()


def test_losses_perfect_and_masked():
    mask = torch.tensor([[True, False]])
    v = torch.randn(1, 4, 2)
    assert cfm_loss(v, v, mask) == 0.0
    assert direct_loss(v, v, mask) == 0.0
    v_bad = v.clone()
    v_bad[..., 1] = 999.0
    assert cfm_loss(v, v_bad, mask) == 0.0


def test_dual_supervision_weights():
    torch.manual_seed(0)
    v_pred, v_t = torch.randn(2, 20, 40), torch.randn(2, 20, 40)
    x0_pred, x0 = torch.randn(2, 20, 40), torch.randn(2, 20, 40)
    mask = torch.ones(2, 40, dtype=torch.bool)
    total, parts = dual_supervision_loss(v_pred, v_t, x0_pred, x0, mask, lambda_cfm=1.0, lambda_direct=2.0)
    assert torch.isclose(total, parts["cfm"] + 2.0 * parts["direct"])
    assert torch.isfinite(total)


def test_duration_loss_masked():
    log_d = torch.randn(2, 6)
    durations = torch.tensor([[5, 5, 5, 5, 5, 5], [5, 5, 0, 0, 0, 0]])
    mask = torch.tensor([[True] * 6, [True, True, False, False, False, False]])
    loss = duration_loss(log_d, durations, mask)
    assert torch.isfinite(loss) and loss >= 0
