import torch

from dsflow.chatterbox.model import DualHeadConditionalDecoder


def test_dual_head_decoder_shapes():
    dec = DualHeadConditionalDecoder(
        in_channels=32,
        out_channels=8,
        causal=True,
        channels=[16],
        dropout=0.0,
        attention_head_dim=8,
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=2,
        act_fn="gelu",
    )
    B, T = 2, 20
    # decoder concatenates x + mu (+ spks/cond): in_channels=32 means 16+16 here
    x = torch.randn(B, 16, T)
    mu = torch.randn(B, 16, T)
    mask = torch.ones(B, 1, T, dtype=torch.bool)
    t = torch.zeros(B)
    v, x0 = dec(x, mask, mu, t)
    assert v.shape == (B, 8, T) and x0.shape == (B, 8, T)
    # x0 head is zero-initialized; velocity head is not
    assert x0.abs().max() < 1e-6
    assert v.abs().max() > 0
