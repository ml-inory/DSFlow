"""Chatterbox S3Gen single-step model.

The official S3Gen decodes S3 speech tokens into mel-spectrograms with a
Matcha/CosyVoice-style conditional flow-matching CFM solved with 10 Euler
steps (``CausalConditionalCFM``). This module adds a dual-head student:
the same encoder/CFM trunk with a velocity head (inherited from the teacher)
plus a direct ``x0`` head, trained by reflow distillation + DSFlow-style
step dropout so that a single Euler step (the ``x0`` head at ``t=1``)
approximates the teacher's 10-step output.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3gen.configs import CFM_PARAMS
from chatterbox.models.s3gen.decoder import ConditionalDecoder
from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec
from chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
from chatterbox.models.s3gen.utils.mask import make_pad_mask


def load_teacher(ckpt_dir, device="cuda") -> S3Gen:
    """Load the official S3Gen (flow vocoder + HiFT) from local safetensors."""
    ckpt_dir = Path(ckpt_dir)
    model = S3Gen()
    state = load_file(ckpt_dir / "s3gen.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[chatterbox] missing keys: {missing[:5]}... ({len(missing)})")
    if unexpected:
        print(f"[chatterbox] unexpected keys: {unexpected[:5]}... ({len(unexpected)})")
    model.to(device).eval()
    return model


class DualHeadConditionalDecoder(ConditionalDecoder):
    """ConditionalDecoder trunk with a second output head predicting x0 directly.

    The velocity head (``final_proj``) is initialized from the teacher; the new
    ``x0_proj`` head starts at zero so early training is driven by the
    teacher-initialized trunk.
    """

    def __init__(self, **kwargs):
        channels = tuple(kwargs["channels"])
        super().__init__(**kwargs)
        self.x0_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        nn.init.zeros_(self.x0_proj.weight)
        nn.init.zeros_(self.x0_proj.bias)

    def forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
        """Return (velocity, x0_prediction) sharing the trunk."""
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        if self.meanflow:
            r = self.time_embeddings(r).to(t.dtype)
            r = self.time_mlp(r)
            concat_embed = torch.cat([t, r], dim=1)
            t = self.time_embed_mixer(concat_embed)

        from einops import pack, rearrange

        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = rearrange(spks, "b c -> b c 1").expand(-1, -1, x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = self._attn_mask(x, mask_down)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = self._attn_mask(x, mask_mid)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, : skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = self._attn_mask(x, mask_up)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        v = self.final_proj(x * mask_up) * mask
        x0 = self.x0_proj(x * mask_up) * mask
        return v, x0

    def _attn_mask(self, x, mask_down):
        from chatterbox.models.s3gen.decoder import add_optional_chunk_mask, mask_to_bias

        attn_mask = add_optional_chunk_mask(
            x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1
        )
        return mask_to_bias(attn_mask == 1, x.dtype)


class OneStepS3Gen(nn.Module):
    """Dual-head student sharing the S3Gen architecture."""

    def __init__(self, teacher: S3Gen):
        super().__init__()
        tflow = teacher.flow
        estimator = DualHeadConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
            meanflow=False,
        )
        cfm = CausalConditionalCFM(
            in_channels=240,
            cfm_params=CFM_PARAMS,
            n_spks=1,
            spk_emb_dim=80,
            estimator=estimator,
        )
        self.flow = CausalMaskedDiffWithXvec(
            encoder=copy.deepcopy(tflow.encoder),
            decoder=cfm,
        )
        # Copy all teacher flow weights; x0_proj stays zero-initialized.
        missing = self.flow.load_state_dict(
            {k[len("flow."):] if k.startswith("flow.") else k: v for k, v in tflow.state_dict().items()},
            strict=False,
        )
        assert any(k.startswith("decoder.estimator.x0_proj") for k in missing.missing_keys), missing.missing_keys[:10]

    def _conditions(self, tokens, token_len, ref_dict):
        return flow_conditions(self.flow, tokens, token_len, ref_dict)

    @torch.inference_mode()
    def one_step(self, tokens, token_len, ref_dict, z, t: float = 1.0) -> torch.Tensor:
        """Single-step synthesis via the x0 head: returns the generated mel region."""
        mu, mask, spks, conds, mel_len1, _ = self._conditions(tokens, token_len, ref_dict)
        t_tensor = torch.full((tokens.size(0),), t, device=z.device, dtype=z.dtype)
        _, x0 = self.flow.decoder.estimator(z, mask, mu, t_tensor, spks, conds)
        return x0[:, :, mel_len1:]

    @torch.inference_mode()
    def euler(self, tokens, token_len, ref_dict, z, steps: int, t_start: float = 1.0) -> torch.Tensor:
        """Multi-step synthesis via the velocity head (Euler)."""
        mu, mask, spks, conds, mel_len1, _ = self._conditions(tokens, token_len, ref_dict)
        x = z
        for i in range(steps):
            t = t_start - (t_start / steps) * i
            t_tensor = torch.full((tokens.size(0),), t, device=z.device, dtype=z.dtype)
            v, _ = self.flow.decoder.estimator(x, mask, mu, t_tensor, spks, conds)
            x = x - (t_start / steps) * v
        return x[:, :, mel_len1:]


@torch.inference_mode()
def teacher_endpoint(cfm: CausalConditionalCFM, mu, mask, spks, cond, z, n_timesteps=10):
    """Solve the teacher CFM ODE from *z* with Euler steps (cosine schedule)."""
    in_dtype = z.dtype
    est_dtype = cfm.estimator.dtype
    z = z.to(est_dtype)
    mu = mu.to(est_dtype)
    spks = spks.to(est_dtype)
    cond = cond.to(est_dtype)

    t_span = torch.linspace(0, 1, n_timesteps + 1, device=z.device, dtype=z.dtype)
    if cfm.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    x = z
    for t, r in zip(t_span[:-1], t_span[1:]):
        t_t = t.reshape(1).to(est_dtype)
        dxdt = cfm.estimator(x, mask, mu, t_t, spks, cond)
        x = x + (r - t) * dxdt
    return x.to(in_dtype)


def flow_conditions(flow: CausalMaskedDiffWithXvec, tokens, token_len, ref_dict, finalize: bool = True):
    """Compute (mu, mask, spks, conds, mel_len1, mel_len2) exactly like
    ``CausalMaskedDiffWithXvec.inference`` conditioning."""
    B = tokens.size(0)
    embedding = torch.atleast_2d(ref_dict["embedding"])
    embedding = nn.functional.normalize(embedding, dim=1)
    embedding = flow.spk_embed_affine_layer(embedding)

    prompt_token = ref_dict.get("prompt_token")
    if prompt_token is not None and prompt_token.numel() > 0:
        prompt_token = prompt_token.to(tokens.device)
        tokens = torch.cat([prompt_token, tokens], dim=1)
        token_len = token_len + ref_dict.get("prompt_token_len", torch.zeros_like(token_len))

    mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
    token_emb = flow.input_embedding(tokens.long()) * mask
    h, h_masks = flow.encoder(token_emb, token_len)
    if finalize is False:
        h = h[:, : -flow.pre_lookahead_len * flow.token_mel_ratio]
    h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)

    mel_len1 = 0
    prompt_feat = ref_dict.get("prompt_feat")
    if prompt_feat is not None and prompt_feat.size(1) > 0:
        mel_len1 = prompt_feat.size(1)
    mel_len2 = h.shape[1] - mel_len1
    h = flow.encoder_proj(h)

    conds = torch.zeros(B, mel_len1 + mel_len2, flow.output_size, device=tokens.device, dtype=h.dtype)
    if mel_len1 > 0:
        conds[:, :mel_len1] = prompt_feat.to(tokens.device)
    conds = conds.transpose(1, 2)

    mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h)
    mu = h.transpose(1, 2).contiguous()
    return mu, mask, embedding, conds, mel_len1, mel_len2
