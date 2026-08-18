"""Dual-supervision losses: CFM velocity + direct one-step data prediction + duration."""

from __future__ import annotations

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the feature dimension for [B, F, T] values and [B, T] mask."""
    denom = mask.sum().clamp(min=1)
    return (values * mask.unsqueeze(1)).sum() / denom


def cfm_loss(v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flow-matching velocity supervision: MSE over valid frames."""
    return _masked_mean((v_pred - v_target).pow(2), mask)


def direct_loss(x0_pred: torch.Tensor, x0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Direct one-step supervision: L1 between predicted and target mel frames."""
    return _masked_mean((x0_pred - x0).abs(), mask)


def duration_loss(
    log_dur_pred: torch.Tensor, durations: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE on log-durations over valid tokens (durations in mel frames)."""
    target = (durations.float() + 1.0).log()
    diff = (log_dur_pred - target).pow(2)
    denom = mask.sum().clamp(min=1)
    return (diff * mask).sum() / denom


def dual_supervision_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    x0_pred: torch.Tensor,
    x0: torch.Tensor,
    mel_mask: torch.Tensor,
    lambda_cfm: float = 1.0,
    lambda_direct: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine the two flow-matching supervisions and return per-part losses."""
    cfm = cfm_loss(v_pred, v_target, mel_mask)
    direct = direct_loss(x0_pred, x0, mel_mask)
    total = lambda_cfm * cfm + lambda_direct * direct
    return total, {"cfm": cfm, "direct": direct, "total": total}
