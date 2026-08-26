from __future__ import annotations

import torch
import torch.nn.functional as F


_REDUCIBILITY_EPS = 1e-8


def huber_loss_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Return the baseline Huber loss without reducing the batch dimension."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim != 1:
        raise ValueError("prediction and target must have shape [batch]")
    return F.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=delta,
    )


@torch.no_grad()
def reducible_loss(
    online_base_loss: torch.Tensor,
    target_base_loss: torch.Tensor,
) -> torch.Tensor:
    """R = relu(L_online - L_target), detached from optimization."""
    if online_base_loss.shape != target_base_loss.shape:
        raise ValueError("online and target losses must have the same shape")
    return torch.relu(online_base_loss.detach() - target_base_loss.detach())


@torch.no_grad()
def reducibility_gate(
    online_base_loss: torch.Tensor,
    target_base_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reducible loss and its parameter-free normalized gate."""
    residual = reducible_loss(online_base_loss, target_base_loss)
    gate = residual / (online_base_loss.detach() + _REDUCIBILITY_EPS)
    gate = gate.clamp(0.0, 1.0)
    if not torch.isfinite(gate).all():
        raise FloatingPointError("reducibility gate must be finite")
    return residual, gate
