from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn


class PosteriorQEstimator(nn.Module, ABC):
    """Interface for replaceable approximate posterior Q estimators."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return Q samples shaped [batch, posterior_sample, action]."""


class BootstrapLastLayerEnsemble(PosteriorQEstimator):
    """Low-cost approximate epistemic ensemble over detached features.

    This is not an exact Bayesian posterior. Each independently initialized
    linear head is trained on its own Bernoulli bootstrap sample.
    """

    def __init__(self, feature_dim: int, num_actions: int, num_heads: int = 5) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.heads = nn.ModuleList(
            nn.Linear(feature_dim, num_actions) for _ in range(num_heads)
        )
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(features) for head in self.heads], dim=1)


def posterior_predictive_policy(
    posterior_q: torch.Tensor, tau: float
) -> torch.Tensor:
    """Average per-Q-sample policies: mean_k softmax(Q_k / tau)."""
    if posterior_q.ndim != 3:
        raise ValueError("posterior_q must have shape [batch, head, action]")
    if posterior_q.shape[1] < 1:
        raise ValueError("posterior_q must contain at least one head")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    head_policies = torch.softmax(posterior_q / tau, dim=-1)
    return head_policies.mean(dim=1)


def posterior_predictive_log_policy(
    posterior_q: torch.Tensor, tau: float, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    policy = posterior_predictive_policy(posterior_q, tau)
    return policy, torch.log(policy.clamp_min(eps))


def sample_bootstrap_mask(
    batch_size: int,
    num_heads: int,
    probability: float,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    if not 0.0 < probability <= 1.0:
        raise ValueError("bootstrap probability must be in (0, 1]")
    return torch.rand(batch_size, num_heads, device=device) < probability


def masked_head_huber_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean of independently mask-normalized per-head Huber losses."""
    if predictions.ndim != 2:
        raise ValueError("predictions must have shape [batch, head]")
    if target.shape != predictions.shape[:1]:
        raise ValueError("target must have shape [batch]")
    if mask.shape != predictions.shape:
        raise ValueError("mask must match predictions")
    element_loss = F.huber_loss(
        predictions,
        target[:, None].expand_as(predictions),
        reduction="none",
        delta=delta,
    )
    weights = mask.to(element_loss.dtype)
    per_head = (weights * element_loss).sum(dim=0) / (weights.sum(dim=0) + eps)
    return per_head.mean()

