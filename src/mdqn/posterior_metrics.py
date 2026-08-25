from __future__ import annotations

import torch

from mdqn.algorithm import scaled_log_softmax
from mdqn.posterior import posterior_predictive_log_policy


@torch.no_grad()
def posterior_mechanism_metrics(
    posterior_q: torch.Tensor,
    point_q: torch.Tensor,
    actions: torch.Tensor,
    *,
    tau: float,
    alpha: float,
    log_policy_min: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Batch-mean diagnostics for the PP-MDQN mechanism."""
    pp_policy, pp_log_policy = posterior_predictive_log_policy(
        posterior_q, tau, eps
    )
    head_policy = torch.softmax(posterior_q / tau, dim=-1)
    point_scaled_log_policy = scaled_log_softmax(point_q, tau)
    point_policy = torch.softmax(point_q / tau, dim=-1)

    q_variance = posterior_q.var(dim=1, correction=0).mean(dim=-1).mean()
    policy_disagreement = (
        head_policy
        * (
            torch.log(head_policy.clamp_min(eps))
            - pp_log_policy.unsqueeze(1)
        )
    ).sum(dim=-1).mean(dim=1).mean()

    point_entropy = -(
        point_policy * (point_scaled_log_policy / tau)
    ).sum(dim=-1).mean()
    pp_entropy = -(pp_policy * pp_log_policy).sum(dim=-1).mean()

    action_index = actions.long().view(-1, 1)
    point_chosen = point_scaled_log_policy.gather(1, action_index).squeeze(1)
    pp_chosen = tau * pp_log_policy.gather(1, action_index).squeeze(1)
    point_bonus = alpha * point_chosen.clamp(log_policy_min, 0.0)
    pp_bonus = alpha * pp_chosen.clamp(log_policy_min, 0.0)

    return {
        "posterior/q_variance": q_variance,
        "posterior/policy_disagreement": policy_disagreement,
        "policy/point_entropy": point_entropy,
        "policy/pp_entropy": pp_entropy,
        "munchausen/point_clip_ratio": (point_chosen < log_policy_min)
        .float()
        .mean(),
        "munchausen/pp_clip_ratio": (pp_chosen < log_policy_min).float().mean(),
        "munchausen/point_bonus_mean": point_bonus.mean(),
        "munchausen/pp_bonus_mean": pp_bonus.mean(),
        "munchausen/bonus_difference": pp_bonus.mean() - point_bonus.mean(),
    }
