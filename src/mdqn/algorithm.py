from __future__ import annotations

import torch
import torch.nn.functional as F


def scaled_log_softmax(q_values: torch.Tensor, tau: float) -> torch.Tensor:
    """Compute tau * log softmax(q / tau) with the paper's stable form."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    maximum = q_values.max(dim=-1, keepdim=True).values
    centered = q_values - maximum
    tau_logsumexp = maximum + tau * torch.log(
        torch.exp(centered / tau).sum(dim=-1, keepdim=True)
    )
    return q_values - tau_logsumexp


@torch.no_grad()
def munchausen_target(
    rewards: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
    target_q_current: torch.Tensor,
    target_q_next: torch.Tensor,
    *,
    gamma: float = 0.99,
    tau: float = 0.03,
    alpha: float = 0.9,
    log_policy_min: float = -1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equation (2)/(7), using the target network at s and s'."""
    current_scaled_log_policy = scaled_log_softmax(target_q_current, tau)
    next_scaled_log_policy = scaled_log_softmax(target_q_next, tau)
    next_policy = torch.softmax(target_q_next / tau, dim=-1)

    chosen_log_policy = current_scaled_log_policy.gather(
        1, actions.long().unsqueeze(1)
    ).squeeze(1)
    # The mathematical upper bound is zero. The official TF max of 1 is a
    # no-op in exact arithmetic; zero expresses the paper's [.]^0_l0 notation.
    chosen_log_policy = chosen_log_policy.clamp(min=log_policy_min, max=0.0)
    munchausen_bonus = alpha * chosen_log_policy

    soft_next_value = (
        next_policy * (target_q_next - next_scaled_log_policy)
    ).sum(dim=-1)
    not_done = 1.0 - dones.to(dtype=rewards.dtype)
    target = rewards + munchausen_bonus + gamma * not_done * soft_next_value
    diagnostics = {
        "munchausen_bonus": munchausen_bonus,
        "soft_next_value": soft_next_value,
        "entropy": -(next_policy * (next_scaled_log_policy / tau)).sum(dim=-1),
    }
    return target, diagnostics


@torch.no_grad()
def posterior_predictive_munchausen_target(
    rewards: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
    target_q_current: torch.Tensor,
    target_q_next: torch.Tensor,
    pp_policy_current: torch.Tensor,
    *,
    pp_policy_next: torch.Tensor | None = None,
    pp_scope: str = "munchausen_only",
    gamma: float = 0.99,
    tau: float = 0.03,
    alpha: float = 0.9,
    log_policy_min: float = -1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PP-MDQN target with PP policy controlling policy terms only.

    The main target Q network remains the value estimator. In
    ``munchausen_only`` only the current-state bonus uses the PP policy. In
    ``full_operator`` the next-state entropy-regularized policy also uses it.
    """
    if pp_scope not in {"munchausen_only", "full_operator"}:
        raise ValueError("unknown pp_scope")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    pp_log_current = torch.log(pp_policy_current.clamp_min(eps))
    pp_scaled_log_current = tau * pp_log_current
    chosen_pp_log_policy = pp_scaled_log_current.gather(
        1, actions.long().unsqueeze(1)
    ).squeeze(1)
    chosen_pp_log_policy = chosen_pp_log_policy.clamp(
        min=log_policy_min, max=0.0
    )
    munchausen_bonus = alpha * chosen_pp_log_policy

    if pp_scope == "munchausen_only":
        next_scaled_log_policy = scaled_log_softmax(target_q_next, tau)
        next_policy = torch.softmax(target_q_next / tau, dim=-1)
    else:
        if pp_policy_next is None:
            raise ValueError("full_operator requires pp_policy_next")
        next_policy = pp_policy_next
        next_scaled_log_policy = tau * torch.log(next_policy.clamp_min(eps))

    soft_next_value = (
        next_policy * (target_q_next - next_scaled_log_policy)
    ).sum(dim=-1)
    target = (
        rewards
        + munchausen_bonus
        + gamma
        * (1.0 - dones.to(dtype=rewards.dtype))
        * soft_next_value
    )
    diagnostics = {
        "munchausen_bonus": munchausen_bonus,
        "soft_next_value": soft_next_value,
        "entropy": -(
            next_policy * (next_scaled_log_policy / tau)
        ).sum(dim=-1),
    }
    return target, diagnostics


@torch.no_grad()
def dqn_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    target_q_next: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    return rewards + gamma * (1.0 - dones.to(rewards.dtype)) * target_q_next.max(
        dim=-1
    ).values


def huber_loss(
    prediction: torch.Tensor, target: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    return F.huber_loss(prediction, target, reduction="mean", delta=delta)
