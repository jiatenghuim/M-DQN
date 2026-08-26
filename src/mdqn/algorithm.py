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
    current_policy = torch.softmax(target_q_current / tau, dim=-1)
    next_scaled_log_policy = scaled_log_softmax(target_q_next, tau)
    next_policy = torch.softmax(target_q_next / tau, dim=-1)

    chosen_unclipped_log_policy = current_scaled_log_policy.gather(
        1, actions.long().unsqueeze(1)
    ).squeeze(1)
    # The mathematical upper bound is zero. The official TF max of 1 is a
    # no-op in exact arithmetic; zero expresses the paper's [.]^0_l0 notation.
    chosen_log_policy = chosen_unclipped_log_policy.clamp(
        min=log_policy_min, max=0.0
    )
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
        "point_policy_entropy": -(
            current_policy * (current_scaled_log_policy / tau)
        ).sum(dim=-1),
        "point_clip_ratio": chosen_unclipped_log_policy < log_policy_min,
        "point_unclipped_bonus": alpha * chosen_unclipped_log_policy,
    }
    return target, diagnostics


@torch.no_grad()
def mdqn_base_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    target_q_next: torch.Tensor,
    *,
    gamma: float = 0.99,
    tau: float = 0.03,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """M-DQN next-state soft target without a current-state bonus."""
    next_scaled_log_policy = scaled_log_softmax(target_q_next, tau)
    next_policy = torch.softmax(target_q_next / tau, dim=-1)
    soft_next_value = (
        next_policy * (target_q_next - next_scaled_log_policy)
    ).sum(dim=-1)
    base_target = rewards + gamma * (
        1.0 - dones.to(dtype=rewards.dtype)
    ) * soft_next_value
    diagnostics = {
        "soft_next_value": soft_next_value,
        "next_policy_entropy": -(
            next_policy * (next_scaled_log_policy / tau)
        ).sum(dim=-1),
    }
    return base_target, diagnostics


@torch.no_grad()
def reducibility_gated_munchausen_target(
    rewards: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
    target_q_current: torch.Tensor,
    target_q_next: torch.Tensor,
    gate: torch.Tensor,
    *,
    gamma: float = 0.99,
    tau: float = 0.03,
    alpha: float = 0.9,
    log_policy_min: float = -1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a detached per-transition gate to the current M-DQN bonus."""
    if gate.shape != rewards.shape:
        raise ValueError("gate must have shape [batch]")
    if not torch.isfinite(gate).all():
        raise FloatingPointError("gate must be finite")
    if not ((gate >= 0.0) & (gate <= 1.0)).all():
        raise ValueError("gate must be in [0, 1]")

    base_target, base_diagnostics = mdqn_base_target(
        rewards,
        dones,
        target_q_next,
        gamma=gamma,
        tau=tau,
    )
    current_scaled_log_policy = scaled_log_softmax(target_q_current, tau)
    current_policy = torch.softmax(target_q_current / tau, dim=-1)
    chosen_unclipped_log_policy = current_scaled_log_policy.gather(
        1, actions.long().unsqueeze(1)
    ).squeeze(1)
    chosen_log_policy = chosen_unclipped_log_policy.clamp(
        min=log_policy_min, max=0.0
    )
    full_bonus = alpha * chosen_log_policy
    gated_bonus = gate.detach() * full_bonus
    target = base_target + gated_bonus
    diagnostics = {
        **base_diagnostics,
        "base_target": base_target,
        "full_bonus": full_bonus,
        "gated_bonus": gated_bonus,
        "point_policy_entropy": -(
            current_policy * (current_scaled_log_policy / tau)
        ).sum(dim=-1),
        "full_clip_mask": chosen_unclipped_log_policy < log_policy_min,
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
