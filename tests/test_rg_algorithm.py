import torch

from mdqn.algorithm import (
    mdqn_base_target,
    munchausen_target,
    reducibility_gated_munchausen_target,
)


def _inputs():
    return (
        torch.tensor([0.5, -0.25]),
        torch.tensor([1, 0]),
        torch.tensor([False, True]),
        torch.tensor([[1.2, 0.4], [-0.2, 0.8]]),
        torch.tensor([[0.3, 1.1], [2.0, -1.0]]),
    )


def test_gate_one_recovers_original_mdqn_target() -> None:
    rewards, actions, dones, q_current, q_next = _inputs()
    mdqn, _ = munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        gamma=0.99,
        tau=0.5,
        alpha=0.9,
        log_policy_min=-1.0,
    )
    rg, diagnostics = reducibility_gated_munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        torch.ones_like(rewards),
        gamma=0.99,
        tau=0.5,
        alpha=0.9,
        log_policy_min=-1.0,
    )
    torch.testing.assert_close(rg, mdqn)
    torch.testing.assert_close(
        diagnostics["gated_bonus"], diagnostics["full_bonus"]
    )


def test_gate_zero_removes_only_the_current_bonus() -> None:
    rewards, actions, dones, q_current, q_next = _inputs()
    base, _ = mdqn_base_target(
        rewards,
        dones,
        q_next,
        gamma=0.99,
        tau=0.5,
    )
    rg, diagnostics = reducibility_gated_munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        torch.zeros_like(rewards),
        gamma=0.99,
        tau=0.5,
        alpha=0.9,
        log_policy_min=-1.0,
    )
    torch.testing.assert_close(rg, base)
    torch.testing.assert_close(
        diagnostics["gated_bonus"], torch.zeros_like(rewards)
    )


def test_rg_next_state_soft_value_matches_mdqn_exactly() -> None:
    rewards, actions, dones, q_current, q_next = _inputs()
    _, mdqn_diagnostics = munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        tau=0.5,
    )
    _, rg_diagnostics = reducibility_gated_munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        torch.tensor([0.2, 0.8]),
        tau=0.5,
    )
    torch.testing.assert_close(
        rg_diagnostics["soft_next_value"],
        mdqn_diagnostics["soft_next_value"],
        atol=0.0,
        rtol=0.0,
    )
