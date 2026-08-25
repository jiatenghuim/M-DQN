import torch

from mdqn.algorithm import munchausen_target, scaled_log_softmax


def test_scaled_log_softmax_matches_torch_and_stays_finite() -> None:
    q = torch.tensor([[1000.0, 999.0, -1000.0], [1.0, 2.0, 3.0]])
    tau = 0.03
    actual = scaled_log_softmax(q, tau)
    expected = tau * torch.log_softmax(q / tau, dim=-1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)


def test_munchausen_target_uses_target_policy_at_current_and_next_state() -> None:
    rewards = torch.tensor([1.0, -1.0])
    actions = torch.tensor([0, 1])
    dones = torch.tensor([False, True])
    q_current = torch.tensor([[2.0, 1.0], [3.0, -4.0]])
    q_next = torch.tensor([[0.5, 1.5], [9.0, 8.0]])
    target, diagnostics = munchausen_target(
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

    current_log_pi = 0.5 * torch.log_softmax(q_current / 0.5, dim=-1)
    next_log_pi = 0.5 * torch.log_softmax(q_next / 0.5, dim=-1)
    next_pi = torch.softmax(q_next / 0.5, dim=-1)
    bonus = 0.9 * current_log_pi.gather(1, actions[:, None]).squeeze(1).clamp(-1.0, 0.0)
    soft_value = (next_pi * (q_next - next_log_pi)).sum(dim=-1)
    expected = rewards + bonus + 0.99 * (~dones).float() * soft_value

    torch.testing.assert_close(target, expected)
    torch.testing.assert_close(diagnostics["munchausen_bonus"], bonus)
    assert target[1] == rewards[1] + bonus[1]


def test_munchausen_bonus_is_clipped_before_alpha_scaling() -> None:
    target, diagnostics = munchausen_target(
        torch.tensor([0.0]),
        torch.tensor([1]),
        torch.tensor([True]),
        torch.tensor([[100.0, -100.0]]),
        torch.zeros(1, 2),
        tau=0.03,
        alpha=0.9,
        log_policy_min=-1.0,
    )
    torch.testing.assert_close(diagnostics["munchausen_bonus"], torch.tensor([-0.9]))
    torch.testing.assert_close(target, torch.tensor([-0.9]))

