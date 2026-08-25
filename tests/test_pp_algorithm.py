import torch

from mdqn.algorithm import posterior_predictive_munchausen_target


def test_munchausen_only_uses_pp_bonus_and_point_next_operator() -> None:
    rewards = torch.tensor([0.5])
    actions = torch.tensor([1])
    dones = torch.tensor([False])
    q_current = torch.tensor([[8.0, -2.0]])
    q_next = torch.tensor([[1.0, 2.0]])
    pp_current = torch.tensor([[0.25, 0.75]])
    target, diagnostics = posterior_predictive_munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        pp_current,
        pp_scope="munchausen_only",
        gamma=0.9,
        tau=0.5,
        alpha=0.8,
        log_policy_min=-1.0,
    )
    point_next = torch.softmax(q_next / 0.5, dim=-1)
    point_next_scaled_log = 0.5 * torch.log_softmax(q_next / 0.5, dim=-1)
    next_value = (point_next * (q_next - point_next_scaled_log)).sum(dim=-1)
    bonus = 0.8 * (0.5 * torch.log(torch.tensor(0.75))).clamp(-1.0, 0.0)
    torch.testing.assert_close(target, rewards + bonus + 0.9 * next_value)
    torch.testing.assert_close(diagnostics["munchausen_bonus"], bonus[None])


def test_full_operator_uses_pp_policy_but_main_q_value() -> None:
    rewards = torch.tensor([0.0])
    actions = torch.tensor([0])
    dones = torch.tensor([False])
    main_q_next = torch.tensor([[10.0, -3.0]])
    pp_current = torch.tensor([[0.6, 0.4]])
    pp_next = torch.tensor([[0.2, 0.8]])
    target, _ = posterior_predictive_munchausen_target(
        rewards,
        actions,
        dones,
        torch.zeros_like(main_q_next),
        main_q_next,
        pp_current,
        pp_policy_next=pp_next,
        pp_scope="full_operator",
        gamma=0.99,
        tau=0.3,
        alpha=0.9,
        log_policy_min=-1.0,
    )
    bonus = 0.9 * (0.3 * torch.log(torch.tensor(0.6))).clamp(-1.0, 0.0)
    pp_value = (
        pp_next * (main_q_next - 0.3 * torch.log(pp_next))
    ).sum(dim=-1)
    torch.testing.assert_close(target, bonus + 0.99 * pp_value)


def test_full_operator_requires_next_pp_policy() -> None:
    try:
        posterior_predictive_munchausen_target(
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.full((1, 2), 0.5),
            pp_scope="full_operator",
        )
    except ValueError as exc:
        assert "pp_policy_next" in str(exc)
    else:
        raise AssertionError("missing pp_policy_next should fail")

