import torch

from mdqn.adaptive import (
    UncertaintyCalibration,
    adaptive_lambda,
    adaptive_munchausen_target,
    adaptive_policy,
    policy_disagreement_per_state,
)
from mdqn.algorithm import munchausen_target, posterior_predictive_munchausen_target


POINT_POLICY = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
PP_POLICY = torch.tensor([[0.5, 0.5], [0.6, 0.4]])


def test_lambda_zero_recovers_point_policy() -> None:
    actual = adaptive_policy(
        POINT_POLICY, PP_POLICY, torch.zeros(2), eps=1e-8
    )
    torch.testing.assert_close(actual, POINT_POLICY)


def test_lambda_one_recovers_pp_policy() -> None:
    actual = adaptive_policy(
        POINT_POLICY, PP_POLICY, torch.ones(2), eps=1e-8
    )
    torch.testing.assert_close(actual, PP_POLICY)


def test_intermediate_lambda_is_exact_convex_mixture() -> None:
    actual = adaptive_policy(
        POINT_POLICY, PP_POLICY, torch.full((2,), 0.5), eps=1e-8
    )
    torch.testing.assert_close(actual, 0.5 * POINT_POLICY + 0.5 * PP_POLICY)


def test_zero_uncertainty_maps_to_zero_after_calibration() -> None:
    actual = adaptive_lambda(
        torch.zeros(3),
        uncertainty_reference=0.25,
        calibration_complete=True,
        eps=1e-8,
    )
    torch.testing.assert_close(actual, torch.zeros(3))


def test_uncertainty_at_or_above_reference_maps_to_one() -> None:
    actual = adaptive_lambda(
        torch.tensor([0.25, 0.5]),
        uncertainty_reference=0.25,
        calibration_complete=True,
        eps=1e-8,
    )
    assert torch.equal(actual, torch.ones(2))


def test_adaptive_policy_is_finite_nonnegative_and_normalized() -> None:
    actual = adaptive_policy(
        POINT_POLICY,
        PP_POLICY,
        torch.tensor([0.2, 0.75]),
        eps=1e-8,
    )
    assert torch.isfinite(actual).all()
    assert (actual >= 0.0).all()
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(2))


def test_policy_disagreement_is_per_state_generalized_js() -> None:
    heads = torch.tensor(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.7, 0.3], [0.7, 0.3]],
        ]
    )
    pp = heads.mean(dim=1)
    actual = policy_disagreement_per_state(heads, pp, eps=1e-8)
    expected = (
        heads * (torch.log(heads) - torch.log(pp).unsqueeze(1))
    ).sum(dim=-1).mean(dim=1)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[1], torch.tensor(0.0))


def test_calibration_uses_lambda_one_then_freezes_running_mean() -> None:
    calibration = UncertaintyCalibration(2, 1e-8)
    torch.testing.assert_close(
        calibration.lambda_for(torch.tensor([0.1, 0.2])), torch.ones(2)
    )
    calibration.observe(torch.tensor([0.2, 0.4]))
    torch.testing.assert_close(
        torch.tensor(calibration.uncertainty_reference), torch.tensor(0.3)
    )
    assert not calibration.calibration_complete

    # This second update still used lambda=1; completion takes effect on the
    # following learner update.
    torch.testing.assert_close(
        calibration.lambda_for(torch.tensor([0.5, 0.7])), torch.ones(2)
    )
    calibration.observe(torch.tensor([0.6, 0.8]))
    assert calibration.calibration_complete
    assert calibration.uncertainty_reference_count == 2
    torch.testing.assert_close(
        torch.tensor(calibration.uncertainty_reference), torch.tensor(0.5)
    )

    frozen = calibration.state_dict().copy()
    calibration.observe(torch.zeros(2))
    assert calibration.state_dict() == frozen
    torch.testing.assert_close(
        calibration.lambda_for(torch.tensor([0.0, 0.25, 0.5])),
        torch.tensor([0.0, 0.5, 1.0]),
    )


def test_target_degenerates_to_mdqn_at_zero_and_pp_mdqn_at_one() -> None:
    rewards = torch.tensor([0.25, -0.5])
    actions = torch.tensor([0, 1])
    dones = torch.tensor([False, True])
    q_current = torch.tensor([[1.0, 0.5], [-0.2, 0.3]])
    q_next = torch.tensor([[0.3, 0.8], [0.9, -0.4]])
    point = torch.softmax(q_current / 0.3, dim=-1)
    pp = torch.tensor([[0.4, 0.6], [0.75, 0.25]])

    app_zero, _ = adaptive_munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        adaptive_policy(point, pp, torch.zeros(2), eps=1e-8),
        gamma=0.99,
        tau=0.3,
        alpha=0.9,
        log_policy_min=-1.0,
        eps=1e-8,
    )
    mdqn, _ = munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        gamma=0.99,
        tau=0.3,
        alpha=0.9,
        log_policy_min=-1.0,
    )
    torch.testing.assert_close(app_zero, mdqn)

    app_one, _ = adaptive_munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        adaptive_policy(point, pp, torch.ones(2), eps=1e-8),
        gamma=0.99,
        tau=0.3,
        alpha=0.9,
        log_policy_min=-1.0,
        eps=1e-8,
    )
    pp_mdqn, _ = posterior_predictive_munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        pp,
        pp_scope="munchausen_only",
        gamma=0.99,
        tau=0.3,
        alpha=0.9,
        log_policy_min=-1.0,
        eps=1e-8,
    )
    torch.testing.assert_close(app_one, pp_mdqn)


def test_adaptive_target_keeps_original_mdqn_next_state_operator() -> None:
    rewards = torch.tensor([0.5])
    actions = torch.tensor([1])
    dones = torch.tensor([False])
    q_current = torch.tensor([[4.0, -3.0]])
    q_next = torch.tensor([[1.0, 2.0]])
    adaptive_current = torch.tensor([[0.3, 0.7]])
    target, _ = adaptive_munchausen_target(
        rewards,
        actions,
        dones,
        q_current,
        q_next,
        adaptive_current,
        gamma=0.9,
        tau=0.5,
        alpha=0.8,
        log_policy_min=-1.0,
        eps=1e-8,
    )
    point_next = torch.softmax(q_next / 0.5, dim=-1)
    point_next_scaled_log = 0.5 * torch.log_softmax(q_next / 0.5, dim=-1)
    point_next_value = (
        point_next * (q_next - point_next_scaled_log)
    ).sum(dim=-1)
    adaptive_bonus = 0.8 * (
        0.5 * torch.log(torch.tensor(0.7))
    ).clamp(-1.0, 0.0)
    torch.testing.assert_close(
        target, rewards + adaptive_bonus + 0.9 * point_next_value
    )
