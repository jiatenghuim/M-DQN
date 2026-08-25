import torch

from mdqn.posterior import (
    BootstrapLastLayerEnsemble,
    masked_head_huber_loss,
    posterior_predictive_policy,
)
from mdqn.posterior_metrics import posterior_mechanism_metrics


def test_identical_heads_equal_point_policy() -> None:
    point_q = torch.tensor([[2.0, -1.0, 0.5], [0.0, 1.0, 4.0]])
    posterior_q = point_q[:, None, :].expand(-1, 5, -1).clone()
    pp_policy = posterior_predictive_policy(posterior_q, tau=0.3)
    point_policy = torch.softmax(point_q / 0.3, dim=-1)
    torch.testing.assert_close(pp_policy, point_policy)


def test_single_head_equals_its_policy() -> None:
    q = torch.tensor([[[1.0, 2.0, -3.0]]])
    torch.testing.assert_close(
        posterior_predictive_policy(q, tau=0.7),
        torch.softmax(q[:, 0] / 0.7, dim=-1),
    )


def test_opposing_heads_produce_half_half_policy() -> None:
    posterior_q = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    pp_policy = posterior_predictive_policy(posterior_q, tau=1.0)
    torch.testing.assert_close(pp_policy, torch.tensor([[0.5, 0.5]]))


def test_policy_is_average_of_softmaxes_not_softmax_of_average_q() -> None:
    posterior_q = torch.tensor([[[10.0, 0.0], [0.0, 1.0]]])
    actual = posterior_predictive_policy(posterior_q, tau=1.0)
    expected = torch.softmax(posterior_q, dim=-1).mean(dim=1)
    wrong = torch.softmax(posterior_q.mean(dim=1), dim=-1)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong)


def test_pp_policy_is_finite_and_normalized() -> None:
    posterior_q = torch.tensor(
        [[[1000.0, -1000.0, 0.0]], [[-20.0, -30.0, -40.0]]]
    ).expand(-1, 4, -1)
    policy = posterior_predictive_policy(posterior_q, tau=0.03)
    assert torch.isfinite(policy).all()
    torch.testing.assert_close(policy.sum(dim=-1), torch.ones(2))


def test_posterior_loss_cannot_backpropagate_into_features() -> None:
    estimator = BootstrapLastLayerEnsemble(8, 3, num_heads=2)
    features = torch.randn(4, 8, requires_grad=True)
    q = estimator(features.detach())
    predictions = q[:, :, 0]
    loss = masked_head_huber_loss(
        predictions,
        torch.ones(4),
        torch.ones(4, 2, dtype=torch.bool),
    )
    loss.backward()
    assert features.grad is None
    assert all(parameter.grad is not None for parameter in estimator.parameters())


def test_zero_bootstrap_mask_gives_zero_gradient_for_that_head() -> None:
    predictions = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0]], requires_grad=True
    )
    mask = torch.tensor([[True, False], [True, False]])
    loss = masked_head_huber_loss(predictions, torch.zeros(2), mask)
    loss.backward()
    assert predictions.grad[:, 0].abs().sum() > 0
    assert predictions.grad[:, 1].abs().sum() == 0


def test_mechanism_metrics_are_finite_for_one_head() -> None:
    q = torch.tensor([[[2.0, 1.0]], [[0.0, 3.0]]])
    metrics = posterior_mechanism_metrics(
        q,
        q[:, 0],
        torch.tensor([0, 1]),
        tau=0.3,
        alpha=0.9,
        log_policy_min=-1.0,
        eps=1e-8,
    )
    assert metrics["posterior/q_variance"] == 0
    assert all(torch.isfinite(value) for value in metrics.values())

