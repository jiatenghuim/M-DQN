import torch

from mdqn.reducibility import (
    huber_loss_per_sample,
    reducibility_gate,
)


def test_huber_loss_is_per_sample_and_matches_baseline_definition() -> None:
    prediction = torch.tensor([0.0, 0.0, 3.0])
    target = torch.tensor([0.5, 2.0, 0.0])
    actual = huber_loss_per_sample(prediction, target, delta=1.0)
    expected = torch.tensor([0.125, 1.5, 2.5])
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, expected)


def test_equal_online_and_target_losses_have_zero_gate() -> None:
    loss = torch.tensor([0.0, 0.5, 2.0])
    residual, gate = reducibility_gate(loss, loss)
    torch.testing.assert_close(residual, torch.zeros_like(loss))
    torch.testing.assert_close(gate, torch.zeros_like(loss))


def test_better_online_loss_has_zero_gate() -> None:
    online = torch.tensor([0.2, 0.5])
    target = torch.tensor([0.4, 1.0])
    residual, gate = reducibility_gate(online, target)
    torch.testing.assert_close(residual, torch.zeros_like(online))
    torch.testing.assert_close(gate, torch.zeros_like(online))


def test_zero_target_loss_makes_the_gate_one() -> None:
    residual, gate = reducibility_gate(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
    )
    torch.testing.assert_close(residual, torch.tensor([1.0]))
    torch.testing.assert_close(gate, torch.tensor([1.0]))


def test_partial_reducibility_uses_normalized_loss_difference() -> None:
    residual, gate = reducibility_gate(
        torch.tensor([1.0]),
        torch.tensor([0.6]),
    )
    torch.testing.assert_close(residual, torch.tensor([0.4]))
    torch.testing.assert_close(gate, torch.tensor([0.4]))


def test_random_nonnegative_losses_produce_detached_bounded_gates() -> None:
    torch.manual_seed(7)
    online = torch.rand(256, requires_grad=True)
    target = torch.rand(256, requires_grad=True)
    residual, gate = reducibility_gate(online, target)
    assert not residual.requires_grad
    assert not gate.requires_grad
    assert (gate >= 0.0).all()
    assert (gate <= 1.0).all()
