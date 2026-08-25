import torch

from mdqn.agent import DQNAgent, linearly_decaying_epsilon
from mdqn.config import AlgorithmConfig


def test_target_update_is_a_hard_copy_only_when_called() -> None:
    agent = DQNAgent(4, AlgorithmConfig(), device="cpu")
    target_before = [parameter.clone() for parameter in agent.target.parameters()]
    with torch.no_grad():
        next(agent.online.parameters()).add_(1.0)
    assert all(torch.equal(a, b) for a, b in zip(target_before, agent.target.parameters()))

    agent.hard_update_target()
    assert all(
        torch.equal(online, target)
        for online, target in zip(agent.online.parameters(), agent.target.parameters())
    )


def test_dopamine_epsilon_schedule() -> None:
    assert linearly_decaying_epsilon(250_000, 0, 20_000, 0.01) == 1.0
    assert linearly_decaying_epsilon(250_000, 20_000, 20_000, 0.01) == 1.0
    assert linearly_decaying_epsilon(250_000, 270_000, 20_000, 0.01) == 0.01
    assert linearly_decaying_epsilon(250_000, 999_999, 20_000, 0.01) == 0.01

