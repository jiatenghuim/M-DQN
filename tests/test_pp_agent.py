import numpy as np
import torch

from mdqn.agent import DQNAgent
from mdqn.config import AlgorithmConfig
from mdqn.pp_agent import PosteriorPredictiveMDQNAgent
from mdqn.replay import ReplayBatch


def _batch(batch_size: int = 4) -> ReplayBatch:
    return ReplayBatch(
        states=torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8),
        actions=torch.arange(batch_size) % 3,
        rewards=torch.linspace(-1.0, 1.0, batch_size),
        next_states=torch.randint(
            0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8
        ),
        dones=torch.tensor([False] * (batch_size - 1) + [True]),
    )


def test_mdqn_baseline_does_not_construct_posterior_modules() -> None:
    agent = DQNAgent(3, AlgorithmConfig(name="mdqn"), device="cpu")
    assert not hasattr(agent, "posterior_online")
    assert set(agent.state_dict()) == {"online", "target", "optimizer"}
    metrics = agent.update(_batch(2))
    assert metrics.diagnostics == {}


def test_pp_behavior_policy_uses_main_q_only() -> None:
    agent = PosteriorPredictiveMDQNAgent(
        3,
        AlgorithmConfig(name="pp_mdqn", num_posterior_heads=2),
        device="cpu",
    )
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        agent.online.head[3].bias[2] = 1.0
        for head in agent.posterior_online.heads:
            head.weight.zero_()
            head.bias.zero_()
            head.bias[0] = 100.0
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    action = agent.act(state, epsilon=0.0, rng=np.random.default_rng(0))
    assert action == 2


def test_pp_update_trains_heads_and_hard_syncs_target_heads() -> None:
    torch.manual_seed(9)
    agent = PosteriorPredictiveMDQNAgent(
        3,
        AlgorithmConfig(
            name="pp_mdqn",
            num_posterior_heads=2,
            bootstrap_prob=1.0,
            tau=0.5,
        ),
        device="cpu",
    )
    online_before = [p.clone() for p in agent.posterior_online.parameters()]
    target_before = [p.clone() for p in agent.posterior_target.parameters()]
    metrics = agent.update(_batch())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(online_before, agent.posterior_online.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, agent.posterior_target.parameters())
    )
    assert set(metrics.diagnostics) >= {
        "posterior/q_variance",
        "posterior/policy_disagreement",
        "munchausen/bonus_difference",
    }

    agent.hard_update_target()
    assert all(
        torch.equal(online, target)
        for online, target in zip(
            agent.posterior_online.parameters(), agent.posterior_target.parameters()
        )
    )
