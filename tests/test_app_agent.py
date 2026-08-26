from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from mdqn.agent import DQNAgent
from mdqn.app_agent import AdaptivePosteriorPredictiveMDQNAgent
from mdqn.config import AdaptivePPConfig, AlgorithmConfig, ExperimentConfig
from mdqn.pp_agent import PosteriorPredictiveMDQNAgent
from mdqn.replay import ReplayBatch
from mdqn.trainer import Trainer


def _batch(batch_size: int = 2) -> ReplayBatch:
    return ReplayBatch(
        states=torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8),
        actions=torch.arange(batch_size) % 3,
        rewards=torch.linspace(-1.0, 1.0, batch_size),
        next_states=torch.randint(
            0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8
        ),
        dones=torch.tensor([False] * (batch_size - 1) + [True]),
    )


def test_app_update_records_metrics_and_checkpoints_calibration() -> None:
    torch.manual_seed(5)
    algorithm = AlgorithmConfig(
        name="app_mdqn",
        num_posterior_heads=2,
        bootstrap_prob=1.0,
        tau=0.5,
    )
    adaptive = AdaptivePPConfig(
        uncertainty_calibration_updates=1, adaptive_eps=1e-8
    )
    agent = AdaptivePosteriorPredictiveMDQNAgent(
        3, algorithm, adaptive, device="cpu"
    )
    metrics = agent.update(_batch())
    assert set(metrics.diagnostics) >= {
        "adaptive/lambda_mean",
        "adaptive/uncertainty_reference",
        "adaptive/calibration_complete",
        "policy/point_entropy",
        "policy/pp_entropy",
        "policy/adaptive_entropy",
        "munchausen/point_bonus_mean",
        "munchausen/pp_bonus_mean",
        "munchausen/adaptive_bonus_mean",
        "posterior/policy_disagreement",
        "posterior/q_variance",
        "adaptive/point_pp_policy_distance",
    }
    assert metrics.diagnostics["adaptive/lambda_mean"] == 1.0
    assert metrics.diagnostics["adaptive/calibration_complete"] == 1.0

    state = agent.state_dict()
    assert set(state["adaptive_state"]) == {
        "uncertainty_reference",
        "uncertainty_reference_sum",
        "uncertainty_reference_count",
        "calibration_complete",
    }
    restored = AdaptivePosteriorPredictiveMDQNAgent(
        3, algorithm, adaptive, device="cpu"
    )
    restored.load_state_dict(state)
    assert restored.calibration.state_dict() == agent.calibration.state_dict()


def test_app_behavior_policy_still_uses_main_online_q_only() -> None:
    agent = AdaptivePosteriorPredictiveMDQNAgent(
        3,
        AlgorithmConfig(name="app_mdqn", num_posterior_heads=2),
        AdaptivePPConfig(),
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
    assert agent.act(state, epsilon=0.0, rng=np.random.default_rng(0)) == 2


def test_baseline_agent_state_shapes_are_unchanged() -> None:
    for name in ("dqn", "mdqn"):
        agent = DQNAgent(3, AlgorithmConfig(name=name), device="cpu")
        assert set(agent.state_dict()) == {"online", "target", "optimizer"}

    pp_agent = PosteriorPredictiveMDQNAgent(
        3,
        AlgorithmConfig(name="pp_mdqn", num_posterior_heads=2),
        device="cpu",
    )
    assert set(pp_agent.state_dict()) == {
        "online",
        "target",
        "optimizer",
        "posterior_online",
        "posterior_target",
        "posterior_optimizer",
    }


def test_trainer_checkpoint_resume_preserves_frozen_adaptive_state(
    tmp_path,
) -> None:
    config = ExperimentConfig(
        algorithm=AlgorithmConfig(
            name="app_mdqn", num_posterior_heads=2, bootstrap_prob=1.0
        ),
        adaptive_pp=AdaptivePPConfig(uncertainty_calibration_updates=2),
    )
    config = replace(
        config,
        training=replace(
            config.training,
            total_agent_steps=8,
            steps_per_iteration=8,
            replay_capacity=32,
            min_replay_history=1,
            checkpoint_every_frames=4,
            metrics_log_period_frames=4,
        ),
    )
    environment = SimpleNamespace(action_space=SimpleNamespace(n=3))
    trainer = Trainer(
        environment,
        config,
        tmp_path,
        seed=7,
        device="cpu",
    )
    trainer.agent.calibration.observe(torch.tensor([0.2, 0.4]))
    trainer.agent.calibration.observe(torch.tensor([0.6, 0.8]))
    frozen = trainer.agent.calibration.state_dict().copy()
    trainer._save_checkpoint()

    checkpoint = torch.load(
        tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["agent"]["adaptive_state"] == frozen

    restored = Trainer(
        environment,
        config,
        tmp_path,
        seed=7,
        device="cpu",
        resume=True,
    )
    assert restored.agent.calibration.state_dict() == frozen
    restored.agent.calibration.observe(torch.zeros(2))
    assert restored.agent.calibration.state_dict() == frozen
