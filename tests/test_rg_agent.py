import numpy as np
import torch

from mdqn.agent import DQNAgent
from mdqn.config import AlgorithmConfig
from mdqn.replay import ReplayBatch
from mdqn.rg_agent import ReducibilityGatedMDQNAgent


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


def test_rg_update_records_mechanism_metrics_without_extra_model_state() -> None:
    torch.manual_seed(5)
    agent = ReducibilityGatedMDQNAgent(
        3,
        AlgorithmConfig(name="rg_mdqn"),
        device="cpu",
    )
    metrics = agent.update(_batch())
    assert set(metrics.diagnostics) == {
        "reducibility/online_base_loss_mean",
        "reducibility/target_base_loss_mean",
        "reducibility/reducible_loss_mean",
        "reducibility/gate_mean",
        "reducibility/gate_std",
        "reducibility/gate_min",
        "reducibility/gate_max",
        "reducibility/positive_fraction",
        "reducibility/gate_zero_fraction",
        "reducibility/mean_abs_base_td_error",
        "munchausen/full_bonus_mean",
        "munchausen/gated_bonus_mean",
        "munchausen/bonus_attenuation_mean",
        "munchausen/point_policy_entropy",
        "munchausen/full_clip_ratio",
    }
    assert 0.0 <= metrics.diagnostics["reducibility/gate_mean"] <= 1.0
    assert set(agent.state_dict()) == {"online", "target", "optimizer"}


def test_rg_and_mdqn_behavior_policies_are_identical() -> None:
    mdqn = DQNAgent(3, AlgorithmConfig(name="mdqn"), device="cpu")
    rg = ReducibilityGatedMDQNAgent(
        3,
        AlgorithmConfig(name="rg_mdqn"),
        device="cpu",
    )
    rg.online.load_state_dict(mdqn.online.state_dict())
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    mdqn_rng = np.random.default_rng(19)
    rg_rng = np.random.default_rng(19)
    mdqn_actions = [mdqn.act(state, 0.35, mdqn_rng) for _ in range(20)]
    rg_actions = [rg.act(state, 0.35, rg_rng) for _ in range(20)]
    assert rg_actions == mdqn_actions


def test_dqn_and_mdqn_model_state_remains_unchanged() -> None:
    for name in ("dqn", "mdqn"):
        agent = DQNAgent(3, AlgorithmConfig(name=name), device="cpu")
        assert set(agent.state_dict()) == {"online", "target", "optimizer"}
