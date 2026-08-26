from dataclasses import replace

import pytest

from mdqn.config import ExperimentConfig, load_config


def test_soft_target_update_is_rejected() -> None:
    config = ExperimentConfig()
    config = replace(config, training=replace(config.training, target_update="soft"))
    with pytest.raises(ValueError, match="hard target updates"):
        config.validate()


def test_rg_mdqn_uses_only_baseline_algorithm_parameters() -> None:
    config = ExperimentConfig()
    config = replace(config, algorithm=replace(config.algorithm, name="rg_mdqn"))
    config.validate()
    assert set(config.to_dict()["algorithm"]) == {
        "name",
        "gamma",
        "tau",
        "alpha",
        "log_policy_min",
        "learning_rate",
        "adam_epsilon",
        "huber_delta",
    }
    assert set(config.to_dict()) == {"algorithm", "training", "environment"}


def test_debug_config_is_exactly_500k_raw_frames() -> None:
    config = load_config("configs/debug_rg_mdqn.yaml")
    assert config.algorithm.name == "rg_mdqn"
    assert config.training.total_frames == 500_000
    assert config.training.total_agent_steps == 125_000
    assert config.training.checkpoint_every_frames == 50_000
    assert config.training.metrics_log_period_frames == 10_000


def test_paper_config_budget_is_unchanged() -> None:
    config = load_config("configs/paper_atari.yaml")
    assert config.training.total_frames is None
    assert config.training.total_agent_steps == 50_000_000
