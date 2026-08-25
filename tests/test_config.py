from dataclasses import replace

import pytest

from mdqn.config import ExperimentConfig


def test_soft_target_update_is_rejected() -> None:
    config = ExperimentConfig()
    config = replace(config, training=replace(config.training, target_update="soft"))
    with pytest.raises(ValueError, match="hard target updates"):
        config.validate()


def test_pp_mdqn_defaults_are_valid() -> None:
    config = ExperimentConfig()
    config = replace(config, algorithm=replace(config.algorithm, name="pp_mdqn"))
    config.validate()
    assert config.algorithm.num_posterior_heads == 5
    assert config.algorithm.bootstrap_prob == 0.8
    assert config.algorithm.pp_scope == "munchausen_only"
