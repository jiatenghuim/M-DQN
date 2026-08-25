from dataclasses import replace

import pytest

from mdqn.config import ExperimentConfig


def test_soft_target_update_is_rejected() -> None:
    config = ExperimentConfig()
    config = replace(config, training=replace(config.training, target_update="soft"))
    with pytest.raises(ValueError, match="hard target updates"):
        config.validate()

