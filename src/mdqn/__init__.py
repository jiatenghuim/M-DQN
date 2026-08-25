"""Paper-faithful PyTorch implementation of Munchausen DQN."""

from mdqn.algorithm import munchausen_target, scaled_log_softmax
from mdqn.networks import NatureDQN
from mdqn.posterior import (
    BootstrapLastLayerEnsemble,
    posterior_predictive_policy,
)

__all__ = [
    "BootstrapLastLayerEnsemble",
    "NatureDQN",
    "munchausen_target",
    "posterior_predictive_policy",
    "scaled_log_softmax",
]
__version__ = "0.1.0"
