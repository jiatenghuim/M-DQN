"""Paper-faithful PyTorch M-DQN and reducibility-gated extension."""

from mdqn.algorithm import (
    mdqn_base_target,
    munchausen_target,
    reducibility_gated_munchausen_target,
    scaled_log_softmax,
)
from mdqn.networks import NatureDQN
from mdqn.reducibility import (
    huber_loss_per_sample,
    reducibility_gate,
    reducible_loss,
)

__all__ = [
    "NatureDQN",
    "huber_loss_per_sample",
    "mdqn_base_target",
    "munchausen_target",
    "reducibility_gate",
    "reducibility_gated_munchausen_target",
    "reducible_loss",
    "scaled_log_softmax",
]
__version__ = "0.1.0"
