"""Paper-faithful PyTorch implementation of Munchausen DQN."""

from mdqn.algorithm import munchausen_target, scaled_log_softmax
from mdqn.networks import NatureDQN

__all__ = ["NatureDQN", "munchausen_target", "scaled_log_softmax"]
__version__ = "0.1.0"

