from __future__ import annotations

import torch
from torch import nn


class NatureDQN(nn.Module):
    """Dopamine 3.0.1 Nature DQN network, including TensorFlow SAME padding."""

    def __init__(self, num_actions: int, stack_size: int = 4) -> None:
        super().__init__()
        # TF SAME padding is asymmetric for the 4x4, stride-2 convolution.
        self.features = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(stack_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self.apply(self._initialize_like_tf_keras)

    @staticmethod
    def _initialize_like_tf_keras(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Replay observations remain uint8 until transfer to the accelerator.
        state = state.to(dtype=torch.float32) / 255.0
        return self.head(self.features(state))
