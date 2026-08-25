from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str = "mdqn"
    gamma: float = 0.99
    tau: float = 0.03
    alpha: float = 0.9
    log_policy_min: float = -1.0
    learning_rate: float = 5e-5
    adam_epsilon: float = 0.0003125
    huber_delta: float = 1.0
    num_posterior_heads: int = 5
    bootstrap_prob: float = 0.8
    pp_scope: str = "munchausen_only"
    posterior_eps: float = 1e-8


@dataclass(frozen=True)
class TrainingConfig:
    total_agent_steps: int = 50_000_000
    steps_per_iteration: int = 250_000
    update_period: int = 4
    target_update_period: int = 8_000
    target_update: str = "hard"
    min_replay_history: int = 20_000
    batch_size: int = 32
    replay_capacity: int = 1_000_000
    epsilon_train: float = 0.01
    epsilon_eval: float = 0.001
    epsilon_decay_period: int = 250_000
    max_steps_per_episode: int = 27_000
    checkpoint_every_iterations: int = 1


@dataclass(frozen=True)
class EnvironmentConfig:
    frame_skip: int = 4
    screen_size: int = 84
    stack_size: int = 4
    sticky_action_probability: float = 0.25
    terminal_on_life_loss: bool = False
    reward_clip: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def validate(self) -> None:
        if self.algorithm.name not in {"mdqn", "dqn", "pp_mdqn"}:
            raise ValueError("algorithm.name must be 'dqn', 'mdqn', or 'pp_mdqn'")
        if not 0.0 <= self.algorithm.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.algorithm.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.algorithm.log_policy_min >= 0.0:
            raise ValueError("log_policy_min must be negative")
        if self.algorithm.num_posterior_heads <= 0:
            raise ValueError("num_posterior_heads must be positive")
        if not 0.0 < self.algorithm.bootstrap_prob <= 1.0:
            raise ValueError("bootstrap_prob must be in (0, 1]")
        if self.algorithm.pp_scope not in {"munchausen_only", "full_operator"}:
            raise ValueError(
                "pp_scope must be 'munchausen_only' or 'full_operator'"
            )
        if self.algorithm.posterior_eps <= 0.0:
            raise ValueError("posterior_eps must be positive")
        if self.training.target_update != "hard":
            raise ValueError(
                "The paper baseline only supports hard target updates; "
                "soft/Polyak updates are intentionally rejected."
            )
        positive = {
            "update_period": self.training.update_period,
            "target_update_period": self.training.target_update_period,
            "batch_size": self.training.batch_size,
            "replay_capacity": self.training.replay_capacity,
            "frame_skip": self.environment.frame_skip,
            "stack_size": self.environment.stack_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    config = ExperimentConfig(
        algorithm=AlgorithmConfig(**raw.get("algorithm", {})),
        training=TrainingConfig(**raw.get("training", {})),
        environment=EnvironmentConfig(**raw.get("environment", {})),
    )
    config.validate()
    return config
