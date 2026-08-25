from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mdqn.algorithm import dqn_target, huber_loss, munchausen_target
from mdqn.config import AlgorithmConfig
from mdqn.networks import NatureDQN
from mdqn.replay import ReplayBatch


@dataclass(frozen=True)
class UpdateMetrics:
    loss: float
    q_mean: float
    target_mean: float
    munchausen_bonus_mean: float = 0.0
    entropy_mean: float = 0.0


class DQNAgent:
    def __init__(
        self,
        num_actions: int,
        config: AlgorithmConfig,
        *,
        stack_size: int = 4,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_actions = num_actions
        self.config = config
        self.device = torch.device(device)
        self.online = NatureDQN(num_actions, stack_size).to(self.device)
        self.target = NatureDQN(num_actions, stack_size).to(self.device)
        self.hard_update_target()
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
        )

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() <= epsilon:
            return int(rng.integers(self.num_actions))
        tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return int(self.online(tensor).argmax(dim=1).item())

    def update(self, batch: ReplayBatch) -> UpdateMetrics:
        chosen_q = self.online(batch.states).gather(
            1, batch.actions.long().unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            target_q_next = self.target(batch.next_states)
            if self.config.name == "mdqn":
                target_q_current = self.target(batch.states)
                target, diagnostics = munchausen_target(
                    batch.rewards,
                    batch.actions,
                    batch.dones,
                    target_q_current,
                    target_q_next,
                    gamma=self.config.gamma,
                    tau=self.config.tau,
                    alpha=self.config.alpha,
                    log_policy_min=self.config.log_policy_min,
                )
            else:
                target = dqn_target(
                    batch.rewards, batch.dones, target_q_next, self.config.gamma
                )
                diagnostics = {}

        loss = huber_loss(chosen_q, target, self.config.huber_delta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return UpdateMetrics(
            loss=float(loss.detach()),
            q_mean=float(chosen_q.detach().mean()),
            target_mean=float(target.mean()),
            munchausen_bonus_mean=float(
                diagnostics.get("munchausen_bonus", torch.zeros(1, device=self.device)).mean()
            ),
            entropy_mean=float(
                diagnostics.get("entropy", torch.zeros(1, device=self.device)).mean()
            ),
        )

    @torch.no_grad()
    def hard_update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])


def linearly_decaying_epsilon(
    decay_period: int,
    step: int,
    warmup_steps: int,
    epsilon_final: float,
) -> float:
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon_final) * steps_left / decay_period
    bonus = float(np.clip(bonus, 0.0, 1.0 - epsilon_final))
    return epsilon_final + bonus

