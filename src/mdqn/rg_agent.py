from __future__ import annotations

import torch

from mdqn.agent import DQNAgent, UpdateMetrics
from mdqn.algorithm import (
    huber_loss,
    mdqn_base_target,
    reducibility_gated_munchausen_target,
)
from mdqn.config import AlgorithmConfig
from mdqn.reducibility import huber_loss_per_sample, reducibility_gate
from mdqn.replay import ReplayBatch


class ReducibilityGatedMDQNAgent(DQNAgent):
    """M-DQN with a detached per-transition gate on its current bonus."""

    def __init__(
        self,
        num_actions: int,
        config: AlgorithmConfig,
        *,
        stack_size: int = 4,
        device: torch.device | str = "cpu",
    ) -> None:
        if config.name != "rg_mdqn":
            raise ValueError(
                "ReducibilityGatedMDQNAgent requires name='rg_mdqn'"
            )
        super().__init__(
            num_actions,
            config,
            stack_size=stack_size,
            device=device,
        )

    def update(self, batch: ReplayBatch) -> UpdateMetrics:
        online_q = self.online(batch.states)
        chosen_online_q = online_q.gather(
            1, batch.actions.long().unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            target_q_current = self.target(batch.states)
            target_q_next = self.target(batch.next_states)
            chosen_target_q = target_q_current.gather(
                1, batch.actions.long().unsqueeze(1)
            ).squeeze(1)

            # Reducibility is defined against one shared target without the
            # current-state Munchausen term, avoiding a circular gate.
            base_target, _ = mdqn_base_target(
                batch.rewards,
                batch.dones,
                target_q_next,
                gamma=self.config.gamma,
                tau=self.config.tau,
            )
            online_base_loss = huber_loss_per_sample(
                chosen_online_q.detach(),
                base_target,
                delta=self.config.huber_delta,
            )
            target_base_loss = huber_loss_per_sample(
                chosen_target_q,
                base_target,
                delta=self.config.huber_delta,
            )
            residual, gate = reducibility_gate(
                online_base_loss,
                target_base_loss,
            )
            target, target_diagnostics = (
                reducibility_gated_munchausen_target(
                    batch.rewards,
                    batch.actions,
                    batch.dones,
                    target_q_current,
                    target_q_next,
                    gate,
                    gamma=self.config.gamma,
                    tau=self.config.tau,
                    alpha=self.config.alpha,
                    log_policy_min=self.config.log_policy_min,
                )
            )

        loss = huber_loss(
            chosen_online_q,
            target,
            self.config.huber_delta,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        full_bonus = target_diagnostics["full_bonus"]
        gated_bonus = target_diagnostics["gated_bonus"]
        diagnostics = {
            "reducibility/online_base_loss_mean": float(
                online_base_loss.mean()
            ),
            "reducibility/target_base_loss_mean": float(
                target_base_loss.mean()
            ),
            "reducibility/reducible_loss_mean": float(residual.mean()),
            "reducibility/gate_mean": float(gate.mean()),
            "reducibility/gate_std": float(gate.std(correction=0)),
            "reducibility/gate_min": float(gate.min()),
            "reducibility/gate_max": float(gate.max()),
            "reducibility/positive_fraction": float(
                (residual > 0.0).float().mean()
            ),
            "reducibility/gate_zero_fraction": float(
                (gate < 1e-6).float().mean()
            ),
            "reducibility/mean_abs_base_td_error": float(
                (base_target - chosen_online_q.detach()).abs().mean()
            ),
            "munchausen/full_bonus_mean": float(full_bonus.mean()),
            "munchausen/gated_bonus_mean": float(gated_bonus.mean()),
            "munchausen/bonus_attenuation_mean": float(gate.mean()),
            "munchausen/point_policy_entropy": float(
                target_diagnostics["point_policy_entropy"].mean()
            ),
            "munchausen/full_clip_ratio": float(
                target_diagnostics["full_clip_mask"].float().mean()
            ),
        }
        return UpdateMetrics(
            loss=float(loss.detach()),
            q_mean=float(chosen_online_q.detach().mean()),
            target_mean=float(target.mean()),
            max_q_value=float(online_q.detach().max()),
            mean_td_error=float(
                (target - chosen_online_q.detach()).abs().mean()
            ),
            munchausen_bonus_mean=float(gated_bonus.mean()),
            entropy_mean=float(
                target_diagnostics["point_policy_entropy"].mean()
            ),
            diagnostics=diagnostics,
        )
