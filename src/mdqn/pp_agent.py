from __future__ import annotations

import torch

from mdqn.agent import DQNAgent, UpdateMetrics
from mdqn.algorithm import huber_loss, posterior_predictive_munchausen_target
from mdqn.config import AlgorithmConfig
from mdqn.posterior import (
    BootstrapLastLayerEnsemble,
    masked_head_huber_loss,
    posterior_predictive_policy,
    sample_bootstrap_mask,
)
from mdqn.posterior_metrics import posterior_mechanism_metrics
from mdqn.replay import ReplayBatch


class PosteriorPredictiveMDQNAgent(DQNAgent):
    """M-DQN whose policy bootstrapping uses an approximate PP policy."""

    def __init__(
        self,
        num_actions: int,
        config: AlgorithmConfig,
        *,
        stack_size: int = 4,
        device: torch.device | str = "cpu",
    ) -> None:
        if config.name != "pp_mdqn":
            raise ValueError("PosteriorPredictiveMDQNAgent requires name='pp_mdqn'")
        super().__init__(num_actions, config, stack_size=stack_size, device=device)
        self.posterior_online = BootstrapLastLayerEnsemble(
            self.online.feature_dim,
            num_actions,
            config.num_posterior_heads,
        ).to(self.device)
        self.posterior_target = BootstrapLastLayerEnsemble(
            self.target.feature_dim,
            num_actions,
            config.num_posterior_heads,
        ).to(self.device)
        self._hard_update_posterior_target()
        self.posterior_target.eval()
        for parameter in self.posterior_target.parameters():
            parameter.requires_grad_(False)
        self.posterior_optimizer = torch.optim.Adam(
            self.posterior_online.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
        )

    def update(self, batch: ReplayBatch) -> UpdateMetrics:
        online_features = self.online.encode(batch.states)
        chosen_q = self.online.q_from_features(online_features).gather(
            1, batch.actions.long().unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            target_q_current, target_features_current = (
                self.target.forward_with_features(batch.states)
            )
            target_q_next, target_features_next = self.target.forward_with_features(
                batch.next_states
            )
            posterior_q_current = self.posterior_target(target_features_current)
            pp_policy_current = posterior_predictive_policy(
                posterior_q_current, self.config.tau
            )
            pp_policy_next = None
            if self.config.pp_scope == "full_operator":
                posterior_q_next = self.posterior_target(target_features_next)
                pp_policy_next = posterior_predictive_policy(
                    posterior_q_next, self.config.tau
                )
            target, target_diagnostics = posterior_predictive_munchausen_target(
                batch.rewards,
                batch.actions,
                batch.dones,
                target_q_current,
                target_q_next,
                pp_policy_current,
                pp_policy_next=pp_policy_next,
                pp_scope=self.config.pp_scope,
                gamma=self.config.gamma,
                tau=self.config.tau,
                alpha=self.config.alpha,
                log_policy_min=self.config.log_policy_min,
                eps=self.config.posterior_eps,
            )
            mechanism_diagnostics = posterior_mechanism_metrics(
                posterior_q_current,
                target_q_current,
                batch.actions,
                tau=self.config.tau,
                alpha=self.config.alpha,
                log_policy_min=self.config.log_policy_min,
                eps=self.config.posterior_eps,
            )

        main_loss = huber_loss(chosen_q, target, self.config.huber_delta)
        self.optimizer.zero_grad(set_to_none=True)
        main_loss.backward()
        self.optimizer.step()

        # Detachment is the causal-isolation boundary: posterior loss updates
        # last-layer heads only and cannot alter the baseline feature encoder.
        posterior_q_online = self.posterior_online(online_features.detach())
        action_index = batch.actions.long().view(-1, 1, 1).expand(
            -1, self.config.num_posterior_heads, 1
        )
        posterior_chosen_q = posterior_q_online.gather(2, action_index).squeeze(2)
        bootstrap_mask = sample_bootstrap_mask(
            batch.states.shape[0],
            self.config.num_posterior_heads,
            self.config.bootstrap_prob,
            device=self.device,
        )
        posterior_loss = masked_head_huber_loss(
            posterior_chosen_q,
            target,
            bootstrap_mask,
            delta=self.config.huber_delta,
            eps=self.config.posterior_eps,
        )
        self.posterior_optimizer.zero_grad(set_to_none=True)
        posterior_loss.backward()
        self.posterior_optimizer.step()

        diagnostics = {
            name: float(value.detach())
            for name, value in mechanism_diagnostics.items()
        }
        diagnostics["posterior/loss"] = float(posterior_loss.detach())
        return UpdateMetrics(
            loss=float(main_loss.detach()),
            q_mean=float(chosen_q.detach().mean()),
            target_mean=float(target.mean()),
            munchausen_bonus_mean=float(
                target_diagnostics["munchausen_bonus"].mean()
            ),
            entropy_mean=float(target_diagnostics["entropy"].mean()),
            diagnostics=diagnostics,
        )

    @torch.no_grad()
    def _hard_update_posterior_target(self) -> None:
        self.posterior_target.load_state_dict(self.posterior_online.state_dict())

    @torch.no_grad()
    def hard_update_target(self) -> None:
        # super().__init__ calls this before posterior modules exist.
        super().hard_update_target()
        if hasattr(self, "posterior_online"):
            self._hard_update_posterior_target()

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "posterior_online": self.posterior_online.state_dict(),
                "posterior_target": self.posterior_target.state_dict(),
                "posterior_optimizer": self.posterior_optimizer.state_dict(),
            }
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        required = {
            "posterior_online",
            "posterior_target",
            "posterior_optimizer",
        }
        missing = required - state.keys()
        if missing:
            raise ValueError(
                "PP-MDQN checkpoint is missing posterior state: "
                + ", ".join(sorted(missing))
            )
        super().load_state_dict(state)
        self.posterior_online.load_state_dict(state["posterior_online"])
        self.posterior_target.load_state_dict(state["posterior_target"])
        self.posterior_optimizer.load_state_dict(state["posterior_optimizer"])
