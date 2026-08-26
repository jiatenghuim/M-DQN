from __future__ import annotations

import torch

from mdqn.adaptive import (
    UncertaintyCalibration,
    adaptive_munchausen_target,
    adaptive_policy,
    policy_disagreement_per_state,
)
from mdqn.agent import DQNAgent, UpdateMetrics
from mdqn.algorithm import huber_loss
from mdqn.config import AdaptivePPConfig, AlgorithmConfig
from mdqn.posterior import (
    BootstrapLastLayerEnsemble,
    masked_head_huber_loss,
    sample_bootstrap_mask,
)
from mdqn.posterior_metrics import posterior_mechanism_metrics
from mdqn.replay import ReplayBatch


class AdaptivePosteriorPredictiveMDQNAgent(DQNAgent):
    """M-DQN with uncertainty-adaptive current-state Munchausen policy."""

    def __init__(
        self,
        num_actions: int,
        config: AlgorithmConfig,
        adaptive_config: AdaptivePPConfig,
        *,
        stack_size: int = 4,
        device: torch.device | str = "cpu",
    ) -> None:
        if config.name != "app_mdqn":
            raise ValueError(
                "AdaptivePosteriorPredictiveMDQNAgent requires name='app_mdqn'"
            )
        if config.pp_scope != "munchausen_only":
            raise ValueError("APP-MDQN currently supports munchausen_only only")
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
        self.calibration = UncertaintyCalibration(
            adaptive_config.uncertainty_calibration_updates,
            adaptive_config.adaptive_eps,
        )

    def update(self, batch: ReplayBatch) -> UpdateMetrics:
        online_features = self.online.encode(batch.states)
        online_q = self.online.q_from_features(online_features)
        chosen_q = online_q.gather(
            1, batch.actions.long().unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            target_q_current, target_features_current = (
                self.target.forward_with_features(batch.states)
            )
            target_q_next = self.target(batch.next_states)
            posterior_q_current = self.posterior_target(target_features_current)

            head_policies = torch.softmax(
                posterior_q_current / self.config.tau, dim=-1
            )
            pp_policy_current = head_policies.mean(dim=1)
            point_policy_current = torch.softmax(
                target_q_current / self.config.tau, dim=-1
            )
            policy_disagreement_per_batch_state = (
                policy_disagreement_per_state(
                    head_policies,
                    pp_policy_current,
                    eps=self.config.posterior_eps,
                )
            )

            # The complete calibration window uses lambda=1. Observation is
            # committed only after this update's target has been constructed.
            lambda_s = self.calibration.lambda_for(
                policy_disagreement_per_batch_state
            )
            policy_current = adaptive_policy(
                point_policy_current,
                pp_policy_current,
                lambda_s,
                eps=self.config.posterior_eps,
            )
            target, target_diagnostics = adaptive_munchausen_target(
                batch.rewards,
                batch.actions,
                batch.dones,
                target_q_current,
                target_q_next,
                policy_current,
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

            adaptive_log_policy = torch.log(
                policy_current.clamp_min(self.config.posterior_eps)
            )
            adaptive_entropy = -(
                policy_current * adaptive_log_policy
            ).sum(dim=-1).mean()
            action_index = batch.actions.long().view(-1, 1)
            adaptive_chosen = self.config.tau * adaptive_log_policy.gather(
                1, action_index
            ).squeeze(1)
            adaptive_bonus = self.config.alpha * adaptive_chosen.clamp(
                self.config.log_policy_min, 0.0
            )
            point_pp_distance = 0.5 * (
                pp_policy_current - point_policy_current
            ).abs().sum(dim=-1).mean()

            self.calibration.observe(policy_disagreement_per_batch_state)
            mechanism_diagnostics.update(
                {
                    "posterior/policy_disagreement": (
                        policy_disagreement_per_batch_state.mean()
                    ),
                    "adaptive/lambda_mean": lambda_s.mean(),
                    "adaptive/lambda_std": lambda_s.std(correction=0),
                    "adaptive/lambda_min": lambda_s.min(),
                    "adaptive/lambda_max": lambda_s.max(),
                    "adaptive/uncertainty_reference": torch.tensor(
                        self.calibration.uncertainty_reference,
                        device=self.device,
                    ),
                    "adaptive/calibration_complete": torch.tensor(
                        float(self.calibration.calibration_complete),
                        device=self.device,
                    ),
                    "adaptive/calibration_update_count": torch.tensor(
                        float(self.calibration.uncertainty_reference_count),
                        device=self.device,
                    ),
                    "policy/adaptive_entropy": adaptive_entropy,
                    "munchausen/adaptive_bonus_mean": adaptive_bonus.mean(),
                    "adaptive/point_pp_policy_distance": point_pp_distance,
                }
            )

        main_loss = huber_loss(chosen_q, target, self.config.huber_delta)
        self.optimizer.zero_grad(set_to_none=True)
        main_loss.backward()
        self.optimizer.step()

        # Match PP-MDQN exactly: detached encoder features, Bernoulli masks,
        # an independent optimizer, and hard-synchronized target heads.
        posterior_q_online = self.posterior_online(online_features.detach())
        action_index = batch.actions.long().view(-1, 1, 1).expand(
            -1, self.config.num_posterior_heads, 1
        )
        posterior_chosen_q = posterior_q_online.gather(
            2, action_index
        ).squeeze(2)
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
            max_q_value=float(online_q.detach().max()),
            mean_td_error=float((target - chosen_q.detach()).abs().mean()),
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
                "adaptive_state": self.calibration.state_dict(),
            }
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        required = {
            "posterior_online",
            "posterior_target",
            "posterior_optimizer",
            "adaptive_state",
        }
        missing = required - state.keys()
        if missing:
            raise ValueError(
                "APP-MDQN checkpoint is missing posterior/adaptive state: "
                + ", ".join(sorted(missing))
            )
        super().load_state_dict(state)
        self.posterior_online.load_state_dict(state["posterior_online"])
        self.posterior_target.load_state_dict(state["posterior_target"])
        self.posterior_optimizer.load_state_dict(state["posterior_optimizer"])
        self.calibration.load_state_dict(state["adaptive_state"])
