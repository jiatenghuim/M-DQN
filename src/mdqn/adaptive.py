from __future__ import annotations

from dataclasses import dataclass

import torch

from mdqn.algorithm import posterior_predictive_munchausen_target


def policy_disagreement_per_state(
    head_policies: torch.Tensor,
    pp_policy: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Return mean_k KL(pi_k || pi_pp) for every state in a batch."""
    if head_policies.ndim != 3:
        raise ValueError("head_policies must have shape [batch, head, action]")
    if pp_policy.shape != (
        head_policies.shape[0],
        head_policies.shape[2],
    ):
        raise ValueError("pp_policy must have shape [batch, action]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return (
        head_policies
        * (
            torch.log(head_policies.clamp_min(eps))
            - torch.log(pp_policy.clamp_min(eps)).unsqueeze(1)
        )
    ).sum(dim=-1).mean(dim=1)


def adaptive_lambda(
    policy_disagreement: torch.Tensor,
    *,
    uncertainty_reference: float,
    calibration_complete: bool,
    eps: float,
) -> torch.Tensor:
    """Map per-state policy disagreement to lambda in [0, 1]."""
    if policy_disagreement.ndim != 1:
        raise ValueError("policy_disagreement must have shape [batch]")
    if uncertainty_reference < 0.0:
        raise ValueError("uncertainty_reference must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not calibration_complete:
        return torch.ones_like(policy_disagreement)
    ratio = torch.clamp(
        policy_disagreement / (uncertainty_reference + eps),
        min=0.0,
        max=1.0,
    )
    if uncertainty_reference == 0.0:
        return ratio
    # Preserve the specified saturation U(s) >= U_ref => lambda(s) = 1
    # exactly; eps must not turn the equality boundary into 1-epsilon.
    return torch.where(
        policy_disagreement >= uncertainty_reference,
        torch.ones_like(ratio),
        ratio,
    )


def adaptive_policy(
    point_policy: torch.Tensor,
    pp_policy: torch.Tensor,
    lambda_s: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Blend point and PP policies independently for every batch state."""
    if point_policy.ndim != 2 or pp_policy.shape != point_policy.shape:
        raise ValueError("point_policy and pp_policy must match [batch, action]")
    if lambda_s.shape != point_policy.shape[:1]:
        raise ValueError("lambda_s must have shape [batch]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not torch.all((lambda_s >= 0.0) & (lambda_s <= 1.0)):
        raise ValueError("lambda_s must be in [0, 1]")

    weight = lambda_s.unsqueeze(-1)
    policy = (1.0 - weight) * point_policy + weight * pp_policy
    policy = policy / policy.sum(dim=-1, keepdim=True).clamp_min(eps)

    assert torch.isfinite(policy).all()
    assert (policy >= 0.0).all()
    assert torch.allclose(
        policy.sum(dim=-1),
        torch.ones(policy.shape[0], device=policy.device, dtype=policy.dtype),
        atol=1e-6,
        rtol=1e-6,
    )
    return policy


@torch.no_grad()
def adaptive_munchausen_target(
    rewards: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
    target_q_current: torch.Tensor,
    target_q_next: torch.Tensor,
    policy_current: torch.Tensor,
    *,
    gamma: float,
    tau: float,
    alpha: float,
    log_policy_min: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """APP target: adaptive current bonus and point-policy next operator."""
    return posterior_predictive_munchausen_target(
        rewards,
        actions,
        dones,
        target_q_current,
        target_q_next,
        policy_current,
        pp_scope="munchausen_only",
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        log_policy_min=log_policy_min,
        eps=eps,
    )


@dataclass
class UncertaintyCalibration:
    """Finite running-mean calibration that freezes after a fixed count."""

    calibration_updates: int
    adaptive_eps: float
    uncertainty_reference: float = 0.0
    uncertainty_reference_sum: float = 0.0
    uncertainty_reference_count: int = 0
    calibration_complete: bool = False

    def __post_init__(self) -> None:
        if self.calibration_updates <= 0:
            raise ValueError("calibration_updates must be positive")
        if self.adaptive_eps <= 0.0:
            raise ValueError("adaptive_eps must be positive")

    def lambda_for(self, disagreement: torch.Tensor) -> torch.Tensor:
        return adaptive_lambda(
            disagreement,
            uncertainty_reference=self.uncertainty_reference,
            calibration_complete=self.calibration_complete,
            eps=self.adaptive_eps,
        )

    def observe(self, disagreement: torch.Tensor) -> None:
        """Accumulate one learner update, then permanently freeze at the limit."""
        if self.calibration_complete:
            return
        batch_mean = float(disagreement.detach().mean())
        if not torch.isfinite(torch.tensor(batch_mean)):
            raise FloatingPointError("non-finite posterior policy disagreement")
        self.uncertainty_reference_sum += batch_mean
        self.uncertainty_reference_count += 1
        self.uncertainty_reference = (
            self.uncertainty_reference_sum / self.uncertainty_reference_count
        )
        if self.uncertainty_reference_count >= self.calibration_updates:
            self.calibration_complete = True

    def state_dict(self) -> dict[str, float | int | bool]:
        return {
            "uncertainty_reference": self.uncertainty_reference,
            "uncertainty_reference_sum": self.uncertainty_reference_sum,
            "uncertainty_reference_count": self.uncertainty_reference_count,
            "calibration_complete": self.calibration_complete,
        }

    def load_state_dict(self, state: dict) -> None:
        required = {
            "uncertainty_reference",
            "uncertainty_reference_sum",
            "uncertainty_reference_count",
            "calibration_complete",
        }
        missing = required - state.keys()
        if missing:
            raise ValueError(
                "APP-MDQN checkpoint is missing adaptive state: "
                + ", ".join(sorted(missing))
            )
        count = int(state["uncertainty_reference_count"])
        complete = bool(state["calibration_complete"])
        if count < 0 or count > self.calibration_updates:
            raise ValueError("invalid uncertainty_reference_count")
        if complete != (count >= self.calibration_updates):
            raise ValueError("inconsistent adaptive calibration checkpoint")
        self.uncertainty_reference = float(state["uncertainty_reference"])
        self.uncertainty_reference_sum = float(
            state["uncertainty_reference_sum"]
        )
        self.uncertainty_reference_count = count
        self.calibration_complete = complete
