# Adaptive Posterior-Predictive Munchausen DQN

APP-MDQN is an experimental extension of PP-MDQN. It keeps the behavior policy,
posterior estimator, optimizer settings, and hard target updates unchanged. Its
only algorithmic change is the policy used by the current-state Munchausen
bonus.

## Adaptive policy

For target-network point policy `pi_point` and the mean of per-head policies
`pi_pp`, APP-MDQN computes generalized Jensen-Shannon disagreement for each
state:

```text
U(s) = (1/K) sum_k KL(pi_k(.|s) || pi_pp(.|s))
```

The first `uncertainty_calibration_updates` learner updates use `lambda=1` and
accumulate the batch-mean disagreement. At the end of that finite window,

```text
U_ref = disagreement_sum / disagreement_count
```

is frozen permanently. All later learner updates use

```text
lambda(s) = clip(U(s) / (U_ref + adaptive_eps), 0, 1)
pi_A(.|s) = (1-lambda(s)) pi_point(.|s) + lambda(s) pi_pp(.|s)
```

The APP target is

```text
y = r + alpha clip(tau log pi_A(a_t|s_t), l0, 0)
      + gamma (1-done) V_point(s_{t+1}).
```

`V_point` is exactly the original M-DQN next-state soft Bellman operator.
Adaptive full-operator mode is intentionally unsupported.

## Isolation and checkpointing

Environment actions remain `epsilon-greedy(main online Q)`. Uncertainty never
changes epsilon or action selection. Posterior heads remain bootstrap last-layer
heads trained on detached encoder features with Bernoulli masks and synchronized
by the existing hard target event.

The agent checkpoint stores `uncertainty_reference`, its finite-window sum and
count, and `calibration_complete`. A resumed completed calibration therefore
cannot restart or drift.

## Diagnostics

In addition to PP-MDQN diagnostics, APP-MDQN records:

- `adaptive/lambda_mean`, `adaptive/lambda_std`, `adaptive/lambda_min`,
  `adaptive/lambda_max`
- `adaptive/uncertainty_reference`
- `adaptive/calibration_complete`, `adaptive/calibration_update_count`
- `policy/point_entropy`, `policy/pp_entropy`, `policy/adaptive_entropy`
- `munchausen/point_bonus_mean`, `munchausen/pp_bonus_mean`,
  `munchausen/adaptive_bonus_mean`
- `posterior/policy_disagreement`, `posterior/q_variance`
- `adaptive/point_pp_policy_distance`
