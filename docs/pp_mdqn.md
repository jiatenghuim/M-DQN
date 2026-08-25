# Posterior-Predictive Munchausen DQN

PP-MDQN is an experimental extension of the paper-faithful M-DQN baseline. It
tests whether replacing an uncertainty-blind point-policy bootstrap signal with
an approximate posterior-predictive policy improves learning while leaving
exploration unchanged.

## Approximate posterior estimator

The first-stage estimator is a `BootstrapLastLayerEnsemble` with five
independently initialized linear Q heads by default. It is a low-cost
approximate epistemic ensemble, not an exact Bayesian posterior.

Each head consumes the baseline network's detached 512-dimensional penultimate
features. Its loss cannot update the convolutional encoder or main Q head. For
each replay transition `i` and head `k`, an independent mask is sampled:

```text
m[i, k] ~ Bernoulli(bootstrap_prob)
```

Each head uses the shared PP-MDQN Bellman target and the baseline Huber loss.
Masked loss is normalized independently per head, then averaged over heads.
The posterior optimizer is Adam with exactly the baseline learning rate and
epsilon. Online posterior heads are hard-copied to target posterior heads at
the same 8000-agent-step events as the main target network.

## Posterior-predictive policy

For posterior Q samples shaped `[batch, K, action]`, the implemented policy is:

```text
pi_pp(a | s) = (1 / K) sum_k softmax(Q_k(s, .) / tau)[a]
```

The probabilities are averaged after each head's softmax. The implementation
does not take the softmax of mean Q. Log probabilities use
`log(pi_pp.clamp_min(posterior_eps))`.

## Scopes

`munchausen_only` changes only the current-state Munchausen reward term:

```text
y = r + alpha clip(tau log pi_pp(a_t | s_t), l0, 0)
      + gamma (1-done) V_point(s_{t+1})
```

`full_operator` additionally uses the PP policy in the next-state soft
operator, while retaining the main target Q network as the value estimator:

```text
V_pp(s') = sum_a pi_pp(a | s')
                 [Q_target-main(s', a) - tau log pi_pp(a | s')]
```

Neither scope uses ensemble-mean Q as the value estimate.

## Behavior policy

All environment actions still come from the inherited baseline behavior:

```text
epsilon-greedy(main online Q)
```

Posterior sampling, voting, UCB, uncertainty bonuses, Thompson sampling and
other exploration changes are intentionally absent.

## Diagnostics

PP runs append iteration-averaged batch diagnostics to `iterations.csv`:

- `posterior/q_variance`
- `posterior/policy_disagreement`
- `policy/point_entropy`
- `policy/pp_entropy`
- `munchausen/point_clip_ratio`
- `munchausen/pp_clip_ratio`
- `munchausen/point_bonus_mean`
- `munchausen/pp_bonus_mean`
- `munchausen/bonus_difference` (PP minus point)

The generalized Jensen-Shannon disagreement is implemented as the mean of
`KL(pi_k || pi_pp)` over posterior heads and batch states.

