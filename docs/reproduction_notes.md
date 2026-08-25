# Reproduction notes

## Sources of truth

The implementation was checked against three primary artifacts:

1. Paper Eq. (2), supplementary Eq. (7), Algorithm 1, and Appendix B.1.
2. Google Research `munchausen_rl/agents/m_dqn.py` and `configs/atari.gin`.
3. Dopamine 3.0.1 `DQNAgent`, `AtariPreprocessing`, `NatureDQNNetwork`, and `TrainRunner`.

The BY571 PyTorch repository is not an algorithmic source of truth. Its README explicitly replaces the paper's periodic hard update with a soft update.

## Formula-to-code map

| Paper quantity | Code |
| --- | --- |
| `tau * log pi_bar(a_t | s_t)` | `scaled_log_softmax(target_q_current, tau)` |
| `[tau * log pi]^0_l0` | `.clamp(min=log_policy_min, max=0)` |
| `alpha [tau * log pi]^0_l0` | `munchausen_bonus` |
| `pi_bar(. | s_{t+1})` | `softmax(target_q_next / tau)` |
| soft next value | `sum(pi * (target_q_next - tau_log_pi_next))` |
| terminal handling | multiplication by `1 - done` on the bootstrap only |
| regression loss | mean Huber loss with `delta=1` |

Both current-state and next-state policy terms are computed by the frozen target network. The online network is used only for the regressed `Q(s_t, a_t)`.

## Target network semantics

- Online parameters are copied in full to the target at initialization, as stated by Algorithm 1.
- During learning, an optimizer step is performed first when due; then the complete online state dict is copied when `agent_steps % 8000 == 0`.
- No Polyak coefficient and no soft update path exist. Configuration validation rejects `target_update: soft`.
- Periods are counted in post-frame-skip agent steps, matching Dopamine's `training_steps`. With frame skip 4, 8000 agent steps represent 32000 raw ALE frames.

## TensorFlow-to-PyTorch details

- Dopamine 3.0.1 uses Keras `Conv2D(..., padding="same")`. Explicit asymmetric zero padding reproduces its `84 -> 21 -> 11 -> 11` spatial sizes in PyTorch.
- Keras defaults are mirrored with Xavier-uniform kernels and zero biases.
- Adam uses the official `epsilon=0.0003125`, not PyTorch's default.
- PyTorch Huber loss with `delta=1` matches TensorFlow's default Huber form.

## Atari protocol

- ALE sticky-action probability 0.25; minimal action set.
- Four raw frames per agent step, max-pooling the final two grayscale frames.
- Area resize to 84x84 and four-frame state stacking.
- True game-over or 27000 agent steps terminates an episode; losing a life does not.
- Raw episode returns are logged, while replay rewards are clipped to `[-1, 1]`.
- Each nominal 1M-frame iteration completes its final episode, just like Dopamine. This can slightly exceed 200M total raw frames.

Modern Gymnasium/ALE replaces the obsolete Gym/atari-py API. This is an interface migration; the listed preprocessing and learning semantics remain explicit in this repository.

