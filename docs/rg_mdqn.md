# Reducibility-Gated Munchausen DQN

RG-MDQN 仅改变 sampled transition 的 current-state Munchausen bonus 强度。
网络、uniform replay、epsilon-greedy behavior policy、optimizer、hard target update
以及 next-state M-DQN soft operator 均保持 baseline 不变。

## 无循环依赖的 base target

使用 main target network 在下一状态构造原始 M-DQN soft value：

```text
pi_target(a|s') = softmax(Q_target(s',a) / tau)
V_target(s') = sum_a pi_target(a|s') [Q_target(s',a) - tau log pi_target(a|s')]
y_base = r + gamma (1-done) V_target(s')
```

`y_base` 不包含 current-state Munchausen bonus。

## Reducibility gate

online 和 target 当前状态动作值共享同一个 `y_base`：

```text
L_online = Huber(Q_online(s,a), y_base)
L_target = Huber(Q_target(s,a), y_base)
R = relu(L_online - L_target)
g = clamp(R / (L_online + 1e-8), 0, 1)
```

所有量均逐 transition 计算，shape 为 `[batch]`；`R` 和 `g` 完全 detached，
不会参与梯度传播。固定的 `1e-8` 只用于避免除零，不暴露为可调参数。

## 最终 target

```text
m_full = alpha * clip(tau log pi_target(a|s), l0, 0)
m_RG = g * m_full
y_RG = y_base + m_RG
```

当 `g=1` 时恢复原始 M-DQN target；当 `g=0` 时只关闭当前 transition 的
Munchausen bonus，下一状态 soft operator仍然保留。

本方法受 Reducible Loss / ReLo 启发，但不使用 replay prioritization、PER、
importance weight 或 priority tree。环境交互仍是 main online Q network 的
epsilon-greedy。
