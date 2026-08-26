# Munchausen DQN 与 RG-MDQN（PyTorch）

本项目以 Munchausen RL 论文公式和 Google Research 官方 TensorFlow/Dopamine
实现为真值，提供论文对齐的 PyTorch DQN、M-DQN baseline，以及
Reducibility-Gated Munchausen DQN（RG-MDQN）研究扩展。

最终支持的算法只有：

```text
dqn
mdqn
rg_mdqn
```

## 论文 baseline 对齐

`configs/paper_atari.yaml` 保持 M-DQN Atari 主实验设置：

- 当前状态使用裁剪后的 Munchausen bonus，下一状态使用 target network 的 soft value；
- `alpha=0.9`、`tau=0.03`、`l0=-1`；
- target network 每 8000 agent steps 整体 hard copy；
- uniform replay，容量 1,000,000，batch size 32，1-step return；
- 每 4 agent steps 更新一次，20,000 agent steps replay warm-up；
- Adam `lr=5e-5, eps=0.0003125`，Huber loss `delta=1`；
- epsilon 在 warm-up 后用 250,000 agent steps 从 1.0 降到 0.01；
- 84×84 灰度图、4 帧堆叠、frame skip 4、sticky actions 0.25；
- 50M agent steps，即 200M 原始 ALE frames。

不使用 PER、importance sampling、n-step、Double DQN、dueling、soft/Polyak
target update 或额外 exploration 方法。详细复现说明见
[`docs/reproduction_notes.md`](docs/reproduction_notes.md)。

## RG-MDQN

RG-MDQN uses a per-transition reducibility gate to control the current-state
Munchausen shaping term while leaving replay sampling, behavior policy, and the
next-state soft Bellman operator unchanged.

它受 Reducible Loss / ReLo 思想启发，但 **不使用 ReLo replay
prioritization，也不实现 PER**。首先构造不含 current-state bonus 的 target：

```text
y_base = r + gamma * (1-done) * V_target(s')
```

然后在完全相同的 `y_base` 上比较 online 与 stale target network 的逐样本
Huber loss：

```text
R = relu(L_online - L_target)
g = clamp(R / (L_online + 1e-8), 0, 1)
y_RG = y_base + g * alpha * clip(tau * log pi_target(a|s), l0, 0)
```

`1e-8` 是固定的数值稳定项，不作为配置参数。RG-MDQN 没有 threshold、
temperature、可学习 gate、额外 optimizer 或新增可调超参数。完整公式与隔离边界见
[`docs/rg_mdqn.md`](docs/rg_mdqn.md)。

## 安装与 GPU

按项目约定使用 `mdqn` Conda 环境：

```powershell
conda env create -f environment.yml
conda activate mdqn
python -m pytest
```

入口会在 CUDA 可用时默认选择 GPU，也可以显式指定 `--device cuda`：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Atari ROM 不随 ALE 分发；请确保拥有合法 ROM 并按 `ale-py`/Gymnasium 文档安装。

## 500K-frame sanity check

默认配置为 `configs/debug_rg_mdqn.yaml`，默认游戏为 Breakout。该配置使用
500,000 原始 frames（125,000 agent steps），并保持三个算法之间的 replay、
epsilon、optimizer、网络和 target-update 参数一致。

```powershell
# DQN
python -m mdqn.train --config configs/debug_rg_mdqn.yaml --algo dqn --game Breakout --seed 0 --frames 500000 --device cuda --use-swanlab

# M-DQN
python -m mdqn.train --config configs/debug_rg_mdqn.yaml --algo mdqn --game Breakout --seed 0 --frames 500000 --device cuda --use-swanlab

# RG-MDQN
python -m mdqn.train --config configs/debug_rg_mdqn.yaml --algo rg_mdqn --game Breakout --seed 0 --frames 500000 --device cuda --use-swanlab
```

以上只是用户手动运行示例；代码修改和测试流程不会自动启动 Atari 训练。

## SwanLab

SwanLab project 为 `RG-MDQN`，实验名自动生成为类似
`rg_mdqn_Breakout_seed0`。首次在线上传前执行：

```powershell
swanlab login
```

基础曲线：

- `episode_return`、`episode_length`、`global_step`；
- `loss/q_loss`、`mean_q_value`、`max_q_value`、`mean_td_error`。

RG 机制曲线：

- `reducibility/online_base_loss_mean`；
- `reducibility/target_base_loss_mean`；
- `reducibility/reducible_loss_mean`；
- `reducibility/gate_mean/std/min/max`；
- `reducibility/positive_fraction`；
- `reducibility/gate_zero_fraction`；
- `reducibility/mean_abs_base_td_error`；
- `target/steps_since_update`；
- `munchausen/full_bonus_mean`；
- `munchausen/gated_bonus_mean`；
- `munchausen/bonus_attenuation_mean`；
- `munchausen/point_policy_entropy`；
- `munchausen/full_clip_ratio`。

`gate_mean≈0` 表示 shaping 基本关闭；`gate_mean≈1` 表示 RG-MDQN 接近原始
M-DQN。`target/steps_since_update` 用于检查 hard target sync 前后的 gate 变化，
不会改变 hard update 时序。

## Checkpoint 与结果

debug 配置每 10,000 frames 聚合一次指标，每跨过 50,000-frame 阈值后在第一个
episode 边界保存 checkpoint。恢复实验需要显式指定原目录：

```powershell
python -m mdqn.train --config configs/debug_rg_mdqn.yaml --algo rg_mdqn --game Breakout --seed 0 --device cuda --use-swanlab --resume --run-dir "runs\rg_mdqn_Breakout_seed0\原时间戳目录"
```

运行目录包含：

- `checkpoint.pt`：online/target network、optimizer、训练计数和 RNG state；
- `replay/`：持久化 uniform frame replay；
- `episodes.csv`、`iterations.csv`；
- `results/metrics.csv`、`results/config_used.yaml`、`results/summary.json`；
- `swanlog/`：启用 SwanLab 时的日志缓存。

## 依据

- [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430)
- [Google Research 官方实现](https://github.com/google-research/google-research/tree/master/munchausen_rl)
