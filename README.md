# Munchausen DQN：论文对齐的 PyTorch baseline

本仓库以论文公式、补充材料和 Google Research 官方 TensorFlow/Dopamine 实现为真值，提供一个干净的 PyTorch M-DQN baseline。非官方 PyTorch 仓库只作为交叉检查来源；它把论文的 hard target update 改成了 soft update，因此本项目不会沿用该改动，也不会把 soft-update 结果标为严格论文复现。

## 对齐范围

默认配置 `configs/paper_atari.yaml` 对齐 M-DQN 的 Atari 主实验：

- 目标严格使用论文 Eq. (2)/(7)：当前状态的裁剪 Munchausen 项，加下一状态 soft value；两部分都由 target network 计算。
- `alpha=0.9`、`tau=0.03`、`l0=-1`，先裁剪 `tau * log pi`，再乘 `alpha`。
- target network 只做整体 hard copy；每 8000 个 **agent step** 更新一次。由于 action repeat/frame skip 为 4，这相当于每 32000 个原始 ALE frame 更新一次。
- 1-step return、uniform replay、容量 1,000,000、batch 32；不使用 PER、n-step、Double DQN、dueling 或 distributional RL。
- 每 4 个 agent step 做一次梯度更新；20,000 agent step warm-up。
- Adam `lr=5e-5, eps=0.0003125`，Huber loss `delta=1`。
- epsilon 在 warm-up 后用 250,000 agent step 从 1.0 线性下降到 0.01。
- sticky actions 概率 0.25；84x84 灰度、最近两帧 max-pool、4 帧堆叠；不因 loss of life 终止 episode。
- replay 中的训练奖励裁剪到 `[-1, 1]`，日志中的 episode return 保留原始奖励。
- 名义训练量为 50M agent step × frame skip 4 = 200M 原始 ALE frame；每个 iteration 至少运行 250k agent step（1M frame）。与 Dopamine 一样，iteration 会跑完最后一个 episode，因此实际帧数会略微超过名义值，超额部分不抵扣下一个 iteration。

网络也保留了 Dopamine 3.0.1 的 TensorFlow `SAME` padding。它与很多 PyTorch DQN 示例采用的 `VALID` padding不同：第三层卷积输出是 `64x11x11`，不是 `64x7x7`。

逐项的公式映射、target update 时序和 TensorFlow→PyTorch 迁移说明见 `docs/reproduction_notes.md`。另提供 `configs/paper_adam_dqn.yaml`，用于论文消融中的 Adam-DQN 对照；它使用普通 max Bellman target，不能用 `alpha=0` 的 Soft-DQN 代替。

## 安装

按要求使用名为 `mdqn` 的 Conda 环境：

```powershell
conda env create -f environment.yml
conda run -n mdqn python -m pytest
```

`environment.yml` 默认安装 CUDA 13.0 版 PyTorch。训练入口会在 `torch.cuda.is_available()` 为真时优先选择 `cuda`，否则自动回退到 CPU；也可用 `--device cuda` 或 `--device cpu` 显式指定。本项目已针对 RTX 4060 驱动环境选择官方 `cu130` wheel。

可用以下命令确认实际设备：

```powershell
conda run -n mdqn python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Atari ROM 不随 ALE 一起分发。请确保你有权使用 ROM，并按 `ale-py`/Gymnasium 的说明安装；安装后可先验证：

```powershell
conda run -n mdqn python -c "import gymnasium as gym; import ale_py; gym.make('ALE/Pong-v5').close()"
```

## 训练

完整论文配置（单个游戏、单个 seed）：

```powershell
conda run -n mdqn python -m mdqn.train `
  --config configs/paper_atari.yaml `
  --game Pong `
  --seed 0 `
  --run-dir runs/pong/seed_0
```

论文聚合结果使用 60 个游戏、每个游戏 3 个 seed。建议每个 `(game, seed)` 使用独立目录。默认 replay 是磁盘映射文件，容量 1M 时仅 frame 文件约占 7.1 GB；请预留足够磁盘空间。

快速检查训练管线时可缩短步数和 replay，但此时结果不属于论文复现：

```powershell
conda run -n mdqn python -m mdqn.train `
  --config configs/paper_atari.yaml `
  --game Pong `
  --seed 0 `
  --run-dir runs/smoke `
  --total-agent-steps 1000 `
  --replay-capacity 5000
```

中断后可在同一目录加 `--resume`。checkpoint 保存 online/target/optimizer、计数器和 replay 元数据；由于 ALE 的内部随机状态不在 Gymnasium 公共 checkpoint API 中，恢复后的环境轨迹不保证与不中断运行逐 frame 相同，但训练状态不会退回。

## 输出与指标

- `episodes.csv`：每个 learning episode 的未裁剪 return。
- `iterations.csv`：每 1M 原始 frame 的训练统计，并包含论文口径的最近 100 个 learning episode 平均 return。
- `checkpoint.pt`：网络、优化器和随机数状态。
- `replay/`：持久化的 memory-efficient frame replay。
- `resolved_config.json`：实际运行配置，并明确 target update 的计量单位。

论文报告的主结果需要 3 个 seed，并按论文的 random/human baseline 做跨游戏归一化；一次短训练只能验证实现和学习管线，不能验证论文最终分数。

## 依据

- [Munchausen Reinforcement Learning（NeurIPS 2020）](https://arxiv.org/abs/2007.14430)
- [Google Research 官方实现](https://github.com/google-research/google-research/tree/master/munchausen_rl)
- [非官方 PyTorch 对照仓库](https://github.com/BY571/Munchausen-RL)（README 明确说明使用 soft update）
