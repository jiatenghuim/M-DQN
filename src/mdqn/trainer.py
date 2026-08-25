from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from mdqn.agent import DQNAgent, linearly_decaying_epsilon
from mdqn.config import ExperimentConfig
from mdqn.replay import FrameReplayBuffer


class Trainer:
    def __init__(
        self,
        environment,
        config: ExperimentConfig,
        run_dir: str | Path,
        *,
        seed: int,
        device: str,
        resume: bool = False,
    ) -> None:
        self.environment = environment
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.agent = DQNAgent(
            environment.action_space.n,
            config.algorithm,
            stack_size=config.environment.stack_size,
            device=self.device,
        )
        self.replay = FrameReplayBuffer(
            config.training.replay_capacity,
            frame_shape=(config.environment.screen_size, config.environment.screen_size),
            stack_size=config.environment.stack_size,
            seed=seed,
            storage_dir=self.run_dir / "replay",
            resume=resume,
        )
        self.agent_steps = 0
        self.episode_number = 0
        self.iteration = 0
        self.recent_returns: deque[float] = deque(maxlen=100)
        self.last_metrics = None
        if resume:
            self._load_checkpoint()
        self._write_config()

    def _write_config(self) -> None:
        payload = self.config.to_dict() | {
            "seed": self.seed,
            "device": str(self.device),
            "target_update_units": "agent_steps",
            "raw_frames_per_agent_step": self.config.environment.frame_skip,
        }
        with (self.run_dir / "resolved_config.json").open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

    def _state_stack(self, frames: deque[np.ndarray]) -> np.ndarray:
        stack_size = self.config.environment.stack_size
        state = np.zeros(
            (stack_size, self.config.environment.screen_size, self.config.environment.screen_size),
            dtype=np.uint8,
        )
        values = list(frames)[-stack_size:]
        state[-len(values) :] = values
        return state

    def _learn_after_transition(self) -> None:
        train = self.config.training
        if len(self.replay) > train.min_replay_history:
            if self.agent_steps % train.update_period == 0:
                batch = self.replay.sample(train.batch_size, self.device)
                self.last_metrics = self.agent.update(batch)
            # Official Dopamine order: optimize first, then hard-copy.
            if self.agent_steps % train.target_update_period == 0:
                self.agent.hard_update_target()

    def _run_episode(self) -> tuple[int, float]:
        initial_seed = self.seed if self.agent_steps == 0 and self.episode_number == 0 else None
        observation, _ = self.environment.reset(seed=initial_seed)
        frames: deque[np.ndarray] = deque(maxlen=self.config.environment.stack_size)
        frames.append(observation)
        episode_return = 0.0
        episode_steps = 0

        while True:
            epsilon = linearly_decaying_epsilon(
                self.config.training.epsilon_decay_period,
                self.agent_steps + 1,
                self.config.training.min_replay_history,
                self.config.training.epsilon_train,
            )
            state = self._state_stack(frames)
            action = self.agent.act(state, epsilon, self.rng)
            result = self.environment.step(action)
            episode_steps += 1
            episode_return += result.reward
            time_limit = episode_steps >= self.config.training.max_steps_per_episode
            done = result.game_over or time_limit
            clipped_reward = float(
                np.clip(
                    result.reward,
                    -self.config.environment.reward_clip,
                    self.config.environment.reward_clip,
                )
            )
            self.replay.add(observation, action, clipped_reward, done)
            self.agent_steps += 1
            self._learn_after_transition()
            if done:
                break
            observation = result.observation
            frames.append(observation)

        self.episode_number += 1
        self.recent_returns.append(episode_return)
        return episode_steps, episode_return

    def train(self) -> None:
        train = self.config.training
        num_iterations = math.ceil(train.total_agent_steps / train.steps_per_iteration)
        while self.iteration < num_iterations:
            nominal_steps_before = self.iteration * train.steps_per_iteration
            phase_minimum = min(
                train.steps_per_iteration,
                train.total_agent_steps - nominal_steps_before,
            )
            returns: list[float] = []
            lengths: list[int] = []
            phase_steps = 0
            started = time.time()
            while phase_steps < phase_minimum:
                length, episode_return = self._run_episode()
                lengths.append(length)
                returns.append(episode_return)
                phase_steps += length
                self._append_episode_log(length, episode_return)
                print(
                    f"steps={self.agent_steps} frames={self.agent_steps * self.config.environment.frame_skip} "
                    f"episode={self.episode_number} length={length} return={episode_return:.1f}",
                    flush=True,
                )
            elapsed = max(time.time() - started, 1e-9)
            self._append_iteration_log(self.iteration, returns, lengths, elapsed)
            self.iteration += 1
            if self.iteration % train.checkpoint_every_iterations == 0:
                self._save_checkpoint()

    def _append_episode_log(self, length: int, episode_return: float) -> None:
        path = self.run_dir / "episodes.csv"
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(["episode", "agent_steps", "raw_frames", "length", "return"])
            writer.writerow(
                [
                    self.episode_number,
                    self.agent_steps,
                    self.agent_steps * self.config.environment.frame_skip,
                    length,
                    episode_return,
                ]
            )

    def _append_iteration_log(
        self, iteration: int, returns: list[float], lengths: list[int], elapsed: float
    ) -> None:
        path = self.run_dir / "iterations.csv"
        new_file = not path.exists()
        last_100 = list(self.recent_returns)
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(
                    [
                        "iteration",
                        "agent_steps",
                        "raw_frames",
                        "episodes",
                        "mean_return",
                        "last_100_mean_return",
                        "mean_length",
                        "agent_steps_per_second",
                        "loss",
                    ]
                )
            writer.writerow(
                [
                    iteration,
                    self.agent_steps,
                    self.agent_steps * self.config.environment.frame_skip,
                    len(returns),
                    float(np.mean(returns)),
                    float(np.mean(last_100)),
                    float(np.mean(lengths)),
                    float(sum(lengths) / elapsed),
                    self.last_metrics.loss if self.last_metrics else "",
                ]
            )

    def _save_checkpoint(self) -> None:
        self.replay.flush()
        checkpoint = {
            "iteration": self.iteration,
            "agent_steps": self.agent_steps,
            "episode_number": self.episode_number,
            "recent_returns": list(self.recent_returns),
            "agent": self.agent.state_dict(),
            "exploration_rng": self.rng.bit_generator.state,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
        }
        torch.save(checkpoint, self.run_dir / "checkpoint.pt")

    def _load_checkpoint(self) -> None:
        path = self.run_dir / "checkpoint.pt"
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint["agent"])
        self.iteration = int(checkpoint["iteration"])
        self.agent_steps = int(checkpoint["agent_steps"])
        self.episode_number = int(checkpoint["episode_number"])
        self.recent_returns.extend(checkpoint.get("recent_returns", []))
        self.rng.bit_generator.state = checkpoint["exploration_rng"]
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
