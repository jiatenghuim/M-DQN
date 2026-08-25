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
from mdqn.pp_agent import PosteriorPredictiveMDQNAgent
from mdqn.replay import FrameReplayBuffer
from mdqn.utils.logger import ExperimentLogger, NullExperimentLogger
from mdqn.utils.results import ResultsExporter


PP_LOG_METRICS = (
    "posterior/loss",
    "posterior/q_variance",
    "posterior/head_q_std",
    "posterior/policy_disagreement",
    "policy/point_entropy",
    "policy/pp_entropy",
    "munchausen/point_clip_ratio",
    "munchausen/pp_clip_ratio",
    "munchausen/point_bonus_mean",
    "munchausen/pp_bonus_mean",
    "munchausen/bonus_difference",
)

SWAN_DIAGNOSTIC_NAMES = {
    "policy/point_entropy": "point_policy_entropy",
    "policy/pp_entropy": "posterior_predictive_entropy",
    "munchausen/point_unclipped_bonus_mean": "point_munchausen_bonus",
    "munchausen/pp_unclipped_bonus_mean": "pp_munchausen_bonus",
    "munchausen/unclipped_bonus_difference": "bonus_difference",
    "munchausen/point_bonus_mean": "actual_point_munchausen_bonus",
    "munchausen/pp_bonus_mean": "actual_pp_munchausen_bonus",
    "munchausen/point_clip_ratio": "point_clip_ratio",
    "munchausen/pp_clip_ratio": "pp_clip_ratio",
    "posterior/q_variance": "posterior_q_variance",
    "posterior/policy_disagreement": "posterior_policy_disagreement",
    "posterior/head_q_std": "posterior_head_q_std",
    "posterior/loss": "loss/posterior_loss",
}


def training_metric_payload(metrics) -> dict[str, float]:
    payload = {
        "loss/q_loss": metrics.loss,
        "mean_q_value": metrics.q_mean,
        "max_q_value": metrics.max_q_value,
        "mean_td_error": metrics.mean_td_error,
    }
    for internal_name, external_name in SWAN_DIAGNOSTIC_NAMES.items():
        if internal_name in metrics.diagnostics:
            payload[external_name] = metrics.diagnostics[internal_name]
    return payload


def next_frame_boundary(current_frame: int, period: int | None) -> int | None:
    if period is None:
        return None
    return (current_frame // period + 1) * period


def validate_algorithm_isolation(algorithm: str, agent: DQNAgent) -> None:
    has_posterior = hasattr(agent, "posterior_online") or hasattr(
        agent, "posterior_target"
    )
    if algorithm in {"dqn", "mdqn"}:
        if type(agent) is not DQNAgent or has_posterior:
            raise RuntimeError(f"{algorithm} must not initialize posterior heads")
    elif algorithm == "pp_mdqn":
        if not isinstance(agent, PosteriorPredictiveMDQNAgent) or not has_posterior:
            raise RuntimeError("pp_mdqn requires online and target posterior heads")
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")


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
        game: str = "Unknown",
        experiment_logger: ExperimentLogger | None = None,
        runtime_metadata: dict | None = None,
    ) -> None:
        self.environment = environment
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.game = game
        self.device = torch.device(device)
        self.experiment_logger = experiment_logger or NullExperimentLogger()
        self.runtime_metadata = runtime_metadata or {}
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        agent_class = (
            PosteriorPredictiveMDQNAgent
            if config.algorithm.name == "pp_mdqn"
            else DQNAgent
        )
        self.agent = agent_class(
            environment.action_space.n,
            config.algorithm,
            stack_size=config.environment.stack_size,
            device=self.device,
        )
        validate_algorithm_isolation(config.algorithm.name, self.agent)
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
        self._diagnostic_sums = {name: 0.0 for name in PP_LOG_METRICS}
        self._diagnostic_updates = 0
        self._training_metric_sums: dict[str, float] = {}
        self._training_metric_updates = 0
        self._latest_compact_metrics: dict[str, float] = {}
        self._iteration_phase_steps = 0
        self._iteration_returns: list[float] = []
        self._iteration_lengths: list[int] = []
        self.final_return: float | None = None
        self.best_return: float | None = None
        self.training_time_seconds = 0.0
        self._active_train_started: float | None = None
        if resume:
            self._load_checkpoint()
        self._write_config()
        runtime_config = self._runtime_config_payload()
        self.results = ResultsExporter(
            self.run_dir, runtime_config, resume=resume
        )
        raw_frames = self.agent_steps * self.config.environment.frame_skip
        self._next_metrics_frame = next_frame_boundary(
            raw_frames, self.config.training.metrics_log_period_frames
        )
        self._next_checkpoint_frame = next_frame_boundary(
            raw_frames, self.config.training.checkpoint_every_frames
        )

    def _runtime_config_payload(self) -> dict:
        return self.config.to_dict() | {
            "game": self.game,
            "seed": self.seed,
            "device": str(self.device),
            "actual_total_frames": self.config.training.total_agent_steps
            * self.config.environment.frame_skip,
            "target_update_units": "agent_steps",
            "raw_frames_per_agent_step": self.config.environment.frame_skip,
        } | self.runtime_metadata

    def _write_config(self) -> None:
        payload = self._runtime_config_payload()
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
                payload = training_metric_payload(self.last_metrics)
                self._training_metric_updates += 1
                for name, value in payload.items():
                    self._training_metric_sums[name] = (
                        self._training_metric_sums.get(name, 0.0) + value
                    )
                if (
                    self.config.algorithm.name == "pp_mdqn"
                    and self.last_metrics.diagnostics
                ):
                    self._diagnostic_updates += 1
                    for name in PP_LOG_METRICS:
                        self._diagnostic_sums[name] += self.last_metrics.diagnostics[
                            name
                        ]
                self._maybe_log_training_metrics()
            # Official Dopamine order: optimize first, then hard-copy.
            if self.agent_steps % train.target_update_period == 0:
                self.agent.hard_update_target()

    def _maybe_log_training_metrics(self) -> None:
        raw_frames = self.agent_steps * self.config.environment.frame_skip
        if (
            self._next_metrics_frame is None
            or raw_frames < self._next_metrics_frame
            or self._training_metric_updates == 0
        ):
            return
        averaged = {
            name: value / self._training_metric_updates
            for name, value in self._training_metric_sums.items()
        }
        averaged["global_step"] = raw_frames
        self.experiment_logger.log(averaged, step=raw_frames)
        self._latest_compact_metrics = averaged
        self.results.append_metrics(
            step=raw_frames,
            episode_return=None,
            loss=averaged.get("loss/q_loss"),
            entropy=self._selected_entropy(averaged),
            uncertainty=averaged.get("posterior_q_variance"),
            bonus=self._selected_bonus(averaged),
        )
        self._training_metric_sums = {}
        self._training_metric_updates = 0
        self._next_metrics_frame = next_frame_boundary(
            raw_frames, self.config.training.metrics_log_period_frames
        )

    def _selected_entropy(self, metrics: dict[str, float]) -> float | None:
        if self.config.algorithm.name == "pp_mdqn":
            return metrics.get("posterior_predictive_entropy")
        if self.config.algorithm.name == "mdqn":
            return metrics.get("point_policy_entropy")
        return None

    def _selected_bonus(self, metrics: dict[str, float]) -> float | None:
        if self.config.algorithm.name == "pp_mdqn":
            return metrics.get("actual_pp_munchausen_bonus")
        if self.config.algorithm.name == "mdqn":
            return metrics.get("actual_point_munchausen_bonus")
        return None

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
        self.final_return = episode_return
        self.best_return = (
            episode_return
            if self.best_return is None
            else max(self.best_return, episode_return)
        )
        return episode_steps, episode_return

    def train(self) -> None:
        train = self.config.training
        num_iterations = math.ceil(train.total_agent_steps / train.steps_per_iteration)
        self._active_train_started = time.time()
        completed = False
        try:
            while self.iteration < num_iterations:
                nominal_steps_before = self.iteration * train.steps_per_iteration
                phase_minimum = min(
                    train.steps_per_iteration,
                    train.total_agent_steps - nominal_steps_before,
                )
                iteration_started = time.time()
                while self._iteration_phase_steps < phase_minimum:
                    length, episode_return = self._run_episode()
                    self._iteration_lengths.append(length)
                    self._iteration_returns.append(episode_return)
                    self._iteration_phase_steps += length
                    self._append_episode_log(length, episode_return)
                    print(
                        f"steps={self.agent_steps} frames={self.agent_steps * self.config.environment.frame_skip} "
                        f"episode={self.episode_number} length={length} return={episode_return:.1f}",
                        flush=True,
                    )
                    self._maybe_save_periodic_checkpoint()
                elapsed = max(time.time() - iteration_started, 1e-9)
                self._append_iteration_log(
                    self.iteration,
                    self._iteration_returns,
                    self._iteration_lengths,
                    elapsed,
                )
                self.iteration += 1
                self._iteration_phase_steps = 0
                self._iteration_returns = []
                self._iteration_lengths = []
                if self.iteration % train.checkpoint_every_iterations == 0:
                    self._save_checkpoint()
            completed = True
        finally:
            self.training_time_seconds = self._elapsed_training_time()
            self._active_train_started = None
        if completed:
            self._save_checkpoint()
            self.results.write_summary(
                {
                    "final_return": self.final_return,
                    "best_return": self.best_return,
                    "training_time_seconds": self.training_time_seconds,
                    "seed": self.seed,
                    "algorithm": self.config.algorithm.name,
                    "game": self.game,
                    "final_global_step": self.agent_steps
                    * self.config.environment.frame_skip,
                }
            )

    def _elapsed_training_time(self) -> float:
        if self._active_train_started is None:
            return self.training_time_seconds
        return self.training_time_seconds + time.time() - self._active_train_started

    def _maybe_save_periodic_checkpoint(self) -> None:
        raw_frames = self.agent_steps * self.config.environment.frame_skip
        if (
            self._next_checkpoint_frame is None
            or raw_frames < self._next_checkpoint_frame
        ):
            return
        # Saving at the first episode boundary after the threshold keeps the
        # replay trajectory boundary valid when ALE state itself is unavailable.
        self._save_checkpoint()
        self._next_checkpoint_frame = next_frame_boundary(
            raw_frames, self.config.training.checkpoint_every_frames
        )

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
        raw_frames = self.agent_steps * self.config.environment.frame_skip
        self.experiment_logger.log(
            {
                "episode_return": episode_return,
                "episode_length": length,
                "global_step": raw_frames,
            },
            step=raw_frames,
        )
        latest = self._latest_compact_metrics
        self.results.append_metrics(
            step=raw_frames,
            episode_return=episode_return,
            loss=latest.get("loss/q_loss"),
            entropy=self._selected_entropy(latest),
            uncertainty=latest.get("posterior_q_variance"),
            bonus=self._selected_bonus(latest),
        )

    def _append_iteration_log(
        self, iteration: int, returns: list[float], lengths: list[int], elapsed: float
    ) -> None:
        path = self.run_dir / "iterations.csv"
        new_file = not path.exists()
        last_100 = list(self.recent_returns)
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            header = [
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
            row = [
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
            if self.config.algorithm.name == "pp_mdqn":
                header.extend(PP_LOG_METRICS)
                if self._diagnostic_updates:
                    row.extend(
                        self._diagnostic_sums[name] / self._diagnostic_updates
                        for name in PP_LOG_METRICS
                    )
                else:
                    row.extend("" for _ in PP_LOG_METRICS)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)
        self._diagnostic_sums = {name: 0.0 for name in PP_LOG_METRICS}
        self._diagnostic_updates = 0

    def _save_checkpoint(self) -> None:
        self.replay.flush()
        checkpoint = {
            "iteration": self.iteration,
            "iteration_phase_steps": self._iteration_phase_steps,
            "iteration_returns": self._iteration_returns,
            "iteration_lengths": self._iteration_lengths,
            "agent_steps": self.agent_steps,
            "episode_number": self.episode_number,
            "recent_returns": list(self.recent_returns),
            "final_return": self.final_return,
            "best_return": self.best_return,
            "training_time_seconds": self._elapsed_training_time(),
            "diagnostic_sums": self._diagnostic_sums,
            "diagnostic_updates": self._diagnostic_updates,
            "training_metric_sums": self._training_metric_sums,
            "training_metric_updates": self._training_metric_updates,
            "latest_compact_metrics": self._latest_compact_metrics,
            "agent": self.agent.state_dict(),
            "replay": {
                "count": self.replay.count,
                "capacity": self.replay.capacity,
                "storage_dir": str(self.replay.storage_dir),
            },
            "exploration_rng": self.rng.bit_generator.state,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            checkpoint["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, self.run_dir / "checkpoint.pt")

    def _load_checkpoint(self) -> None:
        path = self.run_dir / "checkpoint.pt"
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint["agent"])
        self.iteration = int(checkpoint["iteration"])
        self._iteration_phase_steps = int(
            checkpoint.get("iteration_phase_steps", 0)
        )
        self._iteration_returns = list(checkpoint.get("iteration_returns", []))
        self._iteration_lengths = list(checkpoint.get("iteration_lengths", []))
        self.agent_steps = int(checkpoint["agent_steps"])
        self.episode_number = int(checkpoint["episode_number"])
        self.recent_returns.extend(checkpoint.get("recent_returns", []))
        self.final_return = checkpoint.get("final_return")
        self.best_return = checkpoint.get("best_return")
        self.training_time_seconds = float(
            checkpoint.get("training_time_seconds", 0.0)
        )
        self._diagnostic_sums.update(checkpoint.get("diagnostic_sums", {}))
        self._diagnostic_updates = int(checkpoint.get("diagnostic_updates", 0))
        self._training_metric_sums = dict(
            checkpoint.get("training_metric_sums", {})
        )
        self._training_metric_updates = int(
            checkpoint.get("training_metric_updates", 0)
        )
        self._latest_compact_metrics = dict(
            checkpoint.get("latest_compact_metrics", {})
        )
        replay_state = checkpoint.get("replay")
        if replay_state is not None:
            if int(replay_state["count"]) != self.replay.count:
                raise ValueError("checkpoint and replay transition counts differ")
            if int(replay_state["capacity"]) != self.replay.capacity:
                raise ValueError("checkpoint and replay capacities differ")
        self.rng.bit_generator.state = checkpoint["exploration_rng"]
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        # map_location may move this CPU RNG tensor to CUDA together with the
        # model checkpoint, but torch.set_rng_state requires a CPU ByteTensor.
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and "torch_cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["torch_cuda_rng"]]
            )
