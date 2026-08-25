from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtariStep:
    observation: np.ndarray
    reward: float
    game_over: bool
    info: dict


class DopamineAtariPreprocessing:
    """Dopamine 3.0.1 preprocessing on top of a frame-skip-1 ALE env."""

    def __init__(self, environment, frame_skip: int = 4, screen_size: int = 84):
        if frame_skip <= 0 or screen_size <= 0:
            raise ValueError("frame_skip and screen_size must be positive")
        self.environment = environment
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.screen_buffer: list[np.ndarray] = []
        self.game_over = False

    @property
    def action_space(self):
        return self.environment.action_space

    def _grayscale(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.uint8)
        if observation.ndim == 2:
            return observation
        import cv2

        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    def _pool_and_resize(self) -> np.ndarray:
        import cv2

        pooled = np.maximum(self.screen_buffer[0], self.screen_buffer[1])
        return cv2.resize(
            pooled,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8, copy=False)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        observation, info = self.environment.reset(seed=seed)
        gray = self._grayscale(observation)
        self.screen_buffer = [gray.copy(), np.zeros_like(gray)]
        self.game_over = False
        return self._pool_and_resize(), info

    def step(self, action: int) -> AtariStep:
        accumulated_reward = 0.0
        info: dict = {}
        game_over = False
        for time_step in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.environment.step(action)
            accumulated_reward += float(reward)
            game_over = bool(terminated or truncated)
            if game_over:
                break
            if time_step >= self.frame_skip - 2:
                slot = time_step - (self.frame_skip - 2)
                self.screen_buffer[slot][...] = self._grayscale(observation)
        self.game_over = game_over
        return AtariStep(self._pool_and_resize(), accumulated_reward, game_over, info)

    def close(self) -> None:
        self.environment.close()


def make_atari(
    game: str,
    sticky_action_probability: float = 0.25,
    frame_skip: int = 4,
    screen_size: int = 84,
):
    try:
        import ale_py  # noqa: F401
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "Atari dependencies are missing. Install the project with the 'atari' extra."
        ) from exc

    env_id = game if game.startswith("ALE/") else f"ALE/{game}-v5"
    base = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=sticky_action_probability,
        full_action_space=False,
        obs_type="grayscale",
    ).unwrapped
    return DopamineAtariPreprocessing(
        base, frame_skip=frame_skip, screen_size=screen_size
    )
