from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class FrameReplayBuffer:
    """Memory-efficient FIFO replay storing one uint8 frame per transition."""

    def __init__(
        self,
        capacity: int,
        frame_shape: tuple[int, int] = (84, 84),
        stack_size: int = 4,
        *,
        seed: int = 0,
        storage_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        self.capacity = int(capacity)
        self.frame_shape = tuple(frame_shape)
        self.stack_size = int(stack_size)
        self.count = 0
        self.rng = np.random.default_rng(seed)
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None

        if self.capacity <= self.stack_size + 1:
            raise ValueError("capacity is too small for stacked replay")

        if self.storage_dir is None:
            self.frames = np.empty((capacity, *frame_shape), dtype=np.uint8)
            self.actions = np.empty(capacity, dtype=np.int32)
            self.rewards = np.empty(capacity, dtype=np.float32)
            self.dones = np.empty(capacity, dtype=np.bool_)
        else:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            mode = "r+" if resume else "w+"
            self.frames = self._memmap("frames.uint8", np.uint8, (capacity, *frame_shape), mode)
            self.actions = self._memmap("actions.int32", np.int32, (capacity,), mode)
            self.rewards = self._memmap("rewards.float32", np.float32, (capacity,), mode)
            self.dones = self._memmap("dones.bool", np.bool_, (capacity,), mode)
            if resume:
                self._load_metadata()

    def _memmap(self, name: str, dtype: np.dtype, shape: tuple[int, ...], mode: str):
        path = self.storage_dir / name
        if mode == "r+" and not path.exists():
            raise FileNotFoundError(f"Replay file missing: {path}")
        return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

    def __len__(self) -> int:
        return min(self.count, self.capacity)

    @property
    def oldest_global_index(self) -> int:
        return max(0, self.count - self.capacity)

    def add(self, frame: np.ndarray, action: int, reward: float, done: bool) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != self.frame_shape:
            raise ValueError(f"expected frame shape {self.frame_shape}, got {frame.shape}")
        index = self.count % self.capacity
        self.frames[index] = frame
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.count += 1

    def can_sample(self, batch_size: int) -> bool:
        return len(self) > self.stack_size + 1 and batch_size > 0

    def _encode_state(self, global_index: int) -> np.ndarray:
        state = np.zeros((self.stack_size, *self.frame_shape), dtype=np.uint8)
        oldest = self.oldest_global_index
        state[-1] = self.frames[global_index % self.capacity]
        output_channel = self.stack_size - 2
        previous = global_index - 1
        while output_channel >= 0 and previous >= oldest:
            if self.dones[previous % self.capacity]:
                break
            state[output_channel] = self.frames[previous % self.capacity]
            output_channel -= 1
            previous -= 1
        return state

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> ReplayBatch:
        if not self.can_sample(batch_size):
            raise RuntimeError("not enough transitions to sample")
        # The newest transition has no stored successor yet. Excluding the first
        # stack at the circular boundary prevents reading overwritten history.
        low = self.oldest_global_index + self.stack_size - 1
        high = self.count - 1
        if high <= low:
            raise RuntimeError("not enough valid stacked transitions to sample")
        indices = self.rng.integers(low, high, size=batch_size, endpoint=False)

        states = np.stack([self._encode_state(int(i)) for i in indices])
        next_states = np.stack([self._encode_state(int(i + 1)) for i in indices])
        physical = indices % self.capacity
        return ReplayBatch(
            states=torch.as_tensor(states, device=device),
            actions=torch.as_tensor(self.actions[physical].copy(), device=device),
            rewards=torch.as_tensor(self.rewards[physical].copy(), device=device),
            next_states=torch.as_tensor(next_states, device=device),
            dones=torch.as_tensor(self.dones[physical].copy(), device=device),
        )

    def flush(self) -> None:
        for array in (self.frames, self.actions, self.rewards, self.dones):
            if isinstance(array, np.memmap):
                array.flush()
        if self.storage_dir is not None:
            metadata = {
                "capacity": self.capacity,
                "frame_shape": self.frame_shape,
                "stack_size": self.stack_size,
                "count": self.count,
                "rng_state": self.rng.bit_generator.state,
            }
            with (self.storage_dir / "metadata.json").open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream)

    def _load_metadata(self) -> None:
        path = self.storage_dir / "metadata.json"
        with path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        expected = (self.capacity, list(self.frame_shape), self.stack_size)
        actual = (
            metadata["capacity"],
            metadata["frame_shape"],
            metadata["stack_size"],
        )
        if actual != expected:
            raise ValueError(f"Replay metadata mismatch: expected {expected}, got {actual}")
        self.count = int(metadata["count"])
        self.rng.bit_generator.state = metadata["rng_state"]

