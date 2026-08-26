import inspect
from dataclasses import fields

import numpy as np

from mdqn.replay import FrameReplayBuffer, ReplayBatch


def test_replay_zeroes_stack_across_episode_boundaries() -> None:
    replay = FrameReplayBuffer(20, frame_shape=(2, 2), stack_size=4, seed=0)
    for value in range(8):
        replay.add(
            np.full((2, 2), value, dtype=np.uint8),
            action=value,
            reward=float(value),
            done=value == 4,
        )
    state = replay._encode_state(6)
    assert state[:, 0, 0].tolist() == [0, 0, 5, 6]


def test_replay_samples_matching_state_and_successor() -> None:
    replay = FrameReplayBuffer(20, frame_shape=(1, 1), stack_size=4, seed=1)
    for value in range(10):
        replay.add(np.array([[value]], dtype=np.uint8), 0, 0.0, False)
    batch = replay.sample(16)
    np.testing.assert_array_equal(
        batch.states[:, 1:, 0, 0].numpy(),
        batch.next_states[:, :-1, 0, 0].numpy(),
    )


def test_replay_interface_remains_uniform_without_priority_weights() -> None:
    parameters = tuple(inspect.signature(FrameReplayBuffer.sample).parameters)
    assert parameters == ("self", "batch_size", "device")
    assert {field.name for field in fields(ReplayBatch)} == {
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
    }
    source = inspect.getsource(FrameReplayBuffer.sample).lower()
    assert "priority" not in source
    assert "importance" not in source
