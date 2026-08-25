import numpy as np

from mdqn.replay import FrameReplayBuffer


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

