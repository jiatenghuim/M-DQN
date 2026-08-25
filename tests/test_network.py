import torch

from mdqn.networks import NatureDQN


def test_network_matches_dopamine_same_padding_shape() -> None:
    network = NatureDQN(num_actions=18)
    x = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
    assert network.features(x.float()).shape == (2, 64, 11, 11)
    assert network(x).shape == (2, 18)
    assert sum(parameter.numel() for parameter in network.parameters()) == 4_052_658

