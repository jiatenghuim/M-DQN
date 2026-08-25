import torch

from mdqn.networks import NatureDQN


def test_network_matches_dopamine_same_padding_shape() -> None:
    network = NatureDQN(num_actions=18)
    x = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
    assert network.features(x.float()).shape == (2, 64, 11, 11)
    assert network(x).shape == (2, 18)
    assert sum(parameter.numel() for parameter in network.parameters()) == 4_052_658


def test_feature_api_preserves_original_forward_computation() -> None:
    torch.manual_seed(3)
    network = NatureDQN(num_actions=6)
    x = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    original_path = network.head(network.features(x.float() / 255.0))
    q_values, features = network.forward_with_features(x)
    torch.testing.assert_close(q_values, original_path)
    torch.testing.assert_close(network.q_from_features(features), original_path)
