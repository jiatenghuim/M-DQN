from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from mdqn.atari import make_atari
from mdqn.config import load_config
from mdqn.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train paper-faithful PyTorch M-DQN")
    parser.add_argument("--config", default="configs/paper_atari.yaml")
    parser.add_argument("--game", default="Pong")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-agent-steps", type=int)
    parser.add_argument("--replay-capacity", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    training = config.training
    if args.total_agent_steps is not None:
        training = replace(training, total_agent_steps=args.total_agent_steps)
    if args.replay_capacity is not None:
        training = replace(training, replay_capacity=args.replay_capacity)
    config = replace(config, training=training)
    config.validate()

    environment = make_atari(
        args.game,
        sticky_action_probability=config.environment.sticky_action_probability,
        frame_skip=config.environment.frame_skip,
        screen_size=config.environment.screen_size,
    )
    try:
        trainer = Trainer(
            environment,
            config,
            args.run_dir,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
        )
        trainer.train()
    finally:
        environment.close()


if __name__ == "__main__":
    main()
