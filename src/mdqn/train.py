from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

from mdqn.atari import make_atari
from mdqn.config import load_config
from mdqn.trainer import Trainer
from mdqn.utils.logger import create_experiment_logger, make_experiment_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PyTorch DQN, M-DQN, or RG-MDQN"
    )
    parser.add_argument("--config", default="configs/debug_rg_mdqn.yaml")
    parser.add_argument("--game", default="Breakout")
    parser.add_argument("--run-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--algo", choices=("dqn", "mdqn", "rg_mdqn")
    )
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--frames", type=int)
    budget.add_argument("--total-agent-steps", type=int)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--checkpoint-every-frames", type=int)
    parser.add_argument("--log-every-frames", type=int)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "offline", "local", "disabled"),
        default="online",
    )
    return parser


def _agent_steps_for_frames(frames: int, frame_skip: int) -> int:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if frames % frame_skip != 0:
        raise ValueError("frames must be divisible by frame_skip")
    return frames // frame_skip


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    algorithm = config.algorithm
    if args.algo is not None:
        algorithm = replace(algorithm, name=args.algo)
    training = config.training
    try:
        if args.frames is not None:
            training = replace(
                training,
                total_frames=args.frames,
                total_agent_steps=_agent_steps_for_frames(
                    args.frames, config.environment.frame_skip
                ),
            )
        if args.total_agent_steps is not None:
            training = replace(
                training,
                total_agent_steps=args.total_agent_steps,
                total_frames=None,
            )
    except ValueError as exc:
        parser.error(str(exc))
    if args.replay_capacity is not None:
        training = replace(training, replay_capacity=args.replay_capacity)
    if args.checkpoint_every_frames is not None:
        training = replace(
            training, checkpoint_every_frames=args.checkpoint_every_frames
        )
    if args.log_every_frames is not None:
        training = replace(
            training, metrics_log_period_frames=args.log_every_frames
        )
    config = replace(config, algorithm=algorithm, training=training)
    config.validate()

    experiment_name = make_experiment_name(
        config.algorithm.name,
        args.game,
        args.seed,
    )
    if args.run_dir is None:
        if args.resume:
            parser.error("--resume requires an explicit --run-dir")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / experiment_name / timestamp
    else:
        run_dir = Path(args.run_dir)

    actual_frames = (
        config.training.total_agent_steps * config.environment.frame_skip
    )
    print(
        "experiment "
        f"algorithm={config.algorithm.name} game={args.game} seed={args.seed} "
        f"frames={actual_frames} device={args.device} run_dir={run_dir}",
        flush=True,
    )

    runtime_config = config.to_dict() | {
        "game": args.game,
        "seed": args.seed,
        "device": args.device,
        "experiment_name": experiment_name,
        "actual_total_frames": actual_frames,
        "run_dir": str(run_dir),
    }

    environment = make_atari(
        args.game,
        sticky_action_probability=config.environment.sticky_action_probability,
        frame_skip=config.environment.frame_skip,
        screen_size=config.environment.screen_size,
    )
    experiment_logger = None
    try:
        experiment_logger = create_experiment_logger(
            use_swanlab=args.use_swanlab,
            experiment_name=experiment_name,
            config=runtime_config,
            run_dir=run_dir,
            mode=args.swanlab_mode,
            resume=args.resume,
        )
        trainer = Trainer(
            environment,
            config,
            run_dir,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
            game=args.game,
            experiment_logger=experiment_logger,
            runtime_metadata={
                "experiment_name": experiment_name,
                "run_dir": str(run_dir),
                "use_swanlab": args.use_swanlab,
                "swanlab_mode": args.swanlab_mode,
            },
        )
        trainer.train()
    finally:
        environment.close()
        if experiment_logger is not None:
            experiment_logger.finish()


if __name__ == "__main__":
    main()
