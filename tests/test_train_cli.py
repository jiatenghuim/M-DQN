from mdqn.train import build_parser


def test_debug_config_and_breakout_are_cli_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config == "configs/debug_rg_mdqn.yaml"
    assert args.game == "Breakout"
    assert args.run_dir is None


def test_rg_mdqn_is_a_cli_algorithm_choice() -> None:
    args = build_parser().parse_args(["--algo", "rg_mdqn"])
    assert args.algo == "rg_mdqn"


def test_frames_and_agent_steps_are_mutually_exclusive() -> None:
    parser = build_parser()
    try:
        parser.parse_args(["--frames", "100", "--total-agent-steps", "25"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("mutually exclusive budgets should be rejected")
