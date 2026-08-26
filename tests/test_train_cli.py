from mdqn.train import build_parser


def test_pp_scope_accepts_hyphenated_and_underscored_flags() -> None:
    parser = build_parser()
    hyphenated = parser.parse_args(
        ["--run-dir", "run", "--pp-scope", "munchausen_only"]
    )
    underscored = parser.parse_args(
        ["--run-dir", "run", "--pp_scope", "full_operator"]
    )
    assert hyphenated.pp_scope == "munchausen_only"
    assert underscored.pp_scope == "full_operator"


def test_debug_config_and_breakout_are_cli_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config == "configs/debug_pp_mdqn.yaml"
    assert args.game == "Breakout"
    assert args.run_dir is None


def test_app_mdqn_is_a_cli_algorithm_choice() -> None:
    args = build_parser().parse_args(["--algo", "app_mdqn"])
    assert args.algo == "app_mdqn"


def test_frames_and_agent_steps_are_mutually_exclusive() -> None:
    parser = build_parser()
    try:
        parser.parse_args(["--frames", "100", "--total-agent-steps", "25"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("mutually exclusive budgets should be rejected")
