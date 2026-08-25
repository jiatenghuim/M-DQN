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
