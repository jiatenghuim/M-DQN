from __future__ import annotations

import csv
import json

import yaml

from mdqn.agent import UpdateMetrics
from mdqn.trainer import (
    next_frame_boundary,
    target_steps_since_update,
    training_metric_payload,
)
from mdqn.utils.logger import SwanLabExperimentLogger, make_experiment_name
from mdqn.utils.results import RESULT_COLUMNS, ResultsExporter


class _FakeRun:
    id = "fake-run-id"

    def __init__(self) -> None:
        self.logs = []
        self.finished = False

    def log(self, data, step=None) -> None:
        self.logs.append((data, step))

    def finish(self) -> None:
        self.finished = True


class _FakeSwanLab:
    def __init__(self) -> None:
        self.init_calls = []
        self.runs = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run = _FakeRun()
        self.runs.append(run)
        return run


def test_experiment_names_are_comparison_friendly() -> None:
    assert make_experiment_name("mdqn", "Breakout", 0) == (
        "mdqn_Breakout_seed0"
    )
    assert make_experiment_name("rg_mdqn", "Breakout", 0) == (
        "rg_mdqn_Breakout_seed0"
    )


def test_swanlab_wrapper_initializes_logs_and_resumes(tmp_path) -> None:
    fake = _FakeSwanLab()
    logger = SwanLabExperimentLogger(
        experiment_name="mdqn_Breakout_seed0",
        config={"seed": 0},
        run_dir=tmp_path,
        mode="offline",
        swanlab_module=fake,
    )
    assert fake.init_calls[0]["project"] == "RG-MDQN"
    assert fake.init_calls[0]["resume"] == "never"
    logger.log({"episode_return": 1.0, "global_step": 100}, step=100)
    assert fake.runs[0].logs == [
        ({"episode_return": 1.0, "global_step": 100}, 100)
    ]
    logger.finish()
    assert fake.runs[0].finished

    SwanLabExperimentLogger(
        experiment_name="mdqn_Breakout_seed0",
        config={"seed": 0},
        run_dir=tmp_path,
        mode="offline",
        resume=True,
        swanlab_module=fake,
    )
    assert fake.init_calls[1]["id"] == "fake-run-id"
    assert fake.init_calls[1]["resume"] == "allow"


def test_results_export_has_required_artifacts(tmp_path) -> None:
    exporter = ResultsExporter(tmp_path, {"seed": 3}, resume=False)
    exporter.append_metrics(
        step=100,
        episode_return=2.0,
        loss=0.5,
        entropy=0.4,
        uncertainty=0.3,
        bonus=-0.2,
    )
    exporter.write_summary(
        {"final_return": 2.0, "best_return": 2.0, "seed": 3}
    )
    with (tmp_path / "results" / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == RESULT_COLUMNS
    assert rows[0]["step"] == "100"
    with (tmp_path / "results" / "config_used.yaml").open(
        encoding="utf-8"
    ) as stream:
        assert yaml.safe_load(stream)["seed"] == 3
    with (tmp_path / "results" / "summary.json").open(
        encoding="utf-8"
    ) as stream:
        assert json.load(stream)["best_return"] == 2.0


def test_training_metric_names_and_frame_boundaries() -> None:
    payload = training_metric_payload(
        UpdateMetrics(
            loss=1.0,
            q_mean=2.0,
            target_mean=3.0,
            max_q_value=4.0,
            mean_td_error=5.0,
        )
    )
    assert payload == {
        "loss/q_loss": 1.0,
        "mean_q_value": 2.0,
        "max_q_value": 4.0,
        "mean_td_error": 5.0,
    }
    assert next_frame_boundary(0, 50_000) == 50_000
    assert next_frame_boundary(50_000, 50_000) == 100_000
    assert target_steps_since_update(24_004, 24_000) == 4


def test_rg_training_metrics_preserve_required_swanlab_names() -> None:
    diagnostics = {
        "reducibility/online_base_loss_mean": 0.8,
        "reducibility/target_base_loss_mean": 0.3,
        "reducibility/reducible_loss_mean": 0.5,
        "reducibility/gate_mean": 0.625,
        "reducibility/gate_std": 0.1,
        "reducibility/gate_min": 0.4,
        "reducibility/gate_max": 0.8,
        "reducibility/positive_fraction": 0.75,
        "reducibility/gate_zero_fraction": 0.25,
        "reducibility/mean_abs_base_td_error": 1.2,
        "munchausen/full_bonus_mean": -0.7,
        "munchausen/gated_bonus_mean": -0.4,
        "munchausen/bonus_attenuation_mean": 0.625,
        "munchausen/point_policy_entropy": 0.2,
        "munchausen/full_clip_ratio": 0.1,
    }
    payload = training_metric_payload(
        UpdateMetrics(
            loss=1.0,
            q_mean=2.0,
            target_mean=3.0,
            max_q_value=4.0,
            mean_td_error=5.0,
            diagnostics=diagnostics,
        ),
        algorithm="rg_mdqn",
    )
    assert diagnostics.items() <= payload.items()
