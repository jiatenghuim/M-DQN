from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


RESULT_COLUMNS = (
    "step",
    "return",
    "loss",
    "entropy",
    "uncertainty",
    "bonus",
)


class ResultsExporter:
    """Streams compact experiment metrics and writes reproducibility artifacts."""

    def __init__(
        self,
        run_dir: str | Path,
        config: Mapping[str, Any],
        *,
        resume: bool,
    ) -> None:
        self.results_dir = Path(run_dir) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.results_dir / "metrics.csv"
        del resume
        with (self.results_dir / "config_used.yaml").open(
            "w", encoding="utf-8"
        ) as stream:
            yaml.safe_dump(
                dict(config), stream, sort_keys=False, allow_unicode=True
            )

    def append_metrics(
        self,
        *,
        step: int,
        episode_return: float | None,
        loss: float | None,
        entropy: float | None,
        uncertainty: float | None,
        bonus: float | None,
    ) -> None:
        new_file = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "step": int(step),
                    "return": "" if episode_return is None else episode_return,
                    "loss": "" if loss is None else loss,
                    "entropy": "" if entropy is None else entropy,
                    "uncertainty": "" if uncertainty is None else uncertainty,
                    "bonus": "" if bonus is None else bonus,
                }
            )

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        with (self.results_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(dict(summary), stream, indent=2, ensure_ascii=False)
