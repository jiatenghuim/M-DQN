from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol


class ExperimentLogger(Protocol):
    enabled: bool

    def log(self, metrics: Mapping[str, float | int], *, step: int) -> None: ...

    def finish(self) -> None: ...


class NullExperimentLogger:
    """No-op logger preserving the existing local CSV behavior."""

    enabled = False

    def log(self, metrics: Mapping[str, float | int], *, step: int) -> None:
        del metrics, step

    def finish(self) -> None:
        return None


def make_experiment_name(
    algorithm: str, game: str, seed: int, pp_scope: str
) -> str:
    game_name = game.removeprefix("ALE/").removesuffix("-v5")
    game_name = re.sub(r"[^A-Za-z0-9_-]+", "_", game_name)
    if algorithm in {"pp_mdqn", "app_mdqn"}:
        scope = "m_only" if pp_scope == "munchausen_only" else "full_operator"
        prefix = f"{algorithm}_{scope}"
    else:
        prefix = algorithm
    return f"{prefix}_{game_name}_seed{seed}"


class SwanLabExperimentLogger:
    """Thin wrapper around the official SwanLab run API."""

    enabled = True

    def __init__(
        self,
        *,
        experiment_name: str,
        config: Mapping[str, Any],
        run_dir: str | Path,
        mode: str = "online",
        resume: bool = False,
        swanlab_module=None,
    ) -> None:
        if mode not in {"online", "offline", "local", "disabled"}:
            raise ValueError("invalid SwanLab mode")
        # SwanLab prints Unicode status icons. Explicit UTF-8 avoids failures
        # when Windows/conda pipes expose a legacy GBK text stream.
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8")
        if swanlab_module is None:
            try:
                swanlab_module = importlib.import_module("swanlab")
            except ModuleNotFoundError as exc:
                if exc.name == "swanlab":
                    message = "SwanLab is not installed. Run: pip install swanlab"
                else:
                    message = (
                        "SwanLab could not import one of its dependencies "
                        f"({exc.name}). Reinstall the tracking dependencies."
                    )
                raise RuntimeError(message) from exc

        self._swanlab = swanlab_module
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id_path = run_dir / "swanlab_run_id.txt"
        init_kwargs: dict[str, Any] = {
            "project": "PP-MDQN",
            "experiment_name": experiment_name,
            "config": dict(config),
            "logdir": str(run_dir / "swanlog"),
            "mode": mode,
            "resume": "never",
        }
        settings_factory = getattr(self._swanlab, "Settings", None)
        if settings_factory is not None:
            init_kwargs["settings"] = settings_factory(probe={"monitor": False})
        if resume:
            if not run_id_path.exists():
                raise FileNotFoundError(
                    f"SwanLab run id is missing for resume: {run_id_path}"
                )
            init_kwargs["id"] = run_id_path.read_text(encoding="utf-8").strip()
            init_kwargs["resume"] = "allow"

        self._run = self._swanlab.init(**init_kwargs)
        if not resume:
            run_id_path.write_text(str(self._run.id), encoding="utf-8")

    def log(self, metrics: Mapping[str, float | int], *, step: int) -> None:
        clean = {
            name: int(value) if isinstance(value, int) else float(value)
            for name, value in metrics.items()
        }
        self._run.log(clean, step=int(step))

    def finish(self) -> None:
        self._run.finish()


def create_experiment_logger(
    *,
    use_swanlab: bool,
    experiment_name: str,
    config: Mapping[str, Any],
    run_dir: str | Path,
    mode: str,
    resume: bool,
) -> ExperimentLogger:
    if not use_swanlab:
        return NullExperimentLogger()
    return SwanLabExperimentLogger(
        experiment_name=experiment_name,
        config=config,
        run_dir=run_dir,
        mode=mode,
        resume=resume,
    )
