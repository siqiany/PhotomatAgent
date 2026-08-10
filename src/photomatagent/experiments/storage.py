"""Filesystem persistence for experiment configs, runs, and summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from photomatagent.experiments.models import ExperimentResult, ExperimentSummary
from photomatagent.logging.event_logger import redact_secrets


def default_experiments_dir() -> Path:
    return Path(".photomatagent") / "experiments"


def new_experiment_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    slug = "".join(char.lower() if char.isalnum() else "-" for char in name)
    slug = "-".join(part for part in slug.split("-") if part)[:32] or "experiment"
    return f"{timestamp}_{slug}_{uuid4().hex[:6]}"


def save_experiment(
    result: ExperimentResult, experiments_dir: Path | str | None = None
) -> Path:
    base = Path(experiments_dir) if experiments_dir is not None else default_experiments_dir()
    target = base / result.experiment_id
    target.mkdir(parents=True, exist_ok=False)
    payloads = {
        "config.json": {
            "experiment_id": result.experiment_id,
            "experiment": result.config.model_dump(mode="json"),
            "configuration_snapshot": result.summary.configuration.model_dump(mode="json"),
        },
        "summary.json": result.summary.model_dump(mode="json"),
        "runs.json": [run.model_dump(mode="json") for run in result.runs],
    }
    for filename, payload in payloads.items():
        safe = redact_secrets(payload)
        (target / filename).write_text(
            json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return target


def resolve_experiment_path(
    target: str | Path, experiments_dir: Path | str | None = None
) -> Path:
    candidate = Path(target)
    if candidate.is_dir():
        return candidate
    base = Path(experiments_dir) if experiments_dir is not None else default_experiments_dir()
    candidate = base / str(target)
    if candidate.is_dir():
        return candidate
    raise ValueError(f"experiment not found: {target}")


def load_experiment_summary(
    target: str | Path, experiments_dir: Path | str | None = None
) -> ExperimentSummary:
    path = resolve_experiment_path(target, experiments_dir) / "summary.json"
    try:
        return ExperimentSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load experiment summary {path}: {exc}") from exc
