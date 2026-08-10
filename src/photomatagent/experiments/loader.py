"""Load the deliberately small JSON experiment format."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from photomatagent.experiments.models import ExperimentConfig


class ExperimentConfigError(ValueError):
    pass


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    source = Path(path)
    if source.suffix.lower() not in {".json"}:
        raise ExperimentConfigError(
            "experiment config must be JSON; YAML is not enabled because the project "
            "has no YAML dependency"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigError(f"could not read experiment config {source}: {exc}") from exc
    try:
        return ExperimentConfig.model_validate(payload)
    except ValidationError as exc:
        raise ExperimentConfigError(f"invalid experiment config {source}: {exc}") from exc
