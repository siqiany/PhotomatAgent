"""Safe defaults and finite budgets for QD events."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

from photomatagent.capability_acquisition.models import MAX_SCRIPT_ARGV_TOKENS
from photomatagent.scientific.evolution.qd.models import QDModel


class EvolutionBudget(QDModel):
    max_trials: StrictInt = Field(default=6)
    offspring_per_batch: StrictInt = Field(default=2)
    max_proposals: StrictInt = Field(default=8)
    max_model_calls: StrictInt = Field(default=12)
    patience: StrictInt = Field(default=2)
    max_judge_calls: StrictInt = Field(default=1)
    max_model_tokens: StrictInt = Field(default=50_000)
    max_wall_seconds: StrictFloat = Field(default=900, allow_inf_nan=False)
    max_installs: StrictInt = Field(default=2)
    max_download_bytes: StrictInt = Field(default=500_000_000)
    max_environment_bytes: StrictInt = Field(default=1_000_000_000)

    @field_validator("max_wall_seconds", mode="before")
    @classmethod
    def finite_wall_time(cls, value: object) -> object:
        if isinstance(value, (int, float)) and not __import__("math").isfinite(value):
            raise ValueError("BUDGET_INVALID: finite number required")
        if isinstance(value, (int, float)) and value < 0:
            raise ValueError("BUDGET_INVALID")
        return value

    @field_validator("max_trials", "offspring_per_batch", "max_proposals", "max_model_calls", "patience", "max_judge_calls", "max_model_tokens", "max_installs", "max_download_bytes", "max_environment_bytes")
    @classmethod
    def valid_nonnegative_budget(cls, value: int) -> int:
        if value < 0:
            raise ValueError("BUDGET_INVALID")
        return value


class MutableParameterRule(QDModel):
    """User-frozen set of values a proposal may assign to one workflow step."""

    step_id: StrictStr = Field(min_length=1, max_length=200)
    binding_id: StrictStr = Field(min_length=1, max_length=200)
    parameters: Mapping[StrictStr, object]

    @field_validator("parameters")
    @classmethod
    def require_bounded_choices(cls, value):
        if not value:
            raise ValueError("PARAMETER_POLICY_INVALID")
        for key, choices in value.items():
            if not key or len(key) > 100 or not isinstance(choices, (list, tuple)) or not 1 <= len(choices) <= 32:
                raise ValueError("PARAMETER_POLICY_INVALID")
            for choice in choices:
                if key == "argv":
                    if (not isinstance(choice, (list, tuple)) or not 1 <= len(choice) <= MAX_SCRIPT_ARGV_TOKENS
                            or any(not isinstance(token, str) or len(token) > 200
                                   or not re.fullmatch(r"[A-Za-z0-9_.:=+-]+", token)
                                   for token in choice)):
                        raise ValueError("PARAMETER_POLICY_INVALID")
                    continue
                if choice is not None and type(choice) not in (str, int, float, bool):
                    raise ValueError("PARAMETER_POLICY_INVALID")
                if isinstance(choice, float) and not math.isfinite(choice):
                    raise ValueError("PARAMETER_POLICY_INVALID")
        return value

    @field_validator("parameters")
    @classmethod
    def argv_only_for_script_runner(cls, value, info):
        if "argv" in value and info.data.get("binding_id") != "script_run":
            raise ValueError("PARAMETER_POLICY_INVALID")
        return value

class QDConfig(QDModel):
    enabled: StrictBool = False
    budget: EvolutionBudget = Field(default_factory=EvolutionBudget)
    judge_mode: StrictStr = "on_demand"
    discovery_provider: StrictStr | None = None
    sandbox: StrictStr = "podman"
    environment_id: StrictStr | None = None
    environment_storage: StrictStr = ".photomatagent/qd/environments"
    mutable_parameters: tuple[MutableParameterRule, ...] = ()
    published_skills: StrictBool = False
