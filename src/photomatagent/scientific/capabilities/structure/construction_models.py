"""Bounded, model-facing contracts for structure construction."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

_TASK_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class _ConstructionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


def _slug(value: str) -> str:
    if not _TASK_SLUG.fullmatch(value) or value in {".", ".."}:
        raise ValueError("task_slug must be a safe workspace directory name")
    return value


class SupercellRequest(_ConstructionModel):
    path: Annotated[str, Field(min_length=1)]
    scaling: tuple[StrictInt, StrictInt, StrictInt]
    task_slug: Annotated[str, Field(min_length=1, max_length=64)]
    hypothesis_id: str | None = None

    @field_validator("task_slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        return _slug(value)

    @field_validator("scaling")
    @classmethod
    def validate_scaling(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3 or any(isinstance(item, bool) or item < 1 for item in value):
            raise ValueError("scaling must contain three positive integers")
        return value


class SiteReplacement(_ConstructionModel):
    index: StrictInt
    from_element: Annotated[StrictStr, Field(min_length=1, max_length=3)]
    to_element: Annotated[StrictStr, Field(min_length=1, max_length=3)]

    @field_validator("index")
    @classmethod
    def validate_index(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("site index must be a non-negative integer")
        return value

    @model_validator(mode="after")
    def different_elements(self) -> "SiteReplacement":
        if self.from_element == self.to_element:
            raise ValueError("from_element and to_element must differ")
        return self


class SubstitutionRequest(_ConstructionModel):
    path: Annotated[str, Field(min_length=1)]
    replacements: list[SiteReplacement] = Field(min_length=1, max_length=512)
    expected_formula: Annotated[str, Field(min_length=1, max_length=256)]
    hypothesis_id: Annotated[str, Field(min_length=1)]
    task_slug: Annotated[str, Field(min_length=1, max_length=64)]

    @field_validator("task_slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        return _slug(value)

    @model_validator(mode="after")
    def unique_indices(self) -> "SubstitutionRequest":
        indices = [item.index for item in self.replacements]
        if len(indices) != len(set(indices)):
            raise ValueError("replacement indices must be unique")
        return self


class OrderingRequest(_ConstructionModel):
    path: Annotated[str, Field(min_length=1)]
    eligible_indices: list[StrictInt] = Field(min_length=1, max_length=4096)
    from_element: Annotated[StrictStr, Field(min_length=1, max_length=3)]
    to_element: Annotated[StrictStr, Field(min_length=1, max_length=3)]
    replacement_count: StrictInt
    expected_formula: Annotated[str, Field(min_length=1, max_length=256)]
    hypothesis_id: Annotated[str, Field(min_length=1)]
    task_slug: Annotated[str, Field(min_length=1, max_length=64)]

    @field_validator("task_slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        return _slug(value)

    @field_validator("eligible_indices")
    @classmethod
    def validate_indices(cls, value: list[int]) -> list[int]:
        if any(isinstance(item, bool) or item < 0 for item in value):
            raise ValueError("eligible_indices must be non-negative integers")
        if len(value) != len(set(value)):
            raise ValueError("eligible_indices must be unique")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> "OrderingRequest":
        if self.from_element == self.to_element:
            raise ValueError("from_element and to_element must differ")
        if not 1 <= self.replacement_count <= len(self.eligible_indices):
            raise ValueError("replacement_count must be within eligible_indices")
        return self


class ConstructionLimits(_ConstructionModel):
    max_atoms: StrictInt = 128
    max_raw_configurations: StrictInt = 4096
    max_outputs: StrictInt = 32

    @field_validator("max_atoms")
    @classmethod
    def atom_limit(cls, value: int) -> int:
        if isinstance(value, bool) or not 1 <= value <= 512:
            raise ValueError("max_atoms must be between 1 and 512")
        return value

    @field_validator("max_raw_configurations")
    @classmethod
    def bounded_positive(cls, value: int) -> int:
        if isinstance(value, bool) or not 1 <= value <= 4096:
            raise ValueError("max_raw_configurations must be between 1 and 4096")
        return value

    @field_validator("max_outputs")
    @classmethod
    def outputs_bounded(cls, value: int) -> int:
        if isinstance(value, bool) or not 1 <= value <= 32:
            raise ValueError("max_outputs must be between 1 and 32")
        return value
