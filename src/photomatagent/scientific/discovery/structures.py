"""Structure derivation records exchanged with the runtime."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
import re
from typing import Any, Literal, Never
from functools import reduce

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from photomatagent.scientific.capabilities.generation.lineage import CandidateLineage


Operation = Literal["make_supercell", "substitute_sites", "enumerate_orderings"]
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

class _FrozenDict(dict[str, Any]):
    def _deny(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("immutable structure record")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _deny
    __ior__ = _deny

class _FrozenList(list[Any]):
    def _deny(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("immutable structure record")

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = _deny
    __iadd__ = __imul__ = _deny

def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("structure records accept only JSON-like values")

class _ImmutableLineage(CandidateLineage):
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def freeze_nested(self) -> "_ImmutableLineage":
        object.__setattr__(self, "generation_parameters", _freeze(self.generation_parameters))
        object.__setattr__(self, "source_artifacts", _freeze(self.source_artifacts))
        return self


class StructureDerivation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=128)
    parent_candidate_id: str | None = None
    hypothesis_id: str | None = None
    input_sha256: str = Field(min_length=64, max_length=64)
    structure_hash: str = Field(min_length=64, max_length=64)
    output_path: str = Field(min_length=1)
    operation: Operation
    parameters: dict[str, Any] = Field(default_factory=dict)
    normalized_composition: tuple[tuple[str, int], ...]
    lineage: CandidateLineage
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    origin: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "candidate_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("invalid structure identity")
        return value

    @field_validator("parent_candidate_id", "hypothesis_id")
    @classmethod
    def optional_id(cls, value: str | None) -> str | None:
        if value is not None and not _ID.fullmatch(value):
            raise ValueError("invalid related identity")
        return value

    @field_validator("input_sha256", "structure_hash")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA.fullmatch(value):
            raise ValueError("hash must be lowercase hexadecimal SHA-256")
        return value

    @field_validator("output_path")
    @classmethod
    def safe_output_path(cls, value: str) -> str:
        path = value.replace("\\", "/")
        unsafe = (
            path.startswith(("/", "\\"))
            or ":" in path[:3]
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.split("/"))
        )
        if unsafe:
            raise ValueError("output_path must be workspace-relative")
        return path

    @field_validator("normalized_composition", mode="before")
    @classmethod
    def valid_composition(cls, value: Any) -> tuple[tuple[str, int], ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("normalized_composition must be a sequence")
        invalid = any(
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
            for item in value
        )
        if invalid:
            raise ValueError("normalized_composition entries must be strict symbol/integer pairs")
        result = tuple((symbol, amount) for symbol, amount in value)
        from math import gcd

        valid = (
            result
            and tuple(sorted(result)) == result
            and len({symbol for symbol, _ in result}) == len(result)
            and all(
                re.fullmatch(r"[A-Z][a-z]?$", symbol) and amount > 0
                for symbol, amount in result
            )
            and reduce(gcd, (amount for _, amount in result)) == 1
        )
        if not valid:
            raise ValueError("normalized_composition must be sorted, reduced, and positive")
        return result

    @model_validator(mode="before")
    @classmethod
    def defensive_copy(cls, value: Any) -> Any:
        copied = copy.deepcopy(value)
        if isinstance(copied, dict):
            copied["parameters"] = _freeze(copied.get("parameters", {}))
            copied["origin"] = _freeze(copied.get("origin", {}))
            if "lineage" in copied and not isinstance(copied["lineage"], _ImmutableLineage):
                copied["lineage"] = _ImmutableLineage.model_validate(copied["lineage"])
        return copied

    @model_validator(mode="after")
    def freeze_fields(self) -> "StructureDerivation":
        object.__setattr__(self, "parameters", _freeze(self.parameters))
        object.__setattr__(self, "origin", _freeze(self.origin))
        return self


class StructureRegistration(BaseModel):
    """Untrusted command; runtime must inject trusted operation/hash/origin."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    derivation: StructureDerivation
