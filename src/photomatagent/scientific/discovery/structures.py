"""Structure derivation records exchanged with the runtime."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from functools import reduce
from math import gcd
import re
from collections.abc import Mapping
from typing import Any, Literal, Never

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from photomatagent.scientific.capabilities.generation.lineage import CandidateLineage
from photomatagent.scientific.discovery.composition import normalize_composition


Operation = Literal["make_supercell", "substitute_sites", "enumerate_orderings"]
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_OUTPUT = re.compile(
    r"^user_output/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/structures/"
    r"op_[0-9a-f]{32}/structure_[0-9]{4}\.cif$"
)


class _FrozenDict(dict[str, Any]):
    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenDict":
        copied = dict.__new__(_FrozenDict)
        memo[id(self)] = copied
        dict.update(copied, {copy.deepcopy(key, memo): copy.deepcopy(value, memo) for key, value in self.items()})
        return copied

    def _deny(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("immutable structure record")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _deny
    __ior__ = _deny

class _FrozenList(list[Any]):
    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenList":
        copied = list.__new__(_FrozenList)
        memo[id(self)] = copied
        list.extend(copied, [copy.deepcopy(value, memo) for value in self])
        return copied

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

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> "StructureDerivation":
        """Copy only through the full identity and immutability validators."""

        payload = self.model_dump(mode="python")
        if deep:
            payload = copy.deepcopy(payload)
        if update:
            payload.update(dict(update))
        return type(self).model_validate(payload)

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
        if not _OUTPUT.fullmatch(path) or path != value:
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

        valid = (
            result
            and len(result) <= 16
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
        formula = "".join(f"{symbol}{amount}" for symbol, amount in result)
        if normalize_composition(formula) != result:
            raise ValueError("normalized_composition is not a canonical composition")
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

    @model_validator(mode="after")
    def lineage_matches_identity(self) -> "StructureDerivation":
        if self.lineage.candidate_id != self.candidate_id:
            raise ValueError("lineage candidate_id must match candidate_id")
        if self.lineage.parent_candidate_id != self.parent_candidate_id:
            raise ValueError("lineage parent_candidate_id must match parent_candidate_id")
        return self


class StructureRegistration(BaseModel):
    """Untrusted update; runtime rebuilds trusted fields before applying it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    derivation: StructureDerivation
