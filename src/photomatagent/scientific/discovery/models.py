"""Pure data contracts for mechanism hypotheses and discovery constraints."""

from __future__ import annotations

from datetime import datetime
from math import gcd
from typing import Any, Annotated, Literal, Never, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from photomatagent.scientific.capabilities.generation.lineage import CandidateLineage

DesignOperation = Literal[
    "isovalent_substitution",
    "ordering",
    "distortion",
    "prototype_transfer",
    "synthesis_constrained",
    "other",
]
BasisRelation = Literal["analogue", "supports", "contradicts"]
EffectDirection = Literal["increase", "decrease", "change", "unknown"]

NonEmptyId = Annotated[str, Field(min_length=1)]
BoundedText1000 = Annotated[str, Field(min_length=1, max_length=1_000)]
BoundedText1500 = Annotated[str, Field(min_length=1, max_length=1_500)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _ImmutableList(list[Any]):
    def _deny_mutation(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("registered hypothesis containers are immutable")

    __setitem__ = _deny_mutation
    __delitem__ = _deny_mutation
    __iadd__ = _deny_mutation
    __imul__ = _deny_mutation
    append = _deny_mutation
    clear = _deny_mutation
    extend = _deny_mutation
    insert = _deny_mutation
    pop = _deny_mutation
    remove = _deny_mutation
    reverse = _deny_mutation
    sort = _deny_mutation

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self


class _ImmutableDict(dict[str, Any]):
    def _deny_mutation(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("registered hypothesis containers are immutable")

    __setitem__ = _deny_mutation
    __delitem__ = _deny_mutation
    __ior__ = _deny_mutation
    clear = _deny_mutation
    pop = _deny_mutation
    popitem = _deny_mutation
    setdefault = _deny_mutation
    update = _deny_mutation

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _ImmutableDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _ImmutableList(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class BasisReference(_StrictModel):
    """A traceable evidence reference used as a proposal basis."""

    evidence_id: NonEmptyId
    relation: BasisRelation
    anchor: str = ""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class ExpectedEffect(_StrictModel):
    """A qualitative, unvalidated property effect expected by the model."""

    property: Annotated[str, Field(min_length=1)]
    direction: EffectDirection
    rationale: Annotated[str, Field(min_length=1)]

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class HypothesisProposal(_StrictModel):
    """The complete set of hypothesis fields that a model may submit."""

    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    formula: Annotated[str, Field(min_length=1, max_length=256)]
    statement: Annotated[str, Field(min_length=1, max_length=2_000)]
    design_operation: DesignOperation
    parent_hypothesis_ids: list[NonEmptyId] = Field(default_factory=list, max_length=8)
    basis: list[BasisReference] = Field(default_factory=list, max_length=16)
    assumptions: list[BoundedText1000] = Field(default_factory=list, max_length=12)
    expected_effects: list[ExpectedEffect] = Field(default_factory=list, max_length=12)
    counter_hypotheses: list[BoundedText1000] = Field(default_factory=list, max_length=8)
    validation_questions: list[BoundedText1000] = Field(min_length=1, max_length=12)
    synthesis_notes: list[BoundedText1500] = Field(default_factory=list, max_length=8)

    @field_validator("formula")
    @classmethod
    def formula_has_no_unexpanded_variable(cls, formula: str) -> str:
        if "x" in formula:
            raise ValueError("formula must not contain an unexpanded x variable")
        return formula


class _ImmutableHypothesisProposal(HypothesisProposal):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    @model_validator(mode="after")
    def freeze_containers(self) -> "_ImmutableHypothesisProposal":
        for field_name in (
            "parent_hypothesis_ids",
            "basis",
            "assumptions",
            "expected_effects",
            "counter_hypotheses",
            "validation_questions",
            "synthesis_notes",
        ):
            object.__setattr__(self, field_name, _deep_freeze(getattr(self, field_name)))
        return self


class HypothesisRegistration(_StrictModel):
    """Pure update command requesting registration of one proposal."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    proposal: HypothesisProposal


class HypothesisOrigin(_StrictModel):
    """Runtime-owned provenance for a registered hypothesis."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    tool_name: str
    tool_call_id: str
    session_id: str
    run_id: str
    provider: str
    model: str


class _ImmutableCandidateLineage(CandidateLineage):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def freeze_containers(self) -> "_ImmutableCandidateLineage":
        object.__setattr__(
            self,
            "generation_parameters",
            _deep_freeze(self.generation_parameters),
        )
        object.__setattr__(self, "source_artifacts", _deep_freeze(self.source_artifacts))
        return self


class ScientificHypothesis(_StrictModel):
    """Immutable runtime record for one registered scientific hypothesis."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: NonEmptyId
    candidate_id: NonEmptyId
    proposal: HypothesisProposal
    normalized_composition: tuple[tuple[StrictStr, StrictInt], ...]
    request_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage: CandidateLineage
    origin: HypothesisOrigin
    created_at: datetime

    @field_validator("proposal", mode="before")
    @classmethod
    def snapshot_proposal(cls, proposal: object) -> _ImmutableHypothesisProposal:
        payload = proposal.model_dump(mode="python") if isinstance(proposal, BaseModel) else proposal
        return _ImmutableHypothesisProposal.model_validate(payload)

    @field_validator("lineage", mode="before")
    @classmethod
    def snapshot_lineage(cls, lineage: object) -> _ImmutableCandidateLineage:
        payload = lineage.model_dump(mode="python") if isinstance(lineage, BaseModel) else lineage
        return _ImmutableCandidateLineage.model_validate(payload)

    @model_validator(mode="after")
    def normalized_composition_is_canonical(self) -> "ScientificHypothesis":
        if self.lineage.generated_by != "mechanism_reasoning":
            raise ValueError("hypothesis lineage must be generated by mechanism_reasoning")
        if self.lineage.validation_status != "UNVALIDATED_HYPOTHESIS":
            raise ValueError("hypothesis lineage must remain UNVALIDATED_HYPOTHESIS")
        composition = self.normalized_composition
        if not composition or len(composition) > 16:
            raise ValueError("normalized_composition must contain 1 to 16 elements")
        symbols = [symbol for symbol, _ in composition]
        if any(not symbol for symbol in symbols) or symbols != sorted(symbols):
            raise ValueError("normalized_composition elements must be non-empty and sorted")
        if len(set(symbols)) != len(symbols):
            raise ValueError("normalized_composition elements must be unique")
        try:
            from pymatgen.core import Element
        except ImportError as exc:  # pragma: no cover - pymatgen is a core dependency
            raise ValueError("element validation requires pymatgen") from exc
        if any(not Element.is_valid_symbol(symbol) for symbol in symbols):
            raise ValueError("normalized_composition contains an invalid element symbol")
        amounts = [amount for _, amount in composition]
        if any(amount <= 0 for amount in amounts):
            raise ValueError("normalized_composition amounts must be positive integers")
        divisor = amounts[0]
        for amount in amounts[1:]:
            divisor = gcd(divisor, amount)
        if divisor != 1:
            raise ValueError("normalized_composition amounts must be reduced")
        return self


class DiscoveryConstraints(_StrictModel):
    """Optional mechanism-discovery constraints supplied by the caller."""

    required_elements: list[str] = Field(default_factory=list)
    forbidden_elements: list[str] = Field(default_factory=list)
    allow_isovalent_alloy: bool | None = None
    allow_donor_acceptor_doping: bool | None = None
