"""Typed models for chemical identity, provenance and generated structures."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChemicalRole(str, Enum):
    """What the chemical entity is in the study context."""

    MOLECULE = "molecule"
    ION = "ion"
    COMPLEX = "complex"
    POLYMER = "polymer"
    OLIGOMER = "oligomer"
    FRAGMENT = "fragment"
    PROXY = "proxy"


class ProvenanceStatus(str, Enum):
    """How a structure entered the study (never guessed silently)."""

    USER_PROVIDED = "USER_PROVIDED"
    DATABASE_RESOLVED = "DATABASE_RESOLVED"
    GENERATED_FROM_SMILES = "GENERATED_FROM_SMILES"
    HEURISTIC_COMPLEX = "HEURISTIC_COMPLEX"
    ASSUMED_REPRESENTATIVE = "ASSUMED_REPRESENTATIVE"
    ASSUMED_PROXY = "ASSUMED_PROXY"
    GENERATION_FAILED = "GENERATION_FAILED"


class ReliabilityGrade(str, Enum):
    """Report reliability of a structure-derived result."""

    A = "A"  # user-provided or reliable database structure
    B = "B"  # generated from an explicit SMILES
    C = "C"  # representative oligomer or heuristic complex
    D = "D"  # proxy model


class ChemicalIdentity(BaseModel):
    """One unique chemical entity (not a calculation)."""

    system_id: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    formula: str = ""  # Hill formula when known (e.g. C7H8F4O2)
    smiles: str = ""
    inchi: str = ""
    total_charge: int
    spin_multiplicity: int = 1
    role: ChemicalRole = ChemicalRole.MOLECULE

    def canonical_key(self) -> str:
        """Deterministic dedup key: identity + charge + spin."""
        return f"{self.system_id}|q{self.total_charge:+d}|s{self.spin_multiplicity}"


class StructureProvenance(BaseModel):
    """Every assumption behind a structure, recorded, never silent."""

    status: ProvenanceStatus = ProvenanceStatus.GENERATED_FROM_SMILES
    source: str = "smiles"
    source_identifier: str = ""
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = 0.0  # 0..1
    generator: str = ""
    random_seed: int = 0
    conformer_id: str = ""
    parent_structures: list[str] = Field(default_factory=list)

    @property
    def reliability_grade(self) -> ReliabilityGrade:
        if self.status in {
            ProvenanceStatus.USER_PROVIDED,
            ProvenanceStatus.DATABASE_RESOLVED,
        }:
            return ReliabilityGrade.A
        if self.status is ProvenanceStatus.GENERATED_FROM_SMILES:
            return ReliabilityGrade.B
        if self.status in {
            ProvenanceStatus.HEURISTIC_COMPLEX,
            ProvenanceStatus.ASSUMED_REPRESENTATIVE,
        }:
            return ReliabilityGrade.C
        return ReliabilityGrade.D


class GeneratedStructure(BaseModel):
    """A persisted, validated 3D structure ready for downstream use."""

    identity: ChemicalIdentity
    structure_path: Path
    format: str = "xyz"
    atom_count: int = 0
    formal_charge: int = 0
    provenance: StructureProvenance = Field(default_factory=StructureProvenance)
    validation: list[str] = Field(default_factory=list)
    force_field_energy: float | None = None  # kcal/mol, MMFF/UFF

    @field_validator("structure_path")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    def reliability_grade(self) -> ReliabilityGrade:
        return self.provenance.reliability_grade

    def manifest_row(self) -> dict[str, Any]:
        """JSON-safe row for structure_manifest.json (no raw content)."""
        return {
            "system_id": self.identity.system_id,
            "display_name": self.identity.display_name,
            "formula": self.identity.formula,
            "total_charge": self.identity.total_charge,
            "spin_multiplicity": self.identity.spin_multiplicity,
            "role": self.identity.role.value,
            "structure_path": str(self.structure_path),
            "format": self.format,
            "atom_count": self.atom_count,
            "formal_charge": self.formal_charge,
            "force_field_energy": self.force_field_energy,
            "reliability": self.reliability_grade().value,
            "provenance": {
                "status": self.provenance.status.value,
                "source": self.provenance.source,
                "source_identifier": self.provenance.source_identifier,
                "assumptions": list(self.provenance.assumptions),
                "confidence": self.provenance.confidence,
                "generator": self.provenance.generator,
                "random_seed": self.provenance.random_seed,
                "conformer_id": self.provenance.conformer_id,
                "parent_structures": list(self.provenance.parent_structures),
            },
            "validation": list(self.validation),
        }
