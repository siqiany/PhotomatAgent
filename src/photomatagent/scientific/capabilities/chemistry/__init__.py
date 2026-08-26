"""Generic chemical structure capabilities (not VASP-specific).

This package owns chemical-entity parsing, SMILES/InChI/local-structure
resolution, deterministic 3D conformer generation (RDKit ETKDG + MMFF/UFF),
complex initial-geometry generation, representative polymer oligomers and
explicitly-labelled proxy models. Every structure carries a
``StructureProvenance`` so assumptions are never silent; the reliability
grade A/B/C/D flows into study reports.

The molecular VASP layer consumes the outputs (``GeneratedStructure``) but
nothing in this package knows about POTCAR, Slurm or VASP.
"""

from __future__ import annotations

from photomatagent.scientific.capabilities.chemistry.models import (
    ChemicalIdentity,
    ChemicalRole,
    GeneratedStructure,
    ProvenanceStatus,
    ReliabilityGrade,
    StructureProvenance,
)
from photomatagent.scientific.capabilities.chemistry.registry import (
    APPROVED_ALIAS_REGISTRY,
    AliasEntry,
    lookup_alias,
)

__all__ = [
    "APPROVED_ALIAS_REGISTRY",
    "AliasEntry",
    "ChemicalIdentity",
    "ChemicalRole",
    "GeneratedStructure",
    "ProvenanceStatus",
    "ReliabilityGrade",
    "StructureProvenance",
    "lookup_alias",
]
