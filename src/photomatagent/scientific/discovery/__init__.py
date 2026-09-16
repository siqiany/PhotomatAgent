"""Contracts and stable identities for mechanism-guided discovery."""

from photomatagent.scientific.discovery.composition import (
    CompositionCapabilityError,
    composition_key,
    normalize_composition,
)
from photomatagent.scientific.discovery.models import (
    BasisReference,
    DiscoveryConstraints,
    ExpectedEffect,
    HypothesisOrigin,
    HypothesisProposal,
    HypothesisRegistration,
    ScientificHypothesis,
)

__all__ = [
    "BasisReference",
    "CompositionCapabilityError",
    "DiscoveryConstraints",
    "ExpectedEffect",
    "HypothesisOrigin",
    "HypothesisProposal",
    "HypothesisRegistration",
    "ScientificHypothesis",
    "composition_key",
    "normalize_composition",
]
