"""Property-conditioned VAE formula generation (donor migration, section 45).

Migrated from the donor ``VAEFormulaGenerator`` with corrected defaults
(section 46): there is NO default ``forbidden_elements`` (Hg, Pb, Bi, Te,
Sb are legitimate infrared candidates) and NO ``prefer_lower_atomic_number``
filter. Toxicity/cost/atomic number are optional user constraints only.

The torch decoding step is dependency-optional: tests and lightweight callers
may inject a ``decoder`` callable, while normal tool execution loads the real
JARVIS conditional-VAE checkpoint.  Missing dependencies/assets become typed
``missing_prerequisites`` failures instead of guessed compositions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from photomatagent.scientific.errors import MissingScientificPrerequisite

HC_EV_UM = 1.239841984

Decoder = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class FormulaProposal:
    formula: str
    chemical_system: str
    elements: tuple[str, ...]
    atom_counts: tuple[int, ...]
    composition_error: float
    charge_neutral: bool
    oxidation_state_examples: tuple[dict[str, float], ...]
    novel_against_training_data: bool
    decoded_sample_index: int

    def as_dict(self) -> dict[str, object]:
        return {
            "formula": self.formula,
            "chemical_system": self.chemical_system,
            "elements": list(self.elements),
            "atom_counts": list(self.atom_counts),
            "composition_error": self.composition_error,
            "charge_neutral": self.charge_neutral,
            "oxidation_state_examples": [
                dict(item) for item in self.oxidation_state_examples
            ],
            "novel_against_training_data": self.novel_against_training_data,
            "decoded_sample_index": self.decoded_sample_index,
        }


class VAEFormulaGenerator:
    """Decode property-conditioned compositions into constrained formulas."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        sample_count: int = 512,
        max_elements: int = 4,
        max_atoms: int = 12,
        random_seed: int = 42,
        require_charge_neutral: bool = True,
        require_novel: bool = True,
        known_formulas: set[str] | None = None,
        decoder: Decoder | None = None,
        property_fields: list[str] | None = None,
        condition_center: np.ndarray | None = None,
        condition_scale: np.ndarray | None = None,
        vocabulary: list[str] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.sample_count = sample_count
        self.max_elements = max_elements
        self.max_atoms = max_atoms
        self.random_seed = random_seed
        self.require_charge_neutral = require_charge_neutral
        self.require_novel = require_novel
        self.known_formulas = set(known_formulas or [])
        self._known_formulas_configured = known_formulas is not None
        self._decoder = decoder
        self.property_fields = list(property_fields or ["gap_selected_eV"])
        self.center = (
            np.asarray(condition_center, dtype=float) if condition_center is not None else None
        )
        self.scale = (
            np.asarray(condition_scale, dtype=float) if condition_scale is not None else None
        )
        self.vocabulary = list(vocabulary or [])

    # -- decoding -----------------------------------------------------------

    def _load_torch_decoder(self) -> Decoder:
        """Load the deployed CVAE checkpoint and return a sampling closure."""
        if self._decoder is not None:
            return self._decoder
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            raise MissingScientificPrerequisite(
                "VAE checkpoint not configured or missing",
                missing=["vae_checkpoint_path"],
            )
        try:
            import torch
        except ImportError as exc:
            raise MissingScientificPrerequisite(
                "PyTorch is required to decode VAE samples (install "
                "torch in an isolated environment and configure the "
                "checkpoint)",
                missing=["torch", "vae_checkpoint_path"],
            ) from exc
        from photomatagent.scientific.capabilities.generation.conditional_vae import (
            ConditionalVAE,
            VAEConfig,
        )

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            config = VAEConfig(**dict(checkpoint["config"]))
            property_fields = list(checkpoint["property_fields"])
            vocabulary = list(checkpoint["vocabulary"])
            center = np.asarray(checkpoint["condition_center"], dtype=float)
            scale = np.asarray(checkpoint["condition_scale"], dtype=float)
            model = ConditionalVAE(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise MissingScientificPrerequisite(
                f"VAE checkpoint is incompatible or unreadable: {exc}",
                missing=["compatible VAE checkpoint"],
            ) from exc

        if len(vocabulary) != config.composition_dim:
            raise MissingScientificPrerequisite(
                "VAE vocabulary size does not match checkpoint composition_dim",
                missing=["compatible VAE vocabulary"],
            )
        if (
            len(property_fields) != config.condition_dim
            or center.shape != (config.condition_dim,)
            or scale.shape != (config.condition_dim,)
            or "gap_selected_eV" not in property_fields
        ):
            raise MissingScientificPrerequisite(
                "VAE condition schema does not match the checkpoint",
                missing=["compatible VAE condition schema"],
            )

        self.property_fields = property_fields
        self.vocabulary = vocabulary
        self.center = center
        self.scale = scale

        def decode(condition: np.ndarray, count: int) -> np.ndarray:
            torch.manual_seed(self.random_seed)
            with torch.inference_mode():
                tensor = torch.as_tensor(
                    condition,
                    dtype=torch.float32,
                ).unsqueeze(0)
                return model.sample(tensor, count=count).cpu().numpy()

        self._decoder = decode
        return decode

    def _load_novelty_reference(self) -> None:
        """Load training-set formulas used to define novelty."""
        if self._known_formulas_configured or not self.require_novel:
            return
        if self.metadata_path is None or not self.metadata_path.is_file():
            raise MissingScientificPrerequisite(
                "VAE novelty filtering requires candidate metadata",
                missing=["vae_candidate_metadata"],
            )
        try:
            rows = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("candidate metadata must be a JSON list")
            self.known_formulas = {
                str(row["formula"]).strip()
                for row in rows
                if isinstance(row, dict) and str(row.get("formula", "")).strip()
            }
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise MissingScientificPrerequisite(
                f"VAE candidate metadata is unreadable: {exc}",
                missing=["valid VAE candidate metadata"],
            ) from exc
        if not self.known_formulas:
            raise MissingScientificPrerequisite(
                "VAE candidate metadata contains no formulas",
                missing=["non-empty VAE candidate metadata"],
            )

    def decode_samples(
        self, condition: np.ndarray, count: int
    ) -> np.ndarray:
        """Decode ``count`` composition fraction vectors for a condition."""
        decoder = self._load_torch_decoder()
        return np.asarray(decoder(condition, count), dtype=float)

    # -- conditioning -------------------------------------------------------

    def condition(
        self,
        *,
        target_band_gap_eV: float,
        target_wavelength_um: float,
    ) -> np.ndarray:
        """Normalized property condition vector."""
        if self.center is None or self.scale is None:
            # No checkpoint statistics: use a unit-normalized proxy so the
            # decoder interface stays deterministic.
            condition = np.zeros(len(self.property_fields), dtype=np.float32)
            if "gap_selected_eV" in self.property_fields:
                condition[self.property_fields.index("gap_selected_eV")] = (
                    float(target_band_gap_eV) / 1.0
                )
            if "cutoff_wavelength_um_from_gap" in self.property_fields:
                condition[
                    self.property_fields.index("cutoff_wavelength_um_from_gap")
                ] = float(target_wavelength_um)
            return condition
        condition = np.zeros(len(self.property_fields), dtype=np.float32)
        targets = {
            "gap_selected_eV": float(target_band_gap_eV),
            "cutoff_wavelength_um_from_gap": float(target_wavelength_um),
        }
        for field, value in targets.items():
            if field not in self.property_fields:
                continue
            index = self.property_fields.index(field)
            condition[index] = (value - self.center[index]) / max(
                abs(self.scale[index]), 1e-12
            )
        return condition

    # -- filtering ----------------------------------------------------------

    def _integerize(
        self, fractions: np.ndarray
    ) -> tuple[list[str], list[int], float] | None:
        if not self.vocabulary:
            raise MissingScientificPrerequisite(
                "VAE element vocabulary is not configured",
                missing=["vocabulary"],
            )
        order = np.argsort(fractions)[::-1]
        selected = [
            index
            for index in order[: self.max_elements]
            if fractions[index] >= 0.025
        ]
        if len(selected) < 2:
            selected = order[:2].tolist()
        selected_fractions = fractions[selected]
        selected_fractions = selected_fractions / selected_fractions.sum()
        best: tuple[list[int], float] | None = None
        for total_atoms in range(len(selected), self.max_atoms + 1):
            raw = selected_fractions * total_atoms
            counts = np.floor(raw).astype(int)
            counts[counts < 1] = 1
            remaining = total_atoms - int(counts.sum())
            if remaining > 0:
                priorities = np.argsort(raw - np.floor(raw))[::-1]
                for index in priorities[:remaining]:
                    counts[index] += 1
            elif remaining < 0:
                priorities = np.argsort(raw - counts)
                for index in priorities:
                    if remaining == 0:
                        break
                    removable = min(counts[index] - 1, -remaining)
                    counts[index] -= removable
                    remaining += removable
            divisor = reduce(math.gcd, counts.tolist())
            counts = counts // divisor
            reconstructed = counts / counts.sum()
            error = float(np.abs(reconstructed - selected_fractions).sum())
            if best is None or error < best[1]:
                best = (counts.tolist(), error)
        if best is None:
            return None
        return [self.vocabulary[index] for index in selected], best[0], best[1]

    @staticmethod
    def _chemistry(
        elements: list[str], counts: list[int]
    ) -> tuple[str, bool, tuple[dict[str, float], ...]]:
        from pymatgen.core import Composition, Element

        composition = Composition(dict(zip(elements, counts, strict=True)))
        formula = composition.reduced_formula
        try:
            guesses = composition.oxi_state_guesses(max_sites=-1)
        except (ValueError, ArithmeticError):
            guesses = ()
        examples_list: list[dict[str, float]] = []
        for guess in guesses:
            normalized = {
                str(element): float(value) for element, value in guess.items()
            }
            if not all(
                abs(value - round(value)) < 1e-8
                for value in normalized.values()
            ):
                continue
            if not any(abs(value) > 1e-8 for value in normalized.values()):
                continue
            electronegativities = {
                element: Element(element).X for element in normalized
            }
            consistent = all(
                not (
                    electronegativities[first] > electronegativities[second] + 1e-8
                    and normalized[first] > normalized[second] + 1e-8
                )
                for first in normalized
                for second in normalized
            )
            if consistent:
                examples_list.append(normalized)
        examples = tuple(examples_list)
        return formula, bool(examples), examples[:3]

    @staticmethod
    def _solid_state_elements_are_plausible(elements: list[str]) -> bool:
        """Reject noble-gas/molecular artifacts; require a solid-forming element."""
        from pymatgen.core import Element

        if "H" in elements:
            return False
        parsed = [Element(element) for element in elements]
        if any(element.is_noble_gas or element.is_actinoid for element in parsed):
            return False
        return any(element.is_metal or element.is_metalloid for element in parsed)

    # -- generate -----------------------------------------------------------

    def generate(
        self,
        *,
        target_band_gap_eV: float | None = None,
        target_wavelength_um: float | None = None,
        limit: int = 8,
        forbidden_elements: Iterable[str] | None = None,
    ) -> tuple[list[FormulaProposal], dict[str, object]]:
        """Return charge-balanced, optionally-novel integer formulas.

        ``forbidden_elements`` is an explicit optional user constraint; it
        defaults to empty (heavy infrared elements are legitimate).
        """
        if (target_band_gap_eV is None) == (target_wavelength_um is None):
            raise ValueError(
                "provide exactly one of target_band_gap_eV / "
                "target_wavelength_um"
            )
        if target_band_gap_eV is not None:
            band_gap = float(target_band_gap_eV)
            wavelength_um = HC_EV_UM / band_gap
        else:
            assert target_wavelength_um is not None  # mutual exclusion above
            wavelength_um = float(target_wavelength_um)
            band_gap = HC_EV_UM / wavelength_um
        if band_gap <= 0:
            raise ValueError("target band gap must be positive")
        self._load_torch_decoder()
        self._load_novelty_reference()
        condition = self.condition(
            target_band_gap_eV=band_gap,
            target_wavelength_um=wavelength_um,
        )
        decoded = self.decode_samples(condition, self.sample_count)
        if decoded.ndim != 2 or decoded.shape[1] != len(self.vocabulary):
            raise ValueError(
                f"decoder output shape {decoded.shape} does not match "
                f"vocabulary size {len(self.vocabulary)}"
            )
        forbidden = {
            item.strip() for item in (forbidden_elements or []) if item.strip()
        }
        proposals: list[FormulaProposal] = []
        seen: set[str] = set()
        rejection_counts: dict[str, int] = {
            "invalid_stoichiometry": 0,
            "forbidden_element": 0,
            "implausible_solid_elements": 0,
            "not_charge_neutral": 0,
            "known_formula": 0,
            "duplicate": 0,
        }
        for sample_index, fractions in enumerate(decoded):
            integerized = self._integerize(fractions)
            if integerized is None:
                rejection_counts["invalid_stoichiometry"] += 1
                continue
            elements, counts, error = integerized
            if set(elements) & forbidden:
                rejection_counts["forbidden_element"] += 1
                continue
            if not self._solid_state_elements_are_plausible(elements):
                rejection_counts["implausible_solid_elements"] += 1
                continue
            formula, charge_neutral, oxidation_examples = self._chemistry(
                elements, counts
            )
            if formula in seen:
                rejection_counts["duplicate"] += 1
                continue
            seen.add(formula)
            if self.require_charge_neutral and not charge_neutral:
                rejection_counts["not_charge_neutral"] += 1
                continue
            novel = formula not in self.known_formulas
            if self.require_novel and not novel:
                rejection_counts["known_formula"] += 1
                continue
            proposals.append(
                FormulaProposal(
                    formula=formula,
                    chemical_system="-".join(sorted(elements)),
                    elements=tuple(elements),
                    atom_counts=tuple(counts),
                    composition_error=error,
                    charge_neutral=charge_neutral,
                    oxidation_state_examples=oxidation_examples,
                    novel_against_training_data=novel,
                    decoded_sample_index=sample_index,
                )
            )
            if len(proposals) >= limit:
                break
        proposals.sort(key=lambda item: item.composition_error)
        metadata: dict[str, object] = {
            "backend": (
                "conditional_vae_formula_generator"
                if self.checkpoint_path
                else "injected_vae_formula_generator"
            ),
            "checkpoint": (
                str(self.checkpoint_path.resolve())
                if self.checkpoint_path
                else None
            ),
            "target_band_gap_eV": band_gap,
            "target_wavelength_um": wavelength_um,
            "decoded_sample_count": self.sample_count,
            "proposal_count": len(proposals),
            "rejection_counts": rejection_counts,
            "novelty_reference_count": len(self.known_formulas),
            "novelty_definition": (
                "reduced formula absent from the configured reference set"
            ),
            "defaults_note": (
                "no default forbidden elements and no atomic-number "
                "preference; heavy infrared elements (Hg/Pb/Bi/Te/Sb) are "
                "legitimate candidates unless the user forbids them"
            ),
            "scope": (
                "VAE proposes compositions only; it does not predict "
                "responsivity/EQE/detectivity/dark current"
            ),
        }
        return proposals, metadata
