from __future__ import annotations

import csv
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROPERTY_FIELDS = (
    "gap_selected_eV",
    "cutoff_wavelength_um_from_gap",
    "formation_energy_eV_per_atom",
    "energy_above_hull_eV_per_atom",
    "density_g_cm3",
    "dielectric_mean",
    "avg_electron_mass_m0",
    "avg_hole_mass_m0",
    "bulk_modulus_GPa",
    "shear_modulus_GPa",
    "exfoliation_energy_meV_per_atom",
    "max_IR_mode_cm-1",
    "min_IR_mode_cm-1",
    "spillage",
)

PROPERTY_ALIASES = {
    "band_gap_eV": "gap_selected_eV",
    "bandgap_eV": "gap_selected_eV",
    "cutoff_um": "cutoff_wavelength_um_from_gap",
    "formation_energy": "formation_energy_eV_per_atom",
    "ehull": "energy_above_hull_eV_per_atom",
    "density": "density_g_cm3",
    "dielectric": "dielectric_mean",
    "electron_mass": "avg_electron_mass_m0",
    "hole_mass": "avg_hole_mass_m0",
    "bulk_modulus": "bulk_modulus_GPa",
    "shear_modulus": "shear_modulus_GPa",
    "exfoliation_energy": "exfoliation_energy_meV_per_atom",
}

UNSUPPORTED_DEVICE_PROPERTIES = {
    "responsivity_a_w",
    "detectivity_jones",
    "dark_current_a",
    "dark_current_density_a_cm2",
    "response_time_s",
    "external_quantum_efficiency",
    "noise_equivalent_power_w_hz_0p5",
}


def _number(value: Any) -> float:
    """Convert a CSV value to float, returning NaN for absent database values."""
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a median/IQR scaler that is less sensitive to unstable outliers."""
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    standard_deviation = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-12, scale, standard_deviation)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center, scale


def _composition(elements: Iterable[str], vocabulary: list[str]) -> np.ndarray:
    counts = Counter(elements)
    total = sum(counts.values())
    if not total:
        raise ValueError("Structure has no elements")
    return np.asarray([counts.get(element, 0) / total for element in vocabulary], dtype=float)


def _load_raw_structures(
    archives: Iterable[str | Path], wanted_ids: set[str]
) -> dict[str, dict[str, Any]]:
    structures: dict[str, dict[str, Any]] = {}
    for archive in archives:
        archive_path = Path(archive)
        with zipfile.ZipFile(archive_path) as handle:
            names = handle.namelist()
            if len(names) != 1:
                raise ValueError(f"Expected one JSON file in {archive_path}, found {len(names)}")
            with handle.open(names[0]) as stream:
                records = json.load(stream)
        for record in records:
            jid = str(record.get("jid", ""))
            if jid in wanted_ids:
                structures[jid] = record["atoms"]
    missing = wanted_ids - structures.keys()
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Missing {len(missing)} candidate structures; examples: {examples}")
    return structures


def _design_matrix(
    properties: np.ndarray,
    present: np.ndarray,
    dimensionality: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    filled = np.where(present, properties, center)
    standardized = (filled - center) / scale
    missing = (~present).astype(float)
    is_2d = (dimensionality == "2D").astype(float)[:, None]
    intercept = np.ones((len(properties), 1), dtype=float)
    return np.concatenate([intercept, standardized, missing, is_2d], axis=1)


@dataclass(frozen=True)
class TrainingReport:
    record_count: int
    element_count: int
    ridge_alpha: float
    composition_mae: float
    top_element_accuracy: float
    property_fields: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "element_count": self.element_count,
            "ridge_alpha": self.ridge_alpha,
            "validation": {
                "composition_mae": self.composition_mae,
                "top_element_accuracy": self.top_element_accuracy,
                "note": "Random 20% validation split; metrics evaluate composition prior only.",
            },
            "property_fields": list(self.property_fields),
            "model_scope": (
                "DFT property-conditioned composition prior plus nonparametric structure retrieval"
            ),
        }


def train_inverse_index(
    candidate_csv: str | Path,
    raw_archives: Iterable[str | Path],
    output_dir: str | Path,
    ridge_alpha: float = 10.0,
    random_seed: int = 42,
) -> TrainingReport:
    """Train a property-to-composition prior and persist a searchable structure index."""
    candidate_path = Path(candidate_csv)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with candidate_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No candidates found in {candidate_path}")

    candidate_ids = {row["jarvis_id"] for row in rows}
    structures = _load_raw_structures(raw_archives, candidate_ids)
    vocabulary = sorted(
        {element for atoms in structures.values() for element in atoms.get("elements", [])}
    )
    properties = np.asarray(
        [[_number(row.get(field)) for field in PROPERTY_FIELDS] for row in rows], dtype=float
    )
    present = np.isfinite(properties)
    center, scale = _robust_center_scale(properties)
    dimensionality = np.asarray([row["dimensionality"] for row in rows])
    compositions = np.stack(
        [_composition(structures[row["jarvis_id"]]["elements"], vocabulary) for row in rows]
    )
    design = _design_matrix(properties, present, dimensionality, center, scale)

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(rows))
    split = max(1, int(0.8 * len(rows)))
    train_indices = indices[:split]
    validation_indices = indices[split:]
    regularizer = np.eye(design.shape[1], dtype=float) * ridge_alpha
    regularizer[0, 0] = 0.0
    train_design = design[train_indices]
    coefficients = np.linalg.solve(
        train_design.T @ train_design + regularizer,
        train_design.T @ compositions[train_indices],
    )
    if len(validation_indices):
        predictions = np.clip(design[validation_indices] @ coefficients, 0.0, None)
        totals = predictions.sum(axis=1, keepdims=True)
        predictions = predictions / np.where(totals > 0, totals, 1.0)
        composition_mae = float(np.mean(np.abs(predictions - compositions[validation_indices])))
        top_element_accuracy = float(
            np.mean(
                np.argmax(predictions, axis=1)
                == np.argmax(compositions[validation_indices], axis=1)
            )
        )
    else:
        composition_mae = math.nan
        top_element_accuracy = math.nan

    # Refit the deployable model on all records after estimating holdout metrics.
    coefficients = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ compositions,
    )

    metadata_fields = (
        "jarvis_id",
        "dimensionality",
        "formula",
        "elements",
        "n_elements",
        "cutoff_region",
        "gap_method_selected",
        "space_group_number",
        "space_group_symbol",
        "crystal_system",
        "jarvis_url",
        "source_dataset_doi",
    )
    metadata = [{field: row.get(field, "") for field in metadata_fields} for row in rows]
    (destination / "candidate_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    with (destination / "structures.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {"jarvis_id": row["jarvis_id"], "atoms": structures[row["jarvis_id"]]}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    np.savez_compressed(
        destination / "inverse_index.npz",
        properties=properties,
        present=present,
        compositions=compositions,
        center=center,
        scale=scale,
        coefficients=coefficients,
        vocabulary=np.asarray(vocabulary),
        dimensionality=dimensionality,
        property_fields=np.asarray(PROPERTY_FIELDS),
    )
    report = TrainingReport(
        record_count=len(rows),
        element_count=len(vocabulary),
        ridge_alpha=ridge_alpha,
        composition_mae=composition_mae,
        top_element_accuracy=top_element_accuracy,
        property_fields=PROPERTY_FIELDS,
    )
    (destination / "training_report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _normalize_targets(targets: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    unsupported = sorted(set(targets) & UNSUPPORTED_DEVICE_PROPERTIES)
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"Unsupported device-level properties without paired labels: {names}. "
            "Build a normalized experimental device table before training on these targets."
        )
    for key, value in targets.items():
        field = PROPERTY_ALIASES.get(key, key)
        if field not in PROPERTY_FIELDS:
            raise ValueError(f"Unknown target property: {key}")
        normalized[field] = float(value)
    if not normalized:
        raise ValueError("At least one supported target property is required")
    return normalized


def _load_structures(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["atoms"] for line in handle if line.strip()]


def _cif_text(metadata: dict[str, Any], atoms: dict[str, Any]) -> str:
    lattice = np.asarray(atoms["lattice_mat"], dtype=float)
    coordinates = np.asarray(atoms["coords"], dtype=float)
    if bool(atoms.get("cartesian", True)):
        coordinates = coordinates @ np.linalg.inv(lattice)
    coordinates = coordinates - np.floor(coordinates)
    abc = atoms.get("abc")
    angles = atoms.get("angles")
    if not abc or not angles:
        abc = [float(np.linalg.norm(vector)) for vector in lattice]
        angles = []
        for first, second in ((1, 2), (0, 2), (0, 1)):
            cosine = np.dot(lattice[first], lattice[second]) / (abc[first] * abc[second])
            angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    lines = [
        f"data_{metadata['jarvis_id'].replace('-', '_')}",
        f"_chemical_formula_sum '{metadata['formula']}'",
        f"_cell_length_a {float(abc[0]):.8f}",
        f"_cell_length_b {float(abc[1]):.8f}",
        f"_cell_length_c {float(abc[2]):.8f}",
        f"_cell_angle_alpha {float(angles[0]):.8f}",
        f"_cell_angle_beta {float(angles[1]):.8f}",
        f"_cell_angle_gamma {float(angles[2]):.8f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "'x, y, z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    counters: Counter[str] = Counter()
    for element, coordinate in zip(atoms["elements"], coordinates, strict=True):
        counters[element] += 1
        lines.append(
            f"{element}{counters[element]} {element} "
            f"{coordinate[0]:.10f} {coordinate[1]:.10f} {coordinate[2]:.10f}"
        )
    return "\n".join(lines) + "\n"


class InverseMaterialRetriever:
    """Property-conditioned composition model with full-structure candidate retrieval."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        arrays = np.load(self.model_dir / "inverse_index.npz")
        self.properties = arrays["properties"]
        self.present = arrays["present"]
        self.compositions = arrays["compositions"]
        self.center = arrays["center"]
        self.scale = arrays["scale"]
        self.coefficients = arrays["coefficients"]
        self.vocabulary = arrays["vocabulary"].tolist()
        self.dimensionality = arrays["dimensionality"]
        self.property_fields = arrays["property_fields"].tolist()
        self.metadata = json.loads(
            (self.model_dir / "candidate_metadata.json").read_text(encoding="utf-8")
        )
        if len(self.metadata) != len(self.properties):
            raise ValueError("Model arrays and candidate metadata are misaligned")

    def predict(
        self,
        targets: dict[str, float],
        top_k: int = 8,
        dimensionality: str | None = None,
        crystal_system: str | None = None,
        forbidden_elements: Iterable[str] = (),
        max_energy_above_hull_eV_per_atom: float | None = 0.2,
        composition_weight: float = 0.1,
        property_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked existing structures closest to a requested property condition."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized = _normalize_targets(targets)
        indices = [self.property_fields.index(field) for field in normalized]
        target_values = np.asarray([normalized[field] for field in normalized], dtype=float)
        scales = self.scale[indices]
        differences = (self.properties[:, indices] - target_values) / scales
        available = self.present[:, indices]
        differences = np.where(available, differences, 3.0)
        weights = np.asarray(
            [float((property_weights or {}).get(field, 1.0)) for field in normalized], dtype=float
        )
        if np.any(weights <= 0):
            raise ValueError("Property weights must be positive")
        property_score = np.sum(differences**2 * weights, axis=1) / np.sum(weights)

        query_properties = self.center[None, :].copy()
        query_present = np.zeros_like(query_properties, dtype=bool)
        for field, value in normalized.items():
            index = self.property_fields.index(field)
            query_properties[0, index] = value
            query_present[0, index] = True
        query_design = _design_matrix(
            query_properties,
            query_present,
            np.asarray([dimensionality or "3D"]),
            self.center,
            self.scale,
        )
        predicted_composition = np.clip(query_design @ self.coefficients, 0.0, None)[0]
        total = predicted_composition.sum()
        if total:
            predicted_composition /= total
        composition_score = np.mean(np.abs(self.compositions - predicted_composition), axis=1)
        score = property_score + composition_weight * composition_score

        allowed = np.ones(len(score), dtype=bool)
        if dimensionality:
            allowed &= self.dimensionality == dimensionality.upper()
        if crystal_system:
            allowed &= np.asarray(
                [row["crystal_system"].lower() == crystal_system.lower() for row in self.metadata]
            )
        forbidden = {element.strip() for element in forbidden_elements if element.strip()}
        if forbidden:
            allowed &= np.asarray(
                [not (set(row["elements"].split(";")) & forbidden) for row in self.metadata]
            )
        if max_energy_above_hull_eV_per_atom is not None:
            hull_index = self.property_fields.index("energy_above_hull_eV_per_atom")
            allowed &= self.present[:, hull_index]
            allowed &= self.properties[:, hull_index] <= max_energy_above_hull_eV_per_atom
        ranking = np.flatnonzero(allowed)
        ranking = ranking[np.argsort(score[ranking])[:top_k]]
        results = []
        for rank, row_index in enumerate(ranking, start=1):
            property_values = {
                field: (
                    float(self.properties[row_index, index])
                    if self.present[row_index, index]
                    else None
                )
                for index, field in enumerate(self.property_fields)
            }
            results.append(
                {
                    "rank": rank,
                    "score": float(score[row_index]),
                    "property_score": float(property_score[row_index]),
                    "composition_prior_score": float(composition_score[row_index]),
                    "metadata": self.metadata[row_index],
                    "properties": property_values,
                    "model_row_index": int(row_index),
                }
            )
        return results

    def export_results(
        self, results: list[dict[str, Any]], output_dir: str | Path
    ) -> dict[str, Any]:
        """Export ranked structures as CIF/JSON together with a machine-readable manifest."""
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        structures = _load_structures(self.model_dir / "structures.jsonl")
        exported = []
        for result in results:
            atoms = structures[result["model_row_index"]]
            metadata = result["metadata"]
            stem = f"rank_{result['rank']:02d}_{metadata['jarvis_id']}"
            json_path = destination / f"{stem}.json"
            cif_path = destination / f"{stem}.cif"
            json_path.write_text(
                json.dumps(
                    {"metadata": metadata, "properties": result["properties"], "atoms": atoms},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            cif_path.write_text(_cif_text(metadata, atoms), encoding="utf-8")
            public_result = {key: value for key, value in result.items() if key != "model_row_index"}
            public_result["structure_json"] = str(json_path.resolve())
            public_result["structure_cif"] = str(cif_path.resolve())
            exported.append(public_result)
        manifest = {
            "model": "property-conditioned composition prior + structure retrieval",
            "candidate_count": len(exported),
            "candidates": exported,
            "warning": (
                "Candidates are DFT database structures, not validated detector devices. "
                "Relaxation, optical absorption, defect, toxicity, synthesis and device checks remain."
            ),
        }
        (destination / "prediction_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
