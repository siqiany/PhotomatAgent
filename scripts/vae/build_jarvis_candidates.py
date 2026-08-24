#!/usr/bin/env python3
"""Build infrared-bandgap candidate tables from official NIST JARVIS archives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HC_EV_UM = 1.239841984
MISSING = {"", "na", "nan", "none", "null", "-"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DATA = REPOSITORY_ROOT / "data" / "photoelectric_vae" / "training"


FIELDS = [
    "jarvis_id",
    "dimensionality",
    "formula",
    "elements",
    "n_elements",
    "gap_selected_eV",
    "gap_method_selected",
    "cutoff_wavelength_um_from_gap",
    "cutoff_region",
    "optb88vdw_bandgap_eV",
    "mbj_bandgap_eV",
    "hse_bandgap_eV",
    "formation_energy_eV_per_atom",
    "energy_above_hull_eV_per_atom",
    "density_g_cm3",
    "space_group_number",
    "space_group_symbol",
    "crystal_system",
    "dielectric_x",
    "dielectric_y",
    "dielectric_z",
    "dielectric_mean",
    "avg_electron_mass_m0",
    "avg_hole_mass_m0",
    "bulk_modulus_GPa",
    "shear_modulus_GPa",
    "exfoliation_energy_meV_per_atom",
    "max_IR_mode_cm-1",
    "min_IR_mode_cm-1",
    "spillage",
    "icsd_ids",
    "jarvis_url",
    "source_dataset_doi",
    "source_archive",
    "source_type",
    "evidence_scope",
    "candidate_warning",
]


def number(value: Any) -> float | None:
    """Return a finite float, or None for JARVIS missing-value sentinels."""
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, str) and value.strip().lower() in MISSING:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def selected_gap(record: dict[str, Any]) -> tuple[float | None, str]:
    """Choose the highest-level available positive gap: HSE, TBmBJ, then OptB88vdW."""
    for key, method in (
        ("hse_gap", "HSE"),
        ("mbj_bandgap", "TBmBJ"),
        ("optb88vdw_bandgap", "OptB88vdW"),
    ):
        value = number(record.get(key))
        if value is not None and value > 0:
            return value, method
    return None, ""


def cutoff_region(wavelength: float) -> str:
    """Classify a gap-derived cutoff wavelength; boundaries are explicitly documented."""
    if wavelength < 1.0:
        return "VIS/NIR (<1 um)"
    if wavelength < 3.0:
        return "SWIR (1-3 um)"
    if wavelength <= 5.0:
        return "MWIR (3-5 um)"
    if wavelength < 8.0:
        return "extended-MWIR (5-8 um)"
    if wavelength <= 15.0:
        return "LWIR (8-15 um)"
    return "VLWIR (>15 um)"


def mean_numbers(values: list[Any]) -> float | None:
    nums = [n for value in values if (n := number(value)) is not None]
    # Preserve the NumPy mean used to create the committed training table.
    # Python's built-in summation differs by a few ULPs for 667 rows and would
    # make a byte-for-byte rebuild fail despite identical source values.
    return float(np.mean(nums)) if nums else None


def serialize(value: Any) -> str:
    if value in (None, "na", ""):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def load_zip(path: Path) -> tuple[list[dict[str, Any]], str]:
    with zipfile.ZipFile(path) as archive:
        json_members = [m for m in archive.infolist() if m.filename.lower().endswith(".json")]
        if len(json_members) != 1:
            raise ValueError(f"Expected one JSON member in {path}, found {len(json_members)}")
        with archive.open(json_members[0]) as stream:
            return json.load(stream), json_members[0].filename


def transform(
    record: dict[str, Any], dimensionality: str, source_doi: str, archive_name: str
) -> dict[str, Any] | None:
    gap, method = selected_gap(record)
    if gap is None:
        return None
    cutoff = HC_EV_UM / gap
    if not 0.8 <= cutoff <= 15.0:
        return None

    atoms = record.get("atoms") or {}
    elements = sorted(set(atoms.get("elements") or []))
    row = {
        "jarvis_id": record.get("jid"),
        "dimensionality": dimensionality,
        "formula": record.get("formula"),
        "elements": ";".join(elements),
        "n_elements": len(elements),
        "gap_selected_eV": gap,
        "gap_method_selected": method,
        "cutoff_wavelength_um_from_gap": round(cutoff, 6),
        "cutoff_region": cutoff_region(cutoff),
        "optb88vdw_bandgap_eV": number(record.get("optb88vdw_bandgap")),
        "mbj_bandgap_eV": number(record.get("mbj_bandgap")),
        "hse_bandgap_eV": number(record.get("hse_gap")),
        "formation_energy_eV_per_atom": number(record.get("formation_energy_peratom")),
        "energy_above_hull_eV_per_atom": number(record.get("ehull")),
        "density_g_cm3": number(record.get("density")),
        "space_group_number": record.get("spg_number"),
        "space_group_symbol": record.get("spg_symbol") or record.get("spg"),
        "crystal_system": record.get("crys"),
        "dielectric_x": number(record.get("epsx")),
        "dielectric_y": number(record.get("epsy")),
        "dielectric_z": number(record.get("epsz")),
        "dielectric_mean": mean_numbers(
            [record.get("epsx"), record.get("epsy"), record.get("epsz")]
        ),
        "avg_electron_mass_m0": number(record.get("avg_elec_mass")),
        "avg_hole_mass_m0": number(record.get("avg_hole_mass")),
        "bulk_modulus_GPa": number(record.get("bulk_modulus_kv")),
        "shear_modulus_GPa": number(record.get("shear_modulus_gv")),
        "exfoliation_energy_meV_per_atom": number(record.get("exfoliation_energy")),
        "max_IR_mode_cm-1": number(record.get("max_ir_mode")),
        "min_IR_mode_cm-1": number(record.get("min_ir_mode")),
        "spillage": number(record.get("spillage")),
        "icsd_ids": serialize(record.get("icsd")),
        "jarvis_url": f"https://jarvis.nist.gov/jarvisdft/dft_3d/{record.get('jid')}",
        "source_dataset_doi": source_doi,
        "source_archive": archive_name,
        "source_type": "DFT database",
        "evidence_scope": "computed material properties; no detector-device validation implied",
        "candidate_warning": (
            "Gap-derived spectral screening only. Verify direct/indirect gap, optical absorption, "
            "defects, stability, toxicity, growth feasibility, and experimental detector metrics."
        ),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--three-d",
        type=Path,
        default=DEFAULT_TRAINING_DATA / "raw" / "jarvis_dft3d_2025.zip",
    )
    parser.add_argument(
        "--two-d",
        type=Path,
        default=DEFAULT_TRAINING_DATA / "raw" / "jarvis_dft2d_2022.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRAINING_DATA,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (args.three_d, "3D", "10.6084/m9.figshare.6815699.v11"),
        (args.two_d, "2D", "10.6084/m9.figshare.6815705.v8"),
    ]
    all_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path, dimensionality, doi in specs:
        records, member = load_zip(path)
        rows = [
            row
            for record in records
            if (row := transform(record, dimensionality, doi, member)) is not None
        ]
        rows.sort(key=lambda r: (r["cutoff_wavelength_um_from_gap"], r["formula"] or ""))
        write_csv(args.output_dir / f"jarvis_{dimensionality.lower()}_ir_candidates.csv", rows)
        all_rows.extend(rows)
        source_counts[dimensionality] = len(rows)

    all_rows.sort(
        key=lambda r: (r["cutoff_region"], r["cutoff_wavelength_um_from_gap"], r["formula"] or "")
    )
    write_csv(args.output_dir / "jarvis_all_ir_candidates.csv", all_rows)
    summary = {
        "selection": "0.8 <= 1.239841984 / selected_gap_eV <= 15.0 um",
        "gap_priority": ["HSE", "TBmBJ", "OptB88vdW"],
        "counts_by_dimensionality": source_counts,
        "counts_by_cutoff_region": dict(Counter(r["cutoff_region"] for r in all_rows)),
        "total": len(all_rows),
    }
    (args.output_dir / "jarvis_candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
