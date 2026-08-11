"""VASP input generation (Sprint 3 Phase D, donor migration).

Migrated from the donor ``VaspInputGenerator`` with two changes:
* no POTCAR content is ever written; a ``POTCAR.policy`` manifest documents
  how the user's pseudopotential directory resolves each element
  (``PMG_VASP_PSP_DIR`` locally or a remote pseudopotential location)
* profiles (``profiles.py``) replace ad-hoc INCAR branches; stage settings
  are merged deterministically on top of the profile base INCAR
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.profiles import (
    VaspProfile,
    get_profile,
)


def _kpoint_grid_density(
    volume_ang3: float,
    density: int,
    lattice_lengths_ang: list[float],
    num_sites: int = 1,
) -> list[int]:
    """Monkhorst-Pack grid from k-points per atom (pymatgen-style)."""
    del volume_ang3  # kept in the signature for compatibility
    ngrid = float(density) / max(1, int(num_sites))
    multiplier = (
        ngrid
        * lattice_lengths_ang[0]
        * lattice_lengths_ang[1]
        * lattice_lengths_ang[2]
    ) ** (1.0 / 3.0)
    return [
        max(1, math.floor(multiplier / length))
        for length in lattice_lengths_ang
    ]


def _stage_incar(profile: VaspProfile, stage: str) -> dict[str, Any]:
    """Merge profile base INCAR with deterministic stage-specific settings."""
    settings = dict(profile.base_incar)
    stage_rules: dict[str, dict[str, Any]] = {
        "relax": {
            "IBRION": 2,
            "NSW": 100,
            "EDIFFG": -0.02,
            "ISIF": 3,
            "LREAL": "Auto",
            "LWAVE": False,
            "LCHARG": True,
        },
        "static": {
            "IBRION": -1,
            "NSW": 0,
            "LREAL": False,
            "LWAVE": False,
            "LCHARG": True,
        },
        "band": {
            "IBRION": -1,
            "NSW": 0,
            "ICHARG": 11,
            "LORBIT": 11,
            "LREAL": False,
            "LWAVE": False,
        },
        "dos": {
            "IBRION": -1,
            "NSW": 0,
            "ICHARG": 11,
            "LORBIT": 11,
            "NEDOS": 2000,
            "LREAL": False,
            "LWAVE": False,
        },
        "optics": {
            "IBRION": -1,
            "NSW": 0,
            "LOPTICS": True,
            "NEDOS": 2000,
            "CSHIFT": 0.1,
            "LREAL": False,
            "LWAVE": False,
            "LCHARG": False,
        },
        "md": {
            "IBRION": 0,
            "NSW": 1000,
            "POTIM": 1.0,
            "TEBEG": 300,
            "TEEND": 300,
            "SMASS": 0,
            "ISIF": 2,
            "NBLOCK": 1,
            "LWAVE": True,
            "LCHARG": False,
        },
        "snapshot": {
            "IBRION": -1,
            "NSW": 0,
            "ISTART": 1,
            "LWAVE": True,
            "LCHARG": False,
        },
    }
    settings.update(stage_rules.get(stage, {}))
    return settings


def _render_incar(settings: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in settings.items():
        if isinstance(value, bool):
            rendered = ".TRUE." if value else ".FALSE."
        elif isinstance(value, float):
            rendered = f"{value:.6g}"
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def _render_kpoints(grid: list[int]) -> str:
    return (
        "Automatic mesh\n"
        "0\n"
        "Monkhorst-Pack\n"
        f"  {grid[0]}  {grid[1]}  {grid[2]}\n"
        "0.0 0.0 0.0\n"
    )


class VaspInputGenerator:
    """Generate a VASP input directory from a structure + profile."""

    def __init__(
        self,
        *,
        psp_dir: str | None = None,
        jobs_local_dir: str | Path = "output/vasp_inputs",
    ) -> None:
        self.psp_dir = Path(
            psp_dir or os.environ.get("PMG_VASP_PSP_DIR", "")
        ).expanduser()
        self.jobs_local_dir = Path(jobs_local_dir).expanduser()

    # -- POTCAR policy ------------------------------------------------------

    def potcar_policy_text(self, site_symbols: list[str], profile: VaspProfile) -> str:
        """Document how POTCAR resolves; never contains POTCAR content."""
        family = "potpaw_PBE.64"
        lines = [
            "# POTCAR policy (POTCAR files are never committed or generated",
            "# by PhotoMatAgent).",
            f"profile: {profile.name}",
            f"executable: {profile.executable}",
            "resolution_order:",
            "  1. PMG_VASP_PSP_DIR (local) with potpaw_PBE.64/<Element>/POTCAR",
            "  2. remote pseudopotential location configured on SCNet",
            "  3. explicit potcar_overrides",
        ]
        if self.psp_dir.is_dir():
            for symbol in dict.fromkeys(site_symbols):
                candidate = self.psp_dir / family / symbol / "POTCAR"
                lines.append(
                    f"  {symbol}: {'resolved' if candidate.is_file() else 'MISSING'} "
                    f"({candidate})"
                )
        else:
            lines.append(
                "PMG_VASP_PSP_DIR not configured locally; POTCAR must be "
                "resolved on the remote side before submission."
            )
        return "\n".join(lines) + "\n"

    # -- structure ----------------------------------------------------------

    @staticmethod
    def load_structure(structure_path: str | Path) -> Any:
        """Load a CIF/POSCAR/other structure via pymatgen."""
        from pymatgen.core import Structure

        path = Path(structure_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"structure file does not exist: {path}")
        return Structure.from_file(path)

    # -- generation ---------------------------------------------------------

    def generate_stage(
        self,
        *,
        structure_path: str | Path,
        profile: VaspProfile,
        stage: str,
        output_dir: str | Path,
        spec_overrides: dict[str, Any] | None = None,
        source_poscar: str | Path | None = None,
    ) -> dict[str, Any]:
        """Write POSCAR/INCAR/KPOINTS/POTCAR.policy for one stage."""
        overrides = dict(spec_overrides or {})
        structure = self.load_structure(structure_path)
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        incar = _stage_incar(profile, stage)
        incar["ENCUT"] = int(overrides.get("encut_ev", incar.get("ENCUT", 520)))
        if "kpoint_grid" in overrides:
            grid = [int(item) for item in overrides["kpoint_grid"]]
        else:
            density = int(overrides.get("kpoint_density", 1000))
            grid = _kpoint_grid_density(
                float(structure.volume),
                density,
                [float(length) for length in structure.lattice.abc],
                len(structure),
            )
        incar.update(
            {
                key: value
                for key, value in overrides.get("incar", {}).items()
            }
        )
        structure.to(
            filename=str(output / "POSCAR"),
            fmt="poscar",
        )
        if source_poscar:
            source = Path(source_poscar).expanduser().resolve()
            if source.is_file():
                (output / "POSCAR").write_text(source.read_text(encoding="utf-8"))
        (output / "INCAR").write_text(_render_incar(incar), encoding="utf-8")
        (output / "KPOINTS").write_text(_render_kpoints(grid), encoding="utf-8")
        symbols = [str(element.symbol) for element in structure.composition.elements]
        (output / "POTCAR.policy").write_text(
            self.potcar_policy_text(symbols, profile), encoding="utf-8"
        )
        return {
            "stage": stage,
            "directory": str(output),
            "incar": {key: str(value) for key, value in incar.items()},
            "kpoint_grid": grid,
            "elements": symbols,
            "formula": structure.composition.reduced_formula,
            "soc": profile.soc,
            "executable": profile.executable,
            "potcar": "policy-only (resolved at submit time)",
        }

    def prepare_workflow(
        self,
        *,
        structure_path: str | Path,
        profile_name: str,
        output_root: str | Path,
        spec_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare every stage of a profile and write workflow.json."""
        profile = get_profile(profile_name)
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        stages: list[dict[str, Any]] = []
        source_poscar: str | Path | None = None
        dependencies: dict[str, dict[str, Any]] = {
            "relax": {"depends_on": None, "required_outputs": []},
            "static": {"depends_on": "relax", "required_outputs": ["CONTCAR"]},
            "band": {"depends_on": "static", "required_outputs": ["CONTCAR", "CHGCAR"]},
            "dos": {"depends_on": "static", "required_outputs": ["CONTCAR", "CHGCAR"]},
            "optics": {"depends_on": "static", "required_outputs": ["CONTCAR", "CHGCAR"]},
            "md": {"depends_on": None, "required_outputs": []},
            "snapshot": {"depends_on": "md", "required_outputs": ["CONTCAR", "XDATCAR"]},
        }
        for index, stage in enumerate(profile.stages, start=1):
            stage_dir = root / f"{index:02d}_{stage}"
            generated = self.generate_stage(
                structure_path=structure_path,
                profile=profile,
                stage=stage,
                output_dir=stage_dir,
                spec_overrides=spec_overrides,
                source_poscar=source_poscar,
            )
            generated.update(dependencies.get(stage, {}))
            stages.append(generated)
            if stage in {"relax", "md"}:
                source_poscar = stage_dir / "POSCAR"
        manifest = {
            "profile": profile.name,
            "soc": profile.soc,
            "executable": profile.executable,
            "stages": stages,
            "potcar_policy": "POTCAR resolved from PMG_VASP_PSP_DIR or remote "
            "location at submit time; never committed",
            "notes": [
                "downstream stages replace POSCAR with the upstream CONTCAR "
                "and copy required outputs before submission",
            ],
        }
        (root / "workflow.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
