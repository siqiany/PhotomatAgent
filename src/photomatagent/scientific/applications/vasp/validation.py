"""VASP result validation and lightweight parsing (Sprint 3 section 30-31).

Validation never equates "Slurm COMPLETED" with "scientifically valid":
the job state is scheduler state, and ``validate_output`` additionally
requires the vasprun.xml contract:

* vasprun.xml exists and is non-empty
* the file is well-formed XML (ElementTree parse reaches EOF)
* at least one electronic SCF block (``<scstep>`` with ``<energy>``) exists
* electronic convergence marker: the final ``<scstep>`` of the final
  ``<calculation>`` contains the ``<c>`` / ``<v>`` elements that VASP
  appends when the SCF loop exits on convergence (conservative: when the
  marker is absent the run is reported as "cannot confirm", never as
  converged)
* relax stages additionally require an explicit trusted structural
  convergence marker (the OUTCAR ``reached required accuracy`` text).  An
  electronic ``EDIFF`` marker, a low ionic-step count, and ``NSW=0`` are not
  accepted as proof that a relax run completed.

``parse_result`` extracts bounded values (final energy, dielectric
spectrum summary) that enter ScientificEvidence.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

REQUIRED_VASP_RESULT_FILES = (
    "OUTCAR",
    "CONTCAR",
    "CHGCAR",
    "vasprun.xml",
    "OSZICAR",
)

_STAGE_REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "md": ("XDATCAR",),
    "snapshot": ("WAVECAR",),
}


def required_result_files(stage: str | None = None) -> tuple[str, ...]:
    """Return the bounded download list for one execution stage.

    Large stage-specific artifacts (WAVECAR/XDATCAR) are requested only for
    the stage that actually validates them.
    """
    extra = _STAGE_REQUIRED_ARTIFACTS.get(str(stage or ""), ())
    return tuple(dict.fromkeys((*REQUIRED_VASP_RESULT_FILES, *extra)))

class VasprunCheck(BaseModel):
    """Outcome of one vasprun.xml validation pass."""

    exists: bool = False
    well_formed_xml: bool = False
    has_scf_blocks: bool = False
    electronic_converged: bool | None = None
    ionic_converged: bool | None = None
    reasons: list[str] = Field(default_factory=list)


def _load_vasprun(path: Path) -> ET.Element | None:
    try:
        tree = ET.parse(path)
        return tree.getroot()
    except (ET.ParseError, OSError, ValueError):
        return None


def check_vasprun(vasprun_path: str | Path) -> VasprunCheck:
    """Run the structural vasprun.xml contract checks."""
    path = Path(vasprun_path)
    check = VasprunCheck()
    if not path.is_file() or path.stat().st_size == 0:
        check.reasons.append("vasprun.xml missing or empty")
        return check
    check.exists = True
    root = _load_vasprun(path)
    if root is None:
        check.reasons.append("vasprun.xml is not well-formed XML")
        return check
    check.well_formed_xml = True
    calculations = root.findall("calculation")
    if not calculations:
        check.reasons.append("vasprun.xml contains no <calculation> blocks")
        return check
    final = calculations[-1]
    scsteps = final.findall("scstep")
    if not scsteps:
        check.reasons.append("no <scstep> electronic SCF blocks found")
        return check
    check.has_scf_blocks = True
    final_step = scsteps[-1]
    has_c = final_step.findall("c")
    has_v = final_step.findall("v")
    outcar = path.parent / "OUTCAR"
    outcar_marker = False
    electronic_outcar_marker = False
    if outcar.is_file():
        tail = outcar.read_text(encoding="utf-8", errors="replace")[-200000:]
        electronic_outcar_marker = (
            "aborting loop because EDIFF is reached" in tail
            or "EDIFF was reached" in tail
        )
        # Only this explicit structural convergence marker is trusted.  The
        # electronic "EDIFF is reached" text says nothing about ionic
        # relaxation, and neither does a low ionic-step count or NSW=0.
        outcar_marker = "reached required accuracy" in tail
    if (has_c and has_v) or electronic_outcar_marker:
        check.electronic_converged = True
    else:
        check.electronic_converged = None
        check.reasons.append(
            "electronic convergence marker not found in vasprun.xml or "
            "OUTCAR; SCF convergence cannot be confirmed"
        )
    check.ionic_converged = True if outcar_marker else None
    return check


_VALIDATION_STAGES = frozenset(
    {"relax", "static", "band", "dos", "optics", "md", "snapshot"}
)


def _validation_stage(profile_name: str, stage: str | None) -> str | None:
    """Separate the scientific method profile from the execution stage.

    Historical callers passed the stage name as ``profile_name`` (for example
    ``profile_name="relax"``).  Preserve that contract only while it is
    unambiguous.  New callers should pass ``stage`` explicitly, because
    scientific profiles such as ``narrow_gap_soc`` do not select a stage.
    """
    if stage is not None and stage.strip():
        return stage.strip()
    if profile_name in _VALIDATION_STAGES:
        return profile_name
    return None


def _nonfinite_total_energy_problem(directory: Path) -> str | None:
    """Return an explicit problem if a parsed final total energy is not finite."""
    try:
        parsed = parse_result(directory)
    except (TypeError, ValueError, OverflowError) as exc:
        return f"final total energy could not be parsed as a finite number: {exc}"
    energy = parsed.get("final_energy_eV")
    if isinstance(energy, (int, float)) and not math.isfinite(float(energy)):
        return "final total energy is not finite"
    return None


def validate_output(
    result_dir: str | Path,
    *,
    profile_name: str,
    stage: str | None = None,
) -> list[str]:
    """Return a list of validation problems (empty means acceptable).

    ``profile_name`` identifies the scientific method/parameters.  ``stage``
    identifies the workflow stage whose requirements apply.  Passing a stage
    explicitly is required for profiles whose names are not stage names.
    """
    directory = Path(result_dir)
    problems: list[str] = []
    check = check_vasprun(directory / "vasprun.xml")
    problems.extend(check.reasons)
    if check.electronic_converged is False:
        problems.append("electronic SCF did not converge")
    nonfinite = _nonfinite_total_energy_problem(directory)
    if nonfinite is not None:
        problems.append(nonfinite)

    validation_stage = _validation_stage(profile_name, stage)
    if validation_stage is None:
        problems.append(
            "validation stage is required when profile_name="
            f"{profile_name!r} is not a stage name"
        )
        return problems
    if validation_stage not in _VALIDATION_STAGES:
        problems.append(f"unsupported validation stage: {validation_stage!r}")
        return problems
    if validation_stage == "relax" and check.ionic_converged is not True:
        problems.append("relax stage: ionic convergence cannot be confirmed")
    if validation_stage in {"band", "dos", "optics"} and not (
        directory / "CHGCAR"
    ).is_file():
        problems.append(
            f"{validation_stage} stage requires CHGCAR from the static run"
        )
    if validation_stage in {"md", "snapshot"}:
        parsed = parse_result(directory)
        if "final_energy_eV" not in parsed:
            problems.append(
                f"{validation_stage} stage requires a parsed final energy"
            )
        if validation_stage == "md" and not (directory / "XDATCAR").is_file():
            problems.append("md stage requires XDATCAR from the MD trajectory")
        if validation_stage == "snapshot" and not (directory / "WAVECAR").is_file():
            problems.append("snapshot stage requires WAVECAR from the MD snapshot")
    return problems


def parse_band_gap(
    result_dir: str | Path,
    *,
    stage: str,
    profile_name: str,
) -> dict[str, Any]:
    """Return a bounded band-gap evidence candidate or an explicit gap.

    LIMITED (task 14): the repository has no fixed periodic SOC/noncollinear
    band/static vasprun.xml sample against which the installed parser could be
    validated for this project.  This function therefore never fabricates a
    numeric gap.  It returns ``UNSUPPORTED`` with no value until such samples
    exist; callers must keep the target property UNKNOWN.
    """
    return {
        "property": "band_gap",
        "status": "UNSUPPORTED",
        "value_eV": None,
        "unit": "eV",
        "stage": stage,
        "profile_name": profile_name,
        "reason": "NO_APPLICABLE_FIXED_SOC_NONCOLLINEAR_BAND_SAMPLE",
        "limitations": (
            "No fixed periodic SOC/noncollinear band/static vasprun.xml sample "
            "is present in the repository; pymatgen parsed an unrelated "
            "molecular fixture only. Numeric band_gap evidence is not emitted."
        ),
        "result_dir": str(result_dir),
    }


def parse_result(result_dir: str | Path) -> dict[str, Any]:
    """Bounded parse: final energy + dielectric spectrum summary."""
    directory = Path(result_dir)
    parsed: dict[str, Any] = {}
    vasprun = directory / "vasprun.xml"
    root = _load_vasprun(vasprun)
    if root is not None:
        calculations = root.findall("calculation")
        if calculations:
            scsteps = calculations[-1].findall("scstep")
            energies = [
                step.findtext("energy/i[@name='e_fr_energy']", default="")
                for step in scsteps
            ]
            values = [float(value) for value in energies if value.strip()]
            if values:
                parsed["final_energy_eV"] = values[-1]
                parsed["energy_series_first_last_eV"] = [values[0], values[-1]]
        dielectric = root.find("calculation/dielectricfunction")
        if dielectric is not None:
            real = dielectric.find("varray[@name='real']")
            imag = dielectric.find("varray[@name='imag']")
            parsed["dielectric"] = _dielectric_summary(real, imag)
    outcar = directory / "OUTCAR"
    if outcar.is_file():
        text = outcar.read_text(encoding="utf-8", errors="replace")
        toten = re.findall(r"free\s+energy\s+TOTEN\s+=\s+([-+0-9.Ee]+)", text)
        if toten and "final_energy_eV" not in parsed:
            parsed["final_energy_eV"] = float(toten[-1])
        parsed["outcar_size_bytes"] = outcar.stat().st_size
        parsed["nsw_max_ionic_steps_used"] = "reached required accuracy" in text
    return parsed


def _dielectric_summary(
    real: ET.Element | None, imag: ET.Element | None
) -> dict[str, Any] | None:
    """Summarize the isotropic (xx) dielectric spectrum without dumping it."""
    if real is None or imag is None:
        return None
    try:
        real_xx: list[float] = []
        imag_xx: list[float] = []
        for r_row, i_row in zip(
            real.findall("v"), imag.findall("v"), strict=False
        ):
            r_values = [float(value) for value in (r_row.text or "").split()]
            i_values = [float(value) for value in (i_row.text or "").split()]
            if len(r_values) >= 2 and len(i_values) >= 2:
                real_xx.append(r_values[0])
                imag_xx.append(i_values[0])
        if not real_xx:
            return None
        return {
            "points": len(real_xx),
            "real_xx_range": [min(real_xx), max(real_xx)],
            "imag_xx_max": max(imag_xx),
            "imag_xx_max_energy_index": int(imag_xx.index(max(imag_xx))),
            "note": "isotropic (xx) component; full spectrum in artifact",
        }
    except (ValueError, TypeError):
        return None
