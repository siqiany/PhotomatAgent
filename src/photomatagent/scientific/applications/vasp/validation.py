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
* relax stages additionally require ionic convergence (early exit below
  NSW, or the OUTCAR ``reached required accuracy`` marker)

``parse_result`` extracts bounded values (final energy, dielectric
spectrum summary) that enter ScientificEvidence.
"""

from __future__ import annotations

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
_IONIC_STEP = re.compile(r"<i\s+name=\"ionic step\"[^>]*>")
_NSW = re.compile(r"<i\s+name=\"NSW\"[^>]*>\s*(\d+)")


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
    if has_c and has_v:
        check.electronic_converged = True
    else:
        check.electronic_converged = None
        check.reasons.append(
            "electronic convergence marker (<c>/<v> in final <scstep>) not "
            "found; SCF convergence cannot be confirmed"
        )
    # Ionic convergence: relaxation ends before NSW, or OUTCAR marker.
    outcar = path.parent / "OUTCAR"
    final_text = ET.tostring(final, encoding="unicode")
    nsw_match = _NSW.search(ET.tostring(root, encoding="unicode")[:50000])
    ionic_steps = len(_IONIC_STEP.findall(final_text))
    nsw: int | None = int(nsw_match.group(1)) if nsw_match else None
    outcar_marker = False
    if outcar.is_file():
        tail = outcar.read_text(encoding="utf-8", errors="replace")[-200000:]
        outcar_marker = (
            "reached required accuracy" in tail or "EDIFF is reached" in tail
        )
    if outcar_marker:
        check.ionic_converged = True
    elif nsw is not None and nsw > 0 and ionic_steps < nsw:
        check.ionic_converged = True
    elif nsw is not None and nsw == 0:
        check.ionic_converged = True  # static run: no ionic relaxation needed
    else:
        check.ionic_converged = None
    return check


def validate_output(result_dir: str | Path, *, profile_name: str) -> list[str]:
    """Return a list of validation problems (empty means acceptable)."""
    directory = Path(result_dir)
    problems: list[str] = []
    check = check_vasprun(directory / "vasprun.xml")
    problems.extend(check.reasons)
    if check.electronic_converged is False:
        problems.append("electronic SCF did not converge")
    if profile_name == "relax" and check.ionic_converged is not True:
        problems.append("relax stage: ionic convergence cannot be confirmed")
    if profile_name in {"band", "dos", "optics"} and not (directory / "CHGCAR").is_file():
        problems.append(f"{profile_name} stage requires CHGCAR from the static run")
    return problems


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
