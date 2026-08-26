"""Offline parsing and validation of isolated-molecule VASP results.

Every number returned here is grounded in a file (EIGENVAL, OSZICAR,
vasprun.xml, INCAR, POSCAR/CONTCAR, LOCPOT) and every interpretation is
carried by an explicit ``method``/``limitations`` note. Slurm COMPLETED never
enters this module: scheduling and scientific validation are separate, and
scientific evidence is only produced when ``MolecularResults.validated`` is
true.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from photomatagent.scientific.applications.vasp.molecular.render import (
    parse_incar,
)
from photomatagent.scientific.applications.vasp.molecular.structures import (
    StructureError,
    grouped_symbols,
    minimum_image_pairs,
    read_structure,
)

# heuristic covalent radii (A); used ONLY for the "abnormal bond" advisory
COVALENT_RADII_ANG: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Li": 1.28, "Na": 1.66,
    "K": 2.03, "Br": 1.20, "I": 1.39, "B": 0.84, "Si": 1.11,
}

_ANOMALOUS_BOND_FACTOR = 0.72
_DISSOCIATION_NEAREST_ANG = 4.0
_CELL_EDGE_MARGIN_ANG = 2.0


# --------------------------------------------------------------------------
# low-level parsers
# --------------------------------------------------------------------------


@dataclass
class EigenvalData:
    nelect: int
    nkpts: int
    nbnds: int
    ispin: int
    kpoints: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""


def parse_eigenval(path: str | Path) -> EigenvalData:
    """Parse a VASP EIGENVAL (Gamma-only molecular runs)."""
    source = str(path)
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 12:
        raise ValueError(f"EIGENVAL too short: {source}")
    header = lines[5].split()
    if len(header) < 3:
        raise ValueError(f"EIGENVAL header line 6 unreadable: {lines[5]!r}")
    if not all(token.isdigit() for token in header[:3]):
        # Fallback: search the first 12 lines for a "NELECT NKPTS NBANDS" row.
        for line in lines[:12]:
            tokens = line.split()
            if len(tokens) == 3 and all(token.isdigit() for token in tokens):
                header = tokens
                break
        else:
            raise ValueError(f"EIGENVAL NELECT/NKPTS/NBANDS row not found: {source}")
    nelect = int(header[0])
    nkpts = int(header[1])
    nbnds = int(header[2])
    ispin = 1
    for line in lines[:6]:
        tokens = line.split()
        if len(tokens) >= 3 and tokens[0].isdigit() and len(tokens) == 4:
            ispin = int(tokens[2])
            break
    data = EigenvalData(
        nelect=nelect, nkpts=nkpts, nbnds=nbnds, ispin=ispin, source=source
    )
    # Spin-major block order: spin 1's k-points, then spin 2's k-points.
    blocks = nkpts * ispin
    block_index = -1
    band_count = 0
    energies: list[float] = []
    occupations: list[float] = []
    k_tokens: list[str] = []
    def flush_block() -> None:
        data.kpoints.append(
            {
                "spin": block_index // nkpts if nkpts else 0,
                "coords": [float(token) for token in k_tokens[:3]],
                "weight": float(k_tokens[3]) if len(k_tokens) > 3 else 0.0,
                "energies": list(energies),
                "occupations": list(occupations),
            }
        )
    for line in lines[6:]:
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) == 3:
            # band row: index energy occupation
            try:
                energies.append(float(tokens[1]))
                occupations.append(float(tokens[2]))
                band_count += 1
            except ValueError:
                continue
            if band_count == nbnds:
                block_index += 1
                flush_block()
                band_count = 0
                energies, occupations = [], []
                if block_index + 1 >= blocks:
                    break
        else:
            try:
                _ = [float(token) for token in tokens]
                k_tokens = tokens  # k-point header (3 coords + weight)
            except ValueError:
                continue
    if block_index + 1 < blocks:
        raise ValueError(
            f"EIGENVAL has {block_index + 1}/{blocks} k-point blocks: {source}"
        )
    return data


@dataclass
class OsziData:
    scf_steps: int
    ionic_steps: list[dict[str, float]]
    last_scf_de_ev: float | None
    final_f_ev: float | None
    final_e0_ev: float | None
    source: str = ""


def parse_oszicar(path: str | Path) -> OsziData:
    """Parse OSZICAR: SCF (DAV) rows, ionic F/E0 rows and convergence."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    source = str(path)
    scf_de: list[float] = []
    ionic: list[dict[str, float]] = []
    for line in text.splitlines():
        if line.startswith("DAV"):
            tokens = line.split()
            # DAV: n E dE d_eps ncg rms rms(c)
            if len(tokens) >= 4:
                try:
                    scf_de.append(abs(float(tokens[3])))
                except ValueError:
                    pass
            continue
        match = re.match(
            r"^\s*(\d+)\s+F=\s*([-+0-9.Ee]+)\s+E0=\s*([-+0-9.Ee]+)",
            line,
        )
        if match:
            ionic.append(
                {
                    "step": int(match.group(1)),
                    "F": float(match.group(2)),
                    "E0": float(match.group(3)),
                }
            )
    return OsziData(
        scf_steps=len(scf_de),
        ionic_steps=ionic,
        last_scf_de_ev=scf_de[-1] if scf_de else None,
        final_f_ev=ionic[-1]["F"] if ionic else None,
        final_e0_ev=ionic[-1]["E0"] if ionic else None,
        source=source,
    )


@dataclass
class VasprunData:
    final_f_ev: float | None
    final_e0_ev: float | None
    entropy_ts_ev: float | None
    ionic_steps: int
    scf_steps: int
    eigenvalues: EigenvalData | None = None
    n_atoms: int | None = None
    source: str = ""


def parse_vasprun(path: str | Path, *, max_steps_read: int = 200) -> VasprunData:
    """Parse energies/scf/ionic structure of vasprun.xml (small molecule runs)."""
    source = str(path)
    tree = ET.parse(path)
    root = tree.getroot()
    calc = root.find("calculation")
    if calc is None:
        raise ValueError(f"vasprun.xml has no <calculation>: {source}")
    final_energy = calc.find("energy")
    final_f: float | None = None
    final_e0: float | None = None
    entropy: float | None = None
    if final_energy is not None:
        names = {v.get("name"): v.text for v in final_energy}
        final_f = _safe_float(names.get("e_fr_energy"))
        final_e0 = _safe_float(names.get("e_0_energy"))
        entropy = _safe_float(names.get("eentropy"))
    steps = calc.findall("scstep")
    ionic_steps = len(calc.findall("structure"))
    n_atoms: int | None = None
    for structure in root.iter("structure"):
        if structure.get("name") in {None, "finalpos", "initialpos"}:
            positions = structure.find('varray[@name="positions"]')
            if positions is not None:
                n_atoms = len(positions.findall("v"))
                break
    eigenvalues: EigenvalData | None = None
    eig_node = calc.find("eigenvalues")
    if eig_node is not None:
        try:
            eigenvalues = _parse_vasprun_eigenvalues(eig_node)
        except Exception:
            eigenvalues = None
    return VasprunData(
        final_f_ev=final_f,
        final_e0_ev=final_e0,
        entropy_ts_ev=entropy,
        ionic_steps=ionic_steps,
        scf_steps=len(steps),
        eigenvalues=eigenvalues,
        n_atoms=n_atoms,
        source=source,
    )


def _parse_vasprun_eigenvalues(eig_node: ET.Element) -> EigenvalData:
    """Extract eigenvalues/occupations from <eigenvalues><array><set>..."""
    array = eig_node.find("array")
    if array is None:
        raise ValueError("eigenvalues array missing")
    outer = array.find("set")
    if outer is None:
        raise ValueError("eigenvalues set missing")
    spins = list(outer)
    ispin = max(1, len(spins))
    data = EigenvalData(nelect=0, nkpts=0, nbnds=0, ispin=ispin, source="vasprun.xml")
    for spin_index, spin_node in enumerate(spins):
        for kpt in list(spin_node):
            energies: list[float] = []
            occupations: list[float] = []
            for row in kpt:
                tokens = (row.text or "").split()
                if tokens:
                    energies.append(float(tokens[0]))
                    occupations.append(float(tokens[1]) if len(tokens) > 1 else 0.0)
            data.kpoints.append(
                {
                    "spin": spin_index,
                    "coords": [0.0, 0.0, 0.0],
                    "weight": 1.0,
                    "energies": energies,
                    "occupations": occupations,
                }
            )
    data.nkpts = len(data.kpoints) // ispin
    data.nbnds = len(data.kpoints[0]["energies"]) if data.kpoints else 0
    total = sum(
        sum(occ for occ in point["occupations"])
        for point in data.kpoints
    )
    data.nelect = int(round(total))
    return data


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# orbital and vacuum analysis
# --------------------------------------------------------------------------


@dataclass
class OrbitalBands:
    homo_band: int | None
    lumo_band: int | None
    homo_occupation: float | None
    lumo_occupation: float | None
    homo_raw_ev: float | None
    lumo_raw_ev: float | None
    ks_gap_ev: float | None
    occupied_floor: float = 0.5


def determine_orbital_bands(data: EigenvalData) -> OrbitalBands:
    """HOMO/LUMO from occupation numbers (never from the Fermi energy).

    Convention: an occupied-level threshold of 0.5 separates HOMO from LUMO
    at Gamma for ISMEAR=0 runs; for spin-polarized runs the highest occupied /
    lowest unoccupied band is taken over both spin channels.
    """
    threshold = 0.5
    homo_band: int | None = None
    lumo_band: int | None = None
    homo_occ: float | None = None
    lumo_occ: float | None = None
    homo_e: float | None = None
    lumo_e: float | None = None
    for point in data.kpoints:
        for band_index, (energy, occupation) in enumerate(
            zip(point["energies"], point["occupations"], strict=True)
        ):
            band = band_index + 1
            if occupation >= threshold:
                if homo_band is None or energy > homo_e:  # type: ignore[operator]
                    homo_band, homo_occ, homo_e = band, occupation, energy
            else:
                if lumo_band is None or energy < lumo_e:  # type: ignore[operator]
                    lumo_band, lumo_occ, lumo_e = band, occupation, energy
    gap = None
    if homo_e is not None and lumo_e is not None:
        gap = lumo_e - homo_e
    return OrbitalBands(
        homo_band=homo_band,
        lumo_band=lumo_band,
        homo_occupation=homo_occ,
        lumo_occupation=lumo_occ,
        homo_raw_ev=homo_e,
        lumo_raw_ev=lumo_e,
        ks_gap_ev=gap,
    )


@dataclass
class LocpotGrid:
    box_ang: float
    grid: tuple[int, int, int]
    data: np.ndarray  # shape (nx, ny, nz)
    source: str

    @property
    def spacing_ang(self) -> tuple[float, float, float]:
        return (
            self.box_ang / self.grid[0],
            self.box_ang / self.grid[1],
            self.box_ang / self.grid[2],
        )


def read_locpot(path: str | Path, box_ang: float | None = None) -> LocpotGrid:
    """Read a LOCPOT 3D potential grid (VASP5 header; x-fastest ordering).

    The file is read as numpy data only after the header/coordinate block;
    no grid content is ever logged or returned to the model.
    """
    source = str(path)
    text_head = Path(path).read_text(encoding="utf-8", errors="replace")[:20000]
    lines = text_head.splitlines()
    if len(lines) < 9:
        raise ValueError(f"LOCPOT header too short: {source}")
    if box_ang is None:
        try:
            box_ang = float(lines[1].split()[0])
        except (IndexError, ValueError):
            raise ValueError(f"LOCPOT scale line unreadable: {source}")
    box_ang = abs(box_ang)
    counts_line = lines[6].split()
    try:
        n_atoms = sum(int(token) for token in counts_line)
    except ValueError:
        # VASP4-style header: single count on line 7
        counts_line = lines[7].split()
        n_atoms = sum(int(token) for token in counts_line)
    grid_line_index = 8 + n_atoms
    grid_tokens = lines[grid_line_index].split()
    if len(grid_tokens) < 3:
        raise ValueError(f"LOCPOT grid line unreadable at line {grid_line_index + 1}")
    grid = (int(grid_tokens[0]), int(grid_tokens[1]), int(grid_tokens[2]))
    expected = int(np.prod(grid))
    if expected > 40_000_000:
        raise ValueError(f"LOCPOT grid too large for offline analysis: {grid}")
    body = Path(path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()[grid_line_index + 1:]
    tokens = " ".join(body).split()
    if len(tokens) < expected:
        raise ValueError(f"LOCPOT grid data too short: {len(tokens)} < {expected}")
    flat = np.asarray(tokens[:expected], dtype=np.float64)
    # VASP grid data is x-fastest; a C-order reshape of (nx, ny, nz) maps
    # flat[ix + nx*(iy + ny*iz)] -> data[ix, iy, iz].
    data = flat.reshape(grid)
    return LocpotGrid(box_ang=box_ang, grid=grid, data=data, source=source)


def vacuum_level(locpot: LocpotGrid, *, outer_band_fraction: float = 0.12) -> tuple[float, float, int]:
    """Vacuum level from the outer planar-average bands of a cubic cell.

    Returns (level_ev, std_ev, n_samples). For a molecule centred in a fixed
    box the outermost ``outer_band_fraction`` of planes along each axis are
    (close to) the vacuum plateau of a LOCPOT written with LVHAR=.TRUE.
    """
    values: list[float] = []
    for axis in range(3):
        other_axes = [a for a in range(3) if a != axis]
        planar = locpot.data.mean(axis=tuple(other_axes))
        band = max(1, int(planar.size * outer_band_fraction))
        values.extend(planar[:band].tolist())
        values.extend(planar[-band:].tolist())
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std()), int(array.size)


def vacuum_aligned(
    bands: OrbitalBands, vacuum_ev: float
) -> dict[str, float | None]:
    return {
        "vacuum_level_ev": vacuum_ev,
        "aligned_homo_ev": (
            bands.homo_raw_ev - vacuum_ev if bands.homo_raw_ev is not None else None
        ),
        "aligned_lumo_ev": (
            bands.lumo_raw_ev - vacuum_ev if bands.lumo_raw_ev is not None else None
        ),
    }


def read_outcar_corrections(path: str | Path) -> dict[str, Any]:
    """Best-effort monopole/dipole+quadrupole corrections from OUTCAR."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    results: dict[str, Any] = {"monopole_ev": None, "dipole_quadrupole_ev": None}
    for pattern, key in (
        (r"[Dd]ipol\+quadrupol moment[^\n]*?(-?\d+\.\d+[Ee][+-]?\d+)", "dipole_quadrupole_ev"),
        (r"[Mm]onopole[^\n]*?(-?\d+\.\d+[Ee][+-]?\d+)", "monopole_ev"),
    ):
        match = re.search(pattern, text)
        if match:
            results[key] = float(match.group(1))
    results["source"] = "OUTCAR best-effort parse; verify against VASP output"
    return results


def magnetization_from_outcar(path: str | Path) -> float | None:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"number of electron\s+([-\d.]+)\s+magnetization\s+([-\d.]+)",
        text,
    )
    if match:
        return float(match.group(2))
    if "ISPIN" in text and not re.search(r"ISPIN\s*=\s*2", text):
        return 0.0
    return None


def geometry_summary(poscar: str | Path, box_ang: float | None = None) -> dict[str, Any]:
    """Geometry sanity flags + close-pair table for the result structure."""
    try:
        structure = read_structure(poscar, kind="poscar")
    except (StructureError, OSError) as exc:
        return {
            "error": f"result geometry unreadable: {exc}",
            "n_atoms": None,
            "box_ang": box_ang,
            "min_distance_ang": None,
            "min_pair": None,
            "cross_periodic_boundary": None,
            "near_cell_edge": None,
            "possible_dissociation": None,
            "anomalous_bonds": None,
            "coordination_distances": [],
        }
    elements, _ = grouped_symbols(structure.symbols)
    actual_box = structure.box_ang if structure.box_ang is not None else box_ang
    if actual_box is None:
        raise ValueError("geometry box_ang is required")
    pairs = minimum_image_pairs(
        structure.positions, actual_box, 0.0
    )  # all pairs, sorted
    if pairs:
        min_distance, min_pair = pairs[0][2], (pairs[0][0], pairs[0][1])
    else:
        min_distance, min_pair = None, None
    anomalous: list[str] = []
    for i, j, distance in pairs[:30]:
        r1 = COVALENT_RADII_ANG.get(elements[i])
        r2 = COVALENT_RADII_ANG.get(elements[j])
        if r1 is not None and r2 is not None:
            if distance < _ANOMALOUS_BOND_FACTOR * (r1 + r2) - 1e-9:
                anomalous.append(f"{i}-{j}:{distance:.2f}A")
    frac = np.asarray(structure.positions, dtype=float) / actual_box
    margin_frac = _CELL_EDGE_MARGIN_ANG / actual_box
    near_edge = bool(
        np.any(frac < margin_frac) or np.any(frac > 1.0 - margin_frac)
    )
    cross_pbc = bool(np.any(frac <= 0.0) or np.any(frac >= 1.0))
    if len(structure.symbols) > 1 and pairs:
        _, _, nearest = pairs[0]
        dissociated = nearest > _DISSOCIATION_NEAREST_ANG
    else:
        dissociated = False
    return {
        "error": None,
        "n_atoms": len(structure.symbols),
        "box_ang": actual_box,
        "min_distance_ang": min_distance,
        "min_pair": min_pair,
        "cross_periodic_boundary": cross_pbc,
        "near_cell_edge": near_edge,
        "possible_dissociation": dissociated,
        "anomalous_bonds": bool(anomalous),
        "anomalous_pairs": anomalous[:8],
        "coordination_distances": [
            {"atoms": [pairs[k][0], pairs[k][1]], "distance_ang": round(pairs[k][2], 3)}
            for k in range(min(8, len(pairs)))
        ],
    }


def esp_metadata(result_dir: str | Path) -> dict[str, Any]:
    """LOCPOT presence + grid metadata (never the potential itself)."""
    locpot = Path(result_dir) / "LOCPOT"
    if not locpot.is_file():
        return {"has_locpot": False, "grid": None, "spacing_ang": None, "size_bytes": None}
    try:
        grid = read_locpot(locpot)
        return {
            "has_locpot": True,
            "grid": list(grid.grid),
            "spacing_ang": [round(v, 4) for v in grid.spacing_ang],
            "size_bytes": locpot.stat().st_size,
        }
    except Exception as exc:
        return {
            "has_locpot": False,
            "grid": None,
            "spacing_ang": None,
            "size_bytes": locpot.stat().st_size,
            "parse_error": str(exc),
        }


# --------------------------------------------------------------------------
# result assembly + evidence gating
# --------------------------------------------------------------------------


def analyze_result_dir(
    result_dir: str | Path,
    *,
    charge: int,
    spin_multiplicity: int,
    box_ang: float | None = None,
    vacancy_align: bool = True,
    corrections_ev: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Analyze one collected stage directory into a structured result dict.

    ``validated`` is true only when every blocking error is absent; evidence
    is never generated from an unvalidated result (Slurm COMPLETED alone is
    never sufficient).
    """
    directory = Path(result_dir)
    errors: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = [
        "Gamma-only single k-point (isolated molecule in a fixed vacuum box)",
        "no vibrational or thermal corrections; energies are electronic only",
        "raw HOMO/LUMO carry the cell's potential reference; only LOCPOT-"
        "vacuum-aligned values may be compared across molecules",
        "the Fermi level is never used as HOMO; occupations define band edges",
    ]

    files = {path.name: path for path in directory.iterdir() if path.is_file()}

    incar = files.get("INCAR")
    incar_dict: dict[str, Any] = (
        parse_incar(incar.read_text(encoding="utf-8", errors="replace"))
        if incar is not None
        else {}
    )
    encut = _plain(incar_dict.get("ENCUT", ""))
    ediff = _plain(incar_dict.get("EDIFF", "1E-6")) or 1e-6
    ispin_value = _plain(incar_dict.get("ISPIN", ""))
    ispin = int(ispin_value) if ispin_value is not None else 1
    functional = str(incar_dict.get("GGA", "PE"))
    surface = incar_dict.get("SYSTEM", "")

    # -- electron bookkeeping -------------------------------------------------
    declared_nelect: float | None = None
    meta = files.get("POTCAR.meta")
    if meta is not None:
        try:
            import json

            declared_nelect = json.loads(
                meta.read_text(encoding="utf-8")
            ).get("nelect")
        except Exception:
            pass
    if declared_nelect is None:
        raw_nelect = incar_dict.get("NELECT")
        declared_nelect = _plain(raw_nelect) if raw_nelect is not None else None
    if declared_nelect is None:
        errors.append("ELECTRON_COUNT_UNDECLARED: NELECT not found in "
                      "POTCAR.meta or INCAR")

    # -- energies / convergence ----------------------------------------------
    energy: dict[str, Any] = {
        "e_fr_ev": None,
        "e_0_ev": None,
        "source": None,
        "entropy_ts_ev": None,
        "note": "",
    }
    oszi = files.get("OSZICAR")
    oszi_data = parse_oszicar(oszi) if oszi is not None else None
    vasprun = files.get("vasprun.xml")
    vasprun_data = None
    if vasprun is not None:
        try:
            vasprun_data = parse_vasprun(vasprun)
        except Exception as exc:
            warnings.append(f"vasprun.xml parse failed: {exc}")
    if vasprun_data is not None and vasprun_data.final_f_ev is not None:
        energy["e_fr_ev"] = vasprun_data.final_f_ev
        energy["source"] = "vasprun.xml <energy> final ionic step"
        if vasprun_data.final_e0_ev not in (None, 0.0):
            energy["e_0_ev"] = vasprun_data.final_e0_ev
        elif (
            vasprun_data.entropy_ts_ev is not None
            and abs(vasprun_data.entropy_ts_ev) < 1e-8
        ):
            energy["e_0_ev"] = vasprun_data.final_f_ev
            energy["note"] = (
                "vasprun.xml e_0_energy missing/zero; used e_fr_energy "
                "because T*S ~ 0"
            )
        else:
            energy["note"] = "e_0_energy missing; only free energy available"
    elif oszi_data is not None and oszi_data.final_e0_ev is not None:
        energy["e_fr_ev"] = oszi_data.final_f_ev
        energy["e_0_ev"] = oszi_data.final_e0_ev
        energy["source"] = "OSZICAR (last ionic F=/E0= row)"
    else:
        errors.append("ENERGY_MISSING: no OSZICAR/vasprun.xml energy found")
    if (
        energy["e_0_ev"] is None
        and energy["e_fr_ev"] is not None
        and oszi_data is not None
        and oszi_data.final_e0_ev is not None
    ):
        energy["e_0_ev"] = oszi_data.final_e0_ev
        energy["note"] = (
            "vasprun.xml e_0_energy missing/zero; E0 taken from the OSZICAR "
            "F=/E0= row"
        )

    scf: dict[str, Any] = {"converged": False, "steps": 0, "last_dE_ev": None, "ediff_ev": ediff}
    if oszi_data is not None:
        scf["steps"] = oszi_data.scf_steps
        scf["last_dE_ev"] = oszi_data.last_scf_de_ev
        scf["converged"] = (
            oszi_data.last_scf_de_ev is not None
            and oszi_data.last_scf_de_ev <= ediff
        )
    if not scf["converged"]:
        errors.append(
            "SCF_NOT_CONVERGED: last SCF dE "
            f"{scf['last_dE_ev'] if scf['last_dE_ev'] is not None else 'n/a'} "
            f"> EDIFF {ediff:g}"
        )

    nsw_value = _plain(incar_dict.get("NSW", ""))
    nsw = int(nsw_value) if nsw_value is not None else 0
    ionic = {
        "steps": vasprun_data.ionic_steps if vasprun_data is not None else None,
        "static_single_point": nsw == 0,
        "converged": nsw == 0,
        "note": "NSW=0 static single point; ionic convergence not applicable" if nsw == 0 else "",
    }
    if nsw != 0 and oszi_data is not None and len(oszi_data.ionic_steps) >= 2:
        last = oszi_data.ionic_steps[-1]
        prev = oszi_data.ionic_steps[-2]
        ionic["converged"] = abs(last["F"] - prev["F"]) < 1e-5
        ionic["note"] = f"dE between ionic steps = {last['F'] - prev['F']:.3e} eV"

    # -- orbitals -------------------------------------------------------------
    orbitals: dict[str, Any] = {
        "homo_band": None, "lumo_band": None,
        "homo_occupation": None, "lumo_occupation": None,
        "homo_raw_ev": None, "lumo_raw_ev": None, "ks_gap_ev": None,
        "nelect_eigenval": None, "nbands": None, "ispin": None,
        "source": None,
    }
    eigen = files.get("EIGENVAL")
    eigen_data = None
    if eigen is not None:
        try:
            eigen_data = parse_eigenval(eigen)
        except Exception as exc:
            errors.append(f"EIGENVAL_PARSE_FAILED: {exc}")
    if eigen_data is not None:
        bands = determine_orbital_bands(eigen_data)
        orbitals.update(
            {
                "homo_band": bands.homo_band,
                "lumo_band": bands.lumo_band,
                "homo_occupation": bands.homo_occupation,
                "lumo_occupation": bands.lumo_occupation,
                "homo_raw_ev": bands.homo_raw_ev,
                "lumo_raw_ev": bands.lumo_raw_ev,
                "ks_gap_ev": bands.ks_gap_ev,
                "nelect_eigenval": eigen_data.nelect,
                "nbands": eigen_data.nbnds,
                "ispin": eigen_data.ispin,
                "source": eigen_data.source,
            }
        )
        if (
            declared_nelect is not None
            and abs(eigen_data.nelect - declared_nelect) > 1e-6
        ):
            errors.append(
                "ELECTRON_COUNT_MISMATCH: EIGENVAL shows "
                f"{eigen_data.nelect} electrons but declared NELECT = "
                f"{declared_nelect:g}"
            )
    else:
        warnings.append("EIGENVAL missing; orbital analysis skipped")

    # -- vacuum alignment -------------------------------------------------------
    vacuum: dict[str, Any] = {
        "level_ev": None, "std_ev": None, "samples": None,
        "method": None, "aligned_homo_ev": None, "aligned_lumo_ev": None,
    }
    locpot_path = directory / "LOCPOT"
    if locpot_path.is_file():
        try:
            locpot = read_locpot(locpot_path, box_ang=box_ang)
            level, std, count = vacuum_level(locpot)
            vacuum.update(
                {
                    "level_ev": level,
                    "std_ev": std,
                    "samples": count,
                    "method": (
                        "LOCPOT planar average; mean of the outermost "
                        "planes along all three axes of the fixed box"
                    ),
                }
            )
            ref_bands = (
                determine_orbital_bands(eigen_data)
                if eigen_data is not None
                else OrbitalBands(None, None, None, None, None, None, None)
            )
            aligned = vacuum_aligned(ref_bands, level)
            vacuum["aligned_homo_ev"] = aligned["aligned_homo_ev"]
            vacuum["aligned_lumo_ev"] = aligned["aligned_lumo_ev"]
            if std > 0.1:
                warnings.append(
                    f"VACUUM_PLATEAU_UNSTABLE: std {std:.3f} eV in the "
                    "outer planes; aligned energies are unreliable"
                )
        except Exception as exc:
            warnings.append(f"LOCPOT parse failed: {exc}")
    else:
        warnings.append(
            "LOCPOT missing; HOMO/LUMO are NOT vacuum-aligned and must "
            "not be compared across molecules"
        )

    # -- corrections -------------------------------------------------------------
    corrections: dict[str, Any] = {
        "monopole_ev": None,
        "dipole_quadrupole_ev": None,
        "source": None,
    }
    outcar = files.get("OUTCAR")
    if outcar is not None:
        try:
            parsed_corrections = read_outcar_corrections(outcar)
            corrections.update(parsed_corrections)
        except Exception:
            pass
    if corrections_ev:
        corrections.update(corrections_ev)
        corrections["source"] = "explicit metadata (e.g. run record)"

    # -- geometry / ESP ----------------------------------------------------------
    structure_file = files.get("CONTCAR") or files.get("POSCAR")
    geometry: dict[str, Any] = geometry_summary(
        structure_file, box_ang=box_ang
    ) if structure_file is not None else {
        "error": "no structure file (CONTCAR/POSCAR) in result directory"
    }
    if geometry.get("error"):
        errors.append(f"GEOMETRY_UNREADABLE: {geometry['error']}")
    if geometry.get("cross_periodic_boundary"):
        errors.append(
            "GEOMETRY_CROSS_PBC: an atom touches or crosses the periodic "
            "cell boundary"
        )
    if geometry.get("anomalous_bonds"):
        errors.append(
            "GEOMETRY_ANOMALOUS_BONDS: abnormally short interatomic "
            f"distances: {', '.join(geometry.get('anomalous_pairs', [])[:5])}"
        )
    if geometry.get("possible_dissociation"):
        warnings.append(
            "POSSIBLE_DISSOCIATION: nearest-neighbour distance exceeds "
            f"{_DISSOCIATION_NEAREST_ANG} A"
        )
    if geometry.get("near_cell_edge"):
        warnings.append(
            "ATOMS_NEAR_CELL_EDGE: molecular image may interact across the "
            "periodic boundary"
        )
    esp: dict[str, Any] = esp_metadata(directory)
    if incar_dict.get("LVHAR") is True:
        esp["lvhar_declared"] = True
    else:
        esp["lvhar_declared"] = bool(incar_dict.get("LVHAR"))

    magnetization: float | None = None
    if outcar is not None:
        try:
            magnetization = magnetization_from_outcar(outcar)
        except Exception:
            pass

    formula = "unknown"
    elements: list[str] = []
    if geometry.get("n_atoms") and structure_file is not None:
        try:
            structure = read_structure(
                structure_file, kind="poscar"
            )
            elements, counts = grouped_symbols(structure.symbols)
            from photomatagent.scientific.applications.vasp.molecular.structures import (
                formula_text,
            )

            formula = formula_text(elements, counts)
        except Exception:
            pass

    validated = not errors
    return {
        "validated": validated,
        "errors": errors,
        "warnings": warnings,
        "limitations": limitations,
        "identity": {
            "formula": formula,
            "elements": elements,
            "charge": charge,
            "spin_multiplicity": spin_multiplicity,
            "nelect_declared": declared_nelect,
            "magnetization_ub": magnetization,
            "system": str(surface),
        },
        "method": {
            "functional": functional,
            "encut_ev": encut,
            "box_ang": box_ang,
            "gamma_only": True,
            "ispin": ispin if ispin else None,
            "kpoints": "Gamma 1x1x1",
        },
        "energy": energy,
        "scf": scf,
        "ionic": ionic,
        "orbitals": orbitals,
        "vacuum": vacuum,
        "corrections": corrections,
        "geometry": geometry,
        "esp": esp,
    }


def scientific_evidence(
    results: dict[str, Any], *, tool: str = "vasp_molecule"
) -> list[Any]:
    """Evidence is ONLY produced for validated results."""
    if not results.get("validated"):
        return []
    from photomatagent.scientific.capabilities.contracts import (
        ScientificEvidence,
    )

    identity = results["identity"]
    evidence: list[ScientificEvidence] = []
    energy = results.get("energy") or {}
    if energy.get("e_0_ev") is not None:
        evidence.append(
            ScientificEvidence(
                subject=identity["formula"],
                property="total_energy_E0",
                value=energy["e_0_ev"],
                unit="eV",
                source=energy.get("source") or "unknown",
                source_type="dft_calculation",
                method="VASP Gamma-only PBE-D3(BJ) fixed-box static",
                fidelity="dft",
                summary=f"{identity['formula']} E0 = {energy['e_0_ev']:.6f} eV",
                limitations="electronic-only; no zero-point or thermal terms",
            )
        )
    orbitals = results.get("orbitals") or {}
    vacuum = results.get("vacuum") or {}
    if orbitals.get("homo_raw_ev") is not None:
        property_name = "HOMO_energy_vacuum_aligned" if vacuum.get("aligned_homo_ev") is not None else "HOMO_energy_raw"
        evidence.append(
            ScientificEvidence(
                subject=identity["formula"],
                property=property_name,
                value=(
                    vacuum["aligned_homo_ev"]
                    if vacuum.get("aligned_homo_ev") is not None
                    else orbitals["homo_raw_ev"]
                ),
                unit="eV",
                source="EIGENVAL + LOCPOT" if vacuum.get("aligned_homo_ev") is not None else "EIGENVAL",
                source_type="dft_calculation",
                method=(
                    f"occupation-defined band {orbitals['homo_band']}"
                    + ("; vacuum-aligned" if vacuum.get("aligned_homo_ev") is not None else "; NOT vacuum-aligned")
                ),
                fidelity="dft",
                summary=f"HOMO (band {orbitals['homo_band']}) = "
                f"{orbitals['homo_raw_ev']:.4f} eV raw, "
                f"{vacuum.get('aligned_homo_ev') if vacuum.get('aligned_homo_ev') is not None else 'n/a'} eV aligned",
            )
        )
    if orbitals.get("lumo_raw_ev") is not None:
        evidence.append(
            ScientificEvidence(
                subject=identity["formula"],
                property="LUMO_energy_vacuum_aligned"
                if vacuum.get("aligned_lumo_ev") is not None
                else "LUMO_energy_raw",
                value=(
                    vacuum["aligned_lumo_ev"]
                    if vacuum.get("aligned_lumo_ev") is not None
                    else orbitals["lumo_raw_ev"]
                ),
                unit="eV",
                source="EIGENVAL + LOCPOT" if vacuum.get("aligned_lumo_ev") is not None else "EIGENVAL",
                source_type="dft_calculation",
                method=f"occupation-defined band {orbitals['lumo_band']}",
                fidelity="dft",
                summary=f"LUMO (band {orbitals['lumo_band']}) = "
                f"{orbitals['lumo_raw_ev']:.4f} eV raw",
            )
        )
    if orbitals.get("ks_gap_ev") is not None:
        evidence.append(
            ScientificEvidence(
                subject=identity["formula"],
                property="Kohn_Sham_gap",
                value=orbitals["ks_gap_ev"],
                unit="eV",
                source="EIGENVAL occupations",
                source_type="dft_calculation",
                method="LUMO(raw) - HOMO(raw) at Gamma",
                fidelity="dft",
                summary=f"KS gap = {orbitals['ks_gap_ev']:.4f} eV",
            )
        )
    corrections = results.get("corrections") or {}
    if corrections.get("dipole_quadrupole_ev") is not None:
        evidence.append(
            ScientificEvidence(
                subject=identity["formula"],
                property="electrostatic_correction_dipole_quadrupole",
                value=corrections["dipole_quadrupole_ev"],
                unit="eV",
                source=corrections.get("source") or "OUTCAR",
                source_type="dft_calculation",
                method="dipole+quadrupole correction (LDIPOL/IDIPOL=4)",
                fidelity="dft",
                summary=(
                    f"dipole+quadrupole correction = "
                    f"{corrections['dipole_quadrupole_ev']:.6f} eV"
                ),
            )
        )
    return evidence


def _plain(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
