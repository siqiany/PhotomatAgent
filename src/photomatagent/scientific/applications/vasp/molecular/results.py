"""Offline parsing and validation of isolated-molecule VASP results.

Every number returned here is grounded in a file (EIGENVAL, OSZICAR,
vasprun.xml, INCAR, POSCAR/CONTCAR, LOCPOT) and every interpretation is
carried by an explicit ``method``/``limitations`` note. Slurm COMPLETED never
enters this module: scheduling and scientific validation are separate, and
scientific evidence is only produced when ``MolecularResults.validated`` is
true.
"""

from __future__ import annotations

import dataclasses
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
    """Parse energies/scf/ionic structure of vasprun.xml.

    * relax runs: the LAST ``<calculation>`` provides the final energy and
      structure (intermediate steps are statistical noise);
    * ALL ionic steps across the file are counted
      (``n_calculations - 1``, one structure per step);
    * ``iterparse`` with subtree clearing keeps memory bounded for large
      multi-step XML files (no whole-tree ElementTree model).
    """
    source = str(path)
    calc_count = 0
    final_energy_elem: ET.Element | None = None
    final_scsteps = 0
    final_eigenvalues: ET.Element | None = None
    n_atoms: int | None = None
    for event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "structure":
            if n_atoms is None:
                positions = elem.find('varray[@name="positions"]')
                if positions is not None:
                    n_atoms = len(positions.findall("v"))
            elem.clear()
            continue
        if elem.tag != "calculation":
            continue
        calc_count += 1
        final_energy_elem = elem.find("energy")
        final_scsteps = len(elem.findall("scstep"))
        final_eigenvalues = elem.find("eigenvalues")
        # Keep only the LAST calculation's data; release the rest.
        elem.clear()
    if calc_count == 0:
        raise ValueError(f"vasprun.xml has no <calculation>: {source}")
    final_f: float | None = None
    final_e0: float | None = None
    entropy: float | None = None
    if final_energy_elem is not None:
        names = {v.get("name"): v.text for v in final_energy_elem}
        final_f = _safe_float(names.get("e_fr_energy"))
        final_e0 = _safe_float(names.get("e_0_energy"))
        entropy = _safe_float(names.get("eentropy"))
    ionic_steps = max(0, calc_count - 1)
    eigenvalues: EigenvalData | None = None
    if final_eigenvalues is not None:
        try:
            eigenvalues = _parse_vasprun_eigenvalues(final_eigenvalues)
        except Exception:
            eigenvalues = None
    return VasprunData(
        final_f_ev=final_f,
        final_e0_ev=final_e0,
        entropy_ts_ev=entropy,
        ionic_steps=ionic_steps,
        scf_steps=final_scsteps,
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


# In-memory (legacy) LOCPOT read is capped: a real 448^3 ~1.6 GB LOCPOT must
# go through the streaming API below instead of materializing the full grid.
# Raising this number would NOT fix large files; it would only move the OOM.
MAX_IN_MEMORY_GRID_POINTS = 40_000_000

SUPPORTED_VACUUM_THICKNESSES_ANG = (0.5, 1.0, 1.5, 2.0)
VACUUM_STABILITY_THRESHOLD_EV = 0.1


@dataclass
class LocpotHeader:
    """LOCPOT header only: lattice, grid, atom count and data byte offset.

    Parsed by streaming the first ``8 + n_atoms + 1`` lines; the potential
    grid itself is never loaded, read or logged by header parsing (so
    ``esp_metadata`` never touches the 1.6 GB body).
    """

    lattice_lengths_ang: list[float]
    lattice: list[list[float]]  # 3x3 Cartesian rows (already scaled)
    grid: tuple[int, int, int]
    n_atoms: int
    data_offset_bytes: int
    source: str

    @property
    def spacing_ang(self) -> tuple[float, float, float]:
        return tuple(
            self.lattice_lengths_ang[axis] / self.grid[axis]
            for axis in range(3)
        )  # type: ignore[return-value]

    @property
    def n_points(self) -> int:
        return int(np.prod(self.grid))


def _decode_line(line: bytes) -> str:
    return line.decode("ascii", errors="replace")


def read_locpot_header(path: str | Path) -> LocpotHeader:
    """Stream the LOCPOT header region only (memory bounded by n_atoms)."""
    source = str(path)
    lattice_lines: list[bytes] = []
    offset = 0
    with open(path, "rb") as handle:
        for _ in range(8):
            line = handle.readline()
            if not line:
                raise ValueError(f"LOCPOT header too short: {source}")
            lattice_lines.append(line)
            offset += len(line)
        try:
            counts_line = lattice_lines[6]
            n_atoms = sum(int(token) for token in _decode_line(counts_line).split())
        except ValueError:
            # VASP4-style header: a single total atom count on line 7.
            counts_line = lattice_lines[7]
            n_atoms = sum(int(token) for token in _decode_line(counts_line).split())
        for _ in range(n_atoms):
            line = handle.readline()
            if not line:
                raise ValueError(f"LOCPOT header too short: {source}")
            lattice_lines.append(line)
            offset += len(line)
        grid_line = handle.readline()
        if not grid_line:
            raise ValueError(f"LOCPOT grid line missing: {source}")
        offset += len(grid_line)
    grid_tokens = _decode_line(grid_line).split()
    if len(grid_tokens) < 3:
        raise ValueError(
            f"LOCPOT grid line unreadable: {_decode_line(grid_line)!r}"
        )
    grid = (int(grid_tokens[0]), int(grid_tokens[1]), int(grid_tokens[2]))
    scale = float(_decode_line(lattice_lines[1]).split()[0])
    raw_rows: list[list[float]] = []
    lengths: list[float] = []
    for raw in lattice_lines[2:5]:
        tokens = [float(token) for token in _decode_line(raw).split()]
        if len(tokens) != 3:
            raise ValueError(f"LOCPOT lattice row unreadable: {_decode_line(raw)!r}")
        raw_rows.append(tokens)
        lengths.append(float(np.linalg.norm(tokens)) * abs(scale))
    return LocpotHeader(
        lattice_lengths_ang=lengths,
        lattice=[[value * scale for value in row] for row in raw_rows],
        grid=grid,
        n_atoms=n_atoms,
        data_offset_bytes=offset,
        source=source,
    )


def read_locpot(path: str | Path, box_ang: float | None = None) -> LocpotGrid:
    """Legacy in-memory LOCPOT read for SMALL test/analysis files.

    Real production grids (448^3 ~1.6 GB) MUST use :func:`read_locpot_header`
    + :func:`stream_locpot_planar` / :func:`vacuum_summary_all_thicknesses`
    instead: this path keeps its hard in-memory cap at
    ``MAX_IN_MEMORY_GRID_POINTS`` (40,000,000 points) and refuses to raise it.
    """
    source = str(path)
    header = read_locpot_header(path)
    if box_ang is None:
        box_ang = header.lattice_lengths_ang[0]
    box_ang = abs(box_ang)
    grid = header.grid
    expected = int(np.prod(grid))
    if expected > MAX_IN_MEMORY_GRID_POINTS:
        raise ValueError(
            "LOCPOT grid too large for the in-memory reader: "
            f"{grid} ({expected:,} points). Use read_locpot_header + "
            "stream_locpot_planar / vacuum_summary_all_thicknesses instead; "
            "the 40,000,000-point cap is intentional and will not be raised."
        )
    with open(path, "rb") as handle:
        handle.seek(header.data_offset_bytes)
        body = handle.read().decode("ascii", errors="replace")
    tokens = body.split()
    if len(tokens) < expected:
        raise ValueError(f"LOCPOT grid data too short: {len(tokens)} < {expected}")
    flat = np.asarray(tokens[:expected], dtype=np.float64)
    # VASP grid data is x-fastest: token k carries the potential at
    # (ix = k % nx, iy = (k // nx) % ny, iz = k // (nx*ny)), i.e.
    # data[ix, iy, iz] = flat[ix + nx*iy + nx*ny*iz]. That is exactly a
    # FORTRAN-order reshape of the (nx, ny, nz) dimensions; a C-order
    # reshape would interpret the file as z-fastest and scramble planes.
    data = flat.reshape(grid, order="F")
    return LocpotGrid(box_ang=box_ang, grid=grid, data=data, source=source)


@dataclass
class LocpotPlanarStats:
    """Per-plane sums for one axis, accumulated WITHOUT the full 3D grid."""

    axis: int  # 0=x, 1=y, 2=z
    plane_means: np.ndarray  # length grid[axis]
    n_points: int


def stream_locpot_planar(
    path: str | Path,
    header: LocpotHeader | None = None,
    *,
    chunk_bytes: int = 1 << 20,
) -> list[LocpotPlanarStats]:
    """Stream a LOCPOT body in fixed-size chunks, accumulating planar sums.

    The full 3D grid never materializes: only three ~n_axis float arrays are
    kept. The flat index mapping honours VASP's x-fastest ordering
    (``k = ix + nx*(iy + ny*iz)``) exactly, so the planar averages are
    identical to an in-memory C-order reshape.
    """
    if header is None:
        header = read_locpot_header(path)
    n_points = header.n_points
    nx, ny, nz = header.grid
    sums_x = np.zeros(nx, dtype=np.float64)
    sums_y = np.zeros(ny, dtype=np.float64)
    sums_z = np.zeros(nz, dtype=np.float64)
    counts_x = np.zeros(nx, dtype=np.int64)
    counts_y = np.zeros(ny, dtype=np.int64)
    counts_z = np.zeros(nz, dtype=np.int64)
    count = 0
    carry = ""
    with open(path, "rb") as handle:
        handle.seek(header.data_offset_bytes)
        while count < n_points:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            text = carry + chunk.decode("ascii", errors="replace")
            if chunk[-1:] and not chunk.endswith((b" ", b"\n", b"\r", b"\t")):
                parts = text.split()
                if parts:
                    carry = parts[-1]
                    parts = parts[:-1]
                else:
                    carry = ""
            else:
                parts = text.split()
                carry = ""
            for token in parts:
                value = float(token)
                ix = count % nx
                iy = (count // nx) % ny
                iz = count // (nx * ny)
                sums_x[ix] += value
                sums_y[iy] += value
                sums_z[iz] += value
                counts_x[ix] += 1
                counts_y[iy] += 1
                counts_z[iz] += 1
                count += 1
                if count >= n_points:
                    break
    if count < n_points:
        raise ValueError(
            f"LOCPOT grid data too short: {count} < {n_points} points"
        )
    return [
        LocpotPlanarStats(
            axis=axis,
            plane_means=means / counts.astype(np.float64),
            n_points=count,
        )
        for axis, (means, counts) in (
            (0, (sums_x, counts_x)),
            (1, (sums_y, counts_y)),
            (2, (sums_z, counts_z)),
        )
    ]


@dataclass
class VacuumFace:
    """One boundary-layer face of the fixed vacuum box."""

    axis: str  # "x" | "y" | "z"
    side: str  # "low" | "high"
    thickness_ang: float
    mean_ev: float
    std_ev: float
    n_planes: int


@dataclass
class VacuumSummary:
    """Six-face vacuum characterization for one boundary thickness."""

    thickness_ang: float
    lattice_lengths_ang: list[float]
    grid: tuple[int, int, int]
    faces: list[VacuumFace]
    mean_ev: float  # mean of the six face means
    std_ev: float
    range_ev: float
    stability: str  # "stable" | "unstable"


def vacuum_summary_from_planar(
    planar: list[LocpotPlanarStats],
    *,
    thickness_ang: float,
    lattice_lengths_ang: list[float],
    grid: tuple[int, int, int],
) -> VacuumSummary:
    """Six boundary-layer face means at ``thickness_ang`` from planar stats."""
    if thickness_ang <= 0:
        raise ValueError(f"vacuum boundary thickness must be positive: {thickness_ang}")
    faces: list[VacuumFace] = []
    axis_names = ("x", "y", "z")
    for stats in planar:
        axis = stats.axis
        spacing = lattice_lengths_ang[axis] / grid[axis]
        n_planes = max(1, int(round(thickness_ang / spacing)))
        values = stats.plane_means
        for side, selection in (
            ("low", values[:n_planes]),
            ("high", values[-n_planes:] if n_planes else values[-1:]),
        ):
            faces.append(
                VacuumFace(
                    axis=axis_names[axis],
                    side=side,
                    thickness_ang=thickness_ang,
                    mean_ev=float(selection.mean()),
                    std_ev=float(selection.std()) if selection.size > 1 else 0.0,
                    n_planes=int(selection.size),
                )
            )
    face_means = np.asarray([face.mean_ev for face in faces], dtype=float)
    stability = (
        "stable"
        if float(face_means.std()) <= VACUUM_STABILITY_THRESHOLD_EV
        else "unstable"
    )
    return VacuumSummary(
        thickness_ang=thickness_ang,
        lattice_lengths_ang=list(lattice_lengths_ang),
        grid=grid,
        faces=faces,
        mean_ev=float(face_means.mean()),
        std_ev=float(face_means.std()),
        range_ev=float(face_means.max() - face_means.min()),
        stability=stability,
    )


def stream_vacuum_summary(
    path: str | Path,
    *,
    thickness_ang: float = 1.0,
    chunk_bytes: int = 1 << 20,
) -> VacuumSummary:
    """Streaming vacuum summary for ONE boundary thickness (any grid size)."""
    header = read_locpot_header(path)
    planar = stream_locpot_planar(path, header, chunk_bytes=chunk_bytes)
    return vacuum_summary_from_planar(
        planar,
        thickness_ang=thickness_ang,
        lattice_lengths_ang=header.lattice_lengths_ang,
        grid=header.grid,
    )


def vacuum_summary_all_thicknesses(
    path: str | Path,
    *,
    thicknesses: tuple[float, ...] = SUPPORTED_VACUUM_THICKNESSES_ANG,
    chunk_bytes: int = 1 << 20,
) -> dict[float, VacuumSummary]:
    """Six-face summaries for every supported thickness with ONE grid pass."""
    header = read_locpot_header(path)
    planar = stream_locpot_planar(path, header, chunk_bytes=chunk_bytes)
    return {
        thickness: vacuum_summary_from_planar(
            planar,
            thickness_ang=thickness,
            lattice_lengths_ang=header.lattice_lengths_ang,
            grid=header.grid,
        )
        for thickness in thicknesses
    }


def vacuum_summary_dict(summary: VacuumSummary) -> dict[str, Any]:
    """Serializable form of a VacuumSummary (faces included)."""
    return {
        **dataclasses.asdict(summary),
        "faces": [dataclasses.asdict(face) for face in summary.faces],
    }


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
    """Best-effort monopole/dipole+quadrupole corrections from OUTCAR.

    Accepts real VASP 5.4/6.x wording, plain decimals AND scientific
    notation, e.g. ``dipol+quadrupol energy correction           -0.000154 eV``
    (6.x) and ``dipol+quadrupol moment           -0.000161 eAng`` (5.4-style).
    Only energy lines are reported in eV; moment lines (eAng) are reported
    separately and never mixed into the eV fields.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    results: dict[str, Any] = {
        "monopole_ev": None,
        "dipole_quadrupole_ev": None,
        "dipole_quadrupole_moment_eang": None,
        "monopole_moment_eang": None,
    }
    number = r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?"
    quadrupole = r"(?:dipol\+quadrupol|dipole\+quadrupol|dipol\+quadrupole|dipole\+quadrupole)"
    _apply_correction_pattern(
        results,
        "dipole_quadrupole_ev",
        rf"{quadrupole}[^\r\n]*?energy correction\s+({number})\s*e[Vv]\b",
        text,
    )
    _apply_correction_pattern(
        results,
        "dipole_quadrupole_moment_eang",
        rf"{quadrupole}[^\r\n]*?moment\s+({number})\s*e?[Aa][Nn][Gg]\b",
        text,
    )
    _apply_correction_pattern(
        results,
        "monopole_ev",
        rf"monopole[^\r\n]*?energy correction\s+({number})\s*e[Vv]\b",
        text,
    )
    _apply_correction_pattern(
        results,
        "monopole_moment_eang",
        rf"monopole[^\r\n]*?moment\s+({number})\s*e?[Aa][Nn][Gg]\b",
        text,
    )
    results["source"] = "OUTCAR best-effort parse; verify against VASP output"
    return results


def _apply_correction_pattern(
    results: dict[str, Any], key: str, pattern: str, text: str
) -> None:
    match = re.search(pattern, text)
    if match:
        results[key] = float(match.group(1))


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


# --------------------------------------------------------------------------
# relax convergence (OUTCAR-grounded; never Slurm state)
# --------------------------------------------------------------------------

RELAX_ACCURACY_MARKER = (
    "reached required accuracy - stopping structural energy minimisation"
)

# Deterministic VASP failure tokens (lowercase). Detection of any of these
# makes a COLLECTED job NOT converged and forbids blind resubmission.
VASP_FAILURE_TOKENS: tuple[str, ...] = (
    "out of memory",
    "cannot allocate",
    "allocation would exceed",
    "brmix: serious error",
    "zhegv",
    "segmentation",
    "mpi_abort",
    "forrtl: severe",
    "fatal error",
)


@dataclass
class ConvergenceReport:
    """Deterministic ionic/electronic convergence verdict for one stage.

    ``ionic_converged`` is TRUE only when VASP's own formal marker appears
    AND the last TOTAL-FORCE block's maximum atomic force satisfies
    ``max|F| <= |EDIFFG|``. Adjacent ionic-step total-energy differences are
    deliberately NOT a convergence criterion (dE can vanish while forces stay
    large). Slurm COMPLETED never enters this report.
    """

    applicable: bool  # False for NSW=0 static single points
    electronic_converged: bool
    ionic_converged: bool
    reached_required_accuracy: bool
    ionic_steps: int | None
    nsw_limit: int | None
    exhausted_nsw: bool
    last_force_ev_ang: float | None
    max_force_ev_ang: float | None
    ediff_ev: float | None
    ediffg_ev_ang: float | None
    detected_errors: list[str]
    recommended_recovery_reason: str
    practical_convergence: bool = False
    practical_convergence_note: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _parse_total_force_blocks(text: str) -> tuple[list[list[float]], int]:
    """Last TOTAL-FORCE block rows (max force per row) + block count."""
    blocks: list[list[float]] = []
    current: list[float] = []
    in_block = False
    block_count = 0
    for line in text.splitlines():
        # Real VASP OUTCAR: the header line reads
        # "POSITION ... TOTAL-FORCE (eV/Angst)" followed by a dashes row.
        if "TOTAL-FORCE" in line:
            in_block = True
            current = []
            block_count += 1
            continue
        if in_block:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("-"):
                if current:
                    # Trailing separator: the block is complete.
                    blocks.append(current)
                    in_block = False
                # Leading separator (before the first row) is ignored.
                continue
            tokens = stripped.split()
            if len(tokens) >= 6:
                try:
                    fx, fy, fz = (float(tokens[3]), float(tokens[4]), float(tokens[5]))
                    current.append(float(np.linalg.norm([fx, fy, fz])))
                except ValueError:
                    continue
    if in_block and current:
        blocks.append(current)
    return blocks, block_count


def analyze_outcar_convergence(
    path: str | Path,
    *,
    incar_dict: dict[str, Any] | None = None,
    ediff_ev: float = 1e-6,
    nsw: int = 0,
    oszi: Any = None,
) -> ConvergenceReport:
    """Parse one OUTCAR into a ConvergenceReport (streaming-friendly read).

    ``incar_dict`` carries EDIFF/EDIFFG/NSW as submitted (mirrored by the
    collector into the results directory). ``oszi`` is an optional OsziData
    used for the SCF verdict and ionic-step count.
    """
    source = str(path)
    incar = dict(incar_dict or {})
    ediff_value = _plain(incar.get("EDIFF")) if incar.get("EDIFF") is not None else ediff_ev
    ediffg_raw = incar.get("EDIFFG")
    ediffg = abs(_plain(ediffg_raw)) if ediffg_raw is not None else None
    nsw_limit = nsw
    if nsw_limit == 0:
        raw_nsw = _plain(incar.get("NSW")) if incar.get("NSW") is not None else None
        if raw_nsw is not None:
            nsw_limit = int(raw_nsw)

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    reached = RELAX_ACCURACY_MARKER in text
    force_blocks, block_count = _parse_total_force_blocks(text)
    last_block = force_blocks[-1] if force_blocks else []
    max_force = float(max(last_block)) if last_block else None
    last_force = float(last_block[-1]) if last_block else None
    lower = text.lower()
    detected_errors = [
        token for token in VASP_FAILURE_TOKENS if token in lower
    ]
    ionic_steps: int | None = None
    if oszi is not None and getattr(oszi, "ionic_steps", None):
        ionic_steps = len(oszi.ionic_steps)
    elif block_count:
        ionic_steps = max(0, block_count - 1)
    electronic_converged = bool(
        oszi is not None
        and getattr(oszi, "last_scf_de_ev", None) is not None
        and oszi.last_scf_de_ev <= (ediff_value or 1e-6)
    )
    if ediffg is None:
        # EDIFFG missing is itself a validation problem: relax acceptance
        # cannot be proven without the force threshold that was submitted.
        detected_errors.append("EDIFFG missing from INCAR; force criterion unknown")
    force_ok = (
        max_force is not None
        and ediffg is not None
        and max_force <= ediffg + 1e-12
    )
    ionic_converged = bool(reached and force_ok)
    exhausted = bool(
        nsw_limit is not None
        and nsw_limit > 0
        and ionic_steps is not None
        and ionic_steps >= nsw_limit
        and not ionic_converged
    )
    recommendation = "not applicable (static single point)"
    if detected_errors:
        if any(token in detected_errors for token in ("out of memory", "cannot allocate", "allocation would exceed")):
            recommendation = (
                "OOM: do NOT repeat identical resources; increase tasks/memory "
                "or switch LREAL/.FALSE. before any retry, then restart from CONTCAR"
            )
        elif any(
            token in detected_errors
            for token in ("mpi_abort", "forrtl: severe", "segmentation", "fatal error", "brmix: serious error", "zhegv")
        ):
            recommendation = (
                "VASP runtime error detected in OUTCAR; inspect stdout/stderr; do not "
                "blindly resubmit identical inputs"
            )
        elif "EDIFFG missing" in detected_errors:
            recommendation = "EDIFFG missing from INCAR; declare EDIFFG before resubmitting"
    elif exhausted:
        recommendation = (
            "NSW_EXHAUSTED: NSW steps were consumed without reaching |EDIFFG|; "
            "restart from CONTCAR with a NEW attempt (never from the initial POSCAR)"
        )
    elif reached and not force_ok:
        recommendation = (
            "CONVERGENCE_FLAG_MISMATCH: VASP reported 'reached required accuracy' but "
            f"max force {max_force:.6f} eV/A exceeds |EDIFFG| {ediffg:.6f}; re-verify "
            "EDIFFG and restart from CONTCAR"
        )
    elif not ionic_converged and ionic_steps is not None and nsw_limit:
        if ionic_steps < nsw_limit:
            recommendation = (
                f"relax stopped early ({ionic_steps} < NSW {nsw_limit}) without the "
                "formal convergence marker; check walltime/OUTCAR tail and continue "
                "from CONTCAR"
            )
        else:
            recommendation = "relax did not converge; restart from CONTCAR with a new attempt"
    return ConvergenceReport(
        applicable=True,
        electronic_converged=electronic_converged,
        ionic_converged=ionic_converged,
        reached_required_accuracy=reached,
        ionic_steps=ionic_steps,
        nsw_limit=nsw_limit,
        exhausted_nsw=exhausted,
        last_force_ev_ang=last_force,
        max_force_ev_ang=max_force,
        ediff_ev=ediff_value,
        ediffg_ev_ang=ediffg,
        detected_errors=detected_errors,
        recommended_recovery_reason=recommendation,
        source=source,
    )


def esp_metadata(result_dir: str | Path) -> dict[str, Any]:
    """LOCPOT presence + header metadata (never the potential itself).

    Only the header region is parsed: the grid body (potentially ~1.6 GB for
    a 448^3 LOCPOT) is never read, loaded or hashed here.
    """
    locpot = Path(result_dir) / "LOCPOT"
    if not locpot.is_file():
        return {
            "has_locpot": False,
            "grid": None,
            "spacing_ang": None,
            "size_bytes": None,
            "lattice_lengths_ang": None,
            "data_offset_bytes": None,
        }
    try:
        header = read_locpot_header(locpot)
        return {
            "has_locpot": True,
            "grid": list(header.grid),
            "spacing_ang": [round(v, 4) for v in header.spacing_ang],
            "size_bytes": locpot.stat().st_size,
            "lattice_lengths_ang": [round(v, 6) for v in header.lattice_lengths_ang],
            "data_offset_bytes": header.data_offset_bytes,
        }
    except Exception as exc:
        return {
            "has_locpot": False,
            "grid": None,
            "spacing_ang": None,
            "size_bytes": locpot.stat().st_size,
            "lattice_lengths_ang": None,
            "data_offset_bytes": None,
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
    convergence: dict[str, Any] | None = None
    ionic = {
        "steps": vasprun_data.ionic_steps if vasprun_data is not None else None,
        "static_single_point": nsw == 0,
        "converged": nsw == 0,
        "note": "NSW=0 static single point; ionic convergence not applicable" if nsw == 0 else "",
    }
    if nsw != 0:
        # Ionic convergence comes ONLY from the OUTCAR force criterion
        # (max|F| <= |EDIFFG| + formal marker). Adjacent-step total-energy
        # differences are never a substitute for the force criterion.
        outcar = files.get("OUTCAR")
        if outcar is None:
            errors.append(
                "RELAX_OUTCAR_MISSING: OUTCAR is required to verify ionic "
                "convergence (max force vs EDIFFG) for a relax stage"
            )
            convergence = {
                "applicable": True,
                "error": "OUTCAR missing; ionic convergence unverifiable",
                "ionic_converged": False,
            }
        else:
            try:
                report = analyze_outcar_convergence(
                    outcar,
                    incar_dict=incar_dict,
                    ediff_ev=ediff,
                    nsw=nsw,
                    oszi=oszi_data,
                )
                convergence = report.to_dict()
                ionic["steps"] = report.ionic_steps if report.ionic_steps is not None else ionic["steps"]
                ionic["converged"] = report.ionic_converged
                ionic["note"] = (
                    f"max force {report.max_force_ev_ang:.6f} eV/A vs |EDIFFG| "
                    f"{report.ediffg_ev_ang:.6f}; formal marker "
                    f"{'present' if report.reached_required_accuracy else 'absent'}"
                )
                if report.detected_errors:
                    errors.append(
                        "DETECTED_VASP_ERROR: "
                        + "; ".join(report.detected_errors[:5])
                    )
                if report.practical_convergence:
                    warnings.append(
                        "PRACTICAL_CONVERGENCE: " + report.practical_convergence_note
                    )
                if not report.ionic_converged:
                    errors.append(
                        "IONIC_NOT_CONVERGED: max force "
                        f"{report.max_force_ev_ang if report.max_force_ev_ang is not None else 'n/a'} "
                        "eV/A vs |EDIFFG| "
                        f"{report.ediffg_ev_ang if report.ediffg_ev_ang is not None else 'n/a'} eV/A; "
                        f"steps={report.ionic_steps} NSW={report.nsw_limit} "
                        f"exhausted={report.exhausted_nsw}"
                    )
            except Exception as exc:
                errors.append(f"RELAX_CONVERGENCE_UNREADABLE: {exc}")
                convergence = {
                    "applicable": True,
                    "error": str(exc),
                    "ionic_converged": False,
                }
    else:
        convergence = {
            "applicable": False,
            "ionic_converged": True,
            "note": "NSW=0 static single point; ionic convergence not applicable",
        }
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
        "grid": None, "lattice_lengths_ang": None,
        "faces": None, "thicknesses": None, "stability": None,
    }
    locpot_path = directory / "LOCPOT"
    if locpot_path.is_file():
        try:
            # Streaming LOCPOT processing: never materializes the 3D grid,
            # so a real 448^3 (~1.6 GB) LVHAR LOCPOT is analyzed fine.
            header = read_locpot_header(locpot_path)
            by_thickness = vacuum_summary_all_thicknesses(locpot_path)
            summary = by_thickness[1.0]
            vacuum.update(
                {
                    "level_ev": summary.mean_ev,
                    "std_ev": summary.std_ev,
                    "samples": sum(face.n_planes for face in summary.faces),
                    "grid": list(header.grid),
                    "lattice_lengths_ang": [
                        round(v, 6) for v in header.lattice_lengths_ang
                    ],
                    "faces": [
                        dataclasses.asdict(face) for face in summary.faces
                    ],
                    "thicknesses": {
                        str(thickness): vacuum_summary_dict(item)
                        for thickness, item in by_thickness.items()
                    },
                    "stability": summary.stability,
                    "method": (
                        "LOCPOT streaming planar average; mean of six "
                        "vacuum-face boundary layers (0.5/1.0/1.5/2.0 A)"
                    ),
                }
            )
            ref_bands = (
                determine_orbital_bands(eigen_data)
                if eigen_data is not None
                else OrbitalBands(None, None, None, None, None, None, None)
            )
            aligned = vacuum_aligned(ref_bands, summary.mean_ev)
            vacuum["aligned_homo_ev"] = aligned["aligned_homo_ev"]
            vacuum["aligned_lumo_ev"] = aligned["aligned_lumo_ev"]
            if summary.std_ev > VACUUM_STABILITY_THRESHOLD_EV:
                warnings.append(
                    f"VACUUM_PLATEAU_UNSTABLE: six-face std "
                    f"{summary.std_ev:.3f} eV; aligned energies are unreliable"
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
        "convergence": convergence,
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
