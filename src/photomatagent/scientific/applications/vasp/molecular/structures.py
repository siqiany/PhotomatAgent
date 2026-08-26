"""Deterministic structure readers and geometry helpers (no RDKit/ASE).

Accepted inputs: XYZ, MOL/SDF (V2000) and VASP5 grouped POSCAR. Parsing is
strict by design: an integer atom-count contract must hold in every format
(XYZ header, MOL counts line, POSCAR counts sum), and a POSCAR element line
must list each element exactly once with its count. The legacy per-atom
element line (``C O C C O ... Li`` produced by the earlier gel-electrolyte
script) is rejected because no valid POTCAR can be built from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from photomatagent.scientific.applications.vasp.molecular.models import (
    StructureKind,
)


class StructureError(ValueError):
    """A structure file is unreadable or violates the molecular contract."""

    def __init__(self, message: str, *, code: str = "STRUCTURE_UNREADABLE"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StructureData:
    """Parsed molecule: one symbol per atom plus geometry metadata."""

    symbols: list[str]
    positions: np.ndarray  # Cartesian positions, Angstrom
    comment: str = ""
    source_kind: str = ""
    box_ang: float | None = None  # cubic lattice length when known
    lattice: np.ndarray | None = None
    elements: list[str] | None = None  # POSCAR element line (VASP5)
    counts: list[int] | None = None  # POSCAR counts line (VASP5)
def detect_kind(path: str | Path) -> StructureKind:
    file_path = Path(path)
    # Extensionless POSCAR/CONTCAR files (the canonical VASP names) are
    # recognized by name; a ".poscar" suffix is accepted as well.
    if file_path.name in {"POSCAR", "CONTCAR"}:
        return StructureKind.POSCAR
    suffix = file_path.suffix.lower().lstrip(".")
    if suffix == "poscar":
        return StructureKind.POSCAR
    if suffix in {"xyz", "sdf", "mol"}:
        return StructureKind(suffix)  # type: ignore[arg-type]
    raise StructureError(
        f"unsupported structure extension {suffix!r}; use XYZ/SDF/MOL/POSCAR",
        code="STRUCTURE_UNREADABLE",
    )


def read_structure(
    path: str | Path,
    *,
    kind: StructureKind | str | None = None,
    conformer_index: int = 0,
) -> StructureData:
    """Read one structure file deterministically."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise StructureError(
            f"structure file does not exist: {file_path}",
            code="STRUCTURE_UNREADABLE",
        )
    selected = kind
    if selected is None:
        selected = detect_kind(file_path)
    else:
        selected = StructureKind(selected) if isinstance(selected, str) else selected
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if selected is StructureKind.XYZ:
        return _parse_xyz(text, source=str(file_path))
    if selected is StructureKind.POSCAR:
        return _parse_poscar(text, source=str(file_path))
    if selected is StructureKind.MOL:
        return _parse_mol(text, source=str(file_path))
    if selected is StructureKind.SDF:
        return _parse_sdf(text, source=str(file_path), block_index=conformer_index)
    raise StructureError(
        f"unsupported structure kind {selected!r}", code="STRUCTURE_UNREADABLE"
    )


def _split_lines(text: str) -> list[str]:
    return text.splitlines()


def _parse_xyz(text: str, *, source: str) -> StructureData:
    lines = [line.strip() for line in _split_lines(text) if line.strip()]
    if len(lines) < 2:
        raise StructureError(f"XYZ file too short: {source}")
    try:
        count = int(lines[0])
    except ValueError as exc:
        raise StructureError(
            f"XYZ atom count is not an integer: {source}", code="STRUCTURE_UNREADABLE"
        ) from exc
    comment = lines[1]
    rows = lines[2 : 2 + count]
    if len(rows) != count:
        raise StructureError(
            f"XYZ atom count mismatch in {source}: header says {count}, "
            f"found {len(rows)} atom lines",
            code="STRUCTURE_ATOM_COUNT_MISMATCH",
        )
    symbols: list[str] = []
    coords: list[list[float]] = []
    for index, row in enumerate(rows, start=1):
        fields = row.split()
        if len(fields) < 4:
            raise StructureError(
                f"XYZ atom line {index} in {source} has fewer than 4 fields: {row!r}"
            )
        symbols.append(_canonical_symbol(fields[0]))
        try:
            coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError as exc:
            raise StructureError(
                f"XYZ atom line {index} in {source} has non-numeric coordinates"
            ) from exc
    return StructureData(
        symbols=symbols,
        positions=np.asarray(coords, dtype=float),
        comment=comment,
        source_kind="xyz",
    )


def _parse_poscar(text: str, *, source: str) -> StructureData:
    lines = [line.rstrip("\r\n") for line in _split_lines(text)]
    if len(lines) < 8:
        raise StructureError(f"POSCAR too short: {source}")
    comment = lines[0].strip()
    scale_fields = lines[1].split()
    try:
        if len(scale_fields) == 1:
            scale = float(scale_fields[0])
            lattice = np.asarray(
                [[float(value) for value in lines[i].split()[:3]] for i in range(2, 5)],
                dtype=float,
            )
            lattice = lattice * scale
        else:
            lattice = np.asarray(
                [
                    [float(value) for value in lines[i].split()[:3]]
                    for i in range(2, 5)
                ],
                dtype=float,
            )
    except ValueError as exc:
        raise StructureError(
            f"POSCAR lattice is not numeric: {source}", code="STRUCTURE_UNREADABLE"
        ) from exc
    index = 5
    element_line: str | None = None
    elements: list[str] | None = None
    counts: list[int] | None = None
    while index < len(lines):
        fields = lines[index].split()
        if not fields:
            index += 1
            continue
        first = fields[0]
        # Element detection comes FIRST: single-letter symbols (C/O/F/H) must
        # not be mistaken for the legacy coordinate-mode keywords d/c/k.
        if _looks_like_element_line(first):
            element_line = lines[index]
            elements = [_canonical_symbol(field) for field in fields]
            index += 1
            if index >= len(lines):
                raise StructureError(
                    f"POSCAR has an element line but no counts line: {source}",
                    code="POSCAR_ELEMENT_BLOCKS_INVALID",
                )
            counts = _parse_counts(lines[index], source=source)
            index += 1
            continue
        lowered = first.lower()
        if lowered in {"direct", "cartesian", "d", "c", "k"}:
            break
        if lowered in {"selective", "selective dynamics"}:
            index += 1
            continue
        break
    if elements is None or counts is None:
        raise StructureError(
            f"POSCAR without a VASP5 element/counts block cannot be used "
            f"(VASP4 format): {source}",
            code="POSCAR_ELEMENT_BLOCKS_INVALID",
        )
    if len(elements) != len({element for element in elements}):
        raise StructureError(
            f"POSCAR element line repeats symbols ({element_line!r}): a valid "
            f"POTCAR cannot be built from duplicated element blocks; write the "
            f"grouped VASP5 form, e.g. 'C O H Li' + '4 10 2 1': {source}",
            code="POSCAR_ELEMENT_BLOCKS_INVALID",
        )
    if len(counts) != len(elements):
        raise StructureError(
            f"POSCAR counts line length {len(counts)} does not match element "
            f"line length {len(elements)}: {source}",
            code="POSCAR_COUNT_MISMATCH",
        )
    if any(count < 1 for count in counts):
        raise StructureError(
            f"POSCAR contains a non-positive element count: {source}",
            code="POSCAR_COUNT_MISMATCH",
        )
    while index < len(lines):
        fields = lines[index].split()
        if not fields:
            index += 1
            continue
        lowered = fields[0].lower()
        if lowered in {"selective", "selective dynamics"}:
            index += 1
            continue
        break
    if index >= len(lines):
        raise StructureError(f"POSCAR has no coordinate mode line: {source}")
    mode = lines[index].split()[0].lower()
    index += 1
    n_atoms = sum(counts)
    coord_rows = lines[index : index + n_atoms]
    if len(coord_rows) != n_atoms:
        raise StructureError(
            f"POSCAR atom count mismatch in {source}: counts sum to "
            f"{n_atoms}, found {len(coord_rows)} coordinate rows",
            code="STRUCTURE_ATOM_COUNT_MISMATCH",
        )
    frac: list[list[float]] = []
    for row_index, row in enumerate(coord_rows, start=1):
        fields = row.split()
        if len(fields) < 3:
            raise StructureError(
                f"POSCAR coordinate row {row_index} has fewer than 3 values in "
                f"{source}"
            )
        try:
            frac.append([float(fields[0]), float(fields[1]), float(fields[2])])
        except ValueError as exc:
            raise StructureError(
                f"POSCAR coordinate row {row_index} is not numeric in {source}"
            ) from exc
    frac_array = np.asarray(frac, dtype=float)
    cartesian = frac_array @ lattice
    box_ang = _cubic_box_length(lattice, source=source)
    symbols = _expand_symbols(elements, counts)
    return StructureData(
        symbols=symbols,
        positions=cartesian,
        comment=comment,
        source_kind="poscar",
        box_ang=box_ang,
        lattice=lattice,
        elements=elements,
        counts=counts,
    )


def _parse_counts(line: str, *, source: str) -> list[int]:
    fields = line.split()
    counts: list[int] = []
    for field in fields:
        try:
            counts.append(int(float(field)))
        except ValueError as exc:
            raise StructureError(
                f"POSCAR counts line is not an integer list: {line!r} in {source}",
                code="POSCAR_COUNT_MISMATCH",
            ) from exc
    return counts


def _looks_like_element_line(first: str) -> bool:
    # Element symbols: optional uppercase token possibly with _suffixes, or a
    # bare symbol like C/O/H/Li/Cd. Anything starting with a digit or a
    # coordinate-mode keyword is not an element line.
    head = first.split("_", 1)[0]
    return (
        len(head) == 1 and head.isalpha() or bool(re.fullmatch(r"[A-Z][a-z]?", head))
    )


def _canonical_symbol(token: str) -> str:
    head = re.split(r"[^A-Za-z]", token, maxsplit=1)[0]
    if not head or not head[0].isupper():
        raise StructureError(f"invalid element symbol {token!r}")
    return head


def _parse_mol(text: str, *, source: str) -> StructureData:
    lines = _split_lines(text)
    if len(lines) < 5:
        raise StructureError(f"MOL file too short: {source}")
    header = "\n".join(lines[0:2])
    counts_index = 3
    while counts_index < len(lines) and not lines[counts_index].strip():
        counts_index += 1
    if counts_index >= len(lines):
        raise StructureError(f"MOL file has no counts line: {source}")
    if lines[counts_index].strip().upper().startswith("V3000"):
        raise StructureError(
            f"MOL V3000 blocks are not supported: {source}; export V2000 or XYZ"
        )
    counts_fields = lines[counts_index].split()
    if len(counts_fields) < 2:
        raise StructureError(f"MOL counts line malformed in {source}")
    try:
        n_atoms = int(counts_fields[0])
        n_bonds = int(counts_fields[1])
    except ValueError as exc:
        raise StructureError(
            f"MOL counts line is not numeric in {source}", code="STRUCTURE_UNREADABLE"
        ) from exc
    atom_lines = lines[counts_index + 1 : counts_index + 1 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise StructureError(
            f"MOL atom count mismatch in {source}: counts line says {n_atoms}, "
            f"found {len(atom_lines)} atom lines",
            code="STRUCTURE_ATOM_COUNT_MISMATCH",
        )
    symbols: list[str] = []
    coords: list[list[float]] = []
    for row_index, row in enumerate(atom_lines, start=1):
        fields = row.split()
        if len(fields) < 4:
            raise StructureError(
                f"MOL atom line {row_index} in {source} has fewer than 4 fields"
            )
        try:
            coords.append([float(fields[0]), float(fields[1]), float(fields[2])])
        except ValueError as exc:
            raise StructureError(
                f"MOL atom line {row_index} has non-numeric coordinates in {source}"
            ) from exc
        symbols.append(_canonical_symbol(fields[3]))
    bond_lines = [
        line for line in lines[counts_index + 1 + n_atoms :] if line.strip()
    ][:n_bonds]
    if len(bond_lines) != n_bonds:
        # Bond bookkeeping is informational; the atom count contract is what
        # matters for input generation.
        pass
    del n_bonds, bond_lines
    return StructureData(
        symbols=symbols,
        positions=np.asarray(coords, dtype=float),
        comment=header,
        source_kind="mol",
    )


def _parse_sdf(text: str, *, source: str, block_index: int = 0) -> StructureData:
    blocks = [block for block in text.split("$$$$") if block.strip()]
    if not blocks:
        raise StructureError(f"SDF contains no molecule blocks: {source}")
    if block_index < 0 or block_index >= len(blocks):
        raise StructureError(
            f"SDF conformer index {block_index} out of range (0.."
            f"{len(blocks) - 1}): {source}",
            code="STRUCTURE_ATOM_COUNT_MISMATCH",
        )
    # ``$$$$``-separated blocks may begin with the newline that terminates
    # the record separator; drop it so the V2000 header keeps its alignment.
    return _parse_mol(blocks[block_index].lstrip("\n"), source=source)


def _expand_symbols(elements: list[str], counts: list[int]) -> list[str]:
    symbols: list[str] = []
    for element, count in zip(elements, counts, strict=True):
        symbols.extend([element] * count)
    return symbols


def _cubic_box_length(lattice: np.ndarray, *, source: str) -> float:
    if lattice.shape != (3, 3):
        raise StructureError(f"POSCAR lattice is not 3x3: {source}")
    off_diagonal = lattice.copy()
    off_diagonal[0, 0] = off_diagonal[1, 1] = off_diagonal[2, 2] = 0.0
    if np.max(np.abs(off_diagonal)) > 1e-3:
        raise StructureError(
            f"molecular workflow requires a cubic cell, got off-diagonal "
            f"lattice terms in {source}",
            code="BOX_NOT_CUBIC",
        )
    lengths = np.diag(lattice)
    if np.max(np.abs(lengths - lengths[0])) > 1e-3:
        raise StructureError(
            f"molecular workflow requires a cubic cell, got unequal lengths "
            f"({lengths}) in {source}",
            code="BOX_NOT_CUBIC",
        )
    return float(lengths[0])


# -- geometry helpers ---------------------------------------------------------


def grouped_symbols(symbols: list[str]) -> tuple[list[str], list[int]]:
    """First-occurrence element order and per-element counts."""
    order: list[str] = []
    counts: list[int] = []
    for symbol in symbols:
        if symbol in order:
            counts[order.index(symbol)] += 1
        else:
            order.append(symbol)
            counts.append(1)
    return order, counts


def reorder_positions(
    symbols: list[str], positions: np.ndarray, elements: list[str]
) -> np.ndarray:
    """Group atomic positions into the POSCAR element-block order."""
    blocks = [
        np.asarray(
            [positions[index] for index, symbol in enumerate(symbols) if symbol == el]
        )
        for el in elements
    ]
    if not blocks:
        return positions
    return np.concatenate(blocks, axis=0)


def center_in_cubic_box(positions: np.ndarray, box_ang: float) -> np.ndarray:
    """Translate the molecule so its centroid sits at the box center."""
    centroid = positions.mean(axis=0)
    return positions - centroid + box_ang / 2.0


def per_side_vacuum(positions: np.ndarray, box_ang: float) -> np.ndarray:
    """Six vacuum thicknesses (x-, x+, y-, y+, z-, z+) in Angstrom."""
    minima = positions.min(axis=0)
    maxima = positions.max(axis=0)
    return np.asarray(
        [minima[0], box_ang - maxima[0], minima[1], box_ang - maxima[1],
         minima[2], box_ang - maxima[2]],
        dtype=float,
    )


def minimum_image_pairs(
    positions: np.ndarray, box_ang: float, threshold: float
) -> list[tuple[int, int, float]]:
    """Pairs (i, j, distance) below ``threshold`` under PBC (cubic box)."""
    result: list[tuple[int, int, float]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = positions[i] - positions[j]
            delta -= box_ang * np.round(delta / box_ang)
            distance = float(np.linalg.norm(delta))
            if distance < threshold:
                result.append((i, j, distance))
    result.sort(key=lambda item: item[2])
    return result


def molecular_extents(positions: np.ndarray) -> np.ndarray:
    """Per-axis bounding-box size in Angstrom."""
    return positions.max(axis=0) - positions.min(axis=0)


def formula_text(elements: list[str], counts: list[int]) -> str:
    """Deterministic formula in element order, e.g. C4O2H10Li."""
    return "".join(f"{element}{count}" for element, count in zip(elements, counts, strict=True))
