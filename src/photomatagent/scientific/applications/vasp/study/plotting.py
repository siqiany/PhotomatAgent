"""Study figures: orbital levels, PARCHG isosurfaces, ESP maps, bindings.

All figures are persisted PNGs with explicit units and method notes. The
orbital isosurface is derived from a PARCHG grid (VASP 5.4.4 text format)
overlaid on the molecular skeleton; the ESP map samples LOCPOT on a
molecular-surface proxy (vdW + 1.4 A probe) with a colorbar. When a grid
cannot be parsed, a typed placeholder note is persisted instead of a fake
figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from photomatagent.scientific.capabilities.chemistry.storage import read_xyz


VASP_VDW = {
    "C": 1.70, "H": 1.20, "O": 1.52, "N": 1.55, "F": 1.47,
    "S": 1.80, "Li": 1.82, "P": 1.80,
}


def _grid_parser(lines: list[str]) -> dict[str, Any] | None:
    """Parse a VASP text grid (CHGCAR/PARCHG-style) header + values."""
    try:
        scale = float(lines[1].split()[0])
        lattice = np.asarray(
            [list(map(float, line.split())) for line in lines[2:5]]
        )
        index = 5
        while index < len(lines) and not lines[index].strip():
            index += 1
        element_line = lines[index].split()
        counts = [int(value) for value in lines[index + 1].split()]
        natoms = sum(counts)
        index += 2
        while index < len(lines) and "Direct" not in lines[index]:
            index += 1
        index += 1  # "Direct"
        coordinates: list[list[float]] = []
        while index < len(lines) and len(coordinates) < natoms:
            row = lines[index].split()
            if len(row) >= 3:
                coordinates.append([float(row[0]), float(row[1]), float(row[2])])
            index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        dims = [int(value) for value in lines[index].split()]
        if len(dims) != 3:
            return None
        values: list[float] = []
        index += 1
        while index < len(lines) and len(values) < dims[0] * dims[1] * dims[2]:
            values.extend(
                float(value)
                for value in lines[index].split()
                if value.replace(".", "", 1).replace("-", "", 1).replace("E", "")
                .replace("+", "").replace("e", "").isdigit()
            )
            index += 1
        if len(values) != dims[0] * dims[1] * dims[2]:
            return None
        return {
            "scale": scale,
            "lattice": lattice * scale,
            "elements": element_line,
            "counts": counts,
            "coordinates": np.asarray(coordinates, dtype=float),
            "dims": dims,
            "values": np.asarray(values, dtype=float).reshape(dims, order="F"),
        }
    except (ValueError, IndexError):
        return None


def read_grid(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return None
    return _grid_parser(lines)


def plot_orbital_levels(
    rows: list[dict[str, Any]], out_path: Path
) -> Path:
    """Aligned HOMO/LUMO level chart (raw values are shown, never compared)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [
        row for row in rows
        if row.get("homo_aligned_ev") is not None
        and row.get("lumo_aligned_ev") is not None
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(5.0, 0.9 * len(systems) + 1), 4.6))
    for index, row in enumerate(systems):
        x = index
        axis.plot(
            [x, x],
            [row["homo_aligned_ev"], row["lumo_aligned_ev"]],
            color="#1f77b4",
            linewidth=2.0,
        )
        axis.scatter([x], [row["homo_aligned_ev"]], color="#d62728", s=46, zorder=3)
        axis.scatter([x], [row["lumo_aligned_ev"]], color="#2ca02c", s=46, zorder=3)
        axis.annotate(
            f"{row['ks_gap_ev']:.2f}" if row.get("ks_gap_ev") is not None else "?",
            (x, (row["homo_aligned_ev"] + row["lumo_aligned_ev"]) / 2),
            fontsize=7,
            ha="center",
        )
    axis.set_xticks(range(len(systems)))
    axis.set_xticklabels(
        [f"{row['system']}\n({row['reliability']})" for row in systems],
        fontsize=7,
    )
    axis.set_ylabel("energy vs vacuum (eV)")
    axis.set_title(
        "Vacuum-aligned HOMO/LUMO (Gamma-only PBE; raw eigenvalues are "
        "never compared across molecules)",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    import matplotlib.pyplot as _plt

    _plt.close(figure)
    return out_path


def _skeleton_lines(symbols: list[str], coords: Any, cutoff: float = 1.7) -> list[Any]:
    bonds: list[tuple[int, int]] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if np.linalg.norm(coords[i] - coords[j]) < cutoff:
                bonds.append((i, j))
    return bonds


def plot_orbital_isosurface(
    parchg_path: Path,
    structure_path: Path,
    out_path: Path,
    *,
    isovalue: float | None = None,
) -> Path:
    """PARCHG isosurface over the molecular skeleton (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from skimage import measure

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = read_grid(parchg_path)
    if grid is None:
        out_path.with_suffix(".txt").write_text(
            "PARCHG grid unreadable; isosurface not rendered (see "
            "study report for the placeholder reason)",
            encoding="utf-8",
        )
        return out_path
    values = grid["values"]
    threshold = (
        isovalue
        if isovalue is not None
        else float(values.mean() + 0.4 * values.std())
    )
    try:
        vertices, faces, _normals, _values = measure.marching_cubes(
            values, level=threshold, spacing=(
                grid["lattice"][0][0] / grid["dims"][0],
                grid["lattice"][1][1] / grid["dims"][1],
                grid["lattice"][2][2] / grid["dims"][2],
            )
        )
    except Exception:
        out_path.with_suffix(".txt").write_text(
            "no isosurface at the chosen level (or marching cubes failed)",
            encoding="utf-8",
        )
        return out_path
    figure = plt.figure(figsize=(5.2, 5.0))
    axis = figure.add_subplot(111, projection="3d")
    try:
        axis.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=faces, alpha=0.55, color="#1f77b4",
            antialiased=True,
        )
    except Exception:
        axis.scatter(
            vertices[:: max(len(vertices) // 400, 1), 0],
            vertices[:: max(len(vertices) // 400, 1), 1],
            vertices[:: max(len(vertices) // 400, 1), 2],
            s=1.0, color="#1f77b4", alpha=0.7,
        )
    try:
        symbols, coords, _ = read_xyz(structure_path)
    except ValueError:
        symbols, coords = [], np.zeros((0, 3))
    if len(coords):
        axis.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            s=16, color="#d62728", depthshade=True,
        )
        for left, right in _skeleton_lines(symbols, coords):
            axis.plot(
                [coords[left][0], coords[right][0]],
                [coords[left][1], coords[right][1]],
                [coords[left][2], coords[right][2]],
                color="#555555", linewidth=0.8,
            )
    axis.set_title(
        f"{parchg_path.name} isosurface @ {threshold:.4g} "
        "(PARCHG grid, skeleton overlay)",
        fontsize=8,
    )
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return out_path


def plot_esp_surface(
    locpot_path: Path,
    structure_path: Path,
    out_path: Path,
) -> Path:
    """LOCPOT sampled on a vdW+1.4 A molecular-surface proxy, colorbar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from photomatagent.scientific.applications.vasp.molecular.results import (
        read_locpot,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        symbols, coords, _ = read_xyz(structure_path)
        locpot = read_locpot(locpot_path)
    except Exception:
        out_path.with_suffix(".txt").write_text(
            "LOCPOT or structure unreadable; ESP surface not rendered",
            encoding="utf-8",
        )
        return out_path
    if locpot is None or not len(coords):
        out_path.with_suffix(".txt").write_text(
            "LOCPOT grid empty; ESP surface not rendered",
            encoding="utf-8",
        )
        return out_path
    grid = locpot.grid
    origin = np.zeros(3, dtype=float)
    spacing = np.asarray(locpot.spacing_ang, dtype=float)
    data = np.asarray(locpot.data)
    sample_points: list[list[float]] = []
    sample_values: list[float] = []
    for symbol, position in zip(symbols, coords, strict=False):
        radius = VASP_VDW.get(symbol, 1.7) + 1.4
        for _ in range(24):
            direction = np.random.default_rng(
                hash((symbol, tuple(position))) % (2**32)
            ).normal(size=3)
            direction /= np.linalg.norm(direction)
            point = position + radius * direction
            sample_points.append(point.tolist())
            index = np.clip(
                np.rint((point - origin) / spacing).astype(int),
                0,
                np.asarray(data.shape) - 1,
            )
            sample_values.append(float(data[tuple(index)]))
    points = np.asarray(sample_points)
    figure = plt.figure(figsize=(5.4, 5.0))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=sample_values, cmap="coolwarm", s=14,
    )
    axis.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        s=22, color="#888888", depthshade=True,
    )
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.7)
    colorbar.set_label("LOCPOT (eV, LVHAR ionic+Hartree)")
    axis.set_title(
        "ESP surface proxy: LOCPOT sampled at vdW+1.4 A (method note in "
        "report; real CHGCAR isosurface pending SCNet validation)",
        fontsize=7,
    )
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return out_path


def plot_binding_energies(
    binding_rows: list[dict[str, Any]], out_path: Path
) -> Path:
    """Binding-energy bar chart (electronic only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["complex"] for row in binding_rows]
    values = [row["delta_e_ev"] for row in binding_rows]
    figure, axis = plt.subplots(figsize=(max(4.0, 0.7 * len(labels) + 1), 4.0))
    colors = [
        "#2ca02c" if value is not None and value < 0 else "#d62728"
        for value in values
    ]
    axis.bar(labels, [value if value is not None else 0.0 for value in values],
             color=colors, alpha=0.8)
    for index, value in enumerate(values):
        if value is not None:
            axis.annotate(f"{value:.3f}", (index, value), fontsize=7, ha="center")
    axis.set_ylabel("electronic binding energy (eV)")
    axis.set_title(
        "E_binding = E(complex) - sum E(fragments); electronic only, no "
        "vibrational/thermal terms",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return out_path
