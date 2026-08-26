"""Persistence for generated structures: XYZ files, manifest and images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from photomatagent.scientific.capabilities.chemistry.models import (
    GeneratedStructure,
)


ELEMENT_COLORS: dict[str, str] = {
    "C": "#303030",
    "H": "#e8e8e8",
    "O": "#d62728",
    "N": "#1f77b4",
    "F": "#2ca02c",
    "S": "#ffcc00",
    "Li": "#9467bd",
    "P": "#ff7f0e",
}


def write_xyz(
    path: Path,
    symbols: Sequence[str],
    coordinates: Any,
    *,
    comment: str = "",
) -> Path:
    """Write one XYZ file (deterministic ordering, 5-decimal precision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coordinates, dtype=float)
    if len(symbols) != coords.shape[0]:
        raise ValueError(
            f"symbol/count mismatch: {len(symbols)} symbols vs "
            f"{coords.shape[0]} coordinates"
        )
    lines = [str(len(symbols)), comment]
    lines.extend(
        f"{symbol:2s} {x:.5f} {y:.5f} {z:.5f}"
        for symbol, (x, y, z) in zip(symbols, coords, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, str]:
    """Strict XYZ reader (atom-count contract enforced)."""
    lines = [line.strip() for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"XYZ too short: {path}")
    try:
        count = int(lines[0])
    except ValueError as exc:
        raise ValueError(f"XYZ atom count not an integer: {path}") from exc
    comment = lines[1]
    rows = lines[2 : 2 + count]
    if len(rows) != count:
        raise ValueError(
            f"XYZ atom count mismatch in {path}: header {count}, "
            f"found {len(rows)}"
        )
    symbols: list[str] = []
    coords: list[list[float]] = []
    for row in rows:
        fields = row.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ line in {path}: {row!r}")
        symbols.append(fields[0].capitalize())
        try:
            coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError as exc:
            raise ValueError(f"non-numeric XYZ coordinates in {path}") from exc
    return symbols, np.asarray(coords, dtype=float), comment


def write_structure_thumbnails(
    structures: Sequence[GeneratedStructure],
    figure_dir: Path,
) -> list[Path]:
    """Render a simple persistable 2D projection PNG per structure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for structure in structures:
        if not structure.structure_path.is_file():
            continue
        try:
            symbols, coords, _ = read_xyz(structure.structure_path)
        except ValueError:
            continue
        figure_path = figure_dir / (
            f"{structure.identity.system_id}"
            f"{'_' + structure.provenance.conformer_id if structure.provenance.conformer_id else ''}.png"
        )
        plt.figure(figsize=(4.2, 4.2))
        # Simple depth-shaded projection; bonds are omitted for clarity at
        # thumbnail scale (the persistable structure image requirement).
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        alpha = 0.35 + 0.65 * (z - z.min()) / max((z.max() - z.min()), 1e-9)
        for symbol, xi, yi, ai in zip(symbols, x, y, alpha, strict=False):
            plt.scatter(
                xi, yi, s=220, color=ELEMENT_COLORS.get(symbol, "#888888"),
                alpha=float(ai), edgecolors="#222222", linewidths=0.5,
            )
        plt.axis("equal")
        plt.axis("off")
        plt.title(
            f"{structure.identity.display_name} "
            f"({structure.reliability_grade().value})",
            fontsize=8,
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path, dpi=120, bbox_inches="tight")
        plt.close()
        written.append(figure_path)
    return written


def write_structure_manifest(
    structures: Sequence[GeneratedStructure],
    path: Path,
) -> Path:
    """Persist structure_manifest.json (metadata only, never file content)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": 1,
        "generated_at_note": "structure manifest; content bytes never stored",
        "structures": [structure.manifest_row() for structure in structures],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
