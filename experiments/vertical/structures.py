"""Generate real test structures (pymatgen) for the vertical slices."""

from __future__ import annotations

from pathlib import Path


def generate_structures(output_dir: Path) -> dict[str, Path]:
    """Write CIFs for HgTe (zinc blende), PbTe (rock salt), InAs (zinc blende)."""
    from pymatgen.core import Structure

    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "HgTe": ("F-43m", 6.46, {"Hg": [0.0, 0.0, 0.0], "Te": [0.25, 0.25, 0.25]}),
        "PbTe": ("Fm-3m", 6.46, {"Pb": [0.0, 0.0, 0.0], "Te": [0.5, 0.5, 0.5]}),
        "InAs": ("F-43m", 6.058, {"In": [0.0, 0.0, 0.0], "As": [0.25, 0.25, 0.25]}),
    }
    paths: dict[str, Path] = {}
    for formula, (spacegroup, a, species) in specs.items():
        coords = list(species.values())
        species_list = list(species.keys())
        structure = Structure.from_spacegroup(
            spacegroup,
            [[a, 0, 0], [0, a, 0], [0, 0, a]],
            species_list,
            coords,
        )
        path = output_dir / f"{formula}.cif"
        structure.to(filename=str(path), fmt="cif")
        paths[formula] = path
    return paths


def hgte_structure(output_dir: Path) -> Path:
    return generate_structures(output_dir)["HgTe"]
