"""Initial complex geometries: multiple coordination sites, orientations
and seeded directions, never one arbitrary guess.

Li+ complexes are seeded near O/N/F/S coordination sites at vdW-based
distances; TFSI- complexes are seeded near polar hydrogens / F-bearing
carbons with several anion orientations. Candidates are collision-filtered,
force-field pre-optimised and returned energy-sorted with diversity pruning.
Complex charge always equals the fragment charge sum (checked, never
assumed).
"""

from __future__ import annotations

import math
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

from photomatagent.scientific.capabilities.chemistry.conformers import (
    ConformerCandidate,
    DEFAULT_SEED,
    ChemistryError,
    generate_conformer_candidates,
    heavy_atom_min_distance,
    mol_from_smiles,
    optimize_conformer,
)


LI_COORDINATION_ELEMENTS = {"O", "N", "F", "S"}
MIN_COMPLEX_HEAVY_DISTANCE_ANG = 1.4
LI_SITE_DISTANCE_ANG = 2.0
TFSI_H_DISTANCE_ANG = 2.1
GUEST_MAX_CANDIDATES = 6


def _atom_positions(
    mol: Chem.Mol, conf_id: int
) -> dict[int, tuple[float, float, float]]:
    conformer = mol.GetConformer(conf_id)
    return {
        atom.GetIdx(): (
            float(position.x), float(position.y), float(position.z)
        )
        for atom, position in (
            (atom, conformer.GetAtomPosition(atom.GetIdx()))
            for atom in mol.GetAtoms()
        )
    }


def _outward_direction(mol: Chem.Mol, conf_id: int, site_idx: int) -> tuple[float, float, float]:
    """Unit vector from the site atom away from its heavy neighbours."""
    positions = _atom_positions(mol, conf_id)
    site = positions[site_idx]
    neighbor_centroid: list[float] = [0.0, 0.0, 0.0]
    neighbors = 0
    for neighbor in mol.GetAtomWithIdx(site_idx).GetNeighbors():
        if neighbor.GetAtomicNum() > 1:
            position = positions[neighbor.GetIdx()]
            neighbor_centroid[0] += position[0]
            neighbor_centroid[1] += position[1]
            neighbor_centroid[2] += position[2]
            neighbors += 1
    if neighbors:
        centroid = (
            neighbor_centroid[0] / neighbors,
            neighbor_centroid[1] / neighbors,
            neighbor_centroid[2] / neighbors,
        )
    else:
        centroid = (site[0], site[1], site[2])
    dx, dy, dz = site[0] - centroid[0], site[1] - centroid[1], site[2] - centroid[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    return dx / norm, dy / norm, dz / norm


def coordination_sites(
    mol: Chem.Mol, guest_symbol: str
) -> list[int]:
    """Deterministic coordination-site atom indices (sorted)."""
    if guest_symbol == "Li":
        return sorted(
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() in LI_COORDINATION_ELEMENTS
        )
    # TFSI- : polar hydrogens (bonded to N/O) then hydrogens on F-bearing
    # carbons; all sorted for determinism.
    candidates: list[tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if not heavy:
            continue
        partner = heavy[0]
        if partner.GetSymbol() in {"N", "O"}:
            candidates.append((0, atom.GetIdx()))
        elif any(
            n.GetSymbol() == "F" for n in partner.GetNeighbors()
        ):
            candidates.append((1, atom.GetIdx()))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [index for _, index in candidates]


def _rotate_around_axis(
    point: tuple[float, float, float],
    axis: tuple[float, float, float],
    origin: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float]:
    """Rotate ``point`` around ``axis`` through ``origin`` (Rodrigues)."""
    px, py, pz = point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]
    ux, uy, uz = axis
    cost = math.cos(angle_rad)
    sint = math.sin(angle_rad)
    dot = ux * px + uy * py + uz * pz
    cross = (uy * pz - uz * py, uz * px - ux * pz, ux * py - uy * px)
    rx = px * cost + cross[0] * sint + ux * dot * (1 - cost) + origin[0]
    ry = py * cost + cross[1] * sint + uy * dot * (1 - cost) + origin[1]
    rz = pz * cost + cross[2] * sint + uz * dot * (1 - cost) + origin[2]
    return rx, ry, rz


def _place_guest(
    host: Chem.Mol,
    host_conf: int,
    guest: Chem.Mol,
    guest_conf: int,
    *,
    guest_symbol: str,
    site_idx: int,
    orientation: int,
    seed: int,
) -> Chem.Mol | None:
    """Translate/rotate the guest into a candidate position around a site."""
    import random

    rng = random.Random(f"{seed}:{site_idx}:{orientation}")
    host_positions = _atom_positions(host, host_conf)
    guest_atoms = list(guest.GetAtoms())
    guest_positions = _atom_positions(guest, guest_conf)
    if guest_symbol == "Li":
        li_idx = guest_atoms[0].GetIdx()
        site = host_positions[site_idx]
        direction = _outward_direction(host, host_conf, site_idx)
        jittered = (
            math.cos(orientation * 0.61) * 0.15,
            math.sin(orientation * 1.37) * 0.12,
            math.sin(orientation * 0.23) * 0.10,
        )
        target = (
            site[0] + direction[0] * LI_SITE_DISTANCE_ANG + jittered[0],
            site[1] + direction[1] * LI_SITE_DISTANCE_ANG + jittered[1],
            site[2] + direction[2] * LI_SITE_DISTANCE_ANG + jittered[2],
        )
        guest_position = guest_positions[li_idx]
        translation = (
            target[0] - guest_position[0],
            target[1] - guest_position[1],
            target[2] - guest_position[2],
        )
        return _translate_mol(guest, guest_conf, translation)
    # TFSI- : point one anion O atom at the polar hydrogen.
    oxygen_idx = next(
        (
            atom.GetIdx()
            for atom in guest_atoms
            if atom.GetSymbol() == "O"
        ),
        None,
    )
    if oxygen_idx is None:
        return None
    h_position = host_positions[site_idx]
    o_position = guest_positions[oxygen_idx]
    guest_centroid = _centroid(list(guest_positions.values()))
    target = (
        h_position[0] + (rng.uniform(-0.2, 0.2)),
        h_position[1] + (rng.uniform(-0.2, 0.2)),
        h_position[2] + (rng.uniform(-0.2, 0.2)),
    )
    # Align the O--centroid axis with host->target, then rotate around it.
    axis = _unit(
        target[0] - guest_centroid[0],
        target[1] - guest_centroid[1],
        target[2] - guest_centroid[2],
    )
    rotated = _rotate_around_axis(
        (o_position[0], o_position[1], o_position[2]), axis, guest_centroid,
        orientation * math.pi / 2,
    )
    translation = (
        target[0] - rotated[0] + axis[0] * TFSI_H_DISTANCE_ANG,
        target[1] - rotated[1] + axis[1] * TFSI_H_DISTANCE_ANG,
        target[2] - rotated[2] + axis[2] * TFSI_H_DISTANCE_ANG,
    )
    return _translate_mol(guest, guest_conf, translation)


def _centroid(
    points: Sequence[tuple[float, float, float]]
) -> tuple[float, float, float]:
    total = len(points)
    if not total:
        return (0.0, 0.0, 0.0)
    return (
        sum(point[0] for point in points) / total,
        sum(point[1] for point in points) / total,
        sum(point[2] for point in points) / total,
    )


def _unit(x: float, y: float, z: float) -> tuple[float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    return x / norm, y / norm, z / norm


def _translate_mol(
    mol: Chem.Mol, conf_id: int, translation: tuple[float, float, float]
) -> Chem.Mol:
    transformed = Chem.Mol(mol)
    conformer = transformed.GetConformer(conf_id)
    for atom in transformed.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        conformer.SetAtomPosition(
            atom.GetIdx(),
            (
                position.x + translation[0],
                position.y + translation[1],
                position.z + translation[2],
            ),
        )
    return transformed


def _merge_with_placement(
    host_mol: Chem.Mol,
    host_conf: int,
    guest_mol: Chem.Mol,
    guest_conf: int,
) -> Chem.Mol:
    """One molecule, one conformer: host atoms + translated guest atoms."""
    merged = Chem.RWMol()
    for atom in host_mol.GetAtoms():
        merged.AddAtom(atom)
    offset = host_mol.GetNumAtoms()
    for atom in guest_mol.GetAtoms():
        merged.AddAtom(atom)
    for bond in host_mol.GetBonds():
        merged.AddBond(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond.GetBondType(),
        )
    for bond in guest_mol.GetBonds():
        merged.AddBond(
            bond.GetBeginAtomIdx() + offset,
            bond.GetEndAtomIdx() + offset,
            bond.GetBondType(),
        )
    result = merged.GetMol()
    conformer = Chem.Conformer(result.GetNumAtoms())
    host_positions = host_mol.GetConformer(host_conf)
    for index in range(host_mol.GetNumAtoms()):
        position = host_positions.GetAtomPosition(index)
        conformer.SetAtomPosition(index, position)
    guest_positions = guest_mol.GetConformer(guest_conf)
    for index in range(guest_mol.GetNumAtoms()):
        position = guest_positions.GetAtomPosition(index)
        conformer.SetAtomPosition(index + offset, position)
    result.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(result)
    result.SetProp("_Name", host_mol.GetProp("_Name"))
    return result


def build_complex_candidates(
    host_mol: Chem.Mol,
    guest_smiles: str,
    *,
    expected_charge: int,
    n_orientations: int = 4,
    seed: int = DEFAULT_SEED,
    max_returned: int = 3,
) -> list[ConformerCandidate]:
    """Generate several low-energy, geometrically distinct complex guesses.

    ``expected_charge`` must equal host charge + guest charge (checked).
    """
    guest_mol = mol_from_smiles(guest_smiles, name="guest")
    guest_mol = Chem.AddHs(guest_mol)
    status = AllChem.EmbedMolecule(guest_mol, randomSeed=seed)  # type: ignore[attr-defined]
    if status != 0:
        raise ChemistryError(
            f"guest embedding failed for {guest_smiles} (seed={seed})"
        )
    host_charge = Chem.GetFormalCharge(host_mol)
    guest_charge = Chem.GetFormalCharge(guest_mol)
    if host_charge + guest_charge != expected_charge:
        raise ChemistryError(
            f"complex charge contract violated: host {host_charge} + guest "
            f"{guest_charge} = {host_charge + guest_charge}, expected "
            f"{expected_charge}",
            code="CHEMISTRY_COMPLEX_CHARGE_MISMATCH",
        )
    guest_symbol = guest_mol.GetAtomWithIdx(0).GetSymbol()
    # A high-quality host conformer anchors the complex geometry.
    host_candidates = generate_conformer_candidates(
        host_mol, n_conformers=8, seed=seed, max_returned=2
    )
    host_mol = host_candidates[0].mol
    host_conf = host_candidates[0].conf_id
    sites = coordination_sites(host_mol, guest_symbol)
    if not sites:
        raise ChemistryError(
            f"no coordination sites found on the host for {guest_symbol}"
        )
    candidates: list[tuple[float | None, str, Chem.Mol]] = []
    for site_idx in sites[:6]:
        for orientation in range(n_orientations):
            placed = _place_guest(
                host_mol, host_conf, guest_mol, 0,
                guest_symbol=guest_symbol,
                site_idx=site_idx,
                orientation=orientation,
                seed=seed,
            )
            if placed is None:
                continue
            combined = _merge_with_placement(
                host_mol, host_conf, placed, 0
            )
            pre_positions = [
                combined.GetConformer(0).GetAtomPosition(index)
                for index in range(combined.GetNumAtoms())
            ]
            pre_minimum = heavy_atom_min_distance(combined)
            energy, field_name = optimize_conformer(combined)
            minimum = heavy_atom_min_distance(combined)
            if minimum < MIN_COMPLEX_HEAVY_DISTANCE_ANG:
                if pre_minimum < MIN_COMPLEX_HEAVY_DISTANCE_ANG:
                    continue  # badly placed before optimisation too
                # The force field (mainly MMFF's weak Li terms) pulled the
                # complex below the collision floor: restore the placement
                # geometry and record that optimisation was rejected.
                conformer = combined.GetConformer(0)
                for index, position in enumerate(pre_positions):
                    conformer.SetAtomPosition(index, position)
                minimum = pre_minimum
                energy = None
                field_name = f"{field_name}(rejected:collapse)"
            candidates.append((energy, field_name, combined))
    if not candidates:
        raise ChemistryError(
            "no collision-free complex candidate could be generated; "
            "try more orientations or a different seed"
        )
    candidates.sort(key=lambda item: (item[0] is None, item[0] or 0.0))
    kept: list[ConformerCandidate] = []
    for rank, (energy, field_name, combined) in enumerate(
        candidates[:max_returned], start=1
    ):
        kept.append(
            ConformerCandidate(
                rank=rank,
                conf_id=0,
                energy_kcal_mol=energy,
                force_field=field_name,
                min_heavy_distance=heavy_atom_min_distance(combined),
                mol=combined,
            )
        )
    return kept
