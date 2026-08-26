"""Deterministic 3D conformer generation (RDKit ETKDG + MMFF/UFF).

Contract:
* the random seed is fixed and recorded (provenance carries it);
* several conformers are generated per structure;
* MMFF94 optimisation is preferred, UFF is the documented fallback;
* severe heavy-atom collisions are rejected;
* charge and atom mapping are preserved;
* output candidates are sorted by force-field energy (lowest first).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign


class ChemistryError(ValueError):
    """Structure generation failure (typed, deterministic)."""

    def __init__(self, message: str, *, code: str = "CHEMISTRY_GENERATION_FAILED"):
        super().__init__(message)
        self.code = code


DEFAULT_SEED = 20260825
MIN_HEAVY_DISTANCE_ANG = 0.9  # severe collision threshold after embedding


def mol_from_smiles(
    smiles: str, *, expected_charge: int | None = None, name: str = ""
) -> Chem.Mol:
    """Parse a SMILES deterministically; verify the formal charge contract."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ChemistryError(
            f"RDKit could not parse SMILES for {name or smiles!r}"
        )
    Chem.SanitizeMol(mol)
    mol.SetProp("_Name", name)
    if expected_charge is not None:
        actual = Chem.GetFormalCharge(mol)
        if actual != expected_charge:
            raise ChemistryError(
                f"formal charge mismatch for {name}: expected "
                f"{expected_charge}, RDKit parsed {actual}",
                code="CHEMISTRY_CHARGE_MISMATCH",
            )
    return mol


def _resolve_force_field(
    mol: Chem.Mol,
) -> tuple[Any, str] | None:
    """Return (force_field, name); MMFF preferred, UFF fallback."""
    if AllChem.MMFFHasAllMoleculeParams(mol):  # type: ignore[attr-defined]
        properties = AllChem.MMFFGetMoleculeProperties(mol)  # type: ignore[attr-defined]
        if properties is not None:
            field = AllChem.MMFFGetMoleculeForceField(mol, properties)  # type: ignore[attr-defined]
            if field is not None:
                return field, "MMFF94"
    field = AllChem.UFFGetMoleculeForceField(mol)  # type: ignore[attr-defined]
    if field is not None:
        return field, "UFF"
    return None


def optimize_conformer(
    mol: Chem.Mol, conf_id: int = -1, *, max_iterations: int = 500
) -> tuple[float | None, str]:
    """Optimise one conformer; returns (energy_kcal_mol, force_field)."""
    resolved = _resolve_force_field(mol)
    if resolved is None:
        return None, ""
    field, name = resolved
    try:
        field.Minimize(maxIts=max_iterations)
        energy = float(field.CalcEnergy())
    except Exception:
        return None, name
    return energy, name


def heavy_atom_min_distance(mol: Chem.Mol, conf_id: int = -1) -> float:
    """Minimum NON-BONDED pairwise distance between non-hydrogen atoms.

    Covalently bonded heavy atoms (e.g. N-O at ~1.24 A) are excluded: they
    sit at bond distance by construction and must not trip the collision
    filter, which guards against steric clashes only.
    """
    conformer = mol.GetConformer(conf_id)
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    minimum = math.inf
    for i, left in enumerate(heavy):
        position_i = conformer.GetAtomPosition(left)
        for right in heavy[i + 1 :]:
            if mol.GetBondBetweenAtoms(left, right) is not None:
                continue  # covalent pair: allowed at bond distance
            position_j = conformer.GetAtomPosition(right)
            distance = position_i.Distance(position_j)
            if distance < minimum:
                minimum = distance
    return float(minimum) if minimum != math.inf else math.inf


@dataclass(frozen=True)
class ConformerCandidate:
    """One energy-sorted 3D candidate."""

    rank: int
    conf_id: int
    energy_kcal_mol: float | None
    force_field: str
    min_heavy_distance: float
    mol: Chem.Mol


def _conformer_rmsd(mol: Chem.Mol, first: int, second: int) -> float:
    try:
        return float(rdMolAlign.GetBestRMS(mol, mol, first, second))
    except Exception:
        return 0.0


def generate_conformer_candidates(
    mol: Chem.Mol,
    *,
    n_conformers: int = 8,
    seed: int = DEFAULT_SEED,
    prune_rmsd_ang: float = 0.5,
    min_heavy_distance: float = MIN_HEAVY_DISTANCE_ANG,
    max_returned: int = 3,
) -> list[ConformerCandidate]:
    """Deterministic ETKDG conformer ensemble, energy-sorted and pruned.

    Severe collisions are rejected; a smaller, geometrically distinct,
    low-energy set is returned (``max_returned`` candidates).
    """
    if n_conformers < 1:
        raise ValueError("n_conformers must be >= 1")
    original = mol
    work = Chem.AddHs(Chem.Mol(original))
    work.RemoveAllConformers()
    if work.GetNumAtoms() <= 1:
        # Single atoms (Li+) have no torsion space: a trivial conformer.
        conf = Chem.Conformer(1)
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        work.AddConformer(conf, assignId=True)
    else:
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = seed
        params.useSmallRingTorsions = True
        params.enforceChirality = True
        embedded = AllChem.EmbedMultipleConfs(  # type: ignore[attr-defined]
            work, numConfs=n_conformers, params=params
        )
        if len(embedded) == 0:
            # Fall back to a deterministic manual seeding for rigid cases.
            params = AllChem.ETKDGv2()  # type: ignore[attr-defined]
            params.randomSeed = seed
            embedded = AllChem.EmbedMultipleConfs(  # type: ignore[attr-defined]
                work, numConfs=n_conformers, params=params
            )
    if work.GetNumConformers() == 0:
        raise ChemistryError(
            f"ETKDG embedding failed for {original.GetProp('_Name') or 'structure'} "
            f"(seed={seed})"
        )

    scored: list[ConformerCandidate] = []
    for conf in work.GetConformers():
        conf_id = conf.GetId()
        energy, field_name = optimize_conformer(work, conf_id)
        minimum = heavy_atom_min_distance(work, conf_id)
        if minimum < min_heavy_distance:
            continue  # reject severe collisions
        scored.append(
            ConformerCandidate(
                rank=0,
                conf_id=conf_id,
                energy_kcal_mol=energy,
                force_field=field_name,
                min_heavy_distance=minimum,
                mol=work,
            )
        )
    if not scored:
        raise ChemistryError(
            f"all embedded conformers of "
            f"{original.GetProp('_Name') or 'structure'} collided or failed "
            "force-field optimisation"
        )
    scored.sort(key=lambda item: (item.energy_kcal_mol is None, item.energy_kcal_mol or 0.0))

    # Greedy diversity pruning: keep candidates whose geometry differs from
    # every already-kept candidate by at least ``prune_rmsd_ang``.
    kept: list[ConformerCandidate] = []
    for candidate in scored:
        if all(
            _conformer_rmsd(work, candidate.conf_id, other.conf_id)
            >= prune_rmsd_ang
            for other in kept
        ):
            kept.append(candidate)
            if len(kept) >= max_returned:
                break
    for rank, candidate in enumerate(kept, start=1):
        candidate = ConformerCandidate(
            rank=rank,
            conf_id=candidate.conf_id,
            energy_kcal_mol=candidate.energy_kcal_mol,
            force_field=candidate.force_field,
            min_heavy_distance=candidate.min_heavy_distance,
            mol=candidate.mol,
        )
        kept[rank - 1] = candidate
    return kept


def mol_to_xyz(
    mol: Chem.Mol, conf_id: int = -1, *, comment: str = ""
) -> str:
    """Render one conformer as XYZ text (deterministic ordering)."""
    conformer = mol.GetConformer(conf_id)
    atoms = list(mol.GetAtoms())
    lines = [str(len(atoms)), comment or mol.GetProp("_Name")]
    for atom in atoms:
        position = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():2s} {position.x:.5f} {position.y:.5f} {position.z:.5f}"
        )
    return "\n".join(lines) + "\n"


def mol_formula(mol: Chem.Mol) -> str:
    """Hill formula via RDKit's composition machinery."""
    from rdkit.Chem import rdMolDescriptors

    return rdMolDescriptors.CalcMolFormula(mol)


def atom_counts(mol: Chem.Mol) -> dict[str, int]:
    """Element counts (heavy + hydrogen), no tuple-ordering surprises."""
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts
