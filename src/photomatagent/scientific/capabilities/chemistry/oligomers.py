"""Finite representative oligomers for VM/TVM-type polymer proxies.

When the user does not provide the exact polymer connectivity, the study
explicitly builds a small linear oligomer from the available monomer
structures (ASSUMED_REPRESENTATIVE). Every default (``repeat_counts``,
``end_caps``, ``crosslink_position``) is recorded in the provenance
assumptions. If even the monomer connection chemistry is unknown the caller
falls back to an explicitly-labelled ASSUMED_PROXY; a proxy is never called
the real VM/TVM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

from photomatagent.scientific.capabilities.chemistry.conformers import (
    DEFAULT_SEED,
    ChemistryError,
    generate_conformer_candidates,
)


@dataclass(frozen=True)
class OligomerRecipe:
    """Explicit, recorded defaults for one oligomer proxy."""

    monomer_smiles: tuple[str, ...]
    repeat_counts: tuple[int, ...] = ()
    end_caps: tuple[str, ...] = ("H", "H")
    crosslink_position: str = "none (single linear chain proxy)"
    assumption_notes: list[str] = field(default_factory=list)

    def assumptions(self) -> list[str]:
        lines = [
            "finite linear oligomer proxy; exact polymer connectivity was "
            "not user-provided",
            f"repeat_counts per monomer: {list(self.repeat_counts)}",
            f"end_caps: {list(self.end_caps)}",
            f"crosslink_position: {self.crosslink_position}",
        ]
        return lines + list(self.assumption_notes)


_TERMINAL_ALKENE = Chem.MolFromSmarts("[CX3]=[CX3]")


def _terminal_alkene_carbons(mol: Chem.Mol) -> tuple[int, int] | None:
    """Return (head, tail) indices of the first terminal C=C (vinyl).

    ``head`` is the carbon with exactly one heavy neighbour (CH2), ``tail``
    is its alkene partner (CHX). Returns None when no terminal alkene exists.
    """
    matches = mol.GetSubstructMatches(_TERMINAL_ALKENE)
    for left, right in matches:
        left_heavy = sum(
            1 for n in mol.GetAtomWithIdx(left).GetNeighbors()
            if n.GetAtomicNum() > 1
        )
        right_heavy = sum(
            1 for n in mol.GetAtomWithIdx(right).GetNeighbors()
            if n.GetAtomicNum() > 1
        )
        # Head = the carbon with exactly one heavy neighbour (terminal CH2);
        # the alkene partner may carry 2 or 3 heavy neighbours (CHX or CXY,
        # e.g. methacrylates).
        if left_heavy == 1 and right_heavy >= 2:
            return left, right
        if right_heavy == 1 and left_heavy >= 2:
            return right, left
    return None


def build_oligomer(
    recipe: OligomerRecipe,
    *,
    seed: int = DEFAULT_SEED,
    n_conformers: int = 6,
    max_returned: int = 2,
) -> Chem.Mol:
    """Build a linear head-to-tail oligomer and embed/optimise it."""
    if not recipe.monomer_smiles:
        raise ChemistryError(
            "oligomer recipe requires at least one monomer SMILES",
            code="CHEMISTRY_OLIGOMER_MONOMERS_MISSING",
        )
    monomers: list[Chem.Mol] = []
    alkene_sites: list[tuple[int, int]] = []
    for smiles in recipe.monomer_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ChemistryError(
                f"oligomer monomer SMILES unparseable: {smiles!r}",
                code="CHEMISTRY_OLIGOMER_MONOMERS_MISSING",
            )
        Chem.SanitizeMol(mol)
        site = _terminal_alkene_carbons(mol)
        if site is None:
            raise ChemistryError(
                f"monomer {smiles!r} has no terminal alkene for head-to-tail "
                "polymerisation; use an explicit ASSUMED_PROXY instead",
                code="CHEMISTRY_OLIGOMER_NO_POLYMERIZATION_SITE",
            )
        monomers.append(mol)
        alkene_sites.append(site)

    counts = list(recipe.repeat_counts)
    if len(counts) < len(monomers):
        counts = counts + [2] * (len(monomers) - len(counts))
    if any(count < 1 for count in counts):
        raise ChemistryError("repeat_counts must be >= 1")
    units: list[tuple[Chem.Mol, int, int]] = []
    for monomer, (head, tail), count in zip(
        monomers, alkene_sites, counts, strict=True
    ):
        for _ in range(count):
            units.append((monomer, head, tail))

    merged = Chem.RWMol()
    offsets: list[int] = []
    global_sites: list[tuple[int, int]] = []
    for monomer, head, tail in units:
        offset = merged.GetNumAtoms()
        offsets.append(offset)
        for atom in monomer.GetAtoms():
            merged.AddAtom(atom)
        for bond in monomer.GetBonds():
            merged.AddBond(
                bond.GetBeginAtomIdx() + offset,
                bond.GetEndAtomIdx() + offset,
                bond.GetBondType(),
            )
        global_sites.append((head + offset, tail + offset))
    # Head-to-tail linking: tail(unit i) -- head(unit i+1).
    for index in range(len(units) - 1):
        tail_i = global_sites[index][1]
        head_next = global_sites[index + 1][0]
        merged.AddBond(tail_i, head_next, Chem.BondType.SINGLE)
    # The former alkene C=C of every repeat unit becomes a single bond in
    # the saturated chain (radical polymerisation).
    for head_g, tail_g in global_sites:
        bond = merged.GetBondBetweenAtoms(head_g, tail_g)
        if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetBondType(Chem.BondType.SINGLE)
    # End caps: explicit hydrogens on the first head and last tail.
    first_head = global_sites[0][0]
    last_tail = global_sites[-1][1]
    cap_configs: list[tuple[Chem.Atom, int]] = []
    for cap in recipe.end_caps:
        if cap == "H":
            cap_configs.append((Chem.Atom(1), 0))
        elif cap == "methyl":
            cap_configs.append((Chem.Atom(6), 3))
        elif cap == "CF3":
            carbon = Chem.Atom(6)
            cap_configs.append((carbon, 3))
        else:
            raise ChemistryError(
                f"unsupported end cap {cap!r}; use H, methyl or CF3"
            )
    first_cap = merged.AddAtom(cap_configs[0][0])
    merged.AddBond(first_head, first_cap, Chem.BondType.SINGLE)
    last_cap = merged.AddAtom(cap_configs[-1][0])
    merged.AddBond(last_tail, last_cap, Chem.BondType.SINGLE)
    if cap_configs[0][1] == 3:
        _add_fluorines_or_hydrogens(merged, first_cap, "F" if recipe.end_caps[0] == "CF3" else "H")
    if cap_configs[-1][1] == 3:
        _add_fluorines_or_hydrogens(merged, last_cap, "F" if recipe.end_caps[-1] == "CF3" else "H")

    chain = merged.GetMol()
    try:
        Chem.SanitizeMol(chain)
    except Exception as exc:
        raise ChemistryError(
            f"oligomer sanitisation failed: {exc}"
        ) from exc
    chain.SetProp("_Name", "oligomer")
    status = AllChem.EmbedMolecule(chain, randomSeed=seed)  # type: ignore[attr-defined]
    if status != 0:
        raise ChemistryError(
            f"oligomer 3D embedding failed (seed={seed})",
            code="CHEMISTRY_GENERATION_FAILED",
        )
    candidates = generate_conformer_candidates(
        chain, n_conformers=n_conformers, seed=seed, max_returned=max_returned
    )
    return candidates[0].mol


def _add_fluorines_or_hydrogens(merged: Chem.RWMol, cap_idx: int, symbol: str) -> None:
    """Attach three F/H atoms to a methyl-like end cap (CF3 or CH3)."""
    for _ in range(3):
        atom = merged.AddAtom(Chem.Atom(1 if symbol == "H" else 9))
        merged.AddBond(cap_idx, atom, Chem.BondType.SINGLE)
