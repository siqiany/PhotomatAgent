"""Project-approved alias registry for chemical entities.

This is the third resolution step (after user-provided structures and
explicit SMILES/InChI). Every entry carries an explicit SMILES, the total
charge and a documented source so charges are never guessed from names.
Polymer/complex entries reference explicit construction recipes
(``oligomer:...`` / ``complex:...``) instead of pretending to be single
molecules.

Formulas follow the verified gel-electrolyte plan
(``gel_electrolyte_dft/plans/plan.md``) and the TFPMA smoke run record.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AliasEntry:
    """One reviewed alias definition."""

    aliases: tuple[str, ...]
    system_id: str
    display_name: str
    smiles: str
    total_charge: int
    formula: str
    role: str = "molecule"
    note: str = ""
    recipe: str = ""  # "smiles" | "oligomer:..." | "complex:..."


APPROVED_ALIAS_REGISTRY: tuple[AliasEntry, ...] = (
    AliasEntry(
        aliases=("TFPMA", "tetrafluoropropyl methacrylate"),
        system_id="tfpma",
        display_name="TFPMA",
        smiles="C=C(C)C(=O)OCC(F)(F)C(F)F",
        total_charge=0,
        formula="C7H8F4O2",
        role="molecule",
        note=(
            "2,2,3,3-tetrafluoropropyl methacrylate; 76 PAW valence "
            "electrons; formula matches the TFPMA smoke run record"
        ),
    ),
    AliasEntry(
        aliases=("VEC", "vinyl ethylene carbonate", "4-vinyl-1,3-dioxolan-2-one"),
        system_id="vec",
        display_name="VEC",
        smiles="C=C[C@@H]1COC(=O)O1",
        total_charge=0,
        formula="C5H6O3",
    ),
    AliasEntry(
        aliases=("MBA", "methylenebisacrylamide", "N,N'-methylenebisacrylamide"),
        system_id="mba",
        display_name="MBA",
        smiles="C=CC(=O)NCNC(=O)C=C",
        total_charge=0,
        formula="C7H10N2O2",
        note="crosslinker monomer with two acrylamide arms",
    ),
    AliasEntry(
        aliases=("DME", "1,2-dimethoxyethane", "glyme-1"),
        system_id="dme",
        display_name="DME",
        smiles="COCCOC",
        total_charge=0,
        formula="C4H10O2",
    ),
    AliasEntry(
        aliases=("TFSI", "TFSI-", "bis(trifluoromethanesulfonyl)imide"),
        system_id="tfsi",
        display_name="TFSI-",
        smiles="O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
        total_charge=-1,
        formula="C2F6NO4S2",
        role="ion",
        note="charge -1 is explicit, never inferred from the name",
    ),
    AliasEntry(
        aliases=("Li", "Li+", "lithium ion"),
        system_id="li",
        display_name="Li+",
        smiles="[Li+]",
        total_charge=1,
        formula="Li",
        role="ion",
    ),
    AliasEntry(
        aliases=("NO3", "NO3-", "nitrate"),
        system_id="no3",
        display_name="NO3-",
        smiles="[O-][N+](=O)[O-]",
        total_charge=-1,
        formula="NO3",
        role="ion",
    ),
    AliasEntry(
        aliases=("LiNO3", "lithium nitrate"),
        system_id="lino3",
        display_name="LiNO3",
        smiles="[Li+].[O-][N+](=O)[O-]",
        total_charge=0,
        formula="LiNO3",
        role="complex",
        note="contact-ion-pair target; conformers rebuilt by build_complex",
        recipe="complex:li+no3",
    ),
    AliasEntry(
        aliases=("LiTFSI", "lithium bis(trifluoromethanesulfonyl)imide"),
        system_id="litfsi",
        display_name="LiTFSI",
        smiles="[Li+].O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
        total_charge=0,
        formula="C2F6LiNO4S2",
        role="complex",
        note="contact-ion-pair target; conformers rebuilt by build_complex",
        recipe="complex:li+tfsi",
    ),
    AliasEntry(
        aliases=("DME-Li+", "DME_Li", "DME-Li"),
        system_id="dme_li",
        display_name="DME-Li+",
        smiles="COCCOC.[Li+]",
        total_charge=1,
        formula="C4H10O2Li",
        role="complex",
        note="charge +1 equals fragment sum DME(0)+Li+(+1)",
        recipe="complex:dme+li",
    ),
    AliasEntry(
        aliases=("VM", "VEC-MBA polymer"),
        system_id="vm",
        display_name="VM",
        smiles="",
        total_charge=0,
        formula="C12H16N2O5",  # one VEC + one MBA repeat (proxies only)
        role="oligomer",
        note=(
            "polymerized VEC+MBA network; connectivity is not user-provided, "
            "so the study builds a finite linear representative oligomer "
            "with explicit defaults (ASSUMED_REPRESENTATIVE)"
        ),
        recipe="oligomer:vec+mba",
    ),
    AliasEntry(
        aliases=("TVM", "TFPMA-VEC-MBA polymer"),
        system_id="tvm",
        display_name="TVM",
        smiles="",
        total_charge=0,
        formula="C19H24F4N2O7",  # one TFPMA+VEC+MBA repeat (proxies only)
        role="oligomer",
        note=(
            "polymerized TFPMA+VEC+MBA network; connectivity is not "
            "user-provided, so the study builds a finite linear "
            "representative oligomer with explicit defaults "
            "(ASSUMED_REPRESENTATIVE)"
        ),
        recipe="oligomer:tfpma+vec+mba",
    ),
    AliasEntry(
        aliases=("TVM-Li+", "TVM_Li"),
        system_id="tvm_li",
        display_name="TVM-Li+",
        smiles="",
        total_charge=1,
        formula="C19H24F4LiN2O7",
        role="complex",
        note="charge +1 equals fragment sum TVM(0)+Li+(+1)",
        recipe="complex:tvm+li",
    ),
    AliasEntry(
        aliases=("TVM-TFSI-", "TVM_TFSI"),
        system_id="tvm_tfsi",
        display_name="TVM-TFSI-",
        smiles="",
        total_charge=-1,
        formula="C21H24F10N3O11S2",
        role="complex",
        note="charge -1 equals fragment sum TVM(0)+TFSI-(-1)",
        recipe="complex:tvm+tfsi",
    ),
)


_ALIAS_INDEX: dict[str, AliasEntry] = {}
for _entry in APPROVED_ALIAS_REGISTRY:
    for _alias in _entry.aliases:
        _ALIAS_INDEX[_alias.casefold()] = _entry


def lookup_alias(name: str) -> AliasEntry | None:
    """Case-insensitive lookup in the approved registry (never guesses)."""
    return _ALIAS_INDEX.get(name.strip().casefold())
