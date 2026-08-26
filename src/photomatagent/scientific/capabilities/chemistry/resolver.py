"""Structure resolution pipeline with recorded provenance (never silent).

Priority: 1) user-provided structure file; 2) explicit SMILES/InChI;
3) the project-approved alias registry; 4) database resolution (declared,
not enabled offline); 5) representative generation from known fragments;
6) an explicitly-labelled proxy. Every step records its provenance; the
study never blocks entirely because one structure is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rdkit import Chem

from photomatagent.scientific.capabilities.chemistry.complexes import (
    build_complex_candidates,
)
from photomatagent.scientific.capabilities.chemistry.conformers import (
    DEFAULT_SEED,
    ChemistryError,
    generate_conformer_candidates,
    mol_formula,
    mol_from_smiles,
    mol_to_xyz,
)
from photomatagent.scientific.capabilities.chemistry.models import (
    ChemicalIdentity,
    ChemicalRole,
    GeneratedStructure,
    ProvenanceStatus,
    StructureProvenance,
)
from photomatagent.scientific.capabilities.chemistry.oligomers import (
    OligomerRecipe,
    build_oligomer,
)
from photomatagent.scientific.capabilities.chemistry.registry import (
    AliasEntry,
    lookup_alias,
)
from photomatagent.scientific.capabilities.chemistry.storage import (
    read_xyz,
    write_xyz,
)


@dataclass
class StructureRequest:
    """Typed resolution request from the study layer."""

    system_id: str
    display_name: str = ""
    aliases: list[str] = field(default_factory=list)
    smiles: str | None = None
    inchi: str | None = None
    structure_path: Path | None = None
    total_charge: int | None = None
    spin_multiplicity: int = 1
    role: str = "molecule"
    allow_assumed: bool = True
    max_candidates: int = 3
    seed: int = DEFAULT_SEED


def _identity_from_alias(entry: AliasEntry) -> ChemicalIdentity:
    return ChemicalIdentity(
        system_id=entry.system_id,
        display_name=entry.display_name,
        aliases=list(entry.aliases),
        formula=entry.formula,
        smiles=entry.smiles,
        total_charge=entry.total_charge,
        role=ChemicalRole(entry.role),
    )


def _provenance(
    *,
    status: ProvenanceStatus,
    source: str,
    source_identifier: str,
    assumptions: Sequence[str],
    confidence: float,
    generator: str,
    seed: int,
    conformer_id: str = "",
    parents: Sequence[str] = (),
) -> StructureProvenance:
    return StructureProvenance(
        status=status,
        source=source,
        source_identifier=source_identifier,
        assumptions=list(assumptions),
        confidence=confidence,
        generator=generator,
        random_seed=seed,
        conformer_id=conformer_id,
        parent_structures=list(parents),
    )


def _generated(
    identity: ChemicalIdentity,
    *,
    structure_path: Path,
    symbols: Sequence[str],
    provenance: StructureProvenance,
    validation: Sequence[str],
    force_field_energy: float | None,
    format: str = "xyz",
) -> GeneratedStructure:
    return GeneratedStructure(
        identity=identity,
        structure_path=structure_path,
        format=format,
        atom_count=len(symbols),
        formal_charge=identity.total_charge,
        provenance=provenance,
        validation=list(validation),
        force_field_energy=force_field_energy,
    )


def _persist_candidates(
    identity: ChemicalIdentity,
    candidates: Sequence[Any],
    *,
    output_dir: Path,
    provenance_base: StructureProvenance,
    validation: Sequence[str],
    conformer_tag: str = "c",
) -> list[GeneratedStructure]:
    """Persist one XYZ per candidate and build GeneratedStructure rows."""
    structures: list[GeneratedStructure] = []
    for candidate in candidates:
        conformer_id = f"{conformer_tag}{candidate.rank}"
        path = output_dir / f"{identity.system_id}_{conformer_id}.xyz"
        text = mol_to_xyz(
            candidate.mol,
            candidate.conf_id,
            comment=(
                f"{identity.display_name} {conformer_id} "
                f"ff={candidate.force_field}"
            ),
        )
        path.write_text(text, encoding="utf-8")
        symbols, coords, _comment = read_xyz(path)
        provenance = StructureProvenance(
            **{
                **provenance_base.model_dump(),
                "conformer_id": conformer_id,
            }
        )
        structures.append(
            _generated(
                identity,
                structure_path=path,
                symbols=symbols,
                provenance=provenance,
                validation=validation,
                force_field_energy=candidate.energy_kcal_mol,
            )
        )
    return structures


def resolve_structure(
    request: StructureRequest,
    output_dir: Path,
) -> list[GeneratedStructure]:
    """Resolve one chemical entity into persisted structures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # 1) user-provided structure file
    if request.structure_path is not None:
        return _resolve_user_file(request, output_dir)
    # 2) explicit SMILES / InChI
    if request.smiles:
        return _resolve_smiles(request, output_dir)
    # 3) approved alias registry
    entry = lookup_alias(request.system_id) or lookup_alias(
        request.display_name
    )
    if entry is not None:
        if entry.recipe.startswith("complex:"):
            return _resolve_complex(request, entry, output_dir)
        if entry.recipe.startswith("oligomer:"):
            return _resolve_oligomer(request, entry, output_dir)
        if entry.smiles:
            request.smiles = entry.smiles
            if request.total_charge is None:
                request.total_charge = entry.total_charge
            return _resolve_smiles(request, output_dir)
    # 6) explicit proxy marker (never a silent guess)
    if request.allow_assumed:
        return _resolve_proxy(request, output_dir)
    return _resolve_blocked(request, output_dir)


def _resolve_user_file(
    request: StructureRequest, output_dir: Path
) -> list[GeneratedStructure]:
    structure_path = request.structure_path
    if structure_path is None:
        raise ChemistryError(
            "user structure path missing",
            code="CHEMISTRY_STRUCTURE_FILE_MISSING",
        )
    path = structure_path.expanduser().resolve()
    if not path.is_file():
        raise ChemistryError(
            f"user structure file missing: {path}",
            code="CHEMISTRY_STRUCTURE_FILE_MISSING",
        )
    if path.suffix.lower() == ".xyz":
        symbols, coords, comment = read_xyz(path)
    else:
        # Minimal generic fallback: try the shared XYZ parser on any text
        # structure; binary formats are rejected explicitly.
        try:
            symbols, coords, comment = read_xyz(path)
        except ValueError as exc:
            raise ChemistryError(
                f"unsupported user structure format {path.suffix} ({exc})",
                code="CHEMISTRY_STRUCTURE_UNREADABLE",
            ) from exc
    identity = ChemicalIdentity(
        system_id=request.system_id.strip().lower(),
        display_name=request.display_name or request.system_id,
        aliases=list(request.aliases),
        formula="",  # computed below from the file
        total_charge=request.total_charge
        if request.total_charge is not None
        else 0,
        spin_multiplicity=request.spin_multiplicity,
        role=ChemicalRole(request.role or "molecule"),
    )
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    identity = identity.model_copy(
        update={"formula": _hill_formula(counts)}
    )
    stored = output_dir / f"{request.system_id}_user.xyz"
    stored.write_text(
        f"{len(symbols)}\n{comment}\n"
        + "".join(
            f"{symbol:2s} {x:.5f} {y:.5f} {z:.5f}\n"
            for symbol, (x, y, z) in zip(symbols, coords, strict=True)
        ),
        encoding="utf-8",
    )
    provenance = _provenance(
        status=ProvenanceStatus.USER_PROVIDED,
        source="user file",
        source_identifier=str(path),
        assumptions=[],
        confidence=1.0,
        generator="",
        seed=0,
    )
    return [
        _generated(
            identity,
            structure_path=stored,
            symbols=symbols,
            provenance=provenance,
            validation=["user-provided geometry; formula read from file"],
            force_field_energy=None,
        )
    ]


def _resolve_smiles(
    request: StructureRequest, output_dir: Path
) -> list[GeneratedStructure]:
    if request.total_charge is None:
        raise ChemistryError(
            "cannot resolve structure without an explicit total_charge",
            code="CHEMISTRY_CHARGE_REQUIRED",
        )
    smiles = request.smiles
    if smiles is None:
        raise ChemistryError(
            "cannot resolve structure without an explicit SMILES",
            code="CHEMISTRY_SMILES_REQUIRED",
        )
    mol = mol_from_smiles(
        smiles,
        expected_charge=request.total_charge,
        name=request.display_name or request.system_id,
    )
    identity = ChemicalIdentity(
        system_id=request.system_id.strip().lower(),
        display_name=request.display_name or request.system_id,
        aliases=list(request.aliases),
        formula=mol_formula(mol),
        smiles=smiles,
        total_charge=request.total_charge,
        spin_multiplicity=request.spin_multiplicity,
        role=ChemicalRole(request.role or "molecule"),
    )
    candidates = generate_conformer_candidates(
        mol,
        n_conformers=8,
        seed=request.seed,
        max_returned=request.max_candidates,
    )
    provenance = _provenance(
        status=ProvenanceStatus.GENERATED_FROM_SMILES,
        source="smiles",
        source_identifier=smiles,
        assumptions=[
            "3D geometry generated from SMILES with RDKit ETKDG",
            f"formal charge {request.total_charge:+d} enforced",
        ],
        confidence=0.9,
        generator="rdkit-etkdg",
        seed=request.seed,
    )
    return _persist_candidates(
        identity,
        candidates,
        output_dir=output_dir,
        provenance_base=provenance,
        validation=["collision-filtered", "MMFF/UFF optimised"],
    )


def _resolve_complex(
    request: StructureRequest, entry: AliasEntry, output_dir: Path
) -> list[GeneratedStructure]:
    parts = entry.recipe.split(":", 1)[1].split("+")
    fragments: list[Any] = []
    parent_ids: list[str] = []
    for part in parts:
        fragment_entry = lookup_alias(part)
        if fragment_entry is None:
            raise ChemistryError(
                f"complex recipe needs resolvable fragments, got {part!r}",
                code="CHEMISTRY_FRAGMENT_UNRESOLVED",
            )
        fragment_mol = _fragment_mol(fragment_entry, request.seed)
        fragments.append(fragment_mol)
        parent_ids.append(fragment_entry.system_id)
    # The heaviest fragment is the host; all smaller fragments are guests.
    fragments.sort(
        key=lambda mol: (mol.GetNumAtoms(), Chem.GetFormalCharge(mol)),
        reverse=True,
    )
    host = fragments[0]
    expected_charge = sum(
        Chem.GetFormalCharge(mol) for mol in fragments
    )
    if request.total_charge is not None and request.total_charge != expected_charge:
        raise ChemistryError(
            f"complex charge {request.total_charge} contradicts the fragment "
            f"sum {expected_charge}",
            code="CHEMISTRY_COMPLEX_CHARGE_MISMATCH",
        )
    candidates: list[Any] = []
    force_fields: list[str] = []
    for guest in fragments[1:]:
        guest_smiles = Chem.MolToSmiles(guest)
        guest_candidates = build_complex_candidates(
            host,
            guest_smiles,
            expected_charge=expected_charge,
            seed=request.seed,
            max_returned=max(1, request.max_candidates // max(len(fragments) - 1, 1)),
        )
        candidates.extend(guest_candidates)
        force_fields.extend(
            candidate.force_field for candidate in guest_candidates
        )
    if not candidates:
        raise ChemistryError(
            f"no complex candidates for {entry.system_id}",
            code="CHEMISTRY_GENERATION_FAILED",
        )
    identity = ChemicalIdentity(
        system_id=entry.system_id,
        display_name=entry.display_name,
        aliases=list(entry.aliases),
        formula=mol_formula(candidates[0].mol),
        smiles=entry.smiles,
        total_charge=expected_charge,
        spin_multiplicity=request.spin_multiplicity,
        role=ChemicalRole.COMPLEX,
    )
    provenance = _provenance(
        status=ProvenanceStatus.HEURISTIC_COMPLEX,
        source="heuristic complex (fragment sum)",
        source_identifier=entry.recipe,
        assumptions=[
            "complex built from fragment charge sum "
            f"({expected_charge:+d})",
            "multiple coordination sites and orientations were sampled",
            "positions seeded at vdW-based distances, collision-filtered",
            "force-field pre-optimised (MMFF/UFF)",
        ],
        confidence=0.7,
        generator="rdkit-complex-builder",
        seed=request.seed,
        parents=parent_ids,
    )
    return _persist_candidates(
        identity,
        candidates,
        output_dir=output_dir,
        provenance_base=provenance,
        validation=["collision-filtered (>=1.4 A)", "FF pre-optimised"],
        conformer_tag="g",
    )


def _fragment_mol(entry: AliasEntry, seed: int) -> Chem.Mol:
    """Resolve one fragment (SMILES monomer or oligomer chain) to a mol."""
    if entry.smiles:
        mol = mol_from_smiles(
            entry.smiles,
            expected_charge=entry.total_charge,
            name=entry.display_name,
        )
        return Chem.AddHs(mol)
    if entry.recipe.startswith("oligomer:"):
        monomers = entry.recipe.split(":", 1)[1].split("+")
        monomer_entries: list[AliasEntry] = []
        for monomer in monomers:
            monomer_entry = lookup_alias(monomer)
            if monomer_entry is None or not monomer_entry.smiles:
                raise ChemistryError(
                    f"oligomer monomer unresolved: {monomer!r}",
                    code="CHEMISTRY_FRAGMENT_UNRESOLVED",
                )
            monomer_entries.append(monomer_entry)
        recipe = OligomerRecipe(
            monomer_smiles=tuple(item.smiles for item in monomer_entries),
            end_caps=("H", "H"),
            crosslink_position="none (single linear chain proxy)",
            assumption_notes=[
                "MBA acts as the crosslinker in the real network; the proxy "
                "keeps MBA as a linear in-chain unit",
            ],
        )
        return build_oligomer(
            recipe, seed=seed, n_conformers=3, max_returned=1
        )
    raise ChemistryError(
        f"fragment {entry.recipe or entry.smiles!r} is not directly "
        "resolvable; nested complex fragments are not supported",
        code="CHEMISTRY_FRAGMENT_UNRESOLVED",
    )


def _resolve_oligomer(
    request: StructureRequest, entry: AliasEntry, output_dir: Path
) -> list[GeneratedStructure]:
    monomers = entry.recipe.split(":", 1)[1].split("+")
    monomer_entries: list[AliasEntry] = []
    for monomer in monomers:
        fragment_entry = lookup_alias(monomer)
        if fragment_entry is None or not fragment_entry.smiles:
            raise ChemistryError(
                f"oligomer monomer unresolved: {monomer!r}",
                code="CHEMISTRY_FRAGMENT_UNRESOLVED",
            )
        monomer_entries.append(fragment_entry)
    recipe = OligomerRecipe(
        monomer_smiles=tuple(item.smiles for item in monomer_entries),
        end_caps=("H", "H"),
        crosslink_position="none (single linear chain proxy)",
        assumption_notes=[
            "MBA acts as the crosslinker in the real network; the proxy "
            "keeps MBA as a linear in-chain unit",
            "exact user polymer connectivity was not provided",
        ],
    )
    chain = build_oligomer(
        recipe, seed=request.seed, max_returned=request.max_candidates
    )
    identity = ChemicalIdentity(
        system_id=entry.system_id,
        display_name=entry.display_name,
        aliases=list(entry.aliases),
        formula=mol_formula(chain),
        total_charge=entry.total_charge,
        spin_multiplicity=request.spin_multiplicity,
        role=ChemicalRole.OLIGOMER,
    )
    # Re-rank candidates like the conformer path (single best chain kept).
    candidates = generate_conformer_candidates(
        chain, n_conformers=6, seed=request.seed,
        max_returned=request.max_candidates,
    )
    provenance = _provenance(
        status=ProvenanceStatus.ASSUMED_REPRESENTATIVE,
        source="representative oligomer",
        source_identifier=entry.recipe,
        assumptions=recipe.assumptions(),
        confidence=0.5,
        generator="rdkit-oligomer-builder",
        seed=request.seed,
        parents=[item.system_id for item in monomer_entries],
    )
    return _persist_candidates(
        identity,
        candidates,
        output_dir=output_dir,
        provenance_base=provenance,
        validation=["finite linear oligomer", "collision-filtered"],
        conformer_tag="p",
    )


def _resolve_proxy(
    request: StructureRequest, output_dir: Path
) -> list[GeneratedStructure]:
    """Explicit proxy marker: no atoms, no fake geometry, study-not-blocking."""
    identity = ChemicalIdentity(
        system_id=request.system_id.strip().lower(),
        display_name=request.display_name or request.system_id,
        aliases=list(request.aliases),
        formula="",
        total_charge=request.total_charge if request.total_charge is not None else 0,
        spin_multiplicity=request.spin_multiplicity,
        role=ChemicalRole.PROXY,
    )
    provenance = _provenance(
        status=ProvenanceStatus.ASSUMED_PROXY,
        source="explicit proxy marker (no structural model)",
        source_identifier=request.system_id,
        assumptions=[
            "no SMILES, structure file or monomer connectivity is available; "
            "the identity is recorded as an explicit proxy only",
            "VASP tasks for this system are skipped (no guess geometry is "
            "invented); the study continues with the other systems",
        ],
        confidence=0.1,
        generator="proxy-marker",
        seed=request.seed,
    )
    marker = output_dir / f"{request.system_id}_PROXY.txt"
    marker.write_text(
        "ASSUMED_PROXY: no structural model; no VASP input is generated.\n",
        encoding="utf-8",
    )
    return [
        _generated(
            identity,
            structure_path=marker,
            symbols=[],
            provenance=provenance,
            validation=["proxy marker; no geometry"],
            force_field_energy=None,
            format="proxy",
        )
    ]


def _resolve_blocked(
    request: StructureRequest, output_dir: Path
) -> list[GeneratedStructure]:
    identity = ChemicalIdentity(
        system_id=request.system_id.strip().lower(),
        display_name=request.display_name or request.system_id,
        aliases=list(request.aliases),
        formula="",
        total_charge=request.total_charge if request.total_charge is not None else 0,
        spin_multiplicity=request.spin_multiplicity,
        role=ChemicalRole.PROXY,
    )
    provenance = _provenance(
        status=ProvenanceStatus.GENERATION_FAILED,
        source="blocked",
        source_identifier=request.system_id,
        assumptions=[
            "structure resolution refused because allow_assumed_structures "
            "is False and no explicit structure/SMILES was provided"
        ],
        confidence=0.0,
        generator="blocked",
        seed=request.seed,
    )
    return [
        _generated(
            identity,
            structure_path=output_dir / f"{request.system_id}_BLOCKED.txt",
            symbols=[],
            provenance=provenance,
            validation=["BLOCKED_MISSING_STRUCTURE"],
            force_field_energy=None,
            format="blocked",
        )
    ]


def validate_generated(
    structure: GeneratedStructure,
    *,
    min_heavy_distance: float = 0.9,
) -> list[str]:
    """Geometry/identity validation for one persisted structure."""
    problems: list[str] = []
    if structure.format not in {"xyz", "sdf", "mol"}:
        return problems  # proxy/blocked markers have no geometry to check
    if not structure.structure_path.is_file():
        problems.append("structure file missing")
        return problems
    try:
        symbols, coords, _ = read_xyz(structure.structure_path)
    except ValueError as exc:
        problems.append(str(exc))
        return problems
    if structure.atom_count and len(symbols) != structure.atom_count:
        problems.append(
            f"atom_count mismatch: manifest {structure.atom_count}, "
            f"file {len(symbols)}"
        )
    if len(coords):
        distances = _heavy_min_distances(symbols, coords)
        if distances is not None and distances < min_heavy_distance:
            problems.append(
                f"severe collision: min heavy-atom distance "
                f"{distances:.2f} A < {min_heavy_distance:.2f} A"
            )
    return problems


def _heavy_min_distances(
    symbols: Sequence[str], coords: Any
) -> float | None:
    import numpy as np

    heavy = [i for i, symbol in enumerate(symbols) if symbol != "H"]
    if len(heavy) < 2:
        return None
    positions = np.asarray(coords, dtype=float)
    minimum = np.inf
    for index, left in enumerate(heavy):
        if index + 1 >= len(heavy):
            break
        diffs = positions[heavy[index + 1 :]] - positions[left]
        distances = np.linalg.norm(diffs, axis=1)
        bonded_mask = _bonded_pairs_mask(symbols, coords, heavy, index)
        non_bonded = distances[~bonded_mask]
        if non_bonded.size:
            minimum = min(minimum, float(non_bonded.min()))
    return float(minimum)


def _bonded_pairs_mask(
    symbols: Sequence[str],
    coords: Any,
    heavy: list[int],
    left_index: int,
) -> Any:
    """Boolean mask: True where a heavy pair is separated by < 1.35 A
    (covalent single-bond proxy without bond-order information)."""
    import numpy as np

    positions = np.asarray(coords, dtype=float)
    left = positions[heavy[left_index]]
    others = positions[heavy[left_index + 1 :]]
    distances = np.linalg.norm(others - left, axis=1)
    return distances < 1.35


def _hill_formula(counts: dict[str, int]) -> str:
    """Hill order: C, H, then the rest alphabetically."""
    order = ["C", "H"] + sorted(
        element for element in counts if element not in {"C", "H"}
    )
    parts = []
    for element in order:
        count = counts.get(element)
        if count:
            parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)
