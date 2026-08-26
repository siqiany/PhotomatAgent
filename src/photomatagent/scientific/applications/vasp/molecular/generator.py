"""Offline input generation for isolated-molecule VASP stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.molecular.models import WorkflowSpec
from photomatagent.scientific.applications.vasp.molecular.psp_metadata import (
    PspError,
    resolve_potcar_metadata,
)
from photomatagent.scientific.applications.vasp.molecular.render import (
    render_incar,
    render_kpoints_gamma,
)
from photomatagent.scientific.applications.vasp.molecular.structures import (
    center_in_cubic_box,
    grouped_symbols,
    read_structure,
    reorder_positions,
)
from photomatagent.scientific.applications.vasp.psp import (
    resolve_local_psp_library,
)


class MolecularVaspGenerator:
    """Generate a prepared molecular workflow tree (never submits)."""

    def __init__(self, psp_dir: str | Path | None = None) -> None:
        self.psp_dir = Path(psp_dir).expanduser().resolve() if psp_dir else None

    def generate(
        self,
        workflow: WorkflowSpec,
        output_root: str | Path,
        *,
        write_potcar: bool = False,
    ) -> dict[str, Any]:
        """Write every stage into ``output_root`` and run the preflight."""
        molecule = workflow.molecule
        if molecule.structure_path is None:
            raise ValueError(
                "cannot generate inputs without a structure file "
                "(BLOCKED_MISSING_STRUCTURE)"
            )
        structure = read_structure(
            molecule.structure_path,
            kind=molecule.structure_kind,
            conformer_index=_conformer_index(molecule.conformer_id),
        )
        elements, counts = grouped_symbols(structure.symbols)
        positions = center_in_cubic_box(
            reorder_positions(structure.symbols, structure.positions, elements),
            molecule.box_ang,
        )
        frac = positions / molecule.box_ang
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        resolution = None
        try:
            resolution = resolve_potcar_metadata(
                molecule, elements, psp_dir=self.psp_dir
            )
        except PspError:
            resolution = None

        neutral_electrons = None
        nelect = None
        if resolution is not None:
            neutral_electrons = sum(
                block.zval * count
                for block, count in zip(resolution.blocks, counts, strict=True)
            )
            nelect = neutral_electrons - molecule.total_charge

        stage_dirs: dict[Any, Path] = {}
        for index, stage in enumerate(workflow.stages, start=1):
            stage_dir = root / f"{index:02d}_{stage.name.value}"
            stage_dir.mkdir(exist_ok=True)
            _write_poscar(
                stage_dir / "POSCAR",
                name=f"{molecule.name} isolated molecule",
                elements=elements,
                counts=counts,
                frac=frac,
                box_ang=molecule.box_ang,
            )
            (stage_dir / "INCAR").write_text(
                render_incar(stage.incar), encoding="utf-8"
            )
            (stage_dir / "KPOINTS").write_text(
                render_kpoints_gamma(), encoding="utf-8"
            )
            if resolution is not None:
                _write_potcar_meta(stage_dir, elements, resolution, nelect)
            _write_potcar_policy(
                stage_dir,
                elements,
                materialized=write_potcar,
                psp_dir=self.psp_dir,
            )
            if write_potcar:
                if self.psp_dir is None or resolution is None:
                    raise PspError(
                        "cannot materialize POTCAR without a psp_dir",
                        code="PSP_UNRESOLVED",
                    )
                _assemble_potcar(stage_dir / "POTCAR", self.psp_dir, elements)
            stage_dirs[stage.name] = stage_dir

        manifest = {
            "scientific_method": workflow.scientific_method,
            "molecule": molecule.model_dump(mode="json"),
            "stages": [
                {
                    "name": stage.name.value,
                    "depends_on": (
                        stage.depends_on.value if stage.depends_on else None
                    ),
                    "required_upstream_outputs": stage.required_upstream_outputs,
                    "produced_outputs": stage.produced_outputs,
                    "resource_class": stage.resource_class.value,
                    "validator": stage.validator,
                    "directory": f"{index:02d}_{stage.name.value}",
                    "incar_keys": sorted(stage.incar),
                }
                for index, stage in enumerate(workflow.stages, start=1)
            ],
            "resource_ceiling": workflow.resource_ceiling.model_dump(mode="json"),
            "correction_policy": workflow.correction_policy.model_dump(mode="json"),
            "potcar_materialized": write_potcar,
            "potcar": (
                {
                    **resolution.metadata_summary(),
                    "neutral_valence_electrons": neutral_electrons,
                    "nelect": nelect,
                }
                if resolution is not None
                else {"note": "POTCAR metadata unresolved at generation time"}
            ),
            "note": (
                "prepared offline; never submitted. POTCAR content is only "
                "written when explicitly materialized for a local licensed run"
            ),
        }
        (root / "workflow.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        from photomatagent.scientific.applications.vasp.molecular.preflight import (
            run_molecular_preflight,
        )

        report = run_molecular_preflight(
            workflow,
            psp_dir=self.psp_dir,
            stage_dirs=stage_dirs,
            output_dir=root,
        )
        return {
            "output_root": str(root),
            "workflow": manifest,
            "preflight": report.model_dump(mode="json"),
        }


def _conformer_index(conformer_id: str | None) -> int:
    if conformer_id is not None and conformer_id.isdigit():
        return int(conformer_id)
    return 0


def _write_poscar(
    path: Path,
    *,
    name: str,
    elements: list[str],
    counts: list[int],
    frac: Any,
    box_ang: float,
) -> None:
    lines = [
        name,
        "1.0",
        f"{box_ang:.10f} 0.0 0.0",
        f"0.0 {box_ang:.10f} 0.0",
        f"0.0 0.0 {box_ang:.10f}",
        " ".join(elements),
        " ".join(str(count) for count in counts),
        "Direct",
    ]
    lines.extend(" ".join(f"{value:.12f}" for value in row) for row in frac)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_potcar_meta(
    stage_dir: Path,
    elements: list[str],
    resolution: Any,
    nelect: float | None,
) -> None:
    """Write the metadata-only POTCAR.meta (never POTCAR content)."""
    data = resolution.metadata_summary()
    data.update({"sequence": elements, "nelect": nelect})
    (stage_dir / "POTCAR.meta").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_potcar_policy(
    stage_dir: Path,
    elements: list[str],
    *,
    materialized: bool,
    psp_dir: Path | None,
) -> None:
    resolved = resolve_local_psp_library(psp_dir) if psp_dir is not None else None
    lines = [
        "# POTCAR policy: POTCAR content is never committed, logged or",
        "# returned to the model; only TITEL/ZVAL/ENMAX metadata is recorded.",
        "profile: isolated-molecule",
        f"materialized: {'true' if materialized else 'false'}",
        f"sequence: {' '.join(elements)}",
        (
            f"psp_dir: {psp_dir}"
            if psp_dir
            else "psp_dir: (not configured)"
        ),
        (
            f"resolved_library: {resolved[0]} ({resolved[1]})"
            if resolved is not None
            else "resolved_library: (unresolved)"
        ),
        "resolution: <element>/POTCAR in POSCAR element order",
    ]
    (stage_dir / "POTCAR.policy").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _assemble_potcar(target: Path, psp_dir: Path, elements: list[str]) -> None:
    """Concatenate the user's licensed datasets (explicit local-run request)."""
    resolved = resolve_local_psp_library(psp_dir)
    if resolved is None:
        raise PspError(
            "cannot materialize POTCAR: no known PAW-PBE layout under "
            f"{psp_dir} (expected <root>/<element>/POTCAR, "
            "<root>/potpaw_PBE/<element>/POTCAR or "
            "<root>/potpaw_PBE.64/<element>/POTCAR)",
            code="PSP_UNRESOLVED",
        )
    library, _ = resolved
    with target.open("wb") as destination:
        for element in elements:
            source = library / element / "POTCAR"
            if not source.is_file():
                raise PspError(
                    f"missing PAW-PBE dataset for {element}: {source}",
                    code="PSP_DATASET_MISSING",
                )
            destination.write(source.read_bytes())
