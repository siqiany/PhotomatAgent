"""Vertical Slice 2 — HgTe / narrow-gap analysis (scripted, real tools)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from slice_runner import (
    ScriptedStep,
    run_scripted,
    save_result,
    scripted_call,
    default_session_dir,
)
from structures import generate_structures


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    output_dir = repo / "output" / "scientific"
    structures = generate_structures(output_dir)

    steps = [
        ScriptedStep(
            reasoning=(
                "Step 1 — Load the narrow-gap skill. Known: HgTe is a zero-gap "
                "semimetal candidate for IR. Uncertain: whether it deserves "
                "further investigation. Missing: electronic, defect, transport, "
                "optical, and device evidence. Next: load skill + constraints."
            ),
            tool_calls=[scripted_call("skill_view", {"name": "narrow-gap-electronic-analysis"})],
        ),
        ScriptedStep(
            reasoning=(
                "Step 2 — Constraints for 8-14 um. Known: cutoff gap 0.089 eV. "
                "Uncertain: HgTe's zero-gap vs cutoff. Missing: none. Next: "
                "compile constraints."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "ir.compile_constraints",
                        "arguments": {"spectral_min_um": 8, "spectral_max_um": 14},
                    },
                )
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 3 — Structure analysis of HgTe (real pymatgen). Known: "
                "zinc-blende HgTe lattice. Uncertain: symmetry/density/coordination. "
                "Missing: none for structure. Next: run structure tools."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "structure.summary",
                        "arguments": {"path": str(structures["HgTe"])},
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "structure.symmetry",
                        "arguments": {"path": str(structures["HgTe"])},
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "structure.density",
                        "arguments": {"path": str(structures["HgTe"])},
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "structure.neighbors",
                        "arguments": {"path": str(structures["HgTe"]), "element": "Te"},
                    },
                ),
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 4 — Literature for HgTe detectors. Known: structure results. "
                "Uncertain: reported HgTe photodetector evidence. Missing: "
                "literature. Next: arXiv search."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "literature.search_arxiv",
                        "arguments": {"query": "HgTe infrared photodetector", "max_results": 3},
                    },
                )
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 5 — Transport capability discovery. Known: zero-gap means "
                "carrier transport is unusual. Uncertain: mobility. Missing: "
                "transport analysis inputs. Next: discover transport tool and "
                "check prerequisites honestly."
            ),
            tool_calls=[
                scripted_call("tool_search", {"query": "carrier mobility"}),
                scripted_call(
                    "tool_call",
                    {
                        "name": "transport.analyze",
                        "arguments": {"path": "output/scientific/vasprun.xml"},
                    },
                ),
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 6 — Final assessment. Known: HgTe is zinc-blende Fm-3m, "
                "zero-gap, structurally sound; literature shows active HgTe "
                "detector research; transport analysis correctly reports missing "
                "AMSET inputs. Uncertain: electronic structure details (gap "
                "character, band edges), defect behavior, device dark current. "
                "Missing: DFT band structure (HgTe zero-gap needs hybrid/GW "
                "corrections), transport (deformation potentials), defect "
                "formation energies, device-level dark current. "
                "Next highest-value computation: DFT band structure of HgTe with "
                "SOC (VASP SOP from AtomisticSkills), because the zero-gap "
                "electronic structure gates every downstream analysis — a "
                "band-gap match alone would NOT justify an excellent-detector "
                "conclusion."
            ),
            tool_calls=[],
        ),
    ]

    result = asyncio.run(
        run_scripted(
            goal=(
                "Analyze whether HgTe or a related candidate deserves further "
                "investigation for infrared photodetection and identify the next "
                "highest-value computation."
            ),
            steps=steps,
            workspace_root=repo,
            session_dir=default_session_dir(),
        )
    )
    path = save_result(result, repo / "output" / "vertical" / "slice_2_hgte.json")
    print(f"slice 2 saved: {path}")
    print(f"evidence: {result['evidence_count']}")


if __name__ == "__main__":
    main()

