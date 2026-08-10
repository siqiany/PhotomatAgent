"""Vertical Slice 1 — LWIR 8-14 um screening (scripted, real tool execution)."""

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
                "Step 1 — Load the domain skill. Known: task is an 8-14 um LWIR "
                "detector screening. Uncertain: constraints and candidate space. "
                "Missing: physical constraints. Next: load the IR design skill and "
                "compile constraints."
            ),
            tool_calls=[scripted_call("skill_view", {"name": "infrared-photodetector-design"})],
        ),
        ScriptedStep(
            reasoning=(
                "Step 2 — Compile deterministic constraints. Known: band is 8-14 um. "
                "Uncertain: cutoff energy, thermal limits, ideal performance. "
                "Missing: none (deterministic). Next: compile constraints."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "ir.compile_constraints",
                        "arguments": {
                            "spectral_min_um": 8,
                            "spectral_max_um": 14,
                            "detector_type": "photodiode",
                        },
                    },
                )
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 3 — Literature search. Known: cutoff gap ~0.089 eV. "
                "Uncertain: which materials/literature exist. Missing: recent "
                "candidate evidence. Next: search arXiv."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "literature.search_arxiv",
                        "arguments": {
                            "query": "LWIR photodetector HgCdTe dark current",
                            "max_results": 3,
                        },
                    },
                )
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 4 — Database probe. Known: literature candidates. "
                "Uncertain: database availability (needs API key). Missing: "
                "structured database evidence. Next: probe materials.search; "
                "an unconfigured key must be reported, not fabricated."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "materials.search",
                        "arguments": {"formula": "HgTe"},
                    },
                )
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 5 — Structure analysis of a candidate. Known: InAs is a "
                "narrow-gap III-V. Uncertain: whether its gap (~0.35 eV) covers "
                "14 um (it does not: cutoff is 0.089 eV). Missing: absorption and "
                "transport evidence. Next: cheap structure analysis."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "structure.summary",
                        "arguments": {"path": str(structures["InAs"])},
                    },
                )
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 6 — Final evidence-grounded recommendation. Known: cutoff "
                "gap 0.089 eV; InAs gap ~0.35 eV does not reach 14 um; literature "
                "points to HgCdTe/type-II superlattices. Uncertain: transport and "
                "defect behavior of any candidate. Missing: (a) Materials Project "
                "database evidence (key unconfigured), (b) absorption spectra, "
                "(c) transport/defect/device data. Next investigation: HgTe/CdTe "
                "narrow-gap analysis (slice 2), then DFT via AtomisticSkills "
                "VASP SOP when justified."
            ),
            tool_calls=[],
        ),
    ]

    result = asyncio.run(
        run_scripted(
            goal="Design / screen candidates for an 8-14 um infrared photodetector.",
            steps=steps,
            workspace_root=repo,
            session_dir=default_session_dir(),
        )
    )
    path = save_result(result, repo / "output" / "vertical" / "slice_1_lwir_screening.json")
    print(f"slice 1 saved: {path}")
    print(f"evidence: {result['evidence_count']}")


if __name__ == "__main__":
    main()

