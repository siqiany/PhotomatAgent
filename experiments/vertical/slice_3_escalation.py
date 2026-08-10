"""Vertical Slice 3 — tool escalation with partial evidence (scripted, real tools)."""

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


def main() -> None:
    repo = Path(__file__).resolve().parents[2]

    steps = [
        ScriptedStep(
            reasoning=(
                "Step 1 — Candidate has only database evidence (band gap). "
                "Known: gap matches the IR band. Uncertain: transport, defects, "
                "device behavior. Missing: all non-electronic evidence. Next: "
                "discover the right deferred capabilities."
            ),
            tool_calls=[scripted_call("tool_search", {"query": "carrier mobility"})],
        ),
        ScriptedStep(
            reasoning=(
                "Step 2 — Discovered transport capability. Known: transport.analyze "
                "exists. Uncertain: its inputs. Missing: details. Next: describe it."
            ),
            tool_calls=[scripted_call("tool_describe", {"name": "transport.analyze"})],
        ),
        ScriptedStep(
            reasoning=(
                "Step 3 — Attempt transport analysis. Known: tool exists and "
                "expects vasprun.xml. Uncertain: whether DFT inputs exist. "
                "Missing: deformation/dielectric data. Next: call it; a missing "
                "dependency must be reported, not hallucinated."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "transport.analyze",
                        "arguments": {"path": "output/scientific/vasprun.xml"},
                    },
                )
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 4 — Defect capability. Known: transport blocked on inputs. "
                "Uncertain: defect landscape. Missing: defect evidence. Next: "
                "discover defects capability and check prerequisites."
            ),
            tool_calls=[
                scripted_call("tool_search", {"query": "defect formation energy"}),
                scripted_call(
                    "tool_call",
                    {
                        "name": "defects.analyze",
                        "arguments": {
                            "defects_json": "output/scientific/defects.json",
                            "bulk_vasprun": "output/scientific/vasprun.xml",
                        },
                    },
                ),
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 5 — Device capability. Known: defect analysis blocked on "
                "DFT inputs. Uncertain: device performance. Missing: device "
                "evidence. Next: discover device simulation and check prerequisites."
            ),
            tool_calls=[
                scripted_call("tool_search", {"query": "semiconductor device"}),
                scripted_call(
                    "tool_call",
                    {
                        "name": "device.run_script",
                        "arguments": {"path": "output/scientific/device_model.py"},
                    },
                ),
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Step 6 — Final report. Known: the candidate's band gap matches "
                "the band; the correct deferred capabilities were discovered "
                "(transport.analyze, defects.analyze, device.run_script). "
                "Uncertain: mobility, defect formation energies, device dark "
                "current. Missing: DFT inputs for transport/defects (vasprun.xml, "
                "deformation potentials, defects.json) and a DEVSIM device script "
                "with material parameters; AMSET/doped/devsim dependencies must "
                "be installed. No mobility, formation energy, or device current "
                "value is fabricated. Next step: run DFT (AtomisticSkills VASP "
                "SOP) to produce the transport/defect inputs, in order of "
                "cheapest sufficient evidence."
            ),
            tool_calls=[],
        ),
    ]

    result = asyncio.run(
        run_scripted(
            goal=(
                "A candidate material has only a database band-gap match for IR "
                "photodetection. Identify what transport/defect/device evidence "
                "is missing and which capabilities would produce it."
            ),
            steps=steps,
            workspace_root=repo,
            session_dir=default_session_dir(),
        )
    )
    path = save_result(result, repo / "output" / "vertical" / "slice_3_escalation.json")
    print(f"slice 3 saved: {path}")
    print(f"evidence: {result['evidence_count']}")


if __name__ == "__main__":
    main()

