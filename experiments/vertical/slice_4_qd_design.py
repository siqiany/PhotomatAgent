"""Vertical Slice 4 — 2-5 um quantum-dot IR detector design (Sprint 2 demos).

Demos B, C, D run scripted against the real deterministic tools:

* Demo B — QD analytical sweep with sourced parameters (InAs from the local
  parameter registry, never LLM-supplied).
* Demo C — inverse target chain for 2-5 um: ir.compile_constraints ->
  alloy bowing -> qd.solve_size_for_transition / qd.screen_size_composition
  -> photodetector.check_targets.
* Demo D — insufficient evidence: typed missing_prerequisites failures for
  parameter-less calls (no fabricated numbers).

Demo A (Materials Project) runs in two variants: native ``materials.search``
(real API) and the MCP gateway (``mcp status`` / ``mcp test``); the MCP
variant needs a normal Linux host (see ``demo_a_mcp.py``).

Run:  MPLCONFIGDIR=/tmp/mpl .venv/bin/python experiments/vertical/slice_4_qd_design.py
"""

from __future__ import annotations

import asyncio
import json
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
        # ---- Demo D first: the honest refusal path -------------------------
        ScriptedStep(
            reasoning=(
                "Demo D — Insufficient evidence. The user asks for an exact "
                "HgTe QD diameter, levels, and device response with no "
                "parameters. Correct behavior: derive what is derivable, "
                "request the missing inputs, refuse to fabricate."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.brus_transition_energy",
                        "arguments": {},  # missing everything
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.brus_transition_energy",
                        "arguments": {
                            "radius_nm": 3.0,
                            "bulk_band_gap_eV": -0.302,
                            "electron_effective_mass_m0": None,
                            "hole_effective_mass_m0": 0.5,
                            "include_coulomb_term": True,
                        },  # missing mass + dielectric
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.solve_size_for_transition",
                        "arguments": {
                            "bulk_band_gap_eV": 0.31,
                            "electron_effective_mass_m0": 0.2,
                            "hole_effective_mass_m0": 0.24,
                            "relative_dielectric_constant": 414.0,
                            "include_coulomb_term": True,
                        },  # no target energy/wavelength
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.screen_size_composition",
                        "arguments": {
                            "target_wavelength_min_um": 2.0,
                            "target_wavelength_max_um": 5.0,
                        },  # everything else missing
                    },
                ),
            ],
        ),
        # ---- Demo B: sourced-parameter analytical sweep ---------------------
        ScriptedStep(
            reasoning=(
                "Demo B — QD analytical sweep with provenance. Parameters come "
                "from the local registry (InAs, Vurgaftman & Meyer 2001) — "
                "the LLM does not supply them."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.parameter_lookup",
                        "arguments": {"material": "InAs"},
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.size_sweep",
                        "arguments": {
                            "min_size_nm": 2.0,
                            "max_size_nm": 12.0,
                            "points": 11,
                            "bulk_band_gap_eV": 0.354,
                            "electron_effective_mass_m0": 0.026,
                            "hole_effective_mass_m0": 0.41,
                            "relative_dielectric_constant": 15.15,
                            "include_coulomb_term": True,
                        },
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.excitonic_regime",
                        "arguments": {
                            "radius_nm": 3.0,
                            "electron_effective_mass_m0": 0.026,
                            "hole_effective_mass_m0": 0.41,
                            "relative_dielectric_constant": 15.15,
                        },
                    },
                ),
            ],
        ),
        # ---- Demo C: 2-5 um inverse chain ----------------------------------
        ScriptedStep(
            reasoning=(
                "Demo C — 2-5 um inverse design chain. First compile the "
                "spectral constraints, then solve the PbTe QD size for "
                "lambda = 3 um (E = 0.413 eV), then check R/EQE targets."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "ir.compile_constraints",
                        "arguments": {
                            "spectral_min_um": 2.0,
                            "spectral_max_um": 5.0,
                            "detector_type": "photodiode",
                        },
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.parameter_lookup",
                        "arguments": {"material": "PbTe"},
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.solve_size_for_transition",
                        "arguments": {
                            "target_wavelength_um": 3.0,
                            "bulk_band_gap_eV": 0.31,
                            "electron_effective_mass_m0": 0.20,
                            "hole_effective_mass_m0": 0.24,
                            "relative_dielectric_constant": 414.0,
                            "include_coulomb_term": True,
                        },
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.excitonic_regime",
                        "arguments": {
                            "radius_nm": 6.0,
                            "electron_effective_mass_m0": 0.20,
                            "hole_effective_mass_m0": 0.24,
                            "relative_dielectric_constant": 414.0,
                        },
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "alloy.bandgap_bowing",
                        "arguments": {
                            "x": 0.3,
                            "band_gap_a_eV": 0.31,
                            "band_gap_b_eV": 1.475,
                            "bowing_parameter_eV": -0.28,
                        },
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "qd.screen_size_composition",
                        "arguments": {
                            "target_wavelength_min_um": 2.0,
                            "target_wavelength_max_um": 5.0,
                            "composition_min": 0.0,
                            "composition_max": 0.4,
                            "composition_points": 9,
                            "radius_min_nm": 3.0,
                            "radius_max_nm": 12.0,
                            "radius_points": 10,
                            "band_gap_a_eV": 0.31,
                            "band_gap_b_eV": 1.475,
                            "bowing_parameter_eV": -0.28,
                            "electron_mass_a_m0": 0.20,
                            "hole_mass_a_m0": 0.24,
                            "electron_mass_b_m0": 0.09,
                            "hole_mass_b_m0": 0.6,
                            "relative_dielectric_constant": 200.0,
                            "include_coulomb_term": True,
                        },
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "photodetector.check_targets",
                        "arguments": {
                            "spectral_min_um": 2.0,
                            "spectral_max_um": 5.0,
                            "target_responsivity_a_w": 0.8,
                            "eqe_percent": 20.0,
                        },
                    },
                ),
            ],
        ),
        # ---- Demo A: native Materials Project (real database) ----------------
        ScriptedStep(
            reasoning=(
                "Demo A (native) — Materials Project database evidence for "
                "HgTe / InAs / PbTe. The MCP variant is exercised via "
                "`photomatagent mcp status` on a normal Linux host."
            ),
            tool_calls=[
                scripted_call(
                    "tool_call",
                    {
                        "name": "materials.search",
                        "arguments": {"formula": "HgTe", "limit": 3},
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "materials.search",
                        "arguments": {"formula": "InAs", "limit": 3},
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "materials.search",
                        "arguments": {"formula": "PbTe", "limit": 3},
                    },
                ),
                scripted_call(
                    "tool_call",
                    {
                        "name": "materials_mcp.status",
                        "arguments": {},
                    },
                ),
            ],
        ),
        ScriptedStep(
            reasoning=(
                "Synthesis — 2-5 um QD detector: constraints compiled; PbTe "
                "L1 candidate sizes solved; R/EQE targets checked for "
                "consistency; database gaps recorded. Missing high-fidelity "
                "evidence: 3D QD electronic states, Si interface, absorption, "
                "transport, device simulation. No fabricated design values."
            )
        ),
    ]
    result = asyncio.run(
        run_scripted(
            goal=(
                "Design a QD material for a 2-5 um Si-integrated IR detector "
                "with R >= 0.8 A/W and EQE >= 20%; use deterministic tools "
                "only and report missing evidence."
            ),
            steps=steps,
            workspace_root=repo,
            session_dir=default_session_dir(),
        )
    )
    save_result(result, repo / "output" / "vertical" / "slice_4_qd_design.json")
    print(json.dumps(result["final_answer"], ensure_ascii=False, indent=2))
    print(f"evidence_count={result['evidence_count']}")
    for item in result["evidence"]:
        print(
            f"  [{item['source_type']}] {item['subject']} "
            f"{item['property']} = {item['value']} {item['unit']} — "
            f"{item['summary'][:80]}"
        )


if __name__ == "__main__":
    main()
