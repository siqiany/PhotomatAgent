"""Low-fidelity band-alignment tools (namespace ``interface``).

Anderson electron-affinity rule for conduction/valence band offsets at a
heterojunction. LOW FIDELITY by design: interface dipoles, surface
chemistry, ligand layers, Fermi-level pinning, and interface reconstruction
are all ignored. Never a substitute for an interface calculation.

Vacuum-level convention (Sprint 3 correctness hotfix)
-----------------------------------------------------
The electron affinity ``chi`` is the energy an electron needs to escape from
the conduction-band minimum to the vacuum level. Taking the vacuum level as
the energy zero:

    Ec = -chi
    Ev = -(chi + Eg)

Therefore, for materials A and B:

    CBO = Ec(A) - Ec(B) = chi_B - chi_A
    VBO = Ev(A) - Ev(B) = CBO + (Eg_B - Eg_A)

A positive CBO means B's conduction-band minimum lies *below* A's, i.e.
electrons are collected in B; a positive VBO means B's valence-band maximum
lies below A's, i.e. holes are collected in B. The old implementation used
``Ec = chi`` / ``Ev = chi + Eg`` which silently inverted the sign of every
offset; this module now follows the vacuum-level convention above.
"""

from __future__ import annotations

import json
from typing import Any

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure


class InterfaceProbe(CapabilityPack):
    name = "interface"
    description = "Low-fidelity Anderson-rule band alignment."

    def probe(self) -> ProbeResult:
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            detail="pure Python; LOW FIDELITY by design",
            version="1.0",
        )

    def tools(self) -> list[Tool]:
        return [AndersonBandAlignmentTool()]


class AndersonBandAlignmentTool(Tool):
    name = "interface.anderson_band_alignment"
    description = (
        "LOW FIDELITY Anderson-rule band alignment between two materials A "
        "and B using the vacuum-level convention Ec = -chi, Ev = -(chi + Eg): "
        "CBO = Ec(A) - Ec(B) = chi_B - chi_A, VBO = Ev(A) - Ev(B) = CBO + "
        "(Eg_B - Eg_A). A positive CBO collects electrons in B; a positive "
        "VBO collects holes in B. Reports Type I (straddling) / Type II "
        "(staggered) / Type III (broken gap) alignment, including the "
        "Anderson-rule limitations. Ignores interface dipoles, surface "
        "chemistry, ligand effects, Fermi-level pinning, and interface "
        "reconstruction. Do NOT use as a design-grade interface model."
    )
    short_description = (
        "Anderson-rule band offsets, vacuum-level convention (LOW FIDELITY)."
    )
    exposure = ToolExposure.DEFERRED
    namespace = "interface"
    source = "native analytical model"
    tags = ("interface", "band alignment", "anderson", "heterojunction")
    input_schema = {
        "type": "object",
        "properties": {
            "electron_affinity_a_eV": {"type": "number"},
            "band_gap_a_eV": {"type": "number", "minimum": 0},
            "electron_affinity_b_eV": {"type": "number"},
            "band_gap_b_eV": {"type": "number", "minimum": 0},
        },
        "required": [
            "electron_affinity_a_eV",
            "band_gap_a_eV",
            "electron_affinity_b_eV",
            "band_gap_b_eV",
        ],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        try:
            chi_a = float(arguments["electron_affinity_a_eV"])
            eg_a = float(arguments["band_gap_a_eV"])
            chi_b = float(arguments["electron_affinity_b_eV"])
            eg_b = float(arguments["band_gap_b_eV"])
        except (TypeError, ValueError):
            return _invalid("all inputs must be numbers")
        # Vacuum-level convention: vacuum = 0, Ec = -chi, Ev = -(chi + Eg).
        ec_a, ev_a = -chi_a, -(chi_a + eg_a)
        ec_b, ev_b = -chi_b, -(chi_b + eg_b)
        cbo = ec_a - ec_b  # = chi_B - chi_A; positive -> electrons collected in B
        vbo = ev_a - ev_b  # = CBO + (Eg_B - Eg_A); positive -> holes collected in B
        broken = ev_a > ec_b or ev_b > ec_a
        if broken:
            alignment = "Type III (broken gap)"
        elif abs(cbo) < 1e-12 and abs(vbo) < 1e-12:
            alignment = "Type I (straddling; identical materials, zero offsets)"
        elif cbo * vbo < 0:
            alignment = "Type I (straddling)"
        else:
            alignment = "Type II (staggered)"
        payload = {
            "cbo_eV": round(cbo, 4),
            "vbo_eV": round(vbo, 4),
            "alignment_type": alignment,
            "convention": (
                "vacuum level = 0 eV; Ec = -chi; Ev = -(chi + Eg); "
                "CBO = Ec(A) - Ec(B) = chi_B - chi_A; "
                "VBO = Ev(A) - Ev(B) = CBO + (Eg_B - Eg_A); positive CBO "
                "means electrons are collected in B; positive VBO means "
                "holes are collected in B"
            ),
            "vacuum_level_eV": 0.0,
            "conduction_band_edges_eV": {
                "Ec_A": round(ec_a, 4),
                "Ec_B": round(ec_b, 4),
            },
            "valence_band_edges_eV": {
                "Ev_A": round(ev_a, 4),
                "Ev_B": round(ev_b, 4),
            },
            "fidelity": "analytical",
            "fidelity_note": "LOW FIDELITY",
            "assumptions": [
                "Anderson electron-affinity rule",
                "vacuum-level convention Ec = -chi, Ev = -(chi + Eg)",
                "interface dipoles ignored",
                "surface chemistry and ligand effects ignored",
                "Fermi-level pinning ignored",
                "interface reconstruction ignored",
            ],
            "limitations": [
                "electron affinity values depend on surface termination and "
                "measurement method; reported chi are bulk values",
                "no strain effects on band edges",
                "no temperature dependence",
                "no interface dipole or Fermi-level pinning",
                "Type III classification only indicates overlapping gaps in "
                "this rule; a real broken-gap junction requires a "
                "higher-fidelity interface calculation",
            ],
        }
        evidence = [
            ScientificEvidence(
                subject="interface_A_B",
                property="conduction_band_offset",
                value=cbo,
                unit="eV",
                source="photomatagent native analytical model",
                source_type="analytical_model",
                method=(
                    "Anderson electron-affinity rule, vacuum-level convention"
                ),
                fidelity="analytical",
                summary=f"LOW FIDELITY CBO = {cbo:.3f} eV (Anderson rule)",
                limitations=(
                    "interface dipoles, pinning, chemistry, strain ignored"
                ),
                provenance={"tool": self.name, "convention": "vacuum_level"},
            )
        ]
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            evidence=evidence,
        )


def _invalid(message: str) -> ScientificToolResult:
    return ScientificToolResult(
        output=message,
        is_error=True,
        data={"error_type": "invalid_input", "message": message},
    )


def interface_pack() -> CapabilityPack:
    return InterfaceProbe()
