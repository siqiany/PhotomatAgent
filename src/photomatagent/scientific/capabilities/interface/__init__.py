"""Low-fidelity band-alignment tools (namespace ``interface``).

Anderson electron-affinity rule for conduction/valence band offsets at a
heterojunction. LOW FIDELITY by design: interface dipoles, surface
chemistry, ligand layers, Fermi-level pinning, and interface reconstruction
are all ignored. Never a substitute for an interface calculation.
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
        "and B: CBO = chi_A - chi_B, VBO = CBO + (Eg_A - Eg_B). Reports "
        "Type I (straddling) / Type II (staggered) / Type III (broken) "
        "alignment. Ignores interface dipoles, surface chemistry, ligand "
        "effects, Fermi-level pinning, and interface reconstruction. Do NOT "
        "use as a design-grade interface model."
    )
    short_description = "Anderson-rule band offsets (LOW FIDELITY)."
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
        ec_a, ev_a = chi_a, chi_a + eg_a
        ec_b, ev_b = chi_b, chi_b + eg_b
        cbo = ec_a - ec_b  # positive: electrons fall from A into B
        vbo = ev_a - ev_b
        broken = ev_a > ec_b or ev_b > ec_a
        if broken:
            alignment = "Type III (broken gap)"
        elif cbo * vbo > 0:
            alignment = "Type I (straddling)"
        else:
            alignment = "Type II (staggered)"
        payload = {
            "cbo_eV": round(cbo, 4),
            "vbo_eV": round(vbo, 4),
            "alignment_type": alignment,
            "convention": (
                "CBO = Ec(A) - Ec(B); VBO = Ev(A) - Ev(B); positive CBO means "
                "electrons are collected in B"
            ),
            "fidelity": "analytical",
            "fidelity_note": "LOW FIDELITY",
            "assumptions": [
                "Anderson electron-affinity rule",
                "interface dipoles ignored",
                "surface chemistry and ligand effects ignored",
                "Fermi-level pinning ignored",
                "interface reconstruction ignored",
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
                method="Anderson electron-affinity rule",
                fidelity="analytical",
                summary=f"LOW FIDELITY CBO = {cbo:.3f} eV (Anderson rule)",
                limitations="interface dipoles, pinning, chemistry ignored",
                provenance={"tool": self.name},
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
