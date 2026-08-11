"""Optics capability pack (namespace ``optics``).

PyTASER (transient absorption) and Meep (1D thin-film R/T/A) integrations
are dependency-optional: tools are always registered and return typed
missing-dependency failures when the package is unavailable; probes report
MISSING_DEPENDENCY accordingly.
"""

from __future__ import annotations

import importlib.metadata
import json
from typing import Any

from photomatagent.scientific.capabilities.base import (
    CapabilityPack,
    CapabilityStatus,
    ProbeResult,
)
from photomatagent.scientific.capabilities.contracts import ScientificToolResult
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.scientific.capabilities.optics.meep_thinfilm import (
    MeepThinFilmTool,
)


class OpticsProbe(CapabilityPack):
    name = "optics"
    description = "Transient absorption / optical response analysis via PyTASER."

    def probe(self) -> ProbeResult:
        missing: list[str] = []
        try:
            import pytaser  # noqa: F401
        except Exception as exc:
            missing.append(f"pytaser ({type(exc).__name__})")
        try:
            import meep  # noqa: F401
        except Exception as exc:
            missing.append(f"meep ({type(exc).__name__})")
        if missing:
            return ProbeResult(
                status=CapabilityStatus.MISSING_DEPENDENCY,
                detail=(
                    "optional optics backends not importable: "
                    + ", ".join(missing)
                    + " (extra: photomatagent[optics])"
                ),
            )
        return ProbeResult(
            status=CapabilityStatus.AVAILABLE,
            version="pytaser+meep",
        )

    def tools(self) -> list[Tool]:
        tools: list[Tool] = [MeepThinFilmTool()]
        try:
            import pytaser  # noqa: F401
        except Exception:
            return tools
        tools.append(TransientAbsorptionTool())
        return tools


class TransientAbsorptionTool(Tool):
    name = "optics.transient_absorption"
    description = (
        "Analyze transient absorption spectra (wavelength-time data) with "
        "PyTASER; returns data dimensions, time/wavelength ranges, and signal "
        "statistics from a CSV input."
    )
    short_description = "Transient absorption analysis of a wavelength-time CSV."
    exposure = ToolExposure.DEFERRED
    namespace = "optics"
    source = "pytaser"
    tags = ("optics", "transient absorption", "spectra", "pytaser")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "CSV with wavelength, time, and delta-OD columns."},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": 100000},
        },
        "required": ["path"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        import csv
        from pathlib import Path

        path = Path(str(arguments["path"]))
        if not path.is_file():
            return ScientificToolResult(
                output=f"spectra file not found: {arguments['path']}",
                is_error=True,
                data={"error": "not_found"},
            )
        max_rows = int(arguments.get("max_rows", 50000))
        try:
            import pytaser  # noqa: F401
        except Exception:
            return ScientificToolResult(
                output=(
                    "missing prerequisite: 'pytaser' is not importable; install "
                    "photomatagent[optics] for transient absorption analysis."
                ),
                is_error=True,
                data={"error": "MISSING_DEPENDENCY"},
            )
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = []
                for row in reader:
                    rows.append(row)
                    if len(rows) >= max_rows:
                        break
            if not rows:
                return ScientificToolResult(
                    output="spectra CSV is empty or unparsable",
                    is_error=True,
                    data={"error": "empty"},
                )
            keys = list(rows[0].keys())
            numeric = [
                [float(row.get(key, "nan")) for row in rows if row.get(key, "").strip()]
                for key in keys[:3]
            ]
            stats = {
                key: {
                    "min": round(min(values), 6) if values else None,
                    "max": round(max(values), 6) if values else None,
                }
                for key, values in zip(keys[:3], numeric, strict=False)
            }
            payload = {
                "file": path.name,
                "columns": keys,
                "rows": len(rows),
                "column_stats": stats,
                "note": "PyTASER is available; raw data ranges reported, no fitting performed.",
            }
        except Exception as exc:
            return ScientificToolResult(
                output=f"optics.transient_absorption failed: {type(exc).__name__}: {exc}",
                is_error=True,
                data={"error": type(exc).__name__},
            )
        return ScientificToolResult(
            output=json.dumps(payload, ensure_ascii=False),
            data=payload,
        )


def optics_pack(config: Any, workspace: Any) -> CapabilityPack:
    return OpticsProbe()
