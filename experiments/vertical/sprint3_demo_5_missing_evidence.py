"""Sprint 3 Demo 5: no hallucination on missing evidence.

Task: "请精确告诉我一个新生成 QD 的 carrier lifetime、EQE 和 detectivity。"
Correct behavior: do not fabricate; report available evidence, identify
missing evidence, recommend the corresponding capability (via real
tool_search against the registry).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from photomatagent.scientific.state import ScientificState  # noqa: E402
from photomatagent.tools.factory import create_default_registry  # noqa: E402
from photomatagent.tools.surface import ToolCatalog  # noqa: E402
from photomatagent.workspace import Workspace  # noqa: E402


def main() -> int:
    print("=" * 72)
    print("Demo 5: missing evidence -> typed gaps, no fabrication")
    print("=" * 72)
    print("\nTask: precise carrier lifetime, EQE, detectivity of a new QD.")
    print("(no NAMD/device evidence exists for this candidate)")

    registry = create_default_registry(
        ScientificState(), Workspace(ROOT)
    )
    catalog = ToolCatalog(registry)

    print("\n[1] Available evidence (what CAN be computed today):")
    print("    - qd.brus_transition_energy: L1 confinement energy (analytical)")
    print("    - materials_mcp.*: database band gaps (with sources)")
    print("    - photodetector.*: R/EQE conversions (physical consistency only)")

    print("\n[2] Missing evidence and the capabilities that would fill it:")
    for query, label in [
        ("carrier dynamics", "carrier lifetime"),
        ("thin film absorption", "EQE input (absorptance)"),
        ("device simulation", "detectivity / dark current"),
    ]:
        matches = catalog.search(query, limit=3)
        names = [match.entry.name for match in matches]
        print(f"    {label:30s} -> {', '.join(names) or '(no capability)'}")

    print("\n[3] Honest answer contract:")
    print("    - carrier lifetime: NO evidence -> no number. Run namd.prepare")
    print("      (VASP AIMD + WAVECAR snapshots) then Hefei-NAMD on SCNet.")
    print("    - EQE: requires absorptance + collection + gain evidence;")
    print("      optics.meep_thinfilm + device evidence required.")
    print("    - detectivity: requires dark current + noise + R evidence;")
    print("      no device evidence exists -> missing_prerequisites.")
    print("\nDemo 5 done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
