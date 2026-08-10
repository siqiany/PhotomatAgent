"""Vertical Slice 1 (LLM) — real provider run of the LWIR screening goal.

The agent is free to discover capabilities via tool_search, load skills via
skill_view, and execute deferred tools through tool_call. Approval is auto so
the run is headless; iterations are bounded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from slice_runner import run_llm, save_result, default_session_dir
from structures import generate_structures


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    from dotenv import load_dotenv

    load_dotenv(repo / ".env")
    generate_structures(repo / "output" / "scientific")
    goal = (
        "Design and screen candidate materials for an 8-14 um infrared "
        "photodetector. Start by loading the infrared-photodetector-design "
        "skill, then compile physical constraints with ir.compile_constraints, "
        "then gather evidence from literature and any other capability that "
        "reduces the most important uncertainty. End with an evidence-grounded "
        "recommendation that lists what is known, what is uncertain, and what "
        "evidence is still missing. Never fabricate values; report missing "
        "prerequisites explicitly."
    )
    result = asyncio.run(
        run_llm(
            goal=goal,
            workspace_root=repo,
            session_dir=default_session_dir(),
            provider="openai",
            model_name="deepseek-v4-flash",
            max_iterations=14,
        )
    )
    path = save_result(result, repo / "output" / "vertical" / "slice_1_lwir_llm.json")
    print(f"slice 1 (llm) saved: {path}")
    print(f"tool calls: {result['tool_calls']}")
    print(f"usage: {result['usage']}")


if __name__ == "__main__":
    main()
