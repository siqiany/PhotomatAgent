"""Demo A (MCP variant) — Materials Project official MCP live smoke test.

Requires a normal Linux host (the Codex sandbox blocks the official MCP SDK's
stdio plumbing) and a configured ``.photomatagent/mcp.json`` with the
Materials Project server. Searches HgTe / InAs / PbTe and fetches one
structure, then prints latency and evidence. API keys never leave the
gateway config.

Run:
  PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 \\
    MPLCONFIGDIR=/tmp/mpl .venv/bin/python experiments/vertical/demo_a_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


async def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from photomatagent.mcp.manager import MCPServerManager

    manager = MCPServerManager(workspace=repo)
    handle = manager.handles.get("materials-project")
    if handle is None:
        print("no materials-project MCP server configured; see .photomatagent/mcp.json")
        raise SystemExit(2)
    state = await handle.start()
    print(f"server state: {state.value}")
    if state.value != "READY":
        print(f"detail: {handle.detail}")
        print(f"last_error: {handle.last_error}")
        raise SystemExit(1)
    print(f"advertised tools: {[t.name for t in handle.remote_tools]}")
    for query in ("HgTe", "InAs", "PbTe"):
        text, is_error, _ = await handle.invoke("search", {"query": query})
        print(f"\n=== search {query} (error={is_error}) ===")
        print(text[:1200])
    text, is_error, _ = await handle.invoke(
        "fetch", {"idx": "mp-20305"}  # InAs per native search results
    )
    print(f"\n=== fetch mp-20305 (error={is_error}) ===")
    print(text[:1500])
    health = await handle.healthcheck()
    print(f"\nhealth: {json.dumps(health, ensure_ascii=False)}")
    await handle.close()


if __name__ == "__main__":
    if os.environ.get("PHOTOMATAGENT_RUN_LIVE_SCIENCE") != "1":
        print("gated by PHOTOMATAGENT_RUN_LIVE_SCIENCE=1 (needs a normal Linux host)")
        raise SystemExit(0)
    asyncio.run(main())
