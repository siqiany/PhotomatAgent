"""Local SCNet MCP server (Sprint 3 section 17-19).

The server runs on the user's local machine (WSL) and talks to SCNet as a
remote compute backend over SSH/Slurm. It is a normal MCP stdio server
implemented with the official FastMCP SDK; PhotoMatAgent's existing MCP
gateway (``mcp.manager``) spawns it like any other server. The agent never
receives generic remote shell access -- only the narrow application tools
registered here (``vasp.*``, ``namd.*``, ``magus.*``).
"""

from __future__ import annotations
