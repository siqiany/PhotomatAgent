"""Compatibility facade over the Loop Observatory trace analyzer."""

from __future__ import annotations

from pathlib import Path

from photomatagent.observability.analyzer import SessionSummary, analyze_trace
from photomatagent.observability.trace import list_session_paths, load_trace


def list_sessions(sessions_dir: Path | str | None = None) -> list[Path]:
    return list_session_paths(sessions_dir)


def latest_session(sessions_dir: Path | str | None = None) -> Path | None:
    sessions = list_sessions(sessions_dir)
    return sessions[0] if sessions else None


def read_session_stats(session: Path | str) -> SessionSummary:
    return analyze_trace(load_trace(session))


__all__ = [
    "SessionSummary",
    "latest_session",
    "list_sessions",
    "read_session_stats",
]
