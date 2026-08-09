"""A single, explicit filesystem boundary for local tools."""

from __future__ import annotations

from pathlib import Path

from photomatagent.errors import ToolExecutionError


class Workspace:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace root is not a directory: {self.root}")

    def resolve(self, path: str, *, must_exist: bool = True) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        if not self.contains(resolved):
            raise ToolExecutionError(f"path is outside workspace: {path}")
        if must_exist and not resolved.exists():
            raise ToolExecutionError(f"path does not exist: {path}")
        return resolved

    def contains(self, path: Path) -> bool:
        """Whether a resolved path stays inside the workspace boundary."""
        return path == self.root or self.root in path.parents

    def relative(self, path: Path) -> str:
        """Return the workspace-relative display path without following symlinks."""
        candidate = path if path.is_absolute() else self.root / path
        return candidate.relative_to(self.root).as_posix()

    def validate_glob(self, pattern: str) -> str:
        candidate = Path(pattern)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ToolExecutionError(f"glob pattern escapes workspace: {pattern}")
        return pattern
