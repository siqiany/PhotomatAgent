"""Sensitive path checks for model-facing filesystem capabilities."""

from __future__ import annotations

import re
import shlex
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field


class SensitiveAccessError(ValueError):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"blocked access to sensitive file: {path}")


class SensitivePathPolicyConfig(BaseModel):
    sensitive_directories: tuple[str, ...] = (".ssh", ".aws")
    sensitive_names: tuple[str, ...] = (".git-credentials", ".netrc")
    sensitive_suffixes: tuple[str, ...] = (".pem", ".key")
    sensitive_prefixes: tuple[str, ...] = ("credentials", "secrets")
    block_dotenv: bool = True
    max_reported_path_chars: int = Field(default=160, ge=32)


class SensitivePathPolicy:
    """Conservative lexical policy; it complements, but is not, a sandbox."""

    def __init__(self, config: SensitivePathPolicyConfig | None = None) -> None:
        self.config = config or SensitivePathPolicyConfig()

    def is_sensitive(self, raw_path: str | Path) -> bool:
        normalized = str(raw_path).replace("\\", "/")
        parts = [part.casefold() for part in PurePosixPath(normalized).parts]
        directories = {item.casefold() for item in self.config.sensitive_directories}
        if any(part in directories for part in parts):
            return True
        name = parts[-1] if parts else ""
        if name in {item.casefold() for item in self.config.sensitive_names}:
            return True
        if self.config.block_dotenv and (name == ".env" or name.startswith(".env.")):
            return True
        if any(name.endswith(suffix.casefold()) for suffix in self.config.sensitive_suffixes):
            return True
        return any(name.startswith(prefix.casefold()) for prefix in self.config.sensitive_prefixes)

    def check_path(self, raw_path: str | Path) -> None:
        if self.is_sensitive(raw_path):
            raise SensitiveAccessError(self.display_path(raw_path))

    def check_tool_call(self, name: str, arguments: dict[str, object]) -> None:
        if name in {"read", "edit", "write"}:
            for key in ("path",):
                value = arguments.get(key)
                if isinstance(value, str):
                    self.check_path(value)
        elif name == "grep":
            for key in ("path", "glob"):
                value = arguments.get(key)
                if isinstance(value, str):
                    self.check_path(value)
        elif name == "glob":
            for key in ("pattern",):
                value = arguments.get(key)
                if isinstance(value, str):
                    self.check_path(value)
        elif name == "bash":
            command = str(arguments.get("command", ""))
            for token in self._command_tokens(command):
                if self.is_sensitive(token):
                    raise SensitiveAccessError(self.display_path(token))

    def display_path(self, raw_path: str | Path) -> str:
        value = str(raw_path).replace("\n", " ").replace("\r", " ")
        return value[: self.config.max_reported_path_chars]

    @staticmethod
    def _command_tokens(command: str) -> list[str]:
        try:
            tokens = shlex.split(command, comments=False, posix=True)
        except ValueError:
            tokens = re.split(r"\s+", command)
        # Split shell punctuation so `cat .env;pwd` cannot hide the path.
        expanded: list[str] = []
        for token in tokens:
            expanded.extend(part for part in re.split(r"[;|&><()]+", token) if part)
        return expanded
