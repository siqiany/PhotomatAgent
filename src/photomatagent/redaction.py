"""Shared secret redaction for model-visible observations and persisted traces."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any

_SECRET_KEY = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token|auth[_-]?token)"
)
_SECRET_VALUE = re.compile(
    r"\b(?:sk-ant-|sk-)(?:[A-Za-z0-9_-]{8,}|[A-Za-z0-9_-]{3,}\*+[A-Za-z0-9_-]+)\b"
)
_DOTENV_SECRET_LINE = re.compile(
    r"(?im)^(\s*[A-Za-z_][A-Za-z0-9_]*(?:API[_-]?KEY|PASSWORD|SECRET|TOKEN)\s*=\s*)(.*)$"
)
_INLINE_SECRET_ASSIGNMENT = re.compile(
    r'''(?ix)
    (\b(?:[A-Za-z_][A-Za-z0-9_]*)?(?:API[_-]?KEY|PASSWORD|SECRET|TOKEN)\b\s*[:=]\s*)
    (?:"[^"]*"|'[^']*'|[^\s,;]+)
    '''
)
_AUTH_BEARER = re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[^\s,;]+")


def redact_secrets(payload: Any) -> Any:
    """Conservatively redact known credentials in arbitrary nested payloads."""
    known_values = [
        value
        for name, value in os.environ.items()
        if value and len(value) >= 4 and _SECRET_KEY.search(name)
    ]

    def visit(value: Any, key: str | None = None) -> Any:
        if key and _SECRET_KEY.search(key):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {item_key: visit(item, item_key) for item_key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            assignment_lines = [
                line
                for line in value.splitlines()
                if re.match(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*=", line)
            ]
            if len(assignment_lines) >= 2 and any(
                _SECRET_KEY.search(line.split("=", 1)[0])
                for line in assignment_lines
            ):
                return "[REDACTED .env content]"
            redacted = _SECRET_VALUE.sub("[REDACTED]", value)
            redacted = _DOTENV_SECRET_LINE.sub(r"\1[REDACTED]", redacted)
            redacted = _INLINE_SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", redacted)
            redacted = _AUTH_BEARER.sub(r"\1[REDACTED]", redacted)
            for secret in known_values:
                redacted = redacted.replace(secret, "[REDACTED]")
            return redacted
        return value

    return visit(deepcopy(payload))


def redact_text(value: str) -> str:
    redacted = redact_secrets(value)
    return redacted if isinstance(redacted, str) else "[REDACTED]"
