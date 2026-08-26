"""Deterministic INCAR/KPOINTS rendering and parsing for molecular stages.

The renderer converts Python ``list``/``tuple`` values into space-separated
strings, so ``DIPOL = [0.5, 0.5, 0.5]`` can never appear in a generated
INCAR: it always renders as ``DIPOL = 0.5 0.5 0.5``. The preflight additionally
rejects any hand-written INCAR whose DIPOL line uses list syntax.
"""

from __future__ import annotations

import re
from typing import Any


def _render_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot render INCAR value of type {type(value).__name__}: {value!r}")


def render_incar(settings: dict[str, Any]) -> str:
    """Render INCAR settings deterministically (insertion-ordered)."""
    lines: list[str] = []
    for key, value in settings.items():
        if isinstance(value, (list, tuple)):
            rendered = " ".join(_render_scalar(item) for item in value)
        else:
            rendered = _render_scalar(value)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def render_kpoints_gamma(comment: str = "Gamma-only isolated molecule") -> str:
    """Render the mandatory Gamma-only 1x1x1 KPOINTS file."""
    return f"{comment}\n0\nGamma\n1 1 1\n0 0 0\n"


_INCAR_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_incar(text: str) -> dict[str, str]:
    """Parse INCAR text into raw ``{KEY: value-string}`` pairs."""
    settings: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _INCAR_LINE_RE.match(line)
        if match:
            settings[match.group(1)] = match.group(2)
    return settings


def parse_bool(value: str) -> bool | None:
    lowered = value.lower().strip(".")
    if lowered in {"true", "t", "yes", "1"}:
        return True
    if lowered in {"false", "f", "no", "0"}:
        return False
    return None


def parse_float(value: str) -> float | None:
    try:
        return float(value.split()[0].rstrip(","))
    except (ValueError, IndexError):
        return None


def parse_int(value: str) -> int | None:
    parsed = parse_float(value)
    if parsed is None or not float(parsed).is_integer():
        return None
    return int(parsed)


def parse_kpoints(text: str) -> dict[str, Any]:
    """Classify a KPOINTS file: gamma-auto, gamma-explicit, MP or invalid."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result: dict[str, Any] = {"mode": "invalid", "grid": None, "points": []}
    if len(lines) < 3:
        return result
    mode_token = lines[1].lower()
    index = 2
    if mode_token in {"0", "auto", "automatic"}:
        if index >= len(lines):
            return result
        scheme = lines[index].lower()
        index += 1
        if scheme in {"monkhorst-pack", "mp"}:
            result["mode"] = "monkhorst_pack"
            grid = _parse_grid(lines[index] if index < len(lines) else "")
            result["grid"] = grid
            return result
        if scheme not in {"gamma", "g"}:
            return result
        result["mode"] = "gamma_auto"
        grid = _parse_grid(lines[index] if index < len(lines) else "")
        result["grid"] = grid
        return result
    if mode_token in {"gamma", "g", "monkhorst-pack", "mp"}:
        grid = _parse_grid(lines[index] if index < len(lines) else "")
        if mode_token in {"monkhorst-pack", "mp"}:
            result["mode"] = "monkhorst_pack"
        else:
            result["mode"] = "gamma_auto"
        result["grid"] = grid
        return result
    try:
        n_points = int(mode_token)
    except ValueError:
        return result
    explicit: list[tuple[float, float, float]] = []
    for row in lines[index : index + n_points]:
        fields = row.split()
        if len(fields) < 3:
            continue
        try:
            explicit.append(
                (float(fields[0]), float(fields[1]), float(fields[2]))
            )
        except ValueError:
            continue
    result["points"] = explicit
    result["mode"] = "explicit"
    return result


def _parse_grid(line: str) -> list[int] | None:
    fields = line.split()
    if len(fields) < 3:
        return None
    try:
        return [int(float(fields[0])), int(float(fields[1])), int(float(fields[2]))]
    except ValueError:
        return None
