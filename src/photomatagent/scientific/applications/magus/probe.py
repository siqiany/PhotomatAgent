"""Pure parsers for the remote MAGUS CLI probe (Sprint 4, sections 10-15).

Everything here is deterministic and offline-testable; the application
module feeds bounded SSH output into these parsers.
"""

from __future__ import annotations

import re
from typing import Any

_VERSION = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)?[A-Za-z0-9._-]*)\s*$")
_CALCULATOR_ROW = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(\w+Calculator)\s*$")
_FAILED_PLUGIN = re.compile(r"Fail when try to import magus\.calculators\.(\w+)")
_SUBCOMMAND_BLOCK = re.compile(r"\{([a-z,]+)\}")
_EXAMPLE_LEAF = re.compile(r"/examples/\d+--\d+[^/]*/$")


def parse_magus_version(stdout: str) -> str:
    """Extract a semver-like version from ``magus -v`` output."""
    for line in (stdout or "").splitlines():
        match = _VERSION.match(line)
        if match:
            return match.group(1)
    return ""


def parse_magus_help_commands(stdout: str) -> list[str]:
    """Extract the subcommand names from ``magus -h`` (2.1.0 style)."""
    block = _SUBCOMMAND_BLOCK.search(stdout or "")
    if block:
        return [name.strip() for name in block.group(1).split(",") if name.strip()]
    # Fallback: any indented token followed by a description inside the
    # "Valid subcommands" section.
    commands: list[str] = []
    in_valid = False
    for line in (stdout or "").splitlines():
        if "Valid subcommands" in line:
            in_valid = True
            continue
        if in_valid:
            match = re.match(r"^\s{2,}([a-z][a-z0-9_-]*)\s", line)
            if match and match.group(1) not in commands:
                commands.append(match.group(1))
    return commands


def parse_checkpack_calculators(stdout: str, stderr: str = "") -> dict[str, Any]:
    """Parse ``magus checkpack calculators`` into available/failed lists."""
    available: list[str] = []
    for line in (stdout or "").splitlines():
        match = _CALCULATOR_ROW.match(line)
        if match:
            available.append(match.group(1))
    failed: list[str] = []
    for text in (stdout or "", stderr or ""):
        for match in _FAILED_PLUGIN.finditer(text):
            failed.append(match.group(1))
    return {
        "available": sorted(set(available)),
        "failed": sorted(set(failed)),
    }


def parse_example_structure_types(stdout: str) -> list[str]:
    """Parse ``structureType:`` values extracted from installed examples."""
    types: list[str] = []
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("structureType:"):
            continue
        value = stripped.split(":", 1)[1].strip().strip("'\"")
        if value and value not in types:
            types.append(value)
    return types


def parse_example_dirs(stdout: str) -> list[str]:
    """Parse example directory paths from ``unzip -l`` output."""
    dirs: list[str] = []
    for line in (stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[-1]
        # Keep only leaf example dirs (NN--NN-NAME/) or demo dirs; skip
        # container dirs (inputFold/, magus-master-examples/) which carry
        # no input.yaml of their own.
        if name.endswith("/") and (
            _EXAMPLE_LEAF.search(name)
            or ("demo" in name.lower() and "example" not in name.lower())
        ):
            if name not in dirs:
                dirs.append(name)
    return dirs[:24]
