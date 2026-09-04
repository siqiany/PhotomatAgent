"""Pseudopotential library layout resolution (Sprint 4, sections 35-38).

Historical PhotoMatAgent assumed the library directory always contains
``potpaw_PBE.64/``. The user's real SCNet library is
``/public/home/scniv4a4go/potpaw_PBE/`` (the exact PBE library directory),
so the resolver accepts three layouts deterministically:

* ``direct``:       ``<configured>/<setup>/POTCAR``          (SCNET_VASP_PSP_DIR)
* ``potpaw_PBE``:   ``<configured>/potpaw_PBE/<setup>/POTCAR``  (ASE VASP_PP_PATH)
* ``potpaw_PBE.64``:``<configured>/potpaw_PBE.64/<setup>/POTCAR`` (legacy)

No POTCAR content is ever read, copied or logged here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

PSPLayout = Literal["direct", "potpaw_PBE", "potpaw_PBE.64"]

_PROBE_SETUPS = ("In", "Al", "C")

# Deterministic element -> POTCAR setup SUFFIX mapping.
#
# The VASP PBE POTCAR libraries (potpaw_PBE, potpaw_PBE.64) ship no plain
# ``Ba`` dataset: barium only exists as ``Ba_sv`` (10 valence electrons,
# s/p/d semicore), which is the canonical PBE choice. Keeping the suffix
# here means policy generation, local resolution and remote assembly all
# agree on the same full setup name (``Ba`` -> ``Ba_sv``). Elements absent
# from the table stay bare.
ELEMENT_SETUP_DEFAULTS: dict[str, str] = {
    "Ba": "_sv",
}

# POTCAR setup directory names must stay shell/argv safe: letters, digits
# and an optional ``_setup`` suffix only -- never path separators,
# whitespace, quotes or ``..``.
_SAFE_POTCAR_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def potcar_element_name(element: str) -> str:
    """Full POTCAR directory name for a chemical element symbol."""
    return f"{element}{ELEMENT_SETUP_DEFAULTS.get(element, '')}"


def is_safe_potcar_symbol(symbol: str) -> bool:
    """True for a safe POTCAR setup name (bare element or ``X_sv``-style)."""
    return bool(_SAFE_POTCAR_NAME.fullmatch(symbol))


def _layout_for_local(root: Path) -> PSPLayout | None:
    """Detect which layout exists under a local root (filesystem only)."""
    for candidate in _PROBE_SETUPS:
        if (root / candidate / "POTCAR").is_file():
            return "direct"
        if (root / "potpaw_PBE" / candidate / "POTCAR").is_file():
            return "potpaw_PBE"
        if (root / "potpaw_PBE.64" / candidate / "POTCAR").is_file():
            return "potpaw_PBE.64"
    return None


def resolve_local_psp_library(configured: str | Path) -> tuple[Path, PSPLayout] | None:
    """Resolve the real local library directory and its layout.

    Returns ``(resolved_library, layout)`` or ``None`` when no known layout
    can be confirmed. ``resolved_library`` always points at the directory
    whose children are ``<setup>/POTCAR``.
    """
    root = Path(configured).expanduser()
    if not root.is_dir():
        return None
    layout = _layout_for_local(root)
    if layout is None:
        return None
    if layout == "direct":
        return root, layout
    return root / layout, layout


def local_potcar_check(
    configured: str | Path, requirements: list[tuple[str, str]]
) -> list[str]:
    """Return the list of missing ``element+setup`` POTCARs (local)."""
    resolved = resolve_local_psp_library(configured)
    missing: list[str] = []
    if resolved is None:
        return [f"{element}{setup or ''}" for element, setup in requirements]
    library, _ = resolved
    for element, setup in requirements:
        candidate = library / f"{element}{setup}" / "POTCAR"
        if not candidate.is_file():
            missing.append(f"{element}{setup or ''}")
    return missing


async def resolve_remote_psp_library(
    configured: str, backend: Any
) -> tuple[str, PSPLayout] | None:
    """Probe one remote directory for a known layout (read-only ``test -f``).

    ``configured`` is the remote path as configured by the user
    (``SCNET_VASP_PSP_DIR`` or ``SCNET_MAGUS_VASP_PP_PATH``); the returned
    library is the directory whose children are ``<setup>/POTCAR``.
    """
    if not configured:
        return None
    probe = _PROBE_SETUPS[0]
    layouts: tuple[tuple[PSPLayout, str], ...] = (
        ("direct", f"test -f {configured}/{probe}/POTCAR"),
        ("potpaw_PBE", f"test -f {configured}/potpaw_PBE/{probe}/POTCAR"),
        ("potpaw_PBE.64", f"test -f {configured}/potpaw_PBE.64/{probe}/POTCAR"),
    )
    for layout, command in layouts:
        result = await backend._run_ssh(command)
        if result.ok:
            if layout == "direct":
                return configured, layout
            return f"{configured}/{layout}", layout
    return None


async def remote_potcar_check(
    backend: Any,
    configured: str,
    requirements: list[tuple[str, str]],
) -> list[str]:
    """Return missing ``element+setup`` POTCARs on the remote side."""
    resolved = await resolve_remote_psp_library(configured, backend)
    if resolved is None:
        return [f"{element}{setup or ''}" for element, setup in requirements]
    library, _ = resolved
    missing: list[str] = []
    for element, setup in requirements:
        candidate = f"{library}/{element}{setup}/POTCAR"
        result = await backend._run_ssh(f"test -f {candidate}")
        if not result.ok:
            missing.append(f"{element}{setup or ''}")
    return missing
