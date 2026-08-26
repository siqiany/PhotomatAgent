"""POTCAR metadata only: TITEL / ZVAL / ENMAX / ENMIN bookkeeping.

The user's licensed PAW-PBE library is read for metadata alone. POTCAR file
content is never copied, written into prepared inputs, embedded into reports,
logged, or returned to the model. The only objects that leave this module are
``PotcarBlock`` records (element, title, zval, enmax, enmin, source path).

Two metadata sources are supported:
* ``psp_dir`` -- a local library with ``<element>/POTCAR`` datasets (the
  configured root, resolved through the same layout detector as the periodic
  path: direct ``<root>/<element>/POTCAR``, ``<root>/potpaw_PBE/<element>/POTCAR``
  or ``<root>/potpaw_PBE.64/<element>/POTCAR``);
* ``potcar_path`` -- one concatenated POTCAR file (as VASP would read it);
  used to verify POSCAR/POTCAR element order and duplicate blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from photomatagent.scientific.applications.vasp.molecular.models import (
    MoleculeSpec,
)
from photomatagent.scientific.applications.vasp.psp import (
    resolve_local_psp_library,
)


class PspError(ValueError):
    """Pseudopotential metadata resolution failure."""

    def __init__(self, message: str, *, code: str = "PSP_METADATA_UNREADABLE"):
        super().__init__(message)
        self.code = code


_TITEL_RE = re.compile(r"^\s*TITEL\s*=\s*(.+?)\s*$")
_ZVAL_RE = re.compile(r"ZVAL\s*=\s*([-+0-9.eE]+)")
_ENMAX_RE = re.compile(r"ENMAX\s*=\s*([-+0-9.eE]+)")
_ENMIN_RE = re.compile(r"ENMIN\s*=\s*([-+0-9.eE]+)")


@dataclass(frozen=True)
class PotcarBlock:
    """Safe metadata extracted from one POTCAR dataset (never its content)."""

    element: str
    title: str
    zval: float
    enmax: float | None
    enmin: float | None
    source: str


@dataclass(frozen=True)
class PotcarResolution:
    """One ordered metadata resolution result for the whole molecule."""

    blocks: list[PotcarBlock]  # in POSCAR element order
    max_enmax: float | None
    library: Path | None = None  # resolved <element>/POTCAR library dir
    layout: str | None = None  # "direct" | "potpaw_PBE" | "potpaw_PBE.64"

    def metadata_summary(self) -> dict[str, object]:
        """JSON-safe summary containing ONLY title/zval/enmax/enmin."""
        return {
            "datasets": [
                {
                    "element": block.element,
                    "title": block.title,
                    "zval": block.zval,
                    "enmax": block.enmax,
                    "enmin": block.enmin,
                }
                for block in self.blocks
            ],
            "max_enmax": self.max_enmax,
            "note": "metadata only; POTCAR content is never written or logged",
        }


def canonical_element_from_title(title: str) -> str | None:
    """Extract the element symbol from a TITEL value like 'PAW_PBE Li 17Jan2003'.

    Tokens that are all uppercase (PAW, PAW_PBE, GGA, US) are skipped; the
    first capitalized element-like token wins (Li, C, Fe_pv -> Fe).
    """
    for token in title.split():
        head = token.split("_", 1)[0]
        if not head or not head[0].isupper():
            continue
        if any(character.isdigit() for character in head):
            continue
        if len(head) > 1 and not head[1].islower():
            continue
        return head
    return None


def _first_float(pattern: re.Pattern[str], lines: list[str]) -> float | None:
    for line in lines:
        match = pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def parse_potcar_blocks(text: str, *, source: str) -> list[PotcarBlock]:
    """Split a concatenated POTCAR into per-dataset metadata blocks."""
    lines = text.splitlines()
    starts = [
        index for index, line in enumerate(lines)
        if _TITEL_RE.match(line)
    ]
    if not starts:
        raise PspError(f"no TITEL header found in {source}")
    blocks: list[PotcarBlock] = []
    for block_index, start in enumerate(starts):
        end = starts[block_index + 1] if block_index + 1 < len(starts) else len(lines)
        chunk = lines[start:end]
        title_match = _TITEL_RE.match(chunk[0])
        if title_match is None:
            raise PspError(f"malformed TITEL line in {source}")
        title = title_match.group(1).strip()
        element = canonical_element_from_title(title)
        zval = _first_float(_ZVAL_RE, chunk)
        if element is None or zval is None:
            raise PspError(
                f"could not parse element/ZVAL from POTCAR block {block_index} "
                f"in {source}"
            )
        blocks.append(
            PotcarBlock(
                element=element,
                title=title,
                zval=zval,
                enmax=_first_float(_ENMAX_RE, chunk),
                enmin=_first_float(_ENMIN_RE, chunk),
                source=source,
            )
        )
    return blocks


def read_dataset_block(library_dir: Path, element: str) -> PotcarBlock:
    """Read metadata of one ``<library>/<element>/POTCAR`` dataset."""
    potcar = library_dir / element / "POTCAR"
    if not potcar.is_file():
        raise PspError(
            f"missing PAW-PBE dataset for {element}: {potcar}",
            code="PSP_DATASET_MISSING",
        )
    blocks = parse_potcar_blocks(
        potcar.read_text(encoding="utf-8", errors="replace"), source=str(potcar)
    )
    if len(blocks) != 1:
        raise PspError(
            f"expected exactly one POTCAR block in {potcar}, found {len(blocks)}"
        )
    return blocks[0]


def resolve_potcar_metadata(
    molecule: MoleculeSpec,
    elements: list[str],
    *,
    psp_dir: str | Path | None = None,
    potcar_path: str | Path | None = None,
) -> PotcarResolution:
    """Resolve metadata for ``elements`` in deterministic POSCAR order.

    ``potcar_path`` (a real concatenated POTCAR stream) takes priority: it
    lets the preflight verify POSCAR/POTCAR order and duplicate blocks.
    Otherwise each element's dataset is read from ``psp_dir``.
    """
    del molecule
    if potcar_path is not None:
        path = Path(potcar_path).expanduser().resolve()
        if not path.is_file():
            raise PspError(f"POTCAR stream does not exist: {path}")
        blocks = parse_potcar_blocks(
            path.read_text(encoding="utf-8", errors="replace"), source=str(path)
        )
        return _ordered_resolution(blocks, elements)
    if psp_dir is None:
        raise PspError(
            "no pseudopotential source configured: set psp_dir or potcar_path",
            code="PSP_UNRESOLVED",
        )
    resolved = resolve_local_psp_library(psp_dir)
    if resolved is None:
        raise PspError(
            "no known PAW-PBE layout under the configured pseudopotential "
            f"library {Path(psp_dir).expanduser().resolve()} "
            "(expected <root>/<element>/POTCAR, <root>/potpaw_PBE/<element>/POTCAR "
            "or <root>/potpaw_PBE.64/<element>/POTCAR)",
            code="PSP_UNRESOLVED",
        )
    library, layout = resolved
    blocks = [read_dataset_block(library, element) for element in elements]
    resolution = _ordered_resolution(blocks, elements)
    resolution = PotcarResolution(
        blocks=resolution.blocks,
        max_enmax=resolution.max_enmax,
        library=library,
        layout=layout,
    )
    return resolution


def _ordered_resolution(
    blocks: list[PotcarBlock], elements: list[str]
) -> PotcarResolution:
    """Keep the raw block order so the preflight can audit order/duplicates.

    When the metadata comes from a real concatenated POTCAR stream, the block
    order is exactly what VASP would read and must match the POSCAR element
    order. When it comes from per-element dataset reads (``psp_dir`` mode) the
    requested order is already the dataset order. No ordering contract is
    enforced here: the preflight audits block count, duplicates, missing
    elements and order with distinct error codes.
    """
    enmaxes = [block.enmax for block in blocks if block.enmax is not None]
    return PotcarResolution(
        blocks=blocks,
        max_enmax=max(enmaxes) if enmaxes else None,
    )
