"""Scientific numerical evidence extraction from retrieved passages.

Focused on infrared detector materials: responsivity (A/W), detectivity
(Jones), dark current (A, A/cm2), wavelength (nm/um), temperature (K),
bandgap (eV), mobility (cm2/Vs), NETD (mK). Extraction is regex-based and
conservative: nothing is guessed, missing fields are simply omitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from photomatagent.scientific.capabilities.contracts import ScientificEvidence

_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(?:[×x*]\s*10\s*[\^]?\s*(-?\d+))?"
)
_EXP_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*[eE]\s*([-+]?\d+)")
# "1.2 10 10" -> 1.2e10 (lost 'x' in "x 10^10").
_MANTISSA_SPACED_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s+10\s*[\^]?\s*(-?\d+)"
)
# "10 10" -> 10^10 (LaTeX superscript rendered with a space).
_BASE_SPACED_RE = re.compile(r"(?<![\w.])(10)\s+(\d{1,2})\b")

_ALL_NUMBER_RES = (
    _MANTISSA_SPACED_RE,
    _BASE_SPACED_RE,
    _EXP_NUMBER_RE,
    _NUMBER_RE,
)


def _value_for_match(match: re.Match[str]) -> float:
    mantissa = float(match.group(1).replace(",", "."))
    exponent = int(match.group(2)) if match.group(2) else 0
    if match.re is _BASE_SPACED_RE:
        return 10.0**exponent
    if match.re is _MANTISSA_SPACED_RE:
        return mantissa * 10.0**exponent
    if match.re is _EXP_NUMBER_RE:
        return mantissa * 10.0**exponent
    if match.re is _NUMBER_RE:
        return mantissa * 10.0**exponent
    return mantissa


def parse_number(text: str) -> float | None:
    """Parse a plain, scientific-notation, or x10^ form number."""
    for pattern in _ALL_NUMBER_RES:
        match = pattern.search(text)
        if match:
            return _value_for_match(match)
    return None


@dataclass(frozen=True)
class UnitRule:
    pattern: str
    unit: str


@dataclass(frozen=True)
class PropertyRule:
    property: str
    keywords: tuple[str, ...]
    units: tuple[UnitRule, ...]


PROPERTY_RULES: tuple[PropertyRule, ...] = (
    PropertyRule(
        property="responsivity",
        keywords=("responsivity", "responsivities", "response of", "photo-responsivity"),
        units=(
            UnitRule(r"A\s*/\s*W|A\s*W\s*[-⁻–]?\s*1\b|AW[-⁻–]?1\b", "A/W"),
        ),
    ),
    PropertyRule(
        property="detectivity",
        keywords=("detectivity", "specific detectivity", "d-star", r"d\*"),
        units=(
            UnitRule(r"Jones\b|cm\s*[·.]?\s*Hz|cmHz", "Jones"),
        ),
    ),
    PropertyRule(
        property="dark_current",
        keywords=("dark current", "dark-current", "dark current density", "dark-current density"),
        units=(
            UnitRule(r"A\s*/\s*cm\s*2\b|A\s*cm\s*[-⁻–]?\s*2\b|A·cm[-⁻–]?2\b", "A/cm2"),
            UnitRule(r"\bA\b", "A"),
        ),
    ),
    PropertyRule(
        property="wavelength",
        keywords=("wavelength", "peak wavelength", "cutoff wavelength", "cut-off wavelength", "emission wavelength"),
        units=(
            UnitRule(r"nm\b", "nm"),
            UnitRule(r"[uµ]m\b|μm\b|µm\b", "um"),
        ),
    ),
    PropertyRule(
        property="temperature",
        keywords=("temperature", "operating temperature", "measurement temperature"),
        units=(UnitRule(r"\bK\b", "K"),),
    ),
    PropertyRule(
        property="bandgap",
        keywords=("band gap", "bandgap", "band-gap", "optical gap", "energy gap"),
        units=(UnitRule(r"eV\b", "eV"),),
    ),
    PropertyRule(
        property="mobility",
        keywords=("mobility", "carrier mobility", "electron mobility", "hole mobility"),
        units=(
            UnitRule(
                r"cm\s*2\s*/\s*V\s*s\b|cm\s*2\s*V\s*[-⁻–]?\s*1\s*s\s*[-⁻–]?\s*1\b|cm²/Vs\b|cm2/Vs\b",
                "cm2/Vs",
            ),
        ),
    ),
    PropertyRule(
        property="netd",
        keywords=("netd", "noise-equivalent temperature difference", "noise equivalent temperature difference"),
        units=(UnitRule(r"mK\b", "mK"),),
    ),
)

_MATERIAL_RE = re.compile(
    r"\b(HgTe|HgCdTe|PbS|PbSe|PbTe|InAs|InGaAs|GaAs|GaSb|InSb|InP|Si|Ge|"
    r"CdTe|CdSe|CdS|ZnS|ZnSe|HgSe|SnTe|SnS|SnSe|Sb2S3|Sb2Se3|Bi2S3|Bi2Se3|"
    r"Bi2Te3|MoS2|WS2|WSe2|MoSe2|GaN|AlGaN|InGaN|ZnO|TiO2|perovskite|"
    r"quantum dot|colloidal quantum dot|nanowire|photodiode|photodetector|"
    r"detector|device|sensor)\b",
    re.IGNORECASE,
)
_TEMPERATURE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*K\b")
_WAVELENGTH_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*((?:μm|µm|um|nm))\b")
_BIAS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*V\b")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _nearest_number_before(sentence: str, end: int, window: int = 25) -> float | None:
    fragment = sentence[max(0, end - window) : end]
    matches = [
        match
        for pattern in _ALL_NUMBER_RES
        for match in pattern.finditer(fragment)
    ]
    if not matches:
        return None
    # Prefer the number closest to the unit (latest span end wins), but a
    # compound form ("10 10", "1.2e10") beats a plain number it overlaps.
    def rank(match: re.Match[str]) -> int:
        return 1 if match.re is not _NUMBER_RE else 0

    matches.sort(key=lambda item: (item.end(), rank(item)), reverse=True)
    match = matches[0]
    for candidate in matches[1:]:
        overlapping = not (
            candidate.end() < match.start() or candidate.start() > match.end()
        )
        if rank(candidate) > rank(match) and overlapping:
            match = candidate
    return _value_for_match(match)


def _subject(sentence: str) -> str | None:
    match = _MATERIAL_RE.search(sentence)
    return match.group(1) if match else None


def _method(sentence: str) -> str:
    lowered = sentence.casefold()
    if "calculat" in lowered or "simulat" in lowered:
        return "calculated value"
    return "reported experimental value"


def _conditions(sentence: str) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    temperature = _TEMPERATURE_RE.search(sentence)
    if temperature:
        value = parse_number(temperature.group(1))
        if value is not None:
            provenance["temperature_K"] = value
    wavelength = _WAVELENGTH_RE.search(sentence)
    if wavelength:
        value = parse_number(wavelength.group(1))
        if value is not None:
            unit = wavelength.group(2)
            provenance["wavelength_um"] = value / 1000.0 if unit == "nm" else value
    bias = _BIAS_RE.search(sentence)
    if bias:
        value = parse_number(bias.group(1))
        if value is not None:
            provenance["bias_V"] = value
    return provenance


def extract_evidence_from_text(
    text: str,
    *,
    source: str = "",
    page: int | None = None,
    passage_id: str = "",
) -> list[ScientificEvidence]:
    """Extract (property, value, unit, condition, source) evidence from text."""
    evidence: list[ScientificEvidence] = []
    for sentence in _sentences(text):
        lowered = sentence.casefold()
        for rule in PROPERTY_RULES:
            if not any(keyword in lowered for keyword in rule.keywords):
                continue
            for unit_rule in rule.units:
                for unit_match in re.finditer(unit_rule.pattern, sentence):
                    value = _nearest_number_before(sentence, unit_match.start())
                    if value is None:
                        continue
                    provenance = _conditions(sentence)
                    provenance["property_context"] = sentence[:160]
                    if page is not None:
                        provenance["page"] = page
                    if passage_id:
                        provenance["passage_id"] = passage_id
                    evidence.append(
                        ScientificEvidence(
                            subject=_subject(sentence) or "detector",
                            property=rule.property,
                            value=value,
                            unit=unit_rule.unit,
                            source=source,
                            source_type="literature",
                            method=_method(sentence),
                            summary=sentence[:220],
                            limitations=(
                                "Regex extraction; verify value and conditions "
                                "against the source PDF"
                            ),
                            provenance=provenance,
                        )
                    )
                    # One unit hit per property per sentence is enough.
                    break
    return evidence


def extract_evidence_from_passages(
    passages: list[dict[str, Any]],
) -> list[ScientificEvidence]:
    """Extract evidence across a list of passage dicts (tool input shape)."""
    results: list[ScientificEvidence] = []
    for passage in passages:
        text = str(passage.get("text") or "")
        if not text:
            continue
        results.extend(
            extract_evidence_from_text(
                text,
                source=str(passage.get("source") or passage.get("file_name") or ""),
                page=passage.get("page"),
                passage_id=str(passage.get("passage_id") or ""),
            )
        )
    return results
