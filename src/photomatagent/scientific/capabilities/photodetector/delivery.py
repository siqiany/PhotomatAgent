"""Delivery checks driven by the optical Checker receipt."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, FiniteFloat

from photomatagent.tools.base import DeliveryReview

_LOWER_SCOPE = re.compile(r"下限|最低要求|至少|≥|>=|lower bound|minimum requirements?", re.I)
_OPTICAL_SCOPE = re.compile(r"\bEQE\b|外量子效率|external quantum efficiency|响应度|responsivity", re.I)
_CONFLICT = re.compile(r"(?<!不)互斥|冲突|矛盾|mutually exclusive|incompatible|cannot both|\bconflict\b", re.I)
_NO_CONFLICT = re.compile(r"(?:不|无|没有|并无|不存在|不构成|不是|非)(?:目标|实际|真正|任何|直接)?\s*(?:互斥|冲突|矛盾)|"
                           r"\b(?:not|no|without)\s+(?:a\s+)?(?:mutually exclusive|incompatible|conflict)|"
                           r"(?:避免|误判为|误写成|并非|不等于).{0,8}(?:冲突|互斥|矛盾)", re.I)
_SENTENCES = re.compile(r"[。！？；;\n]+|(?<=[.!?])\s+")
_WAVELENGTH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[μµu]m)\b", re.I)
_JOINT = re.compile(r"联合(?:\s*EQE\s*)?(?:下限|下界|门槛)|joint (?:EQE )?(?:minimum|lower bound)", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_PERCENT_SERIES = re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:/|、|,|，)\s*\d+(?:\.\d+)?\s*%")
_DECLARED_PERCENT = re.compile(r"\s*(?:(?:约为|即为|约|为|is|about)\s*)?[：:=≈≥>~]?\s*(\d+(?:\.\d+)?)\s*%", re.I)
_SCOPED_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*[μµu]m\s*(?:处|[:：=→为≥]|EQE|\s)*\s*(\d+(?:\.\d+)?)\s*%", re.I)
_R_TARGET = re.compile(r"\bR\s*(?:≥|>=)\s*(\d+(?:\.\d+)?)\s*A/W", re.I)
_R_HEADER = re.compile(r"^\s*(?:R\b|响应度|responsivity\b)", re.I)
_R_CELL = re.compile(r"^\s*(?:≥|>=)?\s*(\d+(?:\.\d+)?)\s*A/W\b", re.I)
_JOINT_LABEL = re.compile(r"联合|\bjoint\b", re.I)
_CITED = re.compile(
    r"^(?:[^:：]{1,32}[：:]\s*)?(?:原报告|历史(?:报告|正文)|旧报告|the (?:old|previous|historical) report)"
    r".{0,100}?(?:称|写|认为|said|claimed|stated)", re.I,
)


class DetectorCheckReceipt(BaseModel):
    mode: Literal["at_wavelength", "minimum_across_band", "peak_in_band"]
    checked_wavelength_um: FiniteFloat
    target_responsivity_a_w: FiniteFloat
    target_eqe_percent: FiniteFloat
    photoconductive_gain: FiniteFloat
    mutually_consistent: bool
    joint_min_eqe_fraction: FiniteFloat
    responsivity_at_declared_eqe_a_w: FiniteFloat

    def summary(self) -> str:
        status = "相容" if self.mutually_consistent else "超出允许的效率上限"
        return (
            f"**光电 Checker 回执** · {self.mode}，{self.checked_wavelength_um:g} μm，"
            f"增益 {self.photoconductive_gain:g}。R ≥ {self.target_responsivity_a_w:g} A/W 与 "
            f"EQE ≥ {self.target_eqe_percent:g}% 的联合约束{status}，"
            f"联合 EQE 下限约为 {self.joint_min_eqe_fraction * 100:.2f}%。"
            f"固定在给定 EQE 时，计算 R 约为 {self.responsivity_at_declared_eqe_a_w:.4f} A/W。"
            "回执仅核对公式约束，实际性能仍需同器件、同波长和工作条件下的响应及噪声测量。"
        )

    def review(self, text: str, *, explicit_target: bool = False) -> str | None:
        prose = re.sub(r"```[\s\S]*?```|https?://\S+", "", text)
        prose = re.sub(r"\*{1,3}|(?<!\w)_{1,3}|_{1,3}(?!\w)", "", prose)
        if self._table_has_wrong_joint_bound(prose, explicit_target=explicit_target):
            return "CHECKER_JOINT_BOUND_MISMATCH: " + self.summary()
        for sentence in _SENTENCES.split(prose):
            wavelengths = [float(value) for value in _WAVELENGTH.findall(sentence)]
            if wavelengths and self.checked_wavelength_um not in wavelengths:
                continue
            if sentence.lstrip().startswith(">") or _CITED.search(sentence.strip(" *-")) or re.search(
                r"(?:如果|若|假定|\bif\b|\bassuming\b).{0,90}(?:上限|upper bound|maximum|定值|固定|fixed)",
                sentence, re.I,
            ):
                continue
            target = _R_TARGET.search(sentence)
            if (explicit_target and (target is None or not wavelengths)
                    or target and abs(float(target[1]) - self.target_responsivity_a_w) > 1e-6):
                continue
            if (_LOWER_SCOPE.search(sentence) and _OPTICAL_SCOPE.search(sentence)
                    and self.mutually_consistent and _CONFLICT.search(sentence) and not _NO_CONFLICT.search(sentence)):
                return "CHECKER_CONCLUSION_CONTRADICTION: " + self.summary()
            joint = _JOINT.search(sentence)
            if joint and not wavelengths and _OPTICAL_SCOPE.search(sentence) and _PERCENT_SERIES.search(sentence[joint.end():]):
                return (
                    "CHECKER_UNSCOPED_JOINT_BOUNDS: attach the checked wavelength and target conditions "
                    "to each joint EQE lower bound, or refer directly to the Checker receipts."
                )
            if joint and wavelengths:
                portion = sentence[joint.end():]
                pairs = [value for wave, value in _SCOPED_PERCENT.findall(portion)
                         if float(wave) == self.checked_wavelength_um]
                fallback = _DECLARED_PERCENT.match(portion) if len(set(wavelengths)) == 1 and not pairs else None
                value = pairs[0] if pairs else fallback[1] if fallback else None
                if value is not None and self._wrong_joint_value(value):
                    return "CHECKER_JOINT_BOUND_MISMATCH: " + self.summary()
        return None

    def _wrong_joint_value(self, value: str) -> bool:
        # Accept rounded percentages at the precision actually written.
        decimals = len(value.partition(".")[2])
        return abs(float(value) - self.joint_min_eqe_fraction * 100) > 0.5 * 10 ** -decimals + 0.001

    def _table_has_wrong_joint_bound(self, prose: str, *, explicit_target: bool) -> bool:
        joint_column = None
        target_column = None
        table_target = None
        for line in prose.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                joint_column = None
                target_column = None
                table_target = None
                continue
            cells = line.strip("|").split("|")
            wavelengths = {float(value) for value in _WAVELENGTH.findall(line)}
            if not wavelengths:
                for index, cell in enumerate(cells):
                    if _R_HEADER.search(cell):
                        target_column = index
                    if _OPTICAL_SCOPE.search(cell) and (
                        _JOINT.search(cell) or (_JOINT_LABEL.search(cell) and _LOWER_SCOPE.search(cell))
                    ):
                        joint_column = index
                        table_target = _R_TARGET.search(line)
                continue
            if joint_column is None or joint_column >= len(cells) or wavelengths != {self.checked_wavelength_um}:
                continue
            target = _R_TARGET.search(line) or table_target
            if target is None and target_column is not None and target_column < len(cells):
                target = _R_CELL.search(cells[target_column])
            if (explicit_target and target is None
                    or target and abs(float(target[1]) - self.target_responsivity_a_w) > 1e-6):
                continue
            value = _PERCENT.search(cells[joint_column])
            if value and self._wrong_joint_value(value[1]):
                return True
        return False

    def delivery_review(self) -> DeliveryReview:
        return DeliveryReview(
            f"optical:{self.mode}:{self.checked_wavelength_um}:{self.target_responsivity_a_w}:{self.target_eqe_percent}:{self.photoconductive_gain}",
            self.review,
            self.summary() + " For pointwise targets, obtain a separate minimum-mode Checker receipt "
            "for each other requested wavelength before reporting joint bounds across those points.",
            self.summary(),
            ambiguous_scope=f"optical:{self.checked_wavelength_um}",
            explicit_scope=f"optical:{self.checked_wavelength_um}:{self.target_responsivity_a_w}",
            check_explicit=lambda text: self.review(text, explicit_target=True),
        )
