"""Conservative material-architecture presentation checks."""
from __future__ import annotations

import re

from photomatagent.tools.base import DeliveryReview

_ARCHITECTURE = re.compile(r"梯度|核壳|空间分布|组成分布|径向|相分离|\bradial\b|core[-– ]shell|spatial distribution|phase separation", re.I)
_UNCONFIRMED = re.compile(r"未(?:被)?(?:证实|确认|验证|检出|观察到|检测到)|无法(?:证实|确认|分辨)|不能(?:证实|确认|分辨)|"
                          r"缺少.{0,12}(?:证据|数据)|not (?:confirmed|verified|resolved|detected)|"
                          r"unconfirmed|unverified|cannot (?:confirm|resolve)|lack of evidence|below detection", re.I)
_CLASSIFY = re.compile(r"(?:应|须|就|则|可直接|默认)?(?:按|视为|判为|归为|当作|归入).{0,35}(?:均匀|单相|同质|homogeneous|uniform|single.phase)|"
                       r"(?:treat|classif\w*|regard|assume|model).{0,50}(?:homogeneous|uniform|single.phase)", re.I)
_QUALIFIED = re.compile(r"(?:不得|不可|不能|不应|不要|不足以|不等于).{0,8}(?:按|视为|判为|归为|当作|归入)|"
                        r"(?:只|仅)作|假设|参照|对照|基线|敏感性|代理|\b(?:not|never|cannot|reference|baseline|proxy|hypothesis|control|sensitivity)\b", re.I)
_SENTENCES = re.compile(r"[。！？；;\n]+|(?<=[.!?])\s+")


def review_architecture_claim(text: str) -> str | None:
    prose = re.sub(r"```[\s\S]*?```|https?://\S+", "", text)
    for sentence in _SENTENCES.split(prose):
        if sentence.lstrip().startswith(">") or re.match(
            r"\s*(?:历史报告|原报告|the (?:old|historical) report).{0,20}(?:称|写|said|claimed)", sentence, re.I,
        ):
            continue
        if _ARCHITECTURE.search(sentence) and (uncertain := _UNCONFIRMED.search(sentence)):
            consequence = sentence[uncertain.end():]
            if _CLASSIFY.search(consequence) and not _QUALIFIED.search(consequence):
                return (
                    "ARCHITECTURE_EVIDENCE_GAP: an unconfirmed spatial structure stays UNKNOWN. "
                    "Keep competing structural hypotheses distinct. A homogeneous or single-phase calculation "
                    "may be labeled as a separate reference or sensitivity scenario."
                )
    return None


def architecture_review(goal: str) -> DeliveryReview | None:
    if not _ARCHITECTURE.search(goal):
        return None
    return DeliveryReview(
        "material_architecture", review_architecture_claim,
        "Keep an unresolved spatial profile UNKNOWN. Label competing structures as hypotheses "
        "and identify any uniform calculation as a scoped reference.",
    )
