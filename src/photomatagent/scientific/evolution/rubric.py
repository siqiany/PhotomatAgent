"""Deterministic ``expert-review-v1`` rubric and hard-cap suggestions."""

from __future__ import annotations

from pydantic import Field

from photomatagent.scientific.evolution.models import (
    RubricFlags,
    RubricScores,
    StrictModel,
    derive_hard_cap_suggestion,
)

RUBRIC_VERSION = "expert-review-v1"

RUBRIC_DIMENSIONS = {
    "scientific_correctness": "科学正确性",
    "evidence_sufficiency": "证据充分性",
    "novelty": "创新性",
    "actionability": "可执行性",
    "overall": "总体等级",
}

RUBRIC_ANCHORS = {
    "scientific_correctness": (
        "存在根本错误，结果不可使用",
        "关键错误可能改变结论，需要大修",
        "核心方向合理，但需要重要修正",
        "主要结论可靠，仅有局部问题",
        "科学逻辑、假设和不确定性处理稳健",
    ),
    "evidence_sufficiency": (
        "无可追溯证据或疑似伪造",
        "核心结论依赖摘要、二手资料或无依据预测",
        "有相关支持，但仍有重要证据缺口",
        "核心结论有全文或一手证据，局限明确",
        "多源可审计证据链，妥善处理冲突和缺口",
    ),
    "novelty": (
        "把成熟结果包装成创新，无定义和基线",
        "只有表面变化，无机制或比较",
        "组分或工艺创新假设合理，但验证不完整",
        "创新类型、基线、机制和比较清晰",
        "系统检索后仍成立，并有定量优势与验证路线",
    ),
    "actionability": (
        "没有可用流程或下一步",
        "只有方向和少数参数",
        "有主要步骤，但缺重要原料、设备、控制或质检",
        "路线基本可复现，输入、设备、参数和质控齐全",
        "接近执行级，包含备选、失败判据、安全和表征",
    ),
    "overall": (
        "拒绝并重做",
        "大修",
        "完成明确修改后可用",
        "小修后可进入下一阶段",
        "专家认可，可进入当前任务的下一阶段",
    ),
}


class HardCapAssessment(StrictModel):
    original_scores: RubricScores
    suggested_scores: RubricScores
    reasons: list[str] = Field(default_factory=list)


def assess_hard_caps(
    scores: RubricScores,
    flags: RubricFlags,
) -> HardCapAssessment:
    """Suggest deterministic caps while retaining the expert's original scores."""

    suggested, reasons = derive_hard_cap_suggestion(scores, flags)

    return HardCapAssessment(
        original_scores=scores.model_copy(deep=True),
        suggested_scores=suggested,
        reasons=reasons,
    )


def expert_utility(scores: RubricScores) -> float:
    """Compute the approved utility; overall is deliberately not a component."""

    normalized = lambda value: (value - 1) / 4
    return round(
        0.35 * normalized(scores.scientific_correctness)
        + 0.30 * normalized(scores.evidence_sufficiency)
        + 0.15 * normalized(scores.novelty)
        + 0.20 * normalized(scores.actionability),
        6,
    )
