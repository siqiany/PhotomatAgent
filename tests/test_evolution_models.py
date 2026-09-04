from pydantic import ValidationError
import pytest

from photomatagent.scientific.evolution.models import (
    ExpertFeedbackDraft,
    RubricFlags,
    RubricScores,
    new_evolution_id,
)
from photomatagent.scientific.evolution.rubric import assess_hard_caps, expert_utility


def test_feedback_scores_are_bounded_integers():
    with pytest.raises(ValidationError):
        RubricScores(
            scientific_correctness=6,
            evidence_sufficiency=3,
            novelty=3,
            actionability=3,
            overall=3,
        )


def test_hard_caps_are_suggested_without_rewriting_expert_input():
    scores = RubricScores(
        scientific_correctness=5,
        evidence_sufficiency=5,
        novelty=5,
        actionability=5,
        overall=5,
    )
    result = assess_hard_caps(
        scores,
        RubricFlags(fabricated_source=True),
    )
    assert scores.evidence_sufficiency == 5
    assert result.suggested_scores.evidence_sufficiency == 1
    assert result.suggested_scores.overall == 1
    assert result.reasons


def test_expert_utility_uses_approved_weights():
    scores = RubricScores(
        scientific_correctness=5,
        evidence_sufficiency=1,
        novelty=1,
        actionability=1,
        overall=5,
    )
    assert expert_utility(scores) == pytest.approx(0.35)


def test_generated_evolution_ids_are_path_safe():
    value = new_evolution_id()
    assert value.startswith("evo_")
    assert "/" not in value and ".." not in value
