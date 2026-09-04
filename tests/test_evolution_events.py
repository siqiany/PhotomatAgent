from __future__ import annotations

import pytest
from pydantic import ValidationError

from photomatagent.runtime.events import (
    EvolutionEpisodeStarted,
    EvolutionTaskCreated,
    ExpertFeedbackRecorded,
    parse_event,
)
from photomatagent.scientific.evolution.events import bounded_summary


@pytest.mark.parametrize(
    ("payload", "expected_kind"),
    [
        (
            {"kind": "evolution_task_created", "evolution_id": "evo_test"},
            "evolution_task_created",
        ),
        (
            {
                "kind": "evolution_episode_started",
                "evolution_id": "evo_test",
                "episode_version": "v001",
            },
            "evolution_episode_started",
        ),
        (
            {
                "kind": "evolution_episode_completed",
                "evolution_id": "evo_test",
                "episode_version": "v001",
            },
            "evolution_episode_completed",
        ),
        (
            {
                "kind": "expert_feedback_recorded",
                "evolution_id": "evo_test",
                "episode_version": "v001",
                "feedback_id": "fb_test",
                "result_sha256": "a" * 64,
                "scores": {"overall": 3},
            },
            "expert_feedback_recorded",
        ),
        (
            {
                "kind": "expert_feedback_compiled",
                "evolution_id": "evo_test",
                "episode_version": "v001",
            },
            "expert_feedback_compiled",
        ),
        (
            {
                "kind": "revision_plan_confirmed",
                "evolution_id": "evo_test",
                "episode_version": "v001",
            },
            "revision_plan_confirmed",
        ),
        (
            {
                "kind": "evolution_iteration_started",
                "evolution_id": "evo_test",
                "episode_version": "v002",
            },
            "evolution_iteration_started",
        ),
        (
            {
                "kind": "evolution_comparison_completed",
                "evolution_id": "evo_test",
                "episode_version": "v002",
            },
            "evolution_comparison_completed",
        ),
        (
            {"kind": "experience_state_changed", "evolution_id": "evo_test"},
            "experience_state_changed",
        ),
        (
            {
                "kind": "evolution_task_accepted",
                "evolution_id": "evo_test",
                "episode_version": "v002",
            },
            "evolution_task_accepted",
        ),
        (
            {"kind": "evolution_task_stopped", "evolution_id": "evo_test"},
            "evolution_task_stopped",
        ),
    ],
)
def test_evolution_events_round_trip(
    payload: dict[str, object], expected_kind: str
) -> None:
    parsed = parse_event(payload)

    assert parsed.kind == expected_kind
    assert getattr(parsed, "evolution_id") == "evo_test"
    assert not hasattr(parsed, "raw_comments")


def test_expert_feedback_event_round_trips_without_raw_feedback_fields() -> None:
    event = ExpertFeedbackRecorded(
        evolution_id="evo_test",
        episode_version="v001",
        feedback_id="fb_test",
        result_sha256="a" * 64,
        scores={"overall": 3},
    )

    parsed = parse_event(event.model_dump(mode="json"))

    assert parsed.kind == "expert_feedback_recorded"
    assert not hasattr(parsed, "raw_comments")
    with pytest.raises(ValidationError):
        ExpertFeedbackRecorded(
            evolution_id="evo_test",
            episode_version="v001",
            feedback_id="fb_test",
            result_sha256="a" * 64,
            scores={"overall": 3},
            raw_comments="must not enter event payloads",  # type: ignore[call-arg]
        )


def test_evolution_event_payload_fields_are_bounded() -> None:
    with pytest.raises(ValidationError):
        EvolutionTaskCreated(evolution_id="expert feedback must not be an id")
    with pytest.raises(ValidationError):
        EvolutionTaskCreated(evolution_id="evo_test", goal_summary="x" * 241)
    with pytest.raises(ValidationError):
        EvolutionEpisodeStarted(evolution_id="evo_test", episode_version="first")
    with pytest.raises(ValidationError):
        ExpertFeedbackRecorded(
            evolution_id="evo_test",
            episode_version="v001",
            feedback_id="fb_test",
            result_sha256="not-a-hash",
            scores={"overall": 3},
        )
    with pytest.raises(ValidationError):
        ExpertFeedbackRecorded(
            evolution_id="evo_test",
            episode_version="v001",
            feedback_id="fb_test",
            result_sha256="a" * 64,
            scores={"overall": 6},
        )
    with pytest.raises(ValidationError):
        ExpertFeedbackRecorded(
            evolution_id="evo_test",
            episode_version="v001",
            feedback_id="fb_test",
            result_sha256="a" * 64,
            scores={"raw expert comment": 3},
        )


def test_bounded_summary_truncates_only_overlong_text() -> None:
    assert bounded_summary("safe") == "safe"
    assert bounded_summary("x" * 241) == "x" * 240
