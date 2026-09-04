"""Deterministic baseline strategy selection for evolution revisions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from photomatagent.scientific.evolution.models import (
    EvolutionTask,
    RevisionPlan,
    StrategyVersion,
)


class FixedStrategySelector:
    """Materialize the fixed-v1 choice already recorded by the planner."""

    def select(self, task: EvolutionTask, plan: RevisionPlan) -> StrategyVersion:
        if plan.evolution_id != task.evolution_id:
            raise ValueError(
                "revision plan and task must belong to the same evolution task"
            )
        parameters: dict[str, Any] = {
            "selector": "fixed-v1",
            "revision_id": plan.revision_id,
        }
        payload: dict[str, Any] = {
            "evolution_id": task.evolution_id,
            "revision_id": plan.revision_id,
            "arm": plan.strategy_arm,
            "reason": plan.strategy_reason[:1_000],
            "parameters": parameters,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return StrategyVersion(
            strategy_id=f"strategy_{digest[:10]}",
            evolution_id=task.evolution_id,
            arm=plan.strategy_arm,
            reason=plan.strategy_reason[:1_000],
            parameters=parameters,
            strategy_sha256=digest,
            cutoff_at=plan.created_at,
            created_at=plan.created_at,
        )


__all__ = ["FixedStrategySelector"]
