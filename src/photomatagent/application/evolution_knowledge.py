"""Evidence-bound procedural memory for ordinary workbench repair rounds.

This journal describes delivery repairs. It does not create expert feedback or
scientific validation, and procedure candidates require separate human approval.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np

from photomatagent.application.procedural_skills import ProcedureCreate, ProcedureService
from photomatagent.application.store import WorkbenchStore
from photomatagent.scientific.evolution.experience import TaskContext, task_context_from_episode
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.evolution.models import StrategyArm
from photomatagent.scientific.evolution.strategy import (
    ARM_ORDER, BayesianLinearStrategySelector, NOISE_VARIANCE, PRIOR_PRECISION, feature_vector,
)
from photomatagent.workspace import Workspace

_STOP = {"a", "an", "and", "for", "in", "of", "on", "or", "the", "to", "with", "write", "prepare", "report"}
_MAX_GUIDANCE = 2400


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _terms(text: str) -> set[str]:
    words: set[str] = set()
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            words.update(token[index:index + 2] for index in range(len(token) - 1))
        elif len(token) >= 3 and token not in _STOP:
            words.add(token)
    return words


def _machine_context(report: dict[str, Any]) -> TaskContext:
    issues = [item for item in report.get("issues", []) if isinstance(item, dict)]
    checks = report.get("checks", [])
    return TaskContext(
        hard_constraint_ratio=min(1.0, sum(item.get("repairable") is True for item in issues) / 10),
        soft_constraint_ratio=min(1.0, sum(item.get("repairable") is not True for item in issues) / 10),
        objective_ratio=1.0 if report.get("delivery_status") in {"PASS", "FAIL"} else 0.0,
        operating_condition_ratio=min(1.0, len(checks) / 10) if isinstance(checks, list) else 0.0,
        previous_critical_gap_ratio=1.0 if report.get("scientific_status") == "UNKNOWN" else 0.0,
    )


def _arm(action: str | dict[str, Any]) -> str:
    candidate = action.get("strategy_arm", "STATIC") if isinstance(action, dict) else action
    return str(candidate) if candidate in ARM_ORDER else "STATIC"


def _delivery_reward(before: dict[str, Any], after: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    prior = {str(issue["code"]) for issue in before.get("issues", []) if isinstance(issue, dict) and issue.get("code")}
    later = {str(issue["code"]) for issue in after.get("issues", []) if isinstance(issue, dict) and issue.get("code")}
    closed, introduced = sorted(prior - later), sorted(later - prior)
    passed = after.get("delivery_status") == "PASS" and before.get("delivery_status") != "PASS"
    reward = 1.0 if passed else -1.0 if before.get("delivery_status") == "PASS" and after.get("delivery_status") != "PASS" else 0.0
    reward += 0.2 * len(closed) - 0.2 * len(introduced)
    if reward == 0 and before.get("delivery_status") == after.get("delivery_status") == "FAIL" and not closed:
        reward = -0.25
    return max(-1.0, min(1.0, reward)), closed, introduced


class EvolutionKnowledge:
    """Bridge one workbench task's checker results to approved guidance."""

    def __init__(self, workspace: Path | str, store: WorkbenchStore) -> None:
        self.workspace = Workspace(workspace)
        self.store = store
        self.procedures = ProcedureService(self.workspace.root)

    def guidance(self, task_id: str, run_id: str, text: str) -> str:
        if self.store.run(run_id).task_id != task_id:
            raise ValueError("run does not belong to task")
        query = _terms(text)
        if not query:
            return ""
        ranked: list[tuple[int, str]] = []
        try:
            available = self.procedures.list_skills()
        except (OSError, ValueError):
            return ""
        misses = 0
        for item in available:
            if item.get("status") != "APPROVED_PROCEDURE" or item.get("task_id") == task_id:
                continue
            overlap = len(query & _terms(str(item.get("scope", "")) + " " + str(item.get("description", ""))))
            if overlap:
                ranked.append((overlap, str(item["skill_id"])))
            elif misses < 3:
                self.store.append_event_for_run(run_id, "evolution_guidance_scope_miss", {
                    "skill_id": str(item["skill_id"]), "reason": "scope_terms_unmatched"})
                misses += 1
        parts: list[str] = []
        for _score, skill_id in sorted(ranked, key=lambda row: (-row[0], row[1]))[:3]:
            # read() validates candidate and rendered content; its retrieval receipt is distinct from use.
            try:
                item = self.procedures.read(skill_id)
            except (OSError, ValueError):
                continue
            instruction = str(item["instructions"]).strip()
            note = str(item.get("review_note") or "")[:300]
            piece = f"Approved procedural guidance {skill_id} ({item['scope']}):\n{instruction}\nReview note: {note}\n"
            if len("\n".join(parts)) + len(piece) > _MAX_GUIDANCE:
                self.store.append_event_for_run(run_id, "evolution_guidance_out_of_range", {
                    "skill_id": skill_id, "reason": "bounded_guidance_limit"})
                continue
            self.procedures.record_retrieval(skill_id, task_id=task_id, run_id=run_id)
            self.store.append_event_for_run(run_id, "evolution_guidance_retrieved", {
                "skill_id": skill_id, "candidate_sha256": str(item["candidate_sha256"]),
                "status": "GUIDANCE_RETRIEVED", "scientific_status": "UNKNOWN",
            })
            parts.append(piece)
        return "\n".join(parts)

    def observe(self, task_id: str, run_id: str, round_index: int,
                before: dict[str, Any], after: dict[str, Any], action: str | dict[str, Any]) -> dict[str, Any]:
        if self.store.run(run_id).task_id != task_id or round_index < 1:
            raise ValueError("observation task, run or round invalid")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("checker reports required")
        reward, closed, introduced = _delivery_reward(before, after)
        passed = after.get("delivery_status") == "PASS" and before.get("delivery_status") != "PASS"
        action_text = action if isinstance(action, str) else json.dumps(action, ensure_ascii=False, sort_keys=True)
        record: dict[str, Any] = {
            "task_id": task_id, "run_id": run_id, "round_index": round_index,
            "before_sha256": _digest(before), "after_sha256": _digest(after),
            "before": before, "after": after, "action": action_text[:1000],
            "closed_issues": closed, "introduced_issues": introduced,
            "reward": reward, "scientific_status": str(after.get("scientific_status", "UNKNOWN")),
            "strategy_arm": _arm(action), "context": _machine_context(before).model_dump(),
            "evaluation_links": list(after.get("evaluation_links", [])),
            "provenance": "deterministic_delivery_checker",
        }
        record["record_sha256"] = _digest(record)
        for event in self.store.events_after(task_id, 0):
            if event.type == "evolution_observation" and event.payload.get("record_sha256") == record["record_sha256"]:
                return record
        self.store.append_event_for_run(run_id, "evolution_observation", record)
        retrieved = {str(event.payload.get("skill_id")) for event in self.store.events_after(task_id, 0)
                     if event.run_id == run_id and event.type == "evolution_guidance_retrieved"}
        for skill_id in sorted(retrieved):
            self.store.append_event_for_run(run_id, "evolution_guidance_outcome", {
                "skill_id": skill_id, "observation_sha256": record["record_sha256"],
                "status": "IMPROVED" if reward > 0 else "NEGATIVE_USE", "reward": reward})
        if passed and closed:
            self._capture_candidate(task_id, run_id, round_index, before, after, action_text, record)
        return record

    def _capture_candidate(self, task_id: str, run_id: str, round_index: int,
                           before: dict[str, Any], after: dict[str, Any], action: str,
                           observation: dict[str, Any]) -> None:
        # Require a current-run immutable text artifact and verify its bytes, not merely its ID.
        source = None
        for artifact_id in after.get("artifact_ids", []):
            try:
                row = self.store.artifact(str(artifact_id))
                if row["task_id"] != task_id or row["run_id"] != run_id or row["media_type"] not in {"text/plain", "text/markdown"}:
                    continue
                relative = Path(str(row["path"]))
                expected_parent = Path(".photomatagent") / "workbench" / "artifacts" / task_id / run_id
                if relative.parent != expected_parent:
                    continue
                path = self.workspace.resolve(relative.as_posix(), must_exist=True)
                if path.is_symlink() or not path.is_file() or path.stat().st_size != row["size_bytes"]:
                    continue
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                    continue
                source = row
                break
            except (KeyError, OSError, ValueError):
                continue
        if source is None:
            return
        issue = next((item for item in before.get("issues", []) if isinstance(item, dict)
                      and str(item.get("code", "")) in observation["closed_issues"]), None)
        if issue is None:
            return
        code = str(issue["code"])
        name = re.sub(r"[^a-z0-9-]+", "-", code.lower()).strip("-")[:50] or "delivery-repair"
        if len(name) < 2:
            name = "delivery-repair"
        description = f"Repair checked delivery issue {code}"[:500]
        detail = str(issue.get("detail", "unspecified delivery defect"))[:500]
        checks = after.get("checks", [])
        passed_checks = [str(item.get("code", item.get("check_id", "registered check")))
                         for item in checks if isinstance(item, dict) and item.get("status") == "PASS"] if isinstance(checks, list) else []
        checked_by = ", ".join(passed_checks[:5]) if passed_checks else "the frozen delivery contract"
        instructions = (f"Observed defect {code}: {detail}. Correction: complete the affected deliverable "
                        f"for the original task contract; prior repair action: {action[:350]}. "
                        f"Check the new immutable artifact with {checked_by} and confirm this issue is closed. "
                        "This procedural note does not validate scientific claims.")
        task = self.store.task(task_id)
        candidate = self.procedures.create(ProcedureCreate(name=name, description=description,
            scope=task.goal[:200], instructions=instructions, task_id=task_id, run_id=run_id,
            artifact_id=str(source["artifact_id"])), provenance={
                "source": "automatic_delivery_repair", "round_index": round_index,
                "before_sha256": observation["before_sha256"],
                "after_sha256": observation["after_sha256"],
                "observation_sha256": observation["record_sha256"],
                "artifact": {"artifact_id": source["artifact_id"], "sha256": source["sha256"]},
                "approval": "PENDING", "scientific_validity": "UNKNOWN",
            })
        self.store.append_event_for_run(run_id, "evolution_candidate_captured", {
            "skill_id": candidate["skill_id"], "candidate_sha256": candidate["candidate_sha256"],
            "observation_sha256": observation["record_sha256"], "status": "CANDIDATE",
        })

    def recommend(self, task_id: str, text: str, report: dict[str, Any]) -> dict[str, Any]:
        self.store.task(task_id)
        issues = [item for item in report.get("issues", []) if isinstance(item, dict)]
        repairable = [item for item in issues if item.get("repairable") is True]
        action = "repair_delivery" if report.get("delivery_status") == "FAIL" and repairable else "gather_evidence" if report.get("delivery_status") == "UNKNOWN" else "stop"
        observations = [event.payload for event in self.store.events_after(task_id, 0)
                        if event.type == "evolution_observation"]
        rewards = [float(value) for item in observations
                   if isinstance((value := item.get("reward")), (int, float)) and not isinstance(value, bool)]
        reliability = {"observations": len(observations),
                       "improved": sum(value > 0 for value in rewards),
                       "worsened": sum(value < 0 for value in rewards)}
        # A native scientific revision can use the existing reviewed selector.
        # Ordinary reports lack its source episode and must use fixed rules.
        evolution_id = report.get("evolution_id")
        revision_id = report.get("revision_id")
        if isinstance(evolution_id, str) and isinstance(revision_id, str):
            try:
                native = EvolutionStore(self.workspace)
                task = native.load_task(evolution_id)
                plan = native.load_revision(evolution_id, revision_id)
                episode = native.load_episode(evolution_id, plan.source_version)
                context = task_context_from_episode(task.target, episode)
                reviewed = [item for item in native.list_all_strategy_observations()
                            if item.evolution_id != evolution_id and item.created_at < plan.created_at]
                incomplete = native.incomplete_strategy_observation_chains(plan.created_at)
                selector = BayesianLinearStrategySelector().fit(
                    reviewed, incomplete_observation_chains=incomplete)
                if selector.enabled:
                    choice = selector.select(task, plan, context)
                    return {"policy": "reviewed_thompson", "action": action,
                            "strategy_arm": choice.arm, "strategy_sha256": choice.strategy_sha256,
                            "issue_codes": [str(item.get("code", "")) for item in repairable][:10],
                            "reliability": reliability,
                            "diagnostics": {"observation_count": selector.diagnostics.observation_count,
                                            "distinct_tasks": selector.diagnostics.distinct_tasks}}
                reason = "reviewed_strategy_gate_unmet"
            except (OSError, ValueError, KeyError) as exc:
                reason = f"native_strategy_unavailable:{type(exc).__name__}"
        else:
            reason = "no_native_reviewed_strategy_context"
        machine = self._machine_choice(task_id, report)
        if machine is not None:
            arm, diagnostics = machine
            return {"policy": "machine_delivery_thompson", "action": action,
                    "strategy_arm": arm,
                    "issue_codes": [str(item.get("code", "")) for item in repairable][:10],
                    "reliability": reliability, "diagnostics": diagnostics}
        return {"policy": "fixed", "action": action,
                "issue_codes": [str(item.get("code", "")) for item in repairable][:10],
                "reliability": reliability,
                "diagnostics": {"reason": reason,
                                "scientific_status": str(report.get("scientific_status", "UNKNOWN"))}}

    def _machine_choice(self, task_id: str, report: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        """Fit a separate delivery-only posterior from verified journal entries.

        This intentionally does not construct StrategyObservation or expert feedback.
        """
        rows: list[tuple[str, StrategyArm, TaskContext, float]] = []
        seen: set[str] = set()
        try:
            tasks = self.store.tasks()
            for task in tasks:
                if task.task_id == task_id:
                    continue
                for event in self.store.events_after(task.task_id, 0):
                    if event.type != "evolution_observation":
                        continue
                    record = dict(event.payload)
                    digest = record.pop("record_sha256", None)
                    before, after = record.get("before"), record.get("after")
                    if (not isinstance(digest, str) or digest in seen or _digest(record) != digest
                            or not isinstance(before, dict) or not isinstance(after, dict)
                            or record.get("task_id") != task.task_id or record.get("run_id") != event.run_id
                            or record.get("provenance") != "deterministic_delivery_checker"
                            or record.get("before_sha256") != _digest(before)
                            or record.get("after_sha256") != _digest(after)):
                        continue
                    reward, closed, introduced = _delivery_reward(before, after)
                    arm = record.get("strategy_arm")
                    context = _machine_context(before)
                    if (record.get("reward") != reward or record.get("closed_issues") != closed
                            or record.get("introduced_issues") != introduced
                            or record.get("context") != context.model_dump() or arm not in ARM_ORDER):
                        continue
                    seen.add(digest)
                    rows.append((task.task_id, cast(StrategyArm, arm), context, reward))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        groups = {row[0] for row in rows}
        if len(rows) < 20 or len(groups) < 8:
            return None
        counts = {group: sum(row[0] == group for row in rows) for group in groups}
        design = np.array([feature_vector(context, arm) for group, arm, context, _reward in rows], dtype=float)
        rewards = np.asarray([reward for _group, _arm, _context, reward in rows], dtype=float)
        weights = np.asarray([1.0 / counts[group] for group, _arm, _context, _reward in rows], dtype=float)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_rewards = rewards * np.sqrt(weights)
        precision = PRIOR_PRECISION * np.eye(design.shape[1]) + weighted_design.T @ weighted_design / NOISE_VARIANCE
        try:
            covariance = np.linalg.solve(precision, np.eye(design.shape[1]))
            mean = np.linalg.solve(precision, weighted_design.T @ weighted_rewards / NOISE_VARIANCE)
            seed_material = _digest({"task_id": task_id, "report_sha256": _digest(report),
                                     "training_hashes": sorted(seen)})
            sampled = np.random.default_rng(int(seed_material[:16], 16)).multivariate_normal(mean, covariance)
            context = _machine_context(report)
            arm = max(ARM_ORDER, key=lambda choice: float(np.dot(feature_vector(context, choice), sampled)))
        except (np.linalg.LinAlgError, ValueError):
            return None
        return arm, {"observation_count": len(rows), "distinct_tasks": len(groups),
                     "training_sha256": _digest({"hashes": sorted(seen)}),
                     "source": "verified_machine_delivery_observations",
                     "scientific_status": "UNKNOWN"}
