"""Task-scoped automatic repair; checks never delegate their verdict to a model."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from photomatagent.application.models import EvolutionOptions
from photomatagent.application.skills import CHECKERS
from photomatagent.application.store import WorkbenchStore
from photomatagent.workspace import Workspace

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]
# Match actual unfinished control markers, not ordinary prose discussing them.
_INCOMPLETE = re.compile(r"(?:<!--\s*(?:CONTINUE|TODO|PLACEHOLDER)\s*-->|\[TO BE CONTINUED\])\s*$", re.I)
_MAX_CHECK_BYTES = 2 * 1024 * 1024


def check_delivery(store: WorkbenchStore, workspace: Workspace | None, run: Any,
                   options: EvolutionOptions, scientific: Any = None,
                   artifact_ids: set[str] | None = None) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for row in store.artifacts_for_task(run.task_id):
        if row["run_id"] == run.run_id and (artifact_ids is None or row["artifact_id"] in artifact_ids):
            latest[row["filename"]] = row
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    verified: dict[str, Path] = {}

    def issue(code: str, detail: str, repairable: bool = True) -> None:
        issues.append({"code": code, "detail": detail[:1000], "repairable": repairable})

    for name in options.required_artifacts:
        if name not in latest:
            issue("ARTIFACT_MISSING", f"Required file {name} has no registered snapshot in this run.")
    events = [e for e in store.events_after(run.task_id, 0) if e.run_id == run.run_id]
    start = max((i for i, e in enumerate(events) if e.type == "loop_started"), default=-1)
    responses = [e for e in events[start + 1:] if e.type == "model_response_completed" and e.payload.get("tool_call_count") == 0]
    if responses:
        iteration = responses[-1].payload.get("iteration")
        final_text = "".join(str(e.payload.get("text", "")) for e in events[start + 1:]
                             if e.type == "text_delta" and e.payload.get("iteration") == iteration)
        if _INCOMPLETE.search(final_text):
            issue("FINAL_RESPONSE_INCOMPLETE", "Final assistant response ends with an unfinished delivery marker.")
    for name, row in latest.items():
        if workspace is None:
            issue("ARTIFACT_UNAVAILABLE", f"Cannot independently read snapshot {name}.", False)
            continue
        try:
            path = workspace.resolve(row["path"], must_exist=True)
            if path.is_symlink() or any(p.is_symlink() for p in (workspace.root / row["path"]).parents if workspace.contains(p)):
                raise ValueError("snapshot link")
            if path.stat().st_size != row["size_bytes"] or path.stat().st_size > _MAX_CHECK_BYTES:
                raise ValueError("snapshot size")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("snapshot digest")
            verified[name] = path
            if not data.strip():
                issue("ARTIFACT_EMPTY", f"{name} is empty.")
            elif row["media_type"] in {"text/markdown", "text/plain"}:
                if _INCOMPLETE.search(data.decode("utf-8", errors="replace")):
                    issue("ARTIFACT_INCOMPLETE", f"{name} ends with an unfinished delivery marker.")
        except (OSError, ValueError, TypeError):
            issue("ARTIFACT_INTEGRITY_UNKNOWN", f"Snapshot {name} cannot be verified.", False)
    inputs: dict[str, Path] = {}
    if workspace is not None:
        for key, artifact_id in options.input_artifact_ids.items():
            try:
                row = store.artifact(artifact_id)
                if row["task_id"] != run.task_id:
                    raise ValueError("input scope")
                path = workspace.resolve(row["path"], must_exist=True)
                if path.stat().st_size > _MAX_CHECK_BYTES or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError("input digest")
                inputs[key] = path
            except (KeyError, ValueError, OSError):
                issue("CHECKER_INPUT_UNKNOWN", f"Bound input {key} is unavailable.", False)
    for spec in options.checks:
        output_path = verified.get(spec.output_path)
        result = CHECKERS[spec.checker_id](spec, inputs, output_path) if output_path is not None else {"status": "UNKNOWN", "gap": "CHECKER_OUTPUT_MISSING"}
        checks.append({"filename": spec.output_path, **result})
        if result["status"] != "PASS":
            issue("CHECKER_" + result["status"], f"{spec.output_path}: {json.dumps(result, ensure_ascii=False)}", result["status"] == "FAIL" or output_path is None)
    science_status = "UNKNOWN"
    if options.target is not None and scientific is not None:
        from photomatagent.scientific.loop.candidate import extract_candidate_from_state
        from photomatagent.scientific.loop.evaluation import ScientificEvaluator
        evaluated = ScientificEvaluator(options.target).evaluate(extract_candidate_from_state(scientific), scientific)
        checks.append({"kind": "scientific", **evaluated.model_dump(mode="json")})
        if evaluated.verdict == "PASS" and evaluated.hard_constraints_passed and not evaluated.critical_evidence_gaps:
            science_status = "PASS"
        elif evaluated.violations:
            science_status = "FAIL"
        for gap in evaluated.critical_evidence_gaps:
            issue("SCIENTIFIC_EVIDENCE_UNKNOWN", gap, False)
        for violation in evaluated.violations:
            issue("SCIENTIFIC_CONSTRAINT_FAIL", str(violation), True)
        if science_status == "UNKNOWN" and not any(i["code"].startswith("SCIENTIFIC_") for i in issues):
            issue("SCIENTIFIC_EVIDENCE_UNKNOWN", evaluated.rationale or "No independent scientific verdict is available.", False)
    delivery_issues = [i for i in issues if not i["code"].startswith("SCIENTIFIC_")]
    delivery_status = "FAIL" if any(i["repairable"] for i in delivery_issues) else "UNKNOWN" if delivery_issues else "PASS"
    evaluation_links = [str(e.payload["evaluation_id"]) for e in store.events_after(run.task_id, 0)
                        if e.run_id == run.run_id and e.type == "evaluation_experience_recorded" and e.payload.get("evaluation_id")]
    return {"delivery_status": delivery_status, "scientific_status": science_status,
            "issues": issues, "artifact_ids": [row["artifact_id"] for row in latest.values()], "checks": checks,
            "contract_explicit": bool(options.required_artifacts or options.checks),
            "evaluation_links": list(dict.fromkeys(evaluation_links)),
            "limitations": ["Delivery checks do not establish material/device validity."]}


def record_external_action(store: WorkbenchStore, workspace: Workspace | None, run: Any) -> None:
    """Join trusted QD/replay results to the task journal without rerunning them."""
    operation = run.request.host_operation or {}
    name = operation.get("operation")
    if name not in {"qd_revision", "skill_replay"}:
        return
    events = [e for e in store.events_after(run.task_id, 0) if e.run_id == run.run_id]
    selected: set[str] | None = None
    source: dict[str, Any] = {}
    baseline: dict[str, Any] | None = None
    if name == "qd_revision":
        finished = [e.payload for e in events if e.type == "qd_completed"]
        trials = [e.payload for e in events if e.type == "qd_trial"]
        best_id = finished[-1].get("best_trial_id") if finished else None
        source = next((t for t in trials if t.get("trial_id") == best_id), {})
        baseline = trials[0] if trials else None
        refs = source.get("output_refs", [])
        selected = {row["artifact_id"] for row in store.artifacts_for_task(run.task_id)
                    if row["run_id"] == run.run_id and any(row["path"] == ref.get("path") and row["sha256"] == ref.get("sha256") for ref in refs)}
        passed = bool(finished and finished[-1].get("status") == "SOLVED" and source.get("qualification") == "PASS")
    else:
        uses = [e.payload for e in events if e.type == "skill_use"]
        source = uses[-1] if uses else {}
        checker = source.get("checker")
        passed = source.get("code") == "OK" and isinstance(checker, dict) and checker.get("status") == "PASS"
    report = check_delivery(store, workspace, run, run.request.evolution, artifact_ids=selected)
    report.update({"round_index": 1, "action": name, "method_check": "PASS" if passed else "UNKNOWN", "source": source})
    if not passed:
        report["delivery_status"] = "UNKNOWN"
        report["issues"].append({"code": "METHOD_CHECK_UNRESOLVED", "detail": f"{name} did not produce a qualified selected result.", "repairable": False})
    store.append_event_for_run(run.run_id, "closed_loop_checked", report)
    if workspace is not None and baseline is not None:
        from photomatagent.application.evolution_knowledge import EvolutionKnowledge
        prior = {"delivery_status": "PASS" if baseline.get("qualification") == "PASS" else "FAIL", "scientific_status": "UNKNOWN",
                 "issues": [] if baseline.get("qualification") == "PASS" else [{"code": "METHOD_CHECK_UNRESOLVED", "detail": "Baseline independent checker did not pass.", "repairable": True}],
                 "artifact_ids": []}
        EvolutionKnowledge(workspace.root, store).observe(run.task_id, run.run_id, 1, prior, report, {"action": name, "strategy_arm": "EVIDENCE_FIRST"})
    store.append_event_for_run(run.run_id, "closed_loop_completed", {"status": "DELIVERY_COMPLETE" if passed and report["delivery_status"] == "PASS" else "NEEDS_EVIDENCE",
                               "rounds": 1, "delivery_status": report["delivery_status"], "scientific_status": "UNKNOWN"})


class ResearchClosedLoop:
    """Reuses the sole runtime and budget. The durable event journal owns progress."""

    def __init__(self, store: WorkbenchStore, workspace: Workspace | None, knowledge: Any = None):
        self.store, self.workspace, self.knowledge = store, workspace, knowledge

    async def run(self, runtime: Any, run: Any, original: str, emit: Emit,
                  drive: Callable[[str], Awaitable[str]]) -> None:
        options = run.request.evolution
        prompt = original
        if self.knowledge is not None:
            guidance = self.knowledge.guidance(run.task_id, run.run_id, original)
            if guidance:
                prompt += "\n\nRetrieved procedural guidance (unvalidated outside its recorded scope):\n" + guidance
        if options.required_artifacts or options.checks:
            prompt += "\n\nFrozen delivery contract: " + options.model_dump_json(exclude={"enabled", "max_rounds", "patience", "target"})
        previous: dict[str, Any] | None = None
        previous_signature = ""
        repeats = 0
        last: dict[str, Any] = {"delivery_status": "UNKNOWN", "scientific_status": "UNKNOWN"}
        status = "ROUND_LIMIT"
        action = "STATIC"
        policy: dict[str, Any] = {"action": action, "strategy_arm": "STATIC", "policy": "fixed"}
        for round_index in range(1, options.max_rounds + 1):
            await emit("closed_loop_round_started", {"round_index": round_index, "phase": "research" if round_index == 1 else "repair", "action": action})
            reason = await drive(prompt)
            if reason not in {"final_response", "completed"}:
                status = "BUDGET_EXHAUSTED" if reason in {"max_iterations", "token_budget_exhausted"} else "EXECUTION_STOPPED"
                break
            last = check_delivery(self.store, self.workspace, run, options, getattr(runtime, "scientific_state", None))
            last["round_index"] = round_index
            await emit("closed_loop_checked", last)
            if previous is not None and self.knowledge is not None:
                self.knowledge.observe(run.task_id, run.run_id, round_index, previous, last, policy)
            repairable = [i for i in last["issues"] if i["repairable"]]
            if not repairable:
                status = "NEEDS_EVIDENCE" if last["issues"] else "DELIVERY_COMPLETE"
                break
            signature = hashlib.sha256(json.dumps([(i["code"], i["detail"]) for i in repairable], sort_keys=True).encode()).hexdigest()
            repeats = repeats + 1 if signature == previous_signature else 0
            previous_signature = signature
            if repeats >= options.patience:
                status = "STALLED"
                break
            if round_index == options.max_rounds:
                break
            policy = self.knowledge.recommend(run.task_id, original, last) if self.knowledge is not None else {"action": "EVIDENCE_FIRST", "selector": "fixed"}
            action = str(policy.get("action", policy.get("arm", "EVIDENCE_FIRST")))
            await emit("closed_loop_action_selected", {"round_index": round_index, **policy})
            previous = last
            prompt = (original + "\n\nAutomatic revision: preserve every original target and scientific condition. "
                      "Repair the independently checked defects below using the registered tools. "
                      "Do not lower targets, alter the contract, invent evidence, or claim a scientific PASS. "
                      "Write a new complete snapshot of the affected deliverables; remove unfinished control markers by completing the content.\n"
                      + json.dumps({"strategy": policy, "contract": options.model_dump(mode="json"), "issues": last["issues"], "artifact_ids": last["artifact_ids"]}, ensure_ascii=False))
        await emit("closed_loop_completed", {"status": status, "rounds": round_index,
                                             "delivery_status": last["delivery_status"], "scientific_status": last["scientific_status"]})
