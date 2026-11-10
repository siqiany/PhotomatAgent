"""Supplementary paired recall probes and a real workbench-owner restart.

Preserves the original strict audit results and report. Clarifies the initially
ambiguous ticket field without supplying any remembered answers to the model.
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
import shutil
from pathlib import Path

import context_compaction_live as live
from fastapi.testclient import TestClient
from photomatagent.models.types import ModelCompleted, ModelRequest, UserMessage
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.context_engine import ContextEngine, ContextEngineConfig
from photomatagent.sessions.store import SessionSnapshot
from photomatagent.tools.surface import ToolSurfacePlanner


ROOT = live.OUTPUT
manifest = json.loads((ROOT / "manifest.json").read_text())
workspace = live.Workspace(manifest["workspace"])
supplement = ROOT / "supplement"
if (supplement / "result.json").exists():
    raise RuntimeError("Supplement already exists; preserve prior results")
recorder = live.Recorder(supplement)
original_calls = json.loads((ROOT / "calls.json").read_text())
live.LIMIT_CALLS -= len(original_calls)
live.LIMIT_TOKENS -= sum(c.get("usage", {}).get("total_tokens", 0) for c in original_calls)
high_config = ContextEngineConfig(**{**manifest["config"], "compact_trigger_tokens": 100000, "compact_target_tokens": 60000})


async def focused_probe(label, snapshot, expected_pending):
    recorder.phase = label
    engine = ContextEngine(config=high_config)
    engine.restore(**snapshot.engine.model_dump(mode="json"))
    surface = ToolSurfacePlanner(live.registry(workspace, snapshot.scientific)).plan()
    context = await engine.build(conversation=snapshot.conversation, scientific=snapshot.scientific,
        context_builder=ContextBuilder(), capability_manifest=surface.manifest.text, surface=surface.stats,
        session_id=manifest["session_id"])
    prompt = (
        "MEMORY AUDIT: Extract factual memory from this current context only; do not use tools/files or infer missing facts. "
        "Return only JSON with constraint_values (C01 through C10, original literal values), "
        "pending_status (P01,P02,P03, status enum only), evidence {row_count,alcohol_quality_r,source_sha256}, "
        "causal_claim_allowed, external_ticket_status, completed_stages. "
        "Clarification of the query field external_ticket_status: it means the exact status field of the local "
        "ScientificState.pending_tasks entry with task_id pending_external_validation. It does not mean its backend "
        "field, the text-only P02 task or a scheduler status. Use UNKNOWN if not present. "
        "Do not change any task state or repeat completed computations."
    )
    provider = live.RecordedProvider(recorder)
    try:
        response = None
        async for event in provider.stream(ModelRequest(messages=[*context.messages, UserMessage(content=prompt)], tools=[], max_output_tokens=1600)):
            if isinstance(event, ModelCompleted):
                response = event.response
        value = live.parse_json(response.text)
    finally:
        await provider.aclose()
    result = {"phase": label, "recall": value, **live.compare_probe(value, expected_pending, manifest["expected_evidence"])}
    print(f"FOCUSED {label}: {result['passed']}/{result['total']}, failed={result['failed']}", flush=True)
    return result


def normalize_original():
    rows = json.loads((ROOT / "memory-audits.json").read_text())
    results = []
    for row in rows:
        value = json.loads(json.dumps(row["recall"]))
        for key, status in value.get("pending_status", {}).items():
            match = re.match(r"^(PENDING|COMPLETED|WAITING_USER_AUTHORIZATION|UNKNOWN)\b", str(status))
            if match:
                value["pending_status"][key] = match[1]
        expected = {**live.PENDING, **({"P01": "COMPLETED"} if row["phase"] == "after-bootstrap" else {})}
        results.append({"phase": row["phase"], **live.compare_probe(value, expected, manifest["expected_evidence"]),
                        "note": "Only strip explanatory suffix after an exact leading pending-status enum; original strict results retained."})
    live.dump(supplement / "normalized-original-audits.json", results)


async def probe_sources():
    results = []
    for label, name in [("baseline-specific-ticket", "baseline-source"), ("auto-specific-ticket", "after-auto-source"),
                        ("manual-specific-ticket", "after-manual-source"), ("continuation-specific-ticket", "after-continuation-source")]:
        snapshot = SessionSnapshot.model_validate_json((ROOT / "snapshots" / (name + ".json")).read_text())
        results.append(await focused_probe(label, snapshot, live.PENDING))
        live.dump(supplement / "paired-probes.json", results)
    return results


def restart_owner():
    # The original process has exited and released its SQLite owner. Construct
    # a new application owner against the same database, with no live runtime.
    store = live.WorkbenchStore(workspace.root / ".photomatagent/workbench.sqlite3")
    def builder(**kwargs):
        scientific = kwargs.get("scientific_state") or live.ScientificState()
        provider = live.RecordedProvider(recorder)
        tools = live.registry(workspace, scientific)
        runtime = live.AgentRuntime(model=provider, tools=tools, workspace=workspace,
            scientific_state=scientific, conversation_state=kwargs.get("conversation_state"),
            budget=kwargs["budget"], session_id=kwargs["session_id"], context_engine_config=high_config,
            permission_policy=live.PolicyRule({tool.name: live.PermissionDecision.ALLOW for tool in tools.list_tools()}, default=live.PermissionDecision.DENY))
        return runtime, live.RuntimeResources((provider,))
    manager = live.RunManager(store, runtime_builder=builder, runtime_options={"workspace_root": workspace.root})
    service = live.WorkbenchService(store=store, manager=manager, workspace_root=workspace.root,
        sessions_dir=workspace.root / ".photomatagent/sessions", artifacts=live.ArtifactService(store, workspace.root))
    app = live.create_app(service=service, manager=manager)
    recorder.phase = "restarted-owner-continuation"
    with TestClient(app) as client:
        before = service.storage.sessions.load(manifest["session_id"])
        response = client.post(f"/api/tasks/{manifest['task_id']}/runs", json={
            "text": "服务已重新启动。继续此前研究，从已有状态说明哪些分析已完成、哪些待办仍未完成，并指出当前证据限制。不要重新计算、改写文件或调用工具。请用简短中文答复。",
            "budget": {"max_model_calls": 3, "max_tokens": 150000, "wall_seconds": 180}, "evolution": {"enabled": False}},
            headers={"Idempotency-Key": "actual-owner-restart"})
        response.raise_for_status()
        client.portal.call(manager.wait_for_idle)
        run_id = response.json()["run_id"]
        after = service.storage.sessions.load(manifest["session_id"])
        events = [e.model_dump(mode="json") for e in store.events_after(manifest["task_id"], 0) if e.run_id == run_id]
        live.dump(supplement / "restart-events.json", events)
        live.dump(supplement / "restart-snapshot.json", after.model_dump(mode="json"))
        result = {"run_id": run_id, "status": store.run(run_id).status,
                  "durable_history_preserves_prefix": after.conversation.messages[:len(before.conversation.messages)] == before.conversation.messages,
                  "evidence_preserved": before.scientific.evidence == after.scientific.evidence,
                  "pending_tasks_preserved": before.scientific.pending_tasks == after.scientific.pending_tasks,
                  "compaction_count_preserved": before.engine.compaction_count == after.engine.compaction_count,
                  "tools_executed": any(e["type"] == "tool_started" for e in events)}
        live.dump(supplement / "restart-result.json", result)
        probe = asyncio.run(focused_probe("restarted-owner-recall", after, {**live.PENDING, "P01": "COMPLETED"}))
        return result, probe


if __name__ == "__main__":
    normalize_original()
    paired = asyncio.run(probe_sources())
    restart, restarted_probe = restart_owner()
    report = workspace.root / "user_output/research/report.md"
    # Preserve the unedited model report. Detect execution-level violations
    # separately from recall: remembering C07 does not ensure obeying it.
    report_text = report.read_text()
    shutil.copyfile(report, supplement / "model-research-report-original.md")
    precision_violations = re.findall(r"(?<![\w.])-?\d+\.\d{4,}(?![\w.])", report_text)
    result = {"paired_probes": [{k:v for k,v in p.items() if k != "recall"} for p in paired],
              "restart": restart, "restart_probe": {k:v for k,v in restarted_probe.items() if k != "recall"},
              "original_report_precision_violations": precision_violations,
              "combined_model_calls": len(original_calls) + len(recorder.calls),
              "combined_observed_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in original_calls + recorder.calls)}
    live.dump(supplement / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
