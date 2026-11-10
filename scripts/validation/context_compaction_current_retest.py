"""Verify the final code on the same real task, including original declarations.

No injected model answers; retains all earlier attempts and their checkpoints.
It replays the old failed report through the deterministic guard in memory and
then drives real automatic/manual compaction and delivery on the existing task.
"""
from __future__ import annotations

import asyncio
import json
import runpy
from typing import Any

import context_compaction_live as live
from fastapi.testclient import TestClient
from photomatagent.runtime.commitments import collect_user_commitments, delivery_constraint_error

live.OUTPUT = live.REPO / "user_output/context-compaction-fixed-v2-20261029"
manifest = json.loads((live.OUTPUT / "manifest.json").read_text())
workspace = live.Workspace(manifest["workspace"])
output = live.OUTPUT / "current-code-retest"
if (output / "result.json").exists():
    raise RuntimeError("Keep previous verification results")
recorder = live.Recorder(output)
profile = {"config": live.ContextEngineConfig(**manifest["config"])}


def builder(**kwargs):
    scientific = kwargs.get("scientific_state") or live.ScientificState()
    provider = live.RecordedProvider(recorder)
    tools = live.registry(workspace, scientific)
    runtime = live.AgentRuntime(model=provider, tools=tools, workspace=workspace,
        scientific_state=scientific, conversation_state=kwargs.get("conversation_state"),
        budget=kwargs["budget"], session_id=kwargs["session_id"], context_engine_config=profile["config"],
        permission_policy=live.PolicyRule({t.name: live.PermissionDecision.ALLOW for t in tools.list_tools()}, default=live.PermissionDecision.DENY))
    return runtime, live.RuntimeResources((provider,))


async def probe(snapshot):
    engine = live.ContextEngine(config=profile["config"].model_copy(update={"compact_trigger_tokens": 100000}))
    engine.restore(**snapshot.engine.model_dump(mode="json"))
    surface = live.ToolSurfacePlanner(live.registry(workspace, snapshot.scientific)).plan()
    context = await engine.build(conversation=snapshot.conversation, scientific=snapshot.scientific,
        context_builder=live.ContextBuilder(), capability_manifest=surface.manifest.text, surface=surface.stats, session_id=manifest["session_id"])
    provider = live.RecordedProvider(recorder)
    request = ("Return only JSON from current memory without tools: constraint_values C01..C10 original literal JSON values; "
        "pending_status P01,P02,P03 exact enums; evidence {row_count,alcohol_quality_r,source_sha256}; "
        "causal_claim_allowed boolean; external_ticket_status means the exact status property of ScientificState.pending_tasks "
        "record task_id pending_external_validation, never its backend or the separate P02 text commitment. UNKNOWN if absent.")
    try:
        response = None
        async for event in provider.stream(live.ModelRequest(messages=[*context.messages, live.UserMessage(content=request)], tools=[], max_output_tokens=1600)):
            if isinstance(event, live.ModelCompleted):
                response = event.response
        value = live.parse_json(response.text)
        return {"recall": value, **live.compare_probe(value, {**live.PENDING, "P01": "COMPLETED"}, manifest["expected_evidence"])}
    finally:
        await provider.aclose()


if __name__ == "__main__":
    store = live.WorkbenchStore(workspace.root / ".photomatagent/workbench.sqlite3")
    manager = live.RunManager(store, runtime_builder=builder, runtime_options={"workspace_root": workspace.root})
    service = live.WorkbenchService(store=store, manager=manager, workspace_root=workspace.root,
        sessions_dir=workspace.root / ".photomatagent/sessions", artifacts=live.ArtifactService(store, workspace.root))
    app = live.create_app(service=service, manager=manager)
    with TestClient(app) as client:
        assert client.portal is not None
        initial = service.storage.sessions.load(manifest["session_id"])
        phases: list[dict[str, Any]] = []
        def run(label, prompt):
            recorder.phase = label
            response = client.post(f"/api/tasks/{manifest['task_id']}/runs", json={"text": prompt,
                "budget": {"max_model_calls": 20, "max_tokens": 400000, "wall_seconds": 300},
                "evolution": {"enabled": False}}, headers={"Idempotency-Key": "current-retest:" + label})
            response.raise_for_status()
            client.portal.call(manager.wait_for_idle)
            run_id = response.json()["run_id"]
            events = [e.model_dump(mode="json") for e in store.events_after(manifest["task_id"], 0) if e.run_id == run_id]
            live.dump(output / (label + "-events.json"), events)
            snapshot = service.storage.sessions.load(manifest["session_id"])
            live.dump(output / (label + "-snapshot.json"), snapshot.model_dump(mode="json"))
            phases.append({"phase": label, "run_id": run_id, "status": store.run(run_id).status, "compactions": snapshot.engine.compaction_count})
            print(phases[-1], flush=True)
            return snapshot
        # Enough actual report reading to exercise the configured trigger. No
        # padding, scripted model replies, new calculations, or repeated values.
        auto = run("automatic", "继续原研究，仅核验现有证据和三项待办。不要重新计算，不改变冻结约束，不执行外部验证。用简短中文说明已有结果的限制。")
        recorder.phase = "automatic-recall"
        auto_probe = asyncio.run(probe(auto))
        profile["config"] = profile["config"].model_copy(update={"compact_trigger_tokens": 100000, "compact_target_tokens": 60000})
        before_manual = run("read-existing-report", "仅用read读取已约定的现有报告并核对精度、来源和待办状态。尚不改写文件，不重做分析，不改变原约束。")
        recorder.phase = "manual"
        response = client.post(f"/api/tasks/{manifest['task_id']}/context/compact", json={"budget": {"max_model_calls": 8, "max_tokens": 300000, "wall_seconds": 300}}, headers={"Idempotency-Key": "current-retest:manual"})
        response.raise_for_status()
        client.portal.call(manager.wait_for_idle)
        manual_run = response.json()["run_id"]
        manual = service.storage.sessions.load(manifest["session_id"])
        live.dump(output / "manual-events.json", [e.model_dump(mode="json") for e in store.events_after(manifest["task_id"], 0) if e.run_id == manual_run])
        live.dump(output / "manual-snapshot.json", manual.model_dump(mode="json"))
        recorder.phase = "manual-recall"
        manual_probe = asyncio.run(probe(manual))
        final = run("final-delivery", "完成此前研究的最终交付核验。按冻结要求核对报告、原始证据、授权阻塞和已完成事项；必要时仅修正报告表述。保持原路径、精度和科学条件，不重算已完成分析、不执行外部验证。给出简短中文交付说明。")
        requirements = collect_user_commitments(final.conversation.messages, final.scientific)
        old = (live.REPO / "user_output/context-compaction-live-20261029/supplement/model-research-report-original.md").read_text()
        report = (workspace.root / "user_output/research/report.md").read_text()
        assert initial.engine is not None and manual.engine is not None
        result = {"phases": phases, "automatic_recall": auto_probe, "manual_recall": manual_probe,
            "automatic_compaction_observed": auto.engine.compaction_count > initial.engine.compaction_count,
            "manual_compaction_observed": manual.engine.compaction_count > before_manual.engine.compaction_count,
            "manual_history_unchanged": manual.conversation == before_manual.conversation,
            "manual_scientific_state_unchanged": manual.scientific == before_manual.scientific,
            "evidence_unchanged": initial.scientific.evidence == final.scientific.evidence,
            "pending_tasks_unchanged": initial.scientific.pending_tasks == final.scientific.pending_tasks,
            "compiled_requirements": requirements.model_dump(mode="json"),
            "old_real_report_rejected": delivery_constraint_error(old, requirements),
            "new_real_report_check": delivery_constraint_error(report, requirements),
            "wrong_authorization_status_rejected": delivery_constraint_error("| P02 | 外部验证 | PENDING |", requirements),
            "calls": len(recorder.calls), "tokens": recorder.total_tokens}
        live.dump(output / "result.json", result)
        (output / "model-research-report-final.md").write_text(report)
        print(json.dumps({k:v for k,v in result.items() if not k.endswith("recall")}, ensure_ascii=False), flush=True)
