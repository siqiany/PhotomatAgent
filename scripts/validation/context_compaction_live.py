"""Opt-in real-provider compaction experiment; no fake responses or HPC access.

Run from the repository root with ``uv run python scripts/validation/context_compaction_live.py``.
Uses existing model configuration without copying or logging credentials. The
workbench database, tools and study files are confined to a new experiment root.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from photomatagent.application.models import TaskCreate
from photomatagent.application.runner import RunManager
from photomatagent.application.runtime_factory import RuntimeResources
from photomatagent.application.service import WorkbenchService
from photomatagent.application.store import WorkbenchStore
from photomatagent.model_settings import public_settings
from photomatagent.models.factory import create_provider
from photomatagent.models.types import ModelCompleted, ModelRequest, UserMessage
from photomatagent.redaction import redact_secrets
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.context_engine import ContextEngine, ContextEngineConfig
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import PermissionDecision, PolicyRule
from photomatagent.scientific.capabilities.contracts import ScientificEvidence, ScientificToolResult
from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.tasks import ScientificTask
from photomatagent.tools.base import Tool
from photomatagent.tools.bridges import ToolCallBridge, ToolDescribeTool, ToolSearchTool
from photomatagent.tools.edit import EditTool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.read import ReadTool
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool
from photomatagent.tools.surface import ToolCatalog, ToolSurfacePlanner
from photomatagent.tools.write import WriteTool
from photomatagent.web.app import create_app
from photomatagent.web.artifacts import ArtifactService
from photomatagent.workspace import Workspace

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "user_output/context-compaction-live-20261029"
WORKSPACE_ROOT = REPO / "tmp/context-live-20261029/workspace"
SOURCE = "https://archive.ics.uci.edu/dataset/186/wine+quality"
LIMIT_CALLS = 64
LIMIT_TOKENS = 800_000
CONSTRAINTS = {
    "C01": "red_only", "C02": 43127, "C03": 7,
    "C04": False, "C05": "retain_duplicates", "C06": "UNKNOWN",
    "C07": 3, "C08": "offline_no_hpc_no_install", "C09": "zh-CN",
    "C10": "user_output/research/report.md",
}
PENDING = {"P01": "PENDING", "P02": "WAITING_USER_AUTHORIZATION", "P03": "UNKNOWN"}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact_secrets(value), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


class Recorder:
    """Observe only canonical messages; never log transport headers or keys."""
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.phase = "setup"

    def start(self, request: ModelRequest) -> tuple[int, float]:
        with self.lock:
            if len(self.calls) >= LIMIT_CALLS or self.total_tokens >= LIMIT_TOKENS:
                raise RuntimeError("LIVE_EXPERIMENT_BUDGET_EXHAUSTED")
            number = len(self.calls) + 1
            row = {"call": number, "phase": self.phase, "status": "STARTED",
                   "kind": "summary" if "Return only JSON matching CompactionState" in getattr(request.messages[0], "content", "") else "research"}
            self.calls.append(row)
            dump(self.root / "calls" / f"{number:03d}-request.json", request.model_dump(mode="json"))
            dump(self.root / "calls.json", self.calls)
        return number, time.monotonic()

    def finish(self, number: int, started: float, response=None, error=None) -> None:
        with self.lock:
            row = self.calls[number - 1]
            row.update(status="ERROR" if error else "COMPLETED", seconds=round(time.monotonic() - started, 3))
            if response is not None:
                row.update(finish_reason=response.finish_reason, usage=response.usage.model_dump(), tool_calls=len(response.tool_calls))
                self.total_tokens += response.usage.resolved_total_tokens
                dump(self.root / "calls" / f"{number:03d}-response.json", response.model_dump(mode="json"))
            if error:
                row["error"] = str(error)[:500]
            dump(self.root / "calls.json", self.calls)


class RecordedProvider:
    def __init__(self, recorder: Recorder):
        self.inner = create_provider(workspace=REPO)
        if self.inner.provider == "fake":
            raise RuntimeError("A real configured provider is required")
        self.provider, self.model = self.inner.provider, self.inner.model
        self.recorder = recorder

    async def stream(self, request):
        number, started = self.recorder.start(request)
        try:
            async with asyncio.timeout(150):
                async for event in self.inner.stream(request):
                    if isinstance(event, ModelCompleted):
                        self.recorder.finish(number, started, event.response)
                    yield event
        except BaseException as error:
            self.recorder.finish(number, started, error=error)
            raise

    async def aclose(self):
        await self.inner.aclose()


class WineAnalysisTool(Tool):
    """Deterministic calculations on the actual downloaded empirical dataset."""
    name = "research_wine_analyze"
    namespace = "research_test"
    effect_class = "READ_ONLY"
    exposure = ToolExposure.DIRECT
    description = (
        "Analyze the local real UCI red-wine CSV. action overview returns data integrity, "
        "quality counts and correlation evidence; feature returns descriptive statistics, "
        "quality associations and a bounded row-level audit; bootstrap returns the 500-resample "
        "alcohol-quality correlation interval with the specified seed. No network, installs or HPC. "
        "Calculations are reproducible associations, not causal or externally validated findings."
    )
    input_schema = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["overview", "feature", "bootstrap"]},
        "feature": {"type": "string"}, "seed": {"type": "integer"},
    }, "required": ["action"], "additionalProperties": False}

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    async def execute(self, arguments):
        path = self.workspace.resolve("data/winequality-red.csv")
        data = path.read_bytes()
        rows = list(csv.DictReader(data.decode().splitlines(), delimiter=";"))
        columns = {key: [float(row[key]) for row in rows] for key in rows[0]}
        quality, alcohol = columns["quality"], columns["alcohol"]
        sha = hashlib.sha256(data).hexdigest()
        action = arguments["action"]
        evidence = []
        updates = []
        limitations = "Observational sample associations only; no causal inference or external validation. Units absent from source remain UNKNOWN."
        def ev(identifier, prop, value, unit, method):
            return ScientificEvidence(id=identifier, subject="UCI red Vinho Verde wine", property=prop,
                value=value, unit=unit, source=SOURCE, source_type="derived", fidelity="empirical",
                method=method, summary=limitations, limitations=limitations,
                provenance={"sha256": sha, "rows": len(rows), "file": "data/winequality-red.csv"},
                conditions={"variant": "red", "duplicates": "retained"})
        result = {"source": SOURCE, "sha256": sha, "local_data_file": "data/winequality-red.csv", "limitations": limitations, "action": action}
        if action == "overview":
            distribution = dict(sorted(Counter(int(q) for q in quality).items()))
            corr = statistics.correlation(alcohol, quality)
            evidence = [ev("sev_wine_rows", "row_count", len(rows), "rows", "CSV row count; retain duplicate rows"),
                        ev("sev_wine_alcohol_r", "alcohol_quality_pearson_r", corr, "1", "Pearson sample correlation"),
                        ev("sev_wine_quality_counts", "quality_distribution", distribution, "rows", "Count sensory scores")]
            updates = [ScientificTask(task_id="pending_external_validation", backend="awaiting_user_authorization", status="PENDING")]
            result.update(row_count=len(rows), columns=list(columns), quality_distribution=distribution,
                          duplicate_rows=len(rows) - len({tuple(row.items()) for row in rows}), alcohol_quality_pearson_r=corr,
                          missing_cells=sum(not v for row in rows for v in row.values()), high_quality_count=sum(q >= 7 for q in quality),
                          note="pending_external_validation is a local waiting ticket, not a submitted job.")
        elif action == "feature":
            name = arguments.get("feature")
            if name not in columns or name == "quality":
                return ScientificToolResult(output="Unknown feature; choose a physicochemical CSV column", is_error=True)
            x = columns[name]
            mean, sd = statistics.mean(x), statistics.stdev(x)
            result.update(feature=name, n=len(x), mean=mean, sample_sd=sd, minimum=min(x), maximum=max(x),
                          pearson_r=statistics.correlation(x, quality),
                          group_means={str(q): statistics.mean(v for v, y in zip(x, quality) if int(y) == q) for q in sorted(set(map(int, quality)))},
                          value_unit="UNKNOWN", audit_rows=[{"row": i + 1, "value": x[i], "quality": quality[i],
                                "standardized_deviation": round((x[i] - mean) / sd, 8)} for i in range(130)])
            evidence = [ev("sev_wine_feature_" + str(name).replace(" ", "_"), str(name) + "_quality_r", result["pearson_r"], "1", "Pearson sample correlation; no row deletion")]
        else:
            seed = int(arguments.get("seed", 43127))
            rng = random.Random(seed)
            resampled = []
            for _ in range(500):
                indices = [rng.randrange(len(rows)) for _ in rows]
                resampled.append(statistics.correlation([alcohol[i] for i in indices], [quality[i] for i in indices]))
            resampled.sort()
            def quantile(p):
                at = p * (len(resampled) - 1)
                low = math.floor(at)
                return resampled[low] + (at - low) * (resampled[min(low + 1, len(resampled) - 1)] - resampled[low])
            result.update(seed=seed, resamples=500, pearson_r=statistics.correlation(alcohol, quality),
                          interval_95=[quantile(.025), quantile(.975)], method="iid percentile bootstrap, sample original rows with replacement")
            evidence = [ev("sev_wine_bootstrap", "alcohol_quality_bootstrap_interval", result["interval_95"], "1", "500-resample percentile bootstrap, seed=" + str(seed))]
            updates = [ScientificTask(task_id="P01", backend="local_bootstrap", status="COMPLETED",
                result_reference="evidence:sev_wine_bootstrap")]
        return ScientificToolResult(output=json.dumps(result, ensure_ascii=False), evidence=evidence, state_updates=[*evidence, *updates])


def registry(workspace, scientific):
    tools = ToolRegistry()
    tools.register_all([WineAnalysisTool(workspace), ReadTool(workspace), WriteTool(workspace), EditTool(workspace), ScientificStateInspectTool(scientific)])
    catalog = ToolCatalog(tools)
    tools.register_all([ToolSearchTool(catalog), ToolDescribeTool(catalog), ToolCallBridge()])
    return tools


def parse_json(text):
    decoder = json.JSONDecoder()
    for position, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[position:])
                if isinstance(value, dict) and "constraint_values" in value:
                    return value
            except ValueError:
                pass
    raise ValueError("Memory probe did not return its required JSON object")


def compare_probe(value, expected_pending, expected_evidence):
    constraints = value.get("constraint_values", {})
    pending = value.get("pending_status", {})
    facts = value.get("evidence", {})
    checks = {key: constraints.get(key) == answer for key, answer in CONSTRAINTS.items()}
    checks.update({key: pending.get(key) == answer for key, answer in expected_pending.items()})
    checks["E01_rows"] = facts.get("row_count") == expected_evidence["rows"]
    try:
        checks["E02_correlation"] = abs(float(facts.get("alcohol_quality_r")) - expected_evidence["r"]) <= 0.0005
    except (ValueError, TypeError):
        checks["E02_correlation"] = False
    checks["E03_provenance"] = facts.get("source_sha256") == expected_evidence["sha256"]
    checks["E04_no_causal_claim"] = value.get("causal_claim_allowed") is False
    checks["P04_structured_ticket"] = value.get("external_ticket_status") == "PENDING"
    return {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "failed": [key for key, passed in checks.items() if not passed]}


class Experiment:
    def __init__(self):
        if (OUTPUT / "manifest.json").exists():
            raise RuntimeError("Experiment output exists; preserve it and choose a new output directory")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.root = WORKSPACE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = Workspace(self.root)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        for name in ("winequality-red.csv", "winequality.names"):
            (self.root / "data" / name).write_bytes((REPO / "tmp/context-live-20261029" / name).read_bytes())
        self.recorder = Recorder(OUTPUT)
        self.config = ContextEngineConfig(compact_trigger_tokens=18_000, compact_target_tokens=10_000,
            protect_recent_turns=1, recent_transaction_count=1, summary_max_tokens=4096,
            summary_chunk_output_tokens=2048, response_reserve_tokens=4096, summary_max_calls=6)
        self.store = WorkbenchStore(self.root / ".photomatagent/workbench.sqlite3")
        self.manager = RunManager(self.store, runtime_builder=self.build, runtime_options={"workspace_root": self.root})
        self.service = WorkbenchService(store=self.store, manager=self.manager, workspace_root=self.root,
            sessions_dir=self.root / ".photomatagent/sessions", artifacts=ArtifactService(self.store, self.root))
        self.app = create_app(service=self.service, manager=self.manager)
        self.task = self.service.create_task(TaskCreate(title="真实模型长科研任务：红葡萄酒理化关联与不确定性", goal="分析 UCI 红葡萄酒数据的理化关联和不确定性，生成可复核中文研究报告，严格保留未知与未完成事项。"))
        self.session = self.store.task(self.task).session_id
        self.results = []
        self.probes = []
        import numpy as np
        records = list(csv.DictReader((self.root / "data/winequality-red.csv").read_text().splitlines(), delimiter=";"))
        self.expected = {"rows": len(records), "r": float(np.corrcoef([float(r["alcohol"]) for r in records], [float(r["quality"]) for r in records])[0, 1]),
                         "sha256": hashlib.sha256((self.root / "data/winequality-red.csv").read_bytes()).hexdigest()}
        settings = public_settings(REPO)["main"]
        dump(OUTPUT / "manifest.json", {"date": "2026-10-29", "task_id": self.task, "session_id": self.session,
            "workspace": str(self.root), "model": settings["model"], "provider": settings["provider"],
            "config": self.config.model_dump(), "constraints": CONSTRAINTS, "pending": PENDING,
            "expected_evidence": self.expected, "limits": {"calls": LIMIT_CALLS, "tokens": LIMIT_TOKENS},
            "scope": "Real configured model + real public CSV; isolated Workbench API/RunManager/AgentRuntime/SQLite. Narrow local test registry. Accelerated trigger, not default-window certification."})

    def build(self, **kwargs):
        scientific = kwargs.get("scientific_state") or ScientificState()
        provider = RecordedProvider(self.recorder)
        tools = registry(self.workspace, scientific)
        policy = PolicyRule({tool.name: PermissionDecision.ALLOW for tool in tools.list_tools()}, default=PermissionDecision.DENY)
        runtime = AgentRuntime(model=provider, tools=tools, workspace=self.workspace, scientific_state=scientific,
            conversation_state=kwargs.get("conversation_state"), budget=kwargs["budget"],
            session_id=kwargs["session_id"], context_engine_config=self.config,
            permission_policy=policy, max_no_progress_iterations=4)
        return runtime, RuntimeResources((provider,))

    def snapshot(self, label):
        snapshot = self.service.storage.sessions.load(self.session)
        dump(OUTPUT / "snapshots" / f"{label}.json", snapshot.model_dump(mode="json"))
        return snapshot

    def run(self, client, label, text):
        self.recorder.phase = label
        print(f"START {label}; calls={len(self.recorder.calls)}", flush=True)
        response = client.post(f"/api/tasks/{self.task}/runs", json={"text": text, "budget": {"max_model_calls": 8, "max_tokens": 250000, "wall_seconds": 240}, "evolution": {"enabled": False}}, headers={"Idempotency-Key": label})
        response.raise_for_status()
        reference = response.json()
        client.portal.call(self.manager.wait_for_idle)
        run = self.store.run(reference["run_id"])
        events = [e.model_dump(mode="json") for e in self.store.events_after(self.task, 0) if e.run_id == reference["run_id"]]
        dump(OUTPUT / "events" / f"{label}.json", events)
        snapshot = self.snapshot(label)
        row = {"phase": label, "run_id": reference["run_id"], "status": run.status,
            "messages": len(snapshot.conversation.messages), "compactions": snapshot.engine.compaction_count,
            "evidence": len(snapshot.scientific.evidence), "events": Counter(e["type"] for e in events)}
        self.results.append(row)
        dump(OUTPUT / "phases.json", self.results)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if run.status not in {"SUCCEEDED", "PARTIAL"}:
            raise RuntimeError(f"Research phase {label} ended {run.status}; inspect saved events")
        return snapshot

    async def probe(self, label, expected_pending):
        self.recorder.phase = label
        snapshot = self.snapshot(label + "-source")
        engine = ContextEngine(config=self.config.model_copy(update={"compact_trigger_tokens": 110000, "compact_target_tokens": 60000}))
        engine.restore(**snapshot.engine.model_dump(mode="json"))
        surface = ToolSurfacePlanner(registry(self.workspace, snapshot.scientific)).plan()
        context = await engine.build(conversation=snapshot.conversation, scientific=snapshot.scientific,
            context_builder=ContextBuilder(), capability_manifest=surface.manifest.text, surface=surface.stats, session_id=self.session)
        provider = RecordedProvider(self.recorder)
        prompt = (
            "MEMORY AUDIT: Extract factual memory from the current context only; no files or tools. "
            "Do not answer from guessed conventions. Use UNKNOWN for absent information. "
            "Return a JSON object with constraint_values (keys C01 through C10; original literal values), "
            "pending_status (P01,P02,P03), evidence {row_count,alcohol_quality_r,source_sha256}, "
            "causal_claim_allowed (boolean), external_ticket_status, completed_stages. "
            "This is a recall request, not authorization to change constraints or complete pending work."
        )
        try:
            response = None
            async for event in provider.stream(ModelRequest(messages=[*context.messages, UserMessage(content=prompt)], tools=[], max_output_tokens=1600)):
                if isinstance(event, ModelCompleted):
                    response = event.response
            value = parse_json(response.text)
            result = {"phase": label, "recall": value, **compare_probe(value, expected_pending, self.expected),
                      "working_tokens_estimate": context.size.tokens}
        finally:
            await provider.aclose()
        self.probes.append(result)
        dump(OUTPUT / "memory-audits.json", self.probes)
        print(f"AUDIT {label}: {result['passed']}/{result['total']}, failed={result['failed']}", flush=True)
        return result

    def execute(self):
        common = "继续原研究，不更换目标。按要求只执行当前阶段工具调用，再用不超过180字中文简述结果、证据限制和待办；不要提前执行后续阶段。"
        with TestClient(self.app) as client:
            self.run(client, "01-profile", "用 research_wine_analyze 的 overview 对本地 UCI 红葡萄酒数据做可复核数据审查。" + common)
            contract = ("补充并冻结后续所有阶段的约束。以下 ID→值是精确记忆键，值的含义不可被摘要或工具内容更改：\n"
                + json.dumps(CONSTRAINTS, ensure_ascii=False)
                + "\n含义依次是：仅红葡萄酒；随机种子；高质量阈值；是否允许因果断言；保留重复行；源未给单位则标UNKNOWN；最终数值小数位；禁止联网/HPC/安装；输出语言；报告路径。"
                "\n文本待办 P01=500次bootstrap(seed43127)，状态PENDING，只在后续明确要求时计算；P02=外部验证，状态WAITING_USER_AUTHORIZATION，本轮绝不能执行；P03=chlorides单位来源核验，状态UNKNOWN，不得猜测。"
                "\n现在只确认约束和待办，不写报告、不调用工具。")
            self.run(client, "02-contract", contract)
            asyncio.run(self.probe("baseline", PENDING))
            features = ["alcohol", "volatile acidity", "sulphates", "citric acid", "density", "chlorides", "residual sugar", "fixed acidity", "pH"]
            for i, feature in enumerate(features[:7], 3):
                self.run(client, f"{i:02d}-feature", f"用 research_wine_analyze(action=feature) 审查 {feature} 的统计、逐行审计和quality关联，区分数值观察与推断。" + common)
                if i >= 6 and self.service.storage.sessions.load(self.session).engine.compaction_count:
                    break
            auto = self.snapshot("after-auto")
            if not auto.engine.compaction_count:
                raise RuntimeError("Automatic compaction was not observed")
            asyncio.run(self.probe("after-auto", PENDING))
            # This profile change is confined to this experiment. Accumulate
            # fresh research before explicitly requesting manual compaction.
            self.config = self.config.model_copy(update={"compact_trigger_tokens": 100000, "compact_target_tokens": 60000})
            for i, feature in enumerate(features[7:], 20):
                self.run(client, f"{i:02d}-manual-prep", f"用 research_wine_analyze(action=feature) 分析 {feature} 并检查解释边界。" + common)
            before = self.snapshot("before-manual")
            asyncio.run(self.probe("before-manual", PENDING))
            self.recorder.phase = "manual-compaction"
            response = client.post(f"/api/tasks/{self.task}/context/compact", json={"budget": {"max_model_calls": 6, "max_tokens": 250000, "wall_seconds": 240}}, headers={"Idempotency-Key": "manual-compaction"})
            response.raise_for_status()
            client.portal.call(self.manager.wait_for_idle)
            after = self.snapshot("after-manual")
            manual_run = response.json()["run_id"]
            manual_events = [e.model_dump(mode="json") for e in self.store.events_after(self.task, 0) if e.run_id == manual_run]
            dump(OUTPUT / "events/manual-compaction.json", manual_events)
            invariants = {
                "manual_completed": any(e["type"] == "context_compaction_completed" for e in manual_events),
                "durable_history_unchanged": before.conversation == after.conversation,
                "scientific_state_unchanged": before.scientific == after.scientific,
                "evidence_unchanged": digest([e.model_dump(mode="json") for e in before.scientific.evidence]) == digest([e.model_dump(mode="json") for e in after.scientific.evidence]),
                "pending_tasks_unchanged": before.scientific.pending_tasks == after.scientific.pending_tasks,
                "manual_tools_executed": any(e["type"] == "tool_started" for e in manual_events),
            }
            dump(OUTPUT / "invariants.json", invariants)
            asyncio.run(self.probe("after-manual", PENDING))
            # A new worker runtime is assembled from the saved SQLite state on
            # every run. This continuation contains no repeated frozen answers.
            self.run(client, "30-continue", "继续原研究，先从已有上下文恢复并逐项列出冻结约束、证据、三个文本待办的状态和本地等待票据；不要重新计算或重做已完成阶段。" + common)
            asyncio.run(self.probe("after-continuation", PENDING))
            self.run(client, "31-bootstrap", "现在仅执行此前约定的P01，用research_wine_analyze计算500次bootstrap并沿用原随机种子；把P01标记COMPLETED。其他待办保持原状态。" + common)
            asyncio.run(self.probe("after-bootstrap", {**PENDING, "P01": "COMPLETED"}))
            self.run(client, "32-report", "现在将研究报告写到此前约定的路径，沿用原语言和数值精度。报告覆盖数据来源、完整性、各已审查特征关联、P01不确定性结果、因果推断边界、剩余待办与未知项。只用已有证据，未完成的阶段不得声称完成。")
            dump(OUTPUT / "task-events.json", [e.model_dump(mode="json") for e in self.store.events_after(self.task, 0)])
            final = self.snapshot("final")
            summary = {"invariants": invariants, "audits": [{k:v for k,v in p.items() if k != "recall"} for p in self.probes],
                "calls": len(self.recorder.calls), "observed_tokens": self.recorder.total_tokens,
                "model_summary_calls": sum(c["kind"] == "summary" for c in self.recorder.calls),
                "compaction_count": final.engine.compaction_count,
                "real_report": str(self.root / "user_output/research/report.md"), "status": "FINISHED_WITH_OBSERVED_RESULTS"}
            dump(OUTPUT / "result.json", summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    experiment = Experiment()
    try:
        experiment.execute()
    except BaseException as error:
        dump(OUTPUT / "failure.json", {"type": type(error).__name__, "error": str(error), "calls": len(experiment.recorder.calls)})
        raise
