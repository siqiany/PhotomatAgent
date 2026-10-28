from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from datetime import datetime, timezone

import uvicorn
from fastapi import HTTPException

from photomatagent.application.runner import RunManager
from photomatagent.application.runtime_factory import RuntimeFactory
from photomatagent.application.models import FeedbackAnalysis
from photomatagent.application.service import WorkbenchService
from photomatagent.application.store import WorkbenchStore
from photomatagent.models.fake import FakeModelProvider, FakeResponse
from photomatagent.models.types import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from photomatagent.capability_acquisition.installation import create_install_plan
from photomatagent.capability_acquisition.models import ArtifactRecord, EnvironmentRecord, ExecutionRecord, WheelArtifact
from photomatagent.capability_acquisition.sandbox import SandboxService
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.state import ScientificState
from photomatagent.scientific.evolution.models import FeedbackDelta
from photomatagent.scientific.evolution.qd.budget import BudgetAmounts
from photomatagent.scientific.evolution.qd.evaluation import EvaluationExecution
from photomatagent.sessions.store import save_session_snapshot
from photomatagent.web.app import create_app
from photomatagent.web.artifacts import ArtifactService
from photomatagent.web.routes import router as workbench_router
import photomatagent.application.runtime_factory as runtime_factory_module


# Engineering-only fixture: the dependency plan is simulated, while script.py
# is executed by a real local Python subprocess through the normal runtime tool.
ENVIRONMENT_ID = "env_fixture"
ENVIRONMENT_SHA = "d" * 64
SCRIPT = '''import json, os
from pathlib import Path
values = json.loads((Path(os.environ["PHOTOMAT_INPUT_DIR"]) / "data").read_text())
Path("result.json").write_text(json.dumps({"sum": sum(values)}, ensure_ascii=False))
'''


class LocalScriptBackend:
    async def install(self, plan):
        return EnvironmentRecord(
            environment_id=ENVIRONMENT_ID, plan_sha256=plan.plan_sha256,
            status="READY", lock_ref="fixture-lock", installed_manifest_ref="fixture-manifest",
            entrypoints={}, backend="local_dev", created_at=datetime.now(timezone.utc),
            environment_sha256=ENVIRONMENT_SHA,
        )

    async def run(self, request, *, output_directory):
        input_dir = output_directory.parent / f"{request.execution_id}-inputs"
        input_dir.mkdir()
        for key, relative in request.input_refs.items():
            (input_dir / key).write_bytes((Path(request.workspace_root) / relative).read_bytes())
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(request.workspace_root) / request.script_path), *request.argv,
            cwd=output_directory,
            env={**os.environ, "PHOTOMAT_INPUT_DIR": str(input_dir)},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        artifacts = tuple(
            ArtifactRecord(path=path.name, size=path.stat().st_size,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in output_directory.iterdir() if path.is_file()
        )
        now = datetime.now(timezone.utc)
        return ExecutionRecord(
            execution_id=request.execution_id, backend="local_dev",
            state="COMPLETED" if process.returncode == 0 else "FAILED",
            exit_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode(),
            artifacts=artifacts, environment_sha256=ENVIRONMENT_SHA,
            script_sha256=request.script_digest, started_at=now, ended_at=now,
        )

    async def reconcile(self, _execution_id):
        return None


def fixture_plan():
    wheel = WheelArtifact(
        filename="fixture-1.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/fixture-1.0-py3-none-any.whl",
        sha256="a" * 64, size=1, distribution="fixture", version="1.0",
        tags=("py3-none-any",),
    )
    return create_install_plan(
        wheels=[wheel], direct_requirements=["fixture==1.0"],
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        platform_tag="py3-none-any", image_digest="sha256:" + "b" * 64,
        installer_version="fixture", estimated_bytes=1, execution_backend="local_dev",
    )


class FakeFeedbackPlanner:
    async def compile(self, **context: object) -> FeedbackAnalysis:
        feedback_text = str(context.get("feedback_text", ""))
        clarified = "用户补充说明：" in feedback_text
        if not clarified:
            item = FeedbackDelta(
                item_id="issue_1",
                category="TASK_DEFINITION",
                status="QUERY",
                severity="MEDIUM",
                responsible_module="scope clarification",
                problem="尚未说明是否只接受已经实验验证的候选材料",
                requested_actions=("明确候选筛选范围",),
                acceptance_test=None,
                preserve=("保留所有来源及其验证状态",),
                confidence=0.9,
                source_span="请确认目标边界",
            )
        else:
            item = FeedbackDelta(
                item_id="issue_2",
                category="EVIDENCE_SUFFICIENCY",
                status="CORRECTION",
                severity="HIGH",
                responsible_module="evidence retrieval",
                problem="关键候选仍缺可核验的一手来源",
                requested_actions=("补充两篇可核验原始研究",),
                acceptance_test=None,
                preserve=("没有核查前继续标为未验证",),
                confidence=0.9,
                source_span="补充可核验的一手来源",
            )
        return FeedbackAnalysis(
            items=[item],
            warnings=[],
            compiler_provider="fixture",
            compiler_model="deterministic",
        )


class FixtureProvider(FakeModelProvider):
    async def stream(self, request):
        user_messages = [message for message in request.messages if isinstance(message, UserMessage)]
        latest_request = user_messages[-1].content if user_messages else ""
        tool_results = [message for message in request.messages if isinstance(message, ToolResultMessage)]
        if "故意触发运行失败" in latest_request:
            raise RuntimeError("fixture revision failed")
        if "开始一轮长时间分析" in latest_request:
            await asyncio.sleep(2)
        if "执行教学求和脚本" in latest_request and not tool_results:
            match = re.search(r"tmp/workbench/(task_[0-9a-f]{32})/", latest_request)
            if match is None:
                raise RuntimeError("fixture task binding missing")
            task_id = match.group(1)
            self.set_responses([FakeResponse(tool_calls=[ToolCall(
                name="tool_call", arguments={"name": "script_run", "arguments": {
                    "script_path": f"tmp/workbench/{task_id}/script.py",
                    "environment_id": ENVIRONMENT_ID,
                    "input_refs": {"data": f"tmp/workbench/{task_id}/data.json"},
                }},
            )])])
        elif "执行教学求和脚本" in latest_request:
            self.set_responses([FakeResponse(text="工程演示：本地脚本已执行；求和仅验证程序路径，不代表材料科学结论。")])
        elif "补充两篇可核验原始研究" in latest_request:
            self.set_responses([FakeResponse(text="修订演示：新增来源仍待核对，当前不能当作已验证结论。")])
        else:
            self.set_responses([FakeResponse(text="初稿演示：吸收边和载流子输运仍需实验验证。")])
        async for event in super().stream(request):
            yield event


class FixtureEvaluationRunner:
    unsupported_conditions = ("FULL_QD_SKILLS",)

    async def execute(self, invocation):
        case_root = Path(invocation.task_workspace)
        if not (case_root / invocation.input_path).is_file():
            raise ValueError("fixture input missing")
        if list(case_root.rglob("reference*")):
            raise ValueError("reference data entered model workspace")
        return EvaluationExecution(
            output=b'{"required":"present","peak":{"value":1.0,"unit":"nm"}}',
            cost=BudgetAmounts(model_calls=1, tokens=7, trials=1),
            provenance="fake",
            strategy_trace=(f"fixture:{invocation.condition}",),
        )


def build_app(root: Path):
    runtime_factory_module.create_provider = lambda _provider, _model=None: FixtureProvider()
    original_build_owned = RuntimeFactory.build_owned
    plan = fixture_plan()
    backend = LocalScriptBackend()

    async def build_fixture_runtime(**kwargs):
        task_id = kwargs["capability_task_id"]
        run_id = kwargs["capability_run_id"]
        sandbox = SandboxService(
            workspace=root, trial_id=run_id, genome_sha256=hashlib.sha256(run_id.encode()).hexdigest(),
            scripts={}, dynamic_script_prefix=f"tmp/workbench/{task_id}", backend=backend,
        )
        installed = await sandbox.install(plan)
        if installed.code != "OK":
            raise RuntimeError(f"fixture local environment: {installed.code}")
        return await original_build_owned(
            provider="fake", log_events=False, capability_sandbox_service=sandbox, **kwargs,
        )

    RuntimeFactory.build_owned = staticmethod(build_fixture_runtime)
    store = WorkbenchStore(root / ".photomatagent" / "workbench.sqlite3")
    archived_session = root / ".photomatagent" / "sessions" / "e2e-archived-session"
    if not (archived_session / "events.jsonl").exists():
        archived_session.mkdir(parents=True, exist_ok=True)
        (archived_session / "events.jsonl").write_text(
            '{"kind":"loop_started","session_id":"e2e-archived-session","goal":"历史 SAM 分割质量复核"}\n',
            encoding="utf-8",
        )
        save_session_snapshot(
            archived_session,
            conversation=ConversationState(messages=[
                UserMessage(content="请复核这组 SAM 分割结果。"),
                AssistantMessage(text="历史记录：边界误差需要检查。"),
            ]),
            scientific=ScientificState(goal="历史 SAM 分割质量复核"),
        )
    manager = RunManager(store, runtime_options={"workspace_root": root})
    service = WorkbenchService(
        store=store,
        manager=manager,
        workspace_root=root,
        sessions_dir=root / ".photomatagent" / "sessions",
        artifacts=ArtifactService(store, root),
        revision_compiler=FakeFeedbackPlanner(),
    )
    service.skill_service.backend_factory = lambda _root, _kind: backend
    service.evaluation_service.runner_factory = lambda **_: FixtureEvaluationRunner()
    service.evaluation_service.model_id_factory = lambda: "fixture-evaluation-model"
    def add_run_artifact(task_id: str, run_id: str) -> dict[str, str]:
        artifact_id = f"artifact_e2e_{run_id}"
        content = b"SAM demo artifact: boundary metrics are illustrative and unverified.\n"
        relative_path = f"user_output/{task_id}/{artifact_id}.md"
        artifact_path = root / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)
        store.register_artifact(
            artifact_id=artifact_id,
            task_id=task_id,
            run_id=run_id,
            path=relative_path,
            filename="SAM 演示指标.md",
            sha256=hashlib.sha256(content).hexdigest(),
            media_type="text/markdown",
            size_bytes=len(content),
        )
        return {"artifact_id": artifact_id}

    workbench_router.add_api_route(
        "/__e2e__/tasks/{task_id}/runs/{run_id}/artifact",
        add_run_artifact,
        methods=["POST"],
        include_in_schema=False,
    )

    def prepare_script_source(task_id: str, artifact_id: str) -> dict[str, str]:
        if not re.fullmatch(r"task_[0-9a-f]{32}", task_id):
            raise HTTPException(status_code=400, detail="invalid fixture task")
        try:
            row = store.artifact(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="fixture input missing") from exc
        if row["task_id"] != task_id or row["run_id"] is not None:
            raise HTTPException(status_code=400, detail="fixture input belongs to another task")
        values = json.loads((root / row["path"]).read_text())
        if not isinstance(values, list) or not values or any(type(value) not in (int, float) for value in values):
            raise HTTPException(status_code=400, detail="fixture expects numeric array")
        task_root = root / "tmp" / "workbench" / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        (task_root / "script.py").write_text(SCRIPT, encoding="utf-8")
        (task_root / "data.json").write_bytes((root / row["path"]).read_bytes())
        return {"kind": "engineering-only", "input_sha256": hashlib.sha256((task_root / "data.json").read_bytes()).hexdigest()}

    workbench_router.add_api_route(
        "/__e2e__/tasks/{task_id}/script-source/{artifact_id}",
        prepare_script_source, methods=["POST"], include_in_schema=False,
    )

    def fixture_events(task_id: str) -> dict[str, object]:
        store.task(task_id)
        return {"kind": "engineering-only", "items": [
            event.model_dump(mode="json") for event in store.events_after(task_id, 0)
        ]}

    workbench_router.add_api_route(
        "/__e2e__/tasks/{task_id}/events", fixture_events,
        methods=["GET"], include_in_schema=False,
    )
    app = create_app(service=service, manager=manager)

    return app


if __name__ == "__main__":
    root = Path(os.environ.get("WORKBENCH_E2E_ROOT", tempfile.mkdtemp(prefix="photomatagent-webui-")))
    uvicorn.run(build_app(root), host="127.0.0.1", port=int(os.environ.get("WORKBENCH_API_E2E_PORT", "18765")), log_level="warning")
