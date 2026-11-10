from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime

import httpx
import pytest

from photomatagent.application.capability_cases import cases
from photomatagent.application.capability_evolution import CapabilityEvolutionRequest
from photomatagent.application.frozen_methods import FrozenEvaluationSkills
from photomatagent.application.models import FeedbackAnalysis, FeedbackRequest, RunRequest, TaskCreate
from photomatagent.application.reproduction import NumericReference, ReproductionService, ReproductionSpec
from photomatagent.application.runner import RunManager
from photomatagent.application.service import WorkbenchService
from photomatagent.application.skills import LocalSkillService
from photomatagent.application.store import WorkbenchStore
from photomatagent.capability_acquisition import container_backend, web_sources
from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend, _tree_sha256
from photomatagent.capability_acquisition.installation import create_install_plan
from photomatagent.capability_acquisition.method_discovery import MethodDiscoveryService, MethodNeed
from photomatagent.capability_acquisition.models import WheelArtifact
from photomatagent.capability_acquisition.sandbox import SandboxService
from photomatagent.models.types import UserMessage
from photomatagent.runtime.events import LoopCompleted
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.evolution.models import FeedbackDelta
from photomatagent.scientific.evolution.qd.store import QDStore
from photomatagent.scientific.state import ScientificState
from photomatagent.sessions.store import SessionSnapshot
from photomatagent.tools.process_plan import RegisterDecisionTool
from photomatagent.tools.skill_run import SkillRunTool
from photomatagent.tools.skill_search import SkillSearchTool
from photomatagent.web.artifacts import ArtifactService
from photomatagent.workspace import Workspace


HELPER = """import numpy as np

def solve(data):
    nodes = {name: i for i, name in enumerate(data["nodes"])}
    samples = []
    for f in data["frequencies_hz"]:
        matrix = np.zeros((len(nodes), len(nodes)), dtype=complex)
        current = np.zeros(len(nodes), dtype=complex)
        current[nodes[data["injection_node"]]] = 1
        for edge in data["components"]:
            value = edge["value_si"]
            assert value > 0
            y = 1 / value if edge["kind"] == "R" else 2j * np.pi * f * value
            a, b = nodes.get(edge["a"]), nodes.get(edge["b"])
            if a is not None:
                matrix[a, a] += y
            if b is not None:
                matrix[b, b] += y
            if a is not None and b is not None:
                matrix[a, b] -= y
                matrix[b, a] -= y
        result = np.linalg.solve(matrix, current)[nodes[data["output_node"]]]
        samples.append(dict(frequency_hz=f, real_ohm=float(result.real), imag_ohm=float(result.imag)))
    return samples
"""


def workbench(tmp_path, **options):
    store = WorkbenchStore(tmp_path / ".photomatagent/workbench.sqlite3")
    manager = RunManager(store, runtime_options={"workspace_root": tmp_path}, **options)
    return WorkbenchService(store=store, manager=manager, workspace_root=tmp_path,
                            sessions_dir=tmp_path / "sessions", artifacts=ArtifactService(store, tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["valid", "hardcoded", "forged"])
async def test_reproduction_promotion_requires_transfer_and_freezes_full_source(tmp_path, monkeypatch, scenario):
    host = workbench(tmp_path, runtime_builder=lambda **_: pytest.fail("qualification called Maker"))
    task = host.create_task(TaskCreate(title="RC paper", goal="Reproduce and qualify a circuit method"))
    source_run = host.store.reserve_run(task, RunRequest(text="Reproduce"), "source").run_id
    ws = Workspace(tmp_path)
    scratch = f"tmp/workbench/{task}"
    path = tmp_path / scratch
    path.mkdir(parents=True)
    data = cases("rc_network")[0]["inputs"]["data"]
    (path / "data.json").write_text(json.dumps(data))
    (path / "helper.py").write_text(HELPER)
    expression = repr(data) if scenario == "hardcoded" else 'json.loads((Path(os.environ["PHOTOMAT_INPUT_DIR"]) / "data").read_text())'
    (path / "main.py").write_text(
        "import json, os\nfrom pathlib import Path\nfrom helper import solve\n"
        f"data = {expression}\n"
        'Path("result.json").write_text(json.dumps(dict(samples=solve(data), scientific_status="UNKNOWN")))\n'
        'print(json.dumps(dict(metric=2, unit="1")))\n'
    )
    install = create_install_plan(
        wheels=[WheelArtifact(filename="fixture-1.0-py3-none-any.whl",
                              url="https://files.pythonhosted.org/packages/fixture.whl",
                              sha256="a" * 64, size=10, distribution="fixture", version="1.0", tags=("py3-none-any",))],
        direct_requirements=["fixture==1.0"], python_version="3.12", platform_tag="py3-none-any",
        image_digest="sha256:" + "b" * 64, installer_version="uv fixture", estimated_bytes=10,
    )
    env_id = "env_" + install.plan_sha256[:24]
    storage = tmp_path / "test-environments"
    env_root = storage / env_id
    (env_root / "python").mkdir(parents=True)
    (env_root / "python" / "fixture").write_text("locked")
    (env_root / "requirements.txt").write_text("")
    (env_root / "installed-manifest.json").write_text(json.dumps({
        "plan_sha256": install.plan_sha256, "environment_sha256": _tree_sha256(env_root / "python"),
        "image_ref": "python:3.12-slim@sha256:" + "b" * 64, "entrypoints": {},
        "created_at": datetime.now(UTC).isoformat(),
    }))

    class FixtureBackend(PodmanSandboxBackend):
        async def install(self, plan):
            assert plan.plan_sha256 == install.plan_sha256
            return self.restore_environment(env_id)

    backend = FixtureBackend(storage)
    backend.executable = "/usr/bin/podman"

    async def process(arguments, **_limits):
        if arguments[1] != "run":
            return 0, b"", b"", False, False
        stage = next(arg.removesuffix(":/input:ro,Z") for arg in arguments if arg.endswith(":/input:ro,Z"))
        output = next(arg.removesuffix(":/output:rw,Z") for arg in arguments if arg.endswith(":/output:rw,Z"))
        assert "--network=none" in arguments
        assert (tmp_path / stage / "source/helper.py").read_text() == HELPER
        result = await asyncio.to_thread(
            subprocess.run, [sys.executable, str(tmp_path / stage / "source/main.py")],
            cwd=tmp_path / output, env={**os.environ, "PHOTOMAT_INPUT_DIR": str(tmp_path / stage / "inputs")},
            capture_output=True, timeout=20,
        )
        return result.returncode, result.stdout, result.stderr, False, False

    monkeypatch.setattr(container_backend, "_run_bounded_process", process)
    sandbox = SandboxService(workspace=tmp_path, trial_id=source_run, genome_sha256="c" * 64,
                             scripts={}, dynamic_script_prefix=scratch, backend=backend)
    sandbox.register_environment(backend.restore_environment(env_id), plan=install)
    reproduce = ReproductionService(ws, sandbox)
    spec = ReproductionSpec(source="https://example.com/paper", paper_anchor="Equation 1",
                            implementation="author_code", source_commit="d" * 40, environment_id=env_id,
                            script_path=f"{scratch}/main.py", source_files=("helper.py",),
                            input_refs={"data": f"{scratch}/data.json"},
                            reference=NumericReference(metric="metric", expected=2, absolute_tolerance=0,
                                                       unit="1", source_anchor="Equation 1"))
    plan = reproduce.prepare(spec)
    (path / "data.json").write_text("{}")
    changed = await reproduce.execute(plan["plan_id"], plan["plan_sha256"])
    assert changed["diagnostic"] == "INPUT_BINDING_CHANGED"
    (path / "data.json").write_text(json.dumps(data))
    outcome = await reproduce.execute(plan["plan_id"], plan["plan_sha256"])
    assert outcome["checker"]["result"] == "PASS" and outcome["applicability"] == "UNKNOWN"
    next_version = reproduce.prepare(spec.model_copy(update={"source_commit": "e" * 40}))
    assert next_version["plan_id"] != plan["plan_id"]
    assert next_version["checker"]["result"] == "UNKNOWN"
    assert not host.skill_service.list_skills()
    host.store.finish_run(source_run, "SUCCEEDED")
    host.skill_service = LocalSkillService(ws.root, backend_factory=lambda *_: backend)
    if scenario == "forged":
        real_create = host.skill_service.create

        def create_with_false_cache(*args, **kwargs):
            skill = real_create(*args, **kwargs)
            job_task = next(link.source_id for link in host.store.links_for_task(task)
                            if link.source_type == "capability_job_result")
            directory = tmp_path / f".photomatagent/capability-evolution/{job_task}/{skill['skill_id']}"
            directory.mkdir(parents=True)
            for case in cases("rc_network"):
                (directory / f"{case['id']}.json").write_text(json.dumps({
                    "id": case["id"], "role": case["role"], "passed": True,
                    "use_id": "does-not-exist", "wall_seconds": 0.01,
                }))
            return skill

        monkeypatch.setattr(host.skill_service, "create", create_with_false_cache)
    await host.manager.start()
    try:
        request = CapabilityEvolutionRequest(source_run_id=source_run, max_candidates=1)
        job = await host.capability_evolution.start(task, request, "promote")
        await host.manager.wait_for_idle()
        results = host.capability_evolution.list_for_task(task)[0]["candidates"]
        if scenario == "forged":
            assert results == []
            assert host.store.run(job["run_id"]).status == "FAILED"
            assert "CAPABILITY_CASE_RECEIPT_MISSING_OR_CHANGED" in str(host.store.events_after(job["task_id"], 0)[-1].payload)
            return
        candidate = results[0]
        method = host.skill_service.list_skills()[0]
        assert await host.capability_evolution.start(task, request, "promote") == job
        assert method["reproduction"]["plan_sha256"] == plan["plan_sha256"]
        assert candidate["scientific_status"] == "UNKNOWN"
        if scenario == "valid":
            assert candidate["archive"]["action"] == "INSERT"
            assert method["status"] == "REUSED"
            discovery = MethodDiscoveryService(ws)
            result = await discovery.search(MethodNeed(query="rc network"), 3)
            archived = next(row for row in result["candidates"] if row["method"]["method_id"] == method["skill_id"])
            assert archived["sources"][0]["trial_id"] == candidate["archive"]["trial_id"]
            assert archived["unconfirmed_inputs"] == ["data"]
            assert (await discovery.inspect(method["skill_id"]))["applicability"] == "UNKNOWN"
            need = MethodNeed(query="rc network", fixed_unavailable_inputs=("data",))
            blocked = await discovery.search(need, 3)
            rejection = next(row for row in blocked["excluded"] if row["method_id"] == method["skill_id"])
            decision = await RegisterDecisionTool(ScientificState(), discovery).execute({
                "question": "Can the circuit method be reused?", "method_need_basis": "Compare RC methods.",
                "method_need": need.model_dump(mode="json"), "method_disposition": "selected",
                "selected_method_id": method["skill_id"],
            })
            assert decision.is_error and not decision.state_updates
            assert decision.data["reason"] == rejection["reason"] == "REQUIRED_INPUT_FIXED_UNAVAILABLE"
            store = QDStore(ws, trial_factory=lambda *_: pytest.fail("existing trial expected"))
            genome = store.load_genome(candidate["archive"]["genome_id"])
            assert {ref.sha256 for ref in genome.script_refs} == {
                hashlib.sha256(HELPER.encode()).hexdigest(),
                hashlib.sha256((path / "main.py").read_bytes()).hexdigest(),
            }
            frozen = FrozenEvaluationSkills(tmp_path / "frozen", host.skill_service, [method["skill_id"]])
            search = await SkillSearchTool(frozen).execute({"query": "rc_network"})
            assert search.data["matches"][0]["input_keys"] == ["data"]
            target_task = "task_" + "1" * 32
            target_input = f"tmp/workbench/{target_task}/data.json"
            file = frozen.workspace.resolve(target_input, must_exist=False)
            file.parent.mkdir(parents=True)
            file.write_text(json.dumps(cases("rc_network")[1]["inputs"]["data"]))
            replay = await SkillRunTool(frozen, target_task).execute({
                "skill_id": method["skill_id"], "scope": method["scope"], "input_refs": {"data": target_input},
            })
            assert replay.data["checker"]["status"] == "PASS", replay.data
            case_dir = tmp_path / f".photomatagent/capability-evolution/{job['task_id']}/{method['skill_id']}"
            cached_report = ws.resolve(candidate["validation_path"])
            cached_report.unlink()
            (case_dir / "regression.json").unlink()
            before_uses = host.skill_service.uses(method["skill_id"])
            await host.capability_evolution.execute(host.store.run(job["run_id"]))
            assert host.skill_service.uses(method["skill_id"]) == before_uses
            assert (case_dir / "regression.json").exists()
            report = json.loads(cached_report.read_bytes())
            report["cases"][0]["use_id"] = "does-not-exist"
            cached_report.write_text(json.dumps(report))
            with pytest.raises(ValueError, match="CAPABILITY_CASE_RECEIPT_MISSING_OR_CHANGED"):
                await host.capability_evolution.execute(host.store.run(job["run_id"]))
            frozen_helper = ws.resolve(f".photomatagent/local-skills/{method['skill_id']}/input-file_0")
            frozen_helper.write_text("changed")
            assert host.skill_service.list_skills()[0]["status"] == "REVALIDATION_REQUIRED"
        else:
            assert candidate["archive"]["action"] == "REJECT"
            assert method["status"] == "DISABLED"
    finally:
        await host.manager.stop()
        host.store.close()


@pytest.mark.asyncio
async def test_feedback_propagates_through_checkpoints_and_clears_working_state(tmp_path):
    seen = []

    class Runtime:
        def __init__(self, **options):
            self.scientific_state = options["scientific_state"]
            self.conversation_state = options["conversation_state"]

        async def run(self, _goal):
            seen.append(self.scientific_state)
            yield LoopCompleted(iterations=1, reason="completed")

    host = workbench(tmp_path, runtime_builder=Runtime)
    store = host.store
    task = host.create_task(TaskCreate(title="Lab result", goal="Check current conditions"))
    source = store.reserve_run(task, RunRequest(text="Original"), "source").run_id
    store.finish_run(source, "SUCCEEDED")
    session_id = store.task(task).session_id
    repo = host.storage.sessions
    repo.ensure(session_id, task)
    original = SessionSnapshot(conversation=ConversationState(messages=[UserMessage(content="Original")]),
                               scientific=ScientificState(hypotheses=["Historical working assumption"]))
    first_checkpoint = repo.checkpoint(session_id, original, run_id=source, boundary="run_end")
    downstream = store.reserve_run(task, RunRequest(text="Derived"), "derived").run_id
    store.finish_run(downstream, "SUCCEEDED")
    repo.checkpoint(session_id, original, run_id=downstream, boundary="run_end")
    store.append_event_for_run(downstream, "closed_loop_completed", {"scientific_status": "PASS"})
    linked_task = host.create_task(TaskCreate(title="Derived task", goal="Use the original result"))
    metadata = dict(path="derived-reference.json", filename="derived-reference.json", sha256="a" * 64,
                    size_bytes=2, media_type="application/json")
    store.register_artifact(artifact_id="artifact_original", task_id=task, run_id=downstream, **metadata)
    with store.database.transaction():
        store.register_attachment(artifact_id="artifact_linked", task_id=linked_task, **metadata)
        store._connection.execute(
            "UPDATE artifacts SET origin_artifact_id=? WHERE artifact_id=?",
            ("artifact_original", "artifact_linked"),
        )
    linked_run = store.reserve_run(linked_task, RunRequest(text="Use linked data"), "linked").run_id
    store.finish_run(linked_run, "SUCCEEDED")

    class Compiler:
        async def compile(self, **context):
            value = str(context["feedback_text"])
            return FeedbackAnalysis(items=[FeedbackDelta(
                category="TASK_DEFINITION", status="PREFERENCE" if value == "Prefer presentation" else "CORRECTION",
                severity="HIGH", responsible_module="materials", problem=value, requested_actions=(value,),
                source_span=value, confidence=1.0,
            )])

    host.revision_service.compiler = Compiler()
    await host.revision_service.prepare(task, FeedbackRequest(source_run_id=source, text="Prefer presentation"), "preference")
    assert store.result_invalidations(task) == {}
    correction = FeedbackRequest(source_run_id=source, text="Change the operating temperature")
    plan_id = await host.revision_service.prepare(task, correction, "correction")
    assert await host.revision_service.prepare(task, correction, "correction") == plan_id
    assert store.result_invalidations(task) == {source: [plan_id], downstream: [plan_id]}
    assert store.run(linked_run).result_invalidation_ids == [plan_id]
    assert host.artifacts.metadata("artifact_linked")["result_invalidation_ids"] == [plan_id]
    assert "derived-reference.json" not in host.manager._runtime_input(store.run(linked_run))
    assert host.task_detail(task)["evolution"]["latest"]["scientific_status"] == "UNKNOWN"
    assert store.events_after(task, 0)[-1].payload["scientific_status"] == "PASS"

    await host.manager.start()
    try:
        from photomatagent.application.models import BudgetSpec

        revised = await host.revision_service.execute(plan_id, BudgetSpec(), "repair")
        await host.manager.wait_for_idle()
        assert store.run(revised.run_id).status == "SUCCEEDED"
        assert store.run(revised.run_id).result_invalidation_ids == []
        assert seen and not seen[-1].hypotheses
        assert "historical" in seen[-1].evidence_gaps[0]
        assert repo.load(session_id, first_checkpoint).scientific.hypotheses == original.scientific.hypotheses
        next_run = store.reserve_run(task, RunRequest(text="Revised scope"), "after-repair").run_id
        assert store.run(next_run).result_invalidation_ids == []
    finally:
        await host.manager.stop()
        store.close()


@pytest.mark.asyncio
async def test_revision_persists_the_exact_pending_feedback_it_rechecks(tmp_path, monkeypatch):
    from photomatagent.application.models import BudgetSpec

    class Compiler:
        async def compile(self, **context):
            text = str(context["feedback_text"])
            return FeedbackAnalysis(items=[FeedbackDelta(
                category="TASK_DEFINITION", status="CORRECTION", severity="HIGH",
                responsible_module="materials", problem=text, requested_actions=(text,),
                source_span=text, confidence=1.0,
            )])

    host = workbench(tmp_path)
    async def enqueue(_run_id):
        pass
    monkeypatch.setattr(host.manager, "enqueue", enqueue)
    host.revision_service.compiler = Compiler()
    store = host.store
    try:
        task = host.create_task(TaskCreate(title="Detector", goal="Check revised conditions"))
        source = store.reserve_run(task, RunRequest(text="Initial result"), "initial").run_id
        store.finish_run(source, "SUCCEEDED")

        async def prepare(source_run, text, key):
            return await host.revision_service.prepare(task, FeedbackRequest(source_run_id=source_run, text=text), key)

        first = await prepare(source, "Use 80 K", "first")
        second = await prepare(source, "Apply 0.1 V", "second")
        intermediate = await host.revision_service.execute(first, BudgetSpec(), "apply-first")
        store.finish_run(intermediate.run_id, "SUCCEEDED")
        assert store.run(intermediate.run_id).result_invalidation_ids == [second]

        combined = await prepare(intermediate.run_id, "Check the updated noise floor", "combined")
        plan = host.revision_service.get(combined)
        assert plan.carried_feedback_plan_ids == [second]
        assert f"[{second}] Apply 0.1 V" in plan.instruction
        failed = await host.revision_service.execute(combined, BudgetSpec(), "failed")
        store.finish_run(failed.run_id, "FAILED")
        assert second in store.run(failed.run_id).result_invalidation_ids
        revised = await host.revision_service.execute(combined, BudgetSpec(), "revised")
        store.finish_run(revised.run_id, "SUCCEEDED")
        assert store.run(revised.run_id).result_invalidation_ids == []
        assert second in store.run(source).result_invalidation_ids

        later = await prepare(source, "Use measured bandwidth", "later")
        assert later in store.run(revised.run_id).result_invalidation_ids
        assert host.revision_service.get(combined).carried_feedback_plan_ids == [second]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_synthetic_dns_opt_in_uses_pinned_https_and_rechecks_addresses(monkeypatch):
    contacted = []

    async def system_dns(host):
        return ["198.18.1.20"] if host == "api.crossref.org" else ["127.0.0.1"]

    async def bootstrap_reply(request):
        assert request.url.host == "8.8.8.8"
        assert request.headers["Host"] == request.extensions["sni_hostname"] == "dns.google"
        assert request.url.params["name"] == "api.crossref.org"
        return httpx.Response(200, json={"Status": 0, "TC": False, "Question": [
            {"name": "api.crossref.org.", "type": 1},
        ], "Answer": [{"type": 1, "data": "93.184.216.34"}]})

    async def upstream(request):
        contacted.append(request.url.host)
        assert request.headers["Host"] == request.extensions["sni_hostname"] == "api.crossref.org"
        return httpx.Response(302, headers={"Location": "https://redirect.example/resource"})

    async_client = httpx.AsyncClient

    def route_transport(*args, **kwargs):
        if kwargs["transport"] is None:
            assert not kwargs["trust_env"] and not kwargs["follow_redirects"]
            kwargs["transport"] = httpx.MockTransport(bootstrap_reply)
        return async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", route_transport)
    monkeypatch.setattr(web_sources, "_addresses", system_dns)
    client = web_sources.PublicSourceClient(
        frozenset({"api.crossref.org", "redirect.example"}), resolver=web_sources.public_dns_for_synthetic_addresses,
        transport=httpx.MockTransport(upstream),
    )
    with pytest.raises(web_sources.SourceError, match="PRIVATE_OR_UNROUTABLE_ADDRESS"):
        await client.get("https://api.crossref.org/works")
    assert contacted == ["93.184.216.34"]
