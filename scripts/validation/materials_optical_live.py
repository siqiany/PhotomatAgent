"""Opt-in public tmm reference and real rootless Podman qualification."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from photomatagent.application.capability_cases import cases
from photomatagent.application.capability_evolution import CapabilityEvolutionRequest
from photomatagent.application.models import RunRequest, TaskCreate
from photomatagent.application.reproduction import NumericReference, ReproductionService, ReproductionSpec
from photomatagent.application.runner import RunManager
from photomatagent.application.service import WorkbenchService
from photomatagent.application.skills import LocalSkillService
from photomatagent.application.store import WorkbenchStore
from photomatagent.capability_acquisition.container_backend import PodmanSandboxBackend
from photomatagent.capability_acquisition.method_discovery import MethodDiscoveryService, MethodNeed
from photomatagent.capability_acquisition.models import InstallPlan
from photomatagent.capability_acquisition.repository_sources import FetchRepositoryScriptTool, RepositorySearchService
from photomatagent.capability_acquisition.sandbox import SandboxService
from photomatagent.capability_acquisition.web_sources import PublicSourceClient, public_dns_for_synthetic_addresses
from photomatagent.scientific.capabilities.literature.documents import DocumentService
from photomatagent.scientific.capabilities.literature.source_fetch import LiteratureFetchSourceTool
from photomatagent.web.artifacts import ArtifactService
from photomatagent.workspace import Workspace

REPOSITORY = "sbyrnes321/tmm"
PAPER_URL = "https://arxiv.org/pdf/1603.02720v5"
WRAPPER = """import json
import os
from math import inf
from pathlib import Path
from tmm_core import coh_tmm
data = json.loads((Path(os.environ["PHOTOMAT_INPUT_DIR"]) / "data").read_text())
n = [data[key] for key in ("n_incident", "n_film", "n_exit")]
R = float(coh_tmm("s", n, [inf, data["thickness_nm"], inf], 0, data["wavelength_nm"])["R"])
result = {"R": R, "unit": "1", "scientific_status": "UNKNOWN"}
Path("result.json").write_text(json.dumps(result))
print(json.dumps(result))
"""


async def main(args: argparse.Namespace) -> None:
    args.workspace.mkdir(parents=True, exist_ok=False)
    workspace = Workspace(args.workspace.resolve())
    api = PublicSourceClient(frozenset({"api.github.com"}), resolver=public_dns_for_synthetic_addresses, request_quota=5)
    raw = PublicSourceClient(frozenset({"raw.githubusercontent.com"}), resolver=public_dns_for_synthetic_addresses, request_quota=2)
    methods = MethodDiscoveryService(workspace, repositories=RepositorySearchService(api))
    found = await methods.search(MethodNeed(query="thin film spectrum repo:" + REPOSITORY), 5, ("github",))
    assert any(item["method"]["method_id"] == "transfer-matrix" for item in found["candidates"])
    endpoint = "https://api.github.com/repos/" + REPOSITORY
    metadata = await api.json(endpoint)
    version = await api.json(endpoint + "/commits/" + metadata["default_branch"])
    commit = version["sha"]
    source = await api.json(endpoint + "/contents/tmm_core.py?ref=" + commit)
    contents = base64.b64decode(source["content"])
    git_blob = b"blob " + str(len(contents)).encode() + b"\0" + contents
    assert hashlib.sha1(git_blob).hexdigest() == source["sha"]
    source_sha = hashlib.sha256(contents).hexdigest()
    license_name = metadata["license"]["spdx_id"]
    assert license_name == "MIT"

    documents = DocumentService(workspace, wait_seconds=10)
    paper = await LiteratureFetchSourceTool(documents=documents).execute({"url": PAPER_URL, "content_role": "paper"})
    assert not paper.is_error, paper.data

    store = WorkbenchStore(workspace.root / ".photomatagent/workbench.sqlite3")
    manager = RunManager(store, runtime_options={"workspace_root": workspace.root})
    host = WorkbenchService(store=store, manager=manager, workspace_root=workspace.root,
                            sessions_dir=workspace.root / "sessions", artifacts=ArtifactService(store, workspace.root))
    try:
        task = host.create_task(TaskCreate(title="Lossless film reference", goal="Validate a pinned public optical implementation"))
        run = store.reserve_run(task, RunRequest(text="Run the published tmm implementation on an optical reference"), "source").run_id
        scratch = f"tmp/workbench/{task}"
        acquired = await FetchRepositoryScriptTool(workspace, scratch, raw).execute({
            "owner": REPOSITORY.split("/")[0], "repository": REPOSITORY.split("/")[1], "commit": commit,
            "file": "tmm_core.py", "sha256": source_sha, "license": license_name,
            "license_source": endpoint + "/license",
        })
        assert not acquired.is_error, acquired.data
        directory = workspace.root / scratch
        (directory / "main.py").write_text(WRAPPER)
        (directory / "tmm_core.py").write_bytes(workspace.resolve(acquired.data["script_path"]).read_bytes())
        (directory / "data.json").write_text(json.dumps(cases("fresnel_film")[0]["inputs"]["data"]))
        persisted = json.loads(args.installation_request.read_bytes())
        install = InstallPlan.model_validate(persisted["plan"] | {"expires_at": datetime.now(UTC) + timedelta(days=1)})
        storage = workspace.root / "environments"
        backend = PodmanSandboxBackend(storage)
        sandbox = SandboxService(workspace=workspace.root, trial_id=run, genome_sha256="d" * 64,
                                 scripts={}, dynamic_script_prefix=scratch, backend=backend)
        if args.installed_environment is not None:
            environment_dir = storage / args.installed_environment.name
            shutil.copytree(args.installed_environment, environment_dir, symlinks=True)
            environment = backend.restore_environment(environment_dir.name)
            sandbox.register_environment(environment, plan=install)
        else:
            installed = await sandbox.install(install)
            assert installed.code == "OK" and installed.environment is not None, installed.code
            environment = installed.environment
        service = ReproductionService(workspace, sandbox)
        plan = service.prepare(ReproductionSpec(
            source=PAPER_URL, paper_anchor="Coherent transfer matrix and zero-thickness interface limit",
            implementation="third_party", source_commit=commit, target="Lossless normal-incidence film",
            script_path=f"{scratch}/main.py", source_files=("tmm_core.py",), environment_id=environment.environment_id,
            input_refs={"data": f"{scratch}/data.json"}, requested_scope="EXAMPLE",
            reference=NumericReference(metric="R", expected=0.04, absolute_tolerance=1e-9, unit="1",
                                       source_anchor="Normal-incidence Fresnel interface",
                                       tolerance_rationale="Host comparison against independently computed interface R"),
        ))
        result = await service.execute(plan["plan_id"], plan["plan_sha256"])
        assert result["execution"] == "SUCCEEDED" and result["checker"]["result"] == "PASS", result
        print(json.dumps({"commit": commit, "source_sha256": source_sha, "document_id": paper.data["document_id"],
                          "plan_id": plan["plan_id"], "execution": result["execution"],
                          "reference": result["checker"]["result"], "paper_reproduction": result["paper_reproduction"]}), flush=True)
        store.finish_run(run, "SUCCEEDED")
        host.skill_service = LocalSkillService(workspace.root, backend_factory=lambda *_: backend)
        await manager.start()
        job = await host.capability_evolution.start(task, CapabilityEvolutionRequest(source_run_id=run, max_candidates=1), "qualify")
        await manager.wait_for_idle()
        candidates = host.capability_evolution.list_for_task(task)[0]["candidates"]
        assert candidates and candidates[0]["archive"]["action"] == "INSERT", store.events_after(job["task_id"], 0)[-1].payload
        print(json.dumps({"archive": candidates[0]["archive"], "status": host.skill_service.list_skills()[0]["status"],
                          "scientific_validation": candidates[0]["scientific_status"]}), flush=True)
    finally:
        await documents.aclose()
        await manager.stop()
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--installed-environment", type=Path)
    parser.add_argument("--installation-request", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
