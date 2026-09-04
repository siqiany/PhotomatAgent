# Unified VASP Paradigm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` to
> implement this plan task-by-task. Use the repository's TDD workflow. Execute
> one task at a time and stop for review after every task.

**Goal:** Expose one safe `vasp.*` model-facing workflow for periodic,
molecular, and study calculations while routing every real submission through
the existing idempotent HPC lifecycle.

**Architecture:** Ten deferred tools (including the bounded `vasp.wait` tool)
call one `UnifiedVaspService`. The service
loads a versioned manifest, verifies application-level approvals and resources,
routes to one of three internal executors, and persists state. Executors retain
their domain-specific science but share `SubmitOnceSession.submit_once`.

**Tech Stack:** Python 3.12, Pydantic, pytest, SQLite, existing ToolRegistry,
ScientificToolResult, ResourcePolicy, JobRegistry, SubmitOnceSession,
SCNetBackend, and FastMCP adapters.

**Spec:**
`docs/superpowers/specs/2026-08-27-unified-vasp-paradigm-design.md`

## Global Constraints

- Read the root `AGENTS.md` before each task. A worktree must contain that file.
- Work in an isolated worktree after current user-owned changes are resolved.
- Do not modify `.env`, credentials, `user_input/`, or real calculation output.
- Do not connect to SSH/SCNet and do not submit a real Slurm job in tests.
- Keep `AgentRuntime` as the sole executor of model-requested tools.
- Do not introduce another ToolRegistry, runtime, resource hard-cap policy, or
  scientific state model.
- All model-visible VASP tools are `DEFERRED` and use the normal permission,
  validation, event, observation, and state-update path.
- Every submission reaches `SubmitOnceSession.submit_once`; public application
  paths never call `SCNetBackend.submit_script()` directly.
- Preserve `PHOTOMATAGENT_ALLOW_HPC_SUBMIT`, runtime approval,
  `ResourcePolicy`, readiness, path, job-ID, and reconciliation gates.
- `vasp.submit` accepts `workflow_id` and optional `stage`; it does not accept
  workflow directories, approval IDs, fingerprints, or raw Slurm resources.
- Scientific and resource approvals are issued only through a user-controlled
  non-model path and are bound to immutable decision hashes.
- Scheduler `COMPLETED` never creates evidence until collection and
  deterministic validation succeed.
- Do not auto-commit. At the end of each task report a suggested commit message
  and stop for human review.

## Target Public Surface

```text
vasp.capabilities
vasp.plan
vasp.prepare
vasp.preflight
vasp.submit
vasp.status
vasp.resume
vasp.collect
vasp.report
```

## Planned File Structure

```text
src/photomatagent/scientific/applications/vasp/unified/
  __init__.py       public internal imports only
  models.py         typed requests, manifests, states, decisions
  fingerprints.py   canonical scientific/execution hashes
  repository.py     workspace-contained atomic manifest persistence
  approvals.py      pending decisions and SQLite approval receipts
  resources.py      recommendation plus existing ResourcePolicy orchestration
  executors.py      shared Protocol and typed operation results
  periodic.py       VaspApplication adapter
  molecular.py      MolecularVaspRuntime/domain-facade adapter
  study.py          StudyExecutor adapter over the molecular adapter
  router.py         deterministic workflow-kind routing
  service.py        workflow state machine and application operations
  recovery.py       automatic versus confirmation-required recovery policy
  tool_pack.py      the ten model-visible Tool classes, including vasp.wait
```

---

### Task 0: Record the Real Tool-Surface Baseline

**Files:**

- Create: `tests/test_vasp_unified_surface.py`
- Read: `src/photomatagent/tools/factory.py`
- Read: `src/photomatagent/tools/surface.py`
- Read: `src/photomatagent/scientific/applications/vasp/tools.py`
- Read: `src/photomatagent/mcp/manager.py`

**Produces:** Characterization tests that distinguish registry membership from
model visibility in progressive and eager modes.

- [ ] Write a helper that builds a registry with MCP auto-connect disabled and
  returns all registered VASP-family names.
- [ ] Assert the current built-in registry contains the periodic,
  `vasp_molecule.*`, and `vasp_study.*` families. Keep this test descriptive;
  remove it in Task 12 when legacy registration is removed.
- [ ] Add strict-xfail target tests for all of the following:
  progressive definitions, eager definitions, capability search, describe,
  bridged call, and guessed direct call.
- [ ] Use a fake MCP handle for MCP assertions; do not require an online server.

Representative target assertion:

```python
PUBLIC = {
    "vasp.capabilities", "vasp.plan", "vasp.prepare",
    "vasp.preflight", "vasp.submit", "vasp.status",
    "vasp.resume", "vasp.collect", "vasp.report",
}

def assert_unique_vasp_surface(names: set[str]) -> None:
    visible = {name for name in names if name.startswith("vasp")}
    assert visible == PUBLIC
```

Run:

```bash
PHOTOMATAGENT_MCP_AUTO_CONNECT=0 uv run pytest -q tests/test_vasp_unified_surface.py
```

Expected: characterization passes and target assertions are strict XFAIL.

Suggested commit: `test: characterize parallel VASP tool surfaces`

---

### Task 1: Define Typed Requests, Manifests, and Fingerprints

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/__init__.py`
- Create: `src/photomatagent/scientific/applications/vasp/unified/models.py`
- Create: `src/photomatagent/scientific/applications/vasp/unified/fingerprints.py`
- Create: `tests/test_vasp_unified_models.py`

**Consumes:** Existing `WorkflowSpec`, `VaspStudyRequest`, `ResourceRequest`, and
named periodic profiles.

**Produces:**

```python
class VaspWorkflowKind(str, Enum):
    PERIODIC = "periodic"
    MOLECULAR = "molecular"
    STUDY = "study"

class PeriodicScientificSpec(BaseModel):
    kind: Literal["periodic"] = "periodic"
    structure_path: str
    profile: str
    scientific_overrides: dict[str, Any] = Field(default_factory=dict)
    potcar_policy: str = "configured"

class MolecularScientificSpec(BaseModel):
    kind: Literal["molecular"] = "molecular"
    workflow: WorkflowSpec

class StudyScientificSpec(BaseModel):
    kind: Literal["study"] = "study"
    request: VaspStudyRequest

ScientificSpec = Annotated[
    PeriodicScientificSpec | MolecularScientificSpec | StudyScientificSpec,
    Field(discriminator="kind"),
]

class WorkflowState(str, Enum):
    PLANNED = "PLANNED"
    PREPARED = "PREPARED"
    PREFLIGHTED = "PREFLIGHTED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    RECONCILING = "RECONCILING"
    AWAITING_RESOURCE_CONFIRMATION = "AWAITING_RESOURCE_CONFIRMATION"
    AWAITING_SCIENTIFIC_CONFIRMATION = "AWAITING_SCIENTIFIC_CONFIRMATION"
    SCHEDULER_COMPLETED = "SCHEDULER_COMPLETED"
    VALIDATED = "VALIDATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    FAILED = "FAILED"

class ScientificChange(BaseModel):
    parameter: str
    old_value: Any
    new_value: Any
    reason: str

class UnifiedStage(BaseModel):
    name: str
    depends_on: list[str] = Field(default_factory=list)
    state: WorkflowState = WorkflowState.PLANNED
    resource_recommendation: ResourceRequest | None = None
    request_id: str | None = None

class WorkflowEvent(BaseModel):
    event_type: str
    timestamp: datetime
    stage: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

class ReportKind(str, Enum):
    SUMMARY = "summary"
    ORBITALS = "orbitals"
    ESP = "esp"
    BINDING_ENERGY = "binding_energy"
    STUDY = "study"

class ReportRequest(BaseModel):
    kind: ReportKind = ReportKind.SUMMARY
    related_workflow_ids: list[str] = Field(default_factory=list)

class UnifiedVaspRequest(BaseModel):
    workflow_kind: VaspWorkflowKind
    scientific_spec: ScientificSpec

class UnifiedVaspManifest(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    workflow_id: str
    workflow_kind: VaspWorkflowKind
    revision: int = 0
    state: WorkflowState = WorkflowState.PLANNED
    scientific_spec: ScientificSpec
    scientific_fingerprint: str
    execution_fingerprint: str | None = None
    stages: list[UnifiedStage]
    events: list[WorkflowEvent] = Field(default_factory=list)
```

`UnifiedVaspRequest` must reject a mismatch between `workflow_kind` and the
discriminated spec `kind`. Fingerprint fields are created by factory functions,
not accepted from the public request.

- [ ] Test that a molecular request without explicit `total_charge` fails
  Pydantic validation.
- [ ] Test that changing only `ResourceRequest` leaves the scientific hash
  unchanged.
- [ ] Test ENCUT, KPOINTS, SOC, charge, spin, structure content, POTCAR policy,
  and stage changes alter the scientific hash.
- [ ] Test dictionary order and equivalent relative paths do not alter hashes.
- [ ] Implement canonical JSON encoding and SHA-256 fingerprints.

Run:

```bash
uv run pytest -q tests/test_vasp_unified_models.py
```

Suggested commit: `feat: define unified VASP workflow contracts`

---

### Task 2: Add Workspace-Contained Manifest Persistence

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/repository.py`
- Create: `tests/test_vasp_manifest_repository.py`
- Read: `src/photomatagent/workspace.py`

**Consumes:** `UnifiedVaspRequest`, `UnifiedVaspManifest`, fingerprint factories,
and `Workspace`.

**Produces:**

```text
ManifestConflictError(RuntimeError)
ManifestRepository.create(UnifiedVaspRequest) -> UnifiedVaspManifest
ManifestRepository.load(workflow_id: str) -> UnifiedVaspManifest
ManifestRepository.save(
    UnifiedVaspManifest, expected_revision: int
) -> UnifiedVaspManifest
ManifestRepository.workflow_dir(workflow_id: str) -> Path
```

Implementation requirements:

- Generate workflow IDs in code.
- Store under `.photomatagent/vasp/workflows/<workflow_id>/manifest.json`.
- Resolve source paths and destinations through `Workspace.resolve`.
- Snapshot the source structure and record its content hash.
- Write a temporary sibling file, flush it, then replace atomically.
- Increment `revision` on save and reject stale expected revisions.
- Read explicitly supported unversioned legacy manifests through a migration
  function; never silently reinterpret unknown versions.

- [ ] Test absolute and `..` path escapes are rejected.
- [ ] Test a changed source structure does not mutate an existing snapshot.
- [ ] Test interrupted temporary output leaves the previous manifest readable.
- [ ] Test stale revision writes raise `ManifestConflictError`.

Run:

```bash
uv run pytest -q tests/test_vasp_manifest_repository.py
```

Suggested commit: `feat: persist versioned VASP manifests safely`

---

### Task 3: Add Non-Model Approval Decisions and Receipts

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/approvals.py`
- Create: `tests/test_vasp_approval_receipts.py`

**Produces:**

```text
class ApprovalKind(str, Enum):
    RESOURCE = "resource"
    SCIENTIFIC = "scientific"

class PendingDecision(BaseModel):
    decision_id: str
    workflow_id: str
    kind: ApprovalKind
    decision_hash: str
    scientific_fingerprint: str
    execution_fingerprint: str | None = None
    summary: str
    changes: list[ScientificChange] = Field(default_factory=list)

class ApprovalReceipt(BaseModel):
    receipt_id: str
    decision_id: str
    decision_hash: str
    workflow_id: str
    kind: ApprovalKind
    scientific_fingerprint: str
    execution_fingerprint: str | None = None
    approved_at: datetime
    approved_by: str

ApprovalReceiptStore.record_pending(PendingDecision) -> None
ApprovalReceiptStore.approve(
    decision_id: str, approved_by: str
) -> ApprovalReceipt
ApprovalReceiptStore.valid_receipt(
    PendingDecision, UnifiedVaspManifest
) -> ApprovalReceipt | None
```

Use SQLite with parameterized statements and a workspace-controlled database
path. `approve()` is an internal API and must not be exported as a Tool.

- [ ] Test a forged receipt ID is rejected.
- [ ] Test a receipt from another workflow is rejected.
- [ ] Test any bound fingerprint or decision-hash change invalidates a receipt.
- [ ] Test approval is idempotent for one decision ID.
- [ ] Test runtime allow-all state does not create an application receipt.

Run:

```bash
uv run pytest -q tests/test_vasp_approval_receipts.py
```

Suggested commit: `feat: add hash-bound VASP approval receipts`

---

### Task 4: Add the User-Controlled Approval Command

**Files:**

- Modify: `src/photomatagent/cli/commands.py`
- Modify: the existing Typer scientific-command implementation discovered from
  `src/photomatagent/cli/app.py`
- Create: `tests/test_vasp_approval_command.py`

**Consumes:** `ApprovalReceiptStore.approve()` from Task 3.

**Produces:** A user-only command with this behavior:

```text
/scientific approve <decision-id>
```

The command loads the pending decision, displays workflow ID, approval kind,
exact changes, and decision hash, asks for explicit confirmation, and calls
`approve()` only after confirmation. It is not a Tool and is not callable by the
model through `ToolRegistry`.

- [ ] Test unknown decision IDs fail without writing a receipt.
- [ ] Test declining confirmation writes nothing.
- [ ] Test approval records the current local user/session source.
- [ ] Test `/approve -a` does not approve pending scientific decisions.
- [ ] Reuse the central command router; do not add a second chat-command parser.

Run:

```bash
uv run pytest -q tests/test_vasp_approval_command.py tests/test_chat_commands.py tests/test_permissions.py
```

Suggested commit: `feat: add user-only VASP decision approval command`

Review gate: this is an intentional cross-layer change. Confirm it still keeps
`AgentRuntime` as the only executor of model-requested tools before Task 5.

---

### Task 5: Compose Resource Planning with Existing ResourcePolicy

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/resources.py`
- Create: `tests/test_vasp_resource_decision.py`
- Read: `src/photomatagent/scientific/remote/models.py`
- Read: `src/photomatagent/scientific/applications/vasp/profiles.py`
- Read: `src/photomatagent/scientific/applications/vasp/molecular/models.py`

**Produces:**

```text
class ResourceDecisionState(str, Enum):
    ALLOWED = "ALLOWED"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    DENIED = "DENIED"

class ResourceDecision(BaseModel):
    state: ResourceDecisionState
    recommended: ResourceRequest
    effective: ResourceRequest | None
    decision_hash: str
    reasons: list[str] = Field(default_factory=list)
    pending_decision: PendingDecision | None = None

VaspResourcePlanner.recommend(
    UnifiedVaspManifest, UnifiedStage
) -> ResourceRequest

ResourceAuthorizationService.decide(
    UnifiedVaspManifest, UnifiedStage, ResourceRequest
) -> ResourceDecision
```

`ResourceAuthorizationService` must call `ResourcePolicy.violations()` and must
not reproduce its hard-cap logic. Automatic-budget settings are a separate,
stricter threshold. A recommendation is never silently reduced.

- [ ] Test an in-budget recommendation is allowed.
- [ ] Test above-auto but under-hard-cap returns a pending resource decision.
- [ ] Test above-hard-cap is denied even with a receipt.
- [ ] Test disabled `PHOTOMATAGENT_ALLOW_HPC_SUBMIT` is denied.
- [ ] Test partition and calibration constraints remain effective.
- [ ] Test the decision and reason list contain no secrets.

Run:

```bash
uv run pytest -q tests/test_vasp_resource_decision.py tests/test_remote.py
```

Suggested commit: `feat: unify VASP resource authorization`

---

### Task 6: Define the Internal Executor Contract

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/executors.py`
- Create: `tests/test_vasp_executor_contract.py`

**Produces:** The following typed results and a Protocol implemented by all
three adapters:

```python
class OperationResult(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)

class PreflightResult(OperationResult):
    passed: bool

class SubmissionResult(OperationResult):
    request_id: str
    job_id: str | None = None
    submitted: bool = False
    duplicate: bool = False
    needs_reconciliation: bool = False

class StatusResult(OperationResult):
    stage_states: dict[str, str] = Field(default_factory=dict)
    query_failed: bool = False

class RecoveryResult(OperationResult):
    action: str
    pending_decision: PendingDecision | None = None

class CollectionResult(OperationResult):
    validated: bool = False
    evidence: list[ScientificEvidence] = Field(default_factory=list)

class ReportResult(OperationResult):
    report_kind: ReportKind

class ServiceResult(OperationResult):
    workflow_id: str
    state: WorkflowState
    evidence: list[ScientificEvidence] = Field(default_factory=list)
    pending_decision: PendingDecision | None = None
```

```text
VaspWorkflowExecutor.prepare(UnifiedVaspManifest) -> OperationResult
VaspWorkflowExecutor.preflight(UnifiedVaspManifest) -> PreflightResult
VaspWorkflowExecutor.submit(
    UnifiedVaspManifest, UnifiedStage, ResourceRequest
) -> SubmissionResult
VaspWorkflowExecutor.status(UnifiedVaspManifest) -> StatusResult
VaspWorkflowExecutor.reconcile(UnifiedVaspManifest) -> RecoveryResult
VaspWorkflowExecutor.collect(UnifiedVaspManifest) -> CollectionResult
VaspWorkflowExecutor.report(
    UnifiedVaspManifest, ReportRequest
) -> ReportResult
```

Typed result models must carry bounded structured data, artifacts, errors, and
evidence gaps. They do not return `ToolResult`; only the public tool pack does.

- [ ] Create a contract test run against three fake executor implementations.
- [ ] Assert no executor interface accepts `approval_ids`, a Tool instance, or
  raw model arguments.

Run:

```bash
uv run pytest -q tests/test_vasp_executor_contract.py
```

Suggested commit: `feat: define internal VASP executor contract`

---

### Task 7: Route Periodic VASP Through SubmitOnceSession

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/periodic.py`
- Modify: `src/photomatagent/scientific/applications/vasp/application.py`
- Create: `tests/test_vasp_periodic_lifecycle.py`
- Reuse: `src/photomatagent/scientific/remote/lifecycle.py`

**Consumes:** Executor Protocol, manifest, `VaspApplication`,
`SubmitOnceSession`, and authorized `ResourceRequest`.

**Produces:** `PeriodicVaspExecutor`.

Implementation rules:

- Keep input generation, POTCAR resolution, Slurm rendering, output validation,
  and parsing in `VaspApplication`.
- Move public lifecycle ownership to the adapter.
- Derive a stable request ID from workflow ID, stage, scientific fingerprint,
  and effective execution fingerprint.
- Pass a deterministic `SubmissionGate` to `submit_once`.
- Deprecate direct public use of `VaspApplication.submit_stage`; do not delete it
  until Task 14 proves no public caller remains.

- [ ] Test two submits produce one backend submission and one immutable job ID.
- [ ] Test failed preflight causes zero upload and zero sbatch calls.
- [ ] Test client timeout enters reconciliation.
- [ ] Test multiple reconciliation candidates block.
- [ ] Test every new attempt uses a distinct remote directory.

Run:

```bash
uv run pytest -q tests/test_vasp_periodic_lifecycle.py tests/test_remote_lifecycle.py tests/test_vasp.py
```

Suggested commit: `refactor: use submit-once lifecycle for periodic VASP`

---

### Task 8: Add the Molecular Internal Adapter

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/molecular.py`
- Create: `tests/test_vasp_molecular_adapter.py`
- Read/reuse: `src/photomatagent/scientific/applications/vasp/molecular/runtime.py`
- Read/reuse: `src/photomatagent/scientific/applications/vasp/molecular/tools.py`
- Read/reuse: molecular preflight, results, and recovery modules

**Produces:** `MolecularVaspExecutorAdapter` implementing the Task 6 Protocol.

Rules:

- Call internal molecular facade/runtime methods, never
  `MolecularVasp*Tool.execute()`.
- Preserve explicit charge, spin, correction, calibration, HSE screening, and
  workflow DAG checks.
- Reuse the runtime's `JobRegistry` and `SubmitOnceSession`.
- Map orbital, ESP, and binding-energy analyses into typed `ReportRequest`
  variants.

- [ ] Test adapter preparation matches existing molecular generated inputs.
- [ ] Test missing charge, correction policy, calibration, or HSE screen blocks.
- [ ] Test repeated adapter submit is idempotent.
- [ ] Test invalid/placeholder results produce evidence gaps and no evidence.
- [ ] Test report variants preserve current bounded outputs.

Run:

```bash
uv run pytest -q tests/test_vasp_molecular_adapter.py tests/test_vasp_molecular_tools.py tests/test_vasp_molecular_wiring.py
```

Suggested commit: `refactor: adapt molecular VASP to unified executor contract`

---

### Task 9: Add the Study Internal Adapter

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/study.py`
- Modify: `src/photomatagent/scientific/applications/vasp/study/executor.py`
- Create: `tests/test_vasp_study_adapter.py`

**Consumes:** `MolecularVaspExecutorAdapter`, existing study planner/models, and
the executor Protocol.

**Produces:** `VaspStudyExecutorAdapter`.

Rules:

- Study remains orchestration; every child calculation delegates to the
  molecular adapter.
- Parent manifests record child workflow IDs and aggregate states.
- Study does not locate or execute `vasp_molecule.*` Tool objects.
- Budget exhaustion preserves partial results and does not create new jobs.

- [ ] Test duplicate chemical tasks map to one child workflow.
- [ ] Test every study scheduler submission passes the molecular adapter and
  `SubmitOnceSession`.
- [ ] Test resume after process exit creates no duplicate jobs.
- [ ] Test partial report and binding-energy behavior remain available.

Run:

```bash
uv run pytest -q tests/test_vasp_study_adapter.py tests/test_vasp_study.py tests/test_vasp_study_end_to_end.py
```

Do not treat the two documented HOMO-isosurface baseline failures as new
successes or hide them; report exact results.

Suggested commit: `refactor: adapt VASP studies to unified child workflows`

---

### Task 10: Implement the Deterministic Router and Service State Machine

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/router.py`
- Create: `src/photomatagent/scientific/applications/vasp/unified/service.py`
- Create: `tests/test_vasp_unified_service.py`

**Produces:**

```text
UnifiedVaspRouter.executor_for(VaspWorkflowKind) -> VaspWorkflowExecutor

UnifiedVaspService.plan(UnifiedVaspRequest) -> UnifiedVaspManifest
UnifiedVaspService.prepare(workflow_id: str) -> ServiceResult
UnifiedVaspService.preflight(workflow_id: str) -> ServiceResult
UnifiedVaspService.submit(workflow_id: str, stage: str | None) -> ServiceResult
UnifiedVaspService.status(workflow_id: str) -> ServiceResult
UnifiedVaspService.resume(workflow_id: str) -> ServiceResult
UnifiedVaspService.collect(workflow_id: str) -> ServiceResult
UnifiedVaspService.report(
    workflow_id: str, ReportRequest
) -> ServiceResult
```

The router uses only persisted `workflow_kind`; it never guesses from names,
files, formulas, or later-stage data. The service owns the transition table:

```text
PLANNED -> PREPARED -> PREFLIGHTED -> SUBMITTED -> RUNNING
RUNNING -> SCHEDULER_COMPLETED -> VALIDATED | VALIDATION_FAILED
any nonterminal state -> AWAITING_RESOURCE_CONFIRMATION
any nonterminal state -> AWAITING_SCIENTIFIC_CONFIRMATION
ambiguous submission -> RECONCILING
```

- [ ] Test all legal and illegal transitions.
- [ ] Test submit before preflight is blocked.
- [ ] Test stale manifest revision cannot overwrite newer status.
- [ ] Test service checks receipts and ResourcePolicy immediately before submit.
- [ ] Test optional executor unavailability returns a typed diagnostic.

Run:

```bash
uv run pytest -q tests/test_vasp_unified_service.py
```

Suggested commit: `feat: add unified VASP router and state machine`

---

### Task 11: Implement the Ten Deferred Tools and Evidence Mapping

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/tool_pack.py`
- Create: `tests/test_vasp_unified_tools.py`
- Read: `src/photomatagent/scientific/capabilities/contracts.py`

**Consumes:** `UnifiedVaspService`.

**Produces:** Exactly ten `Tool` classes and one capability pack. Each has
`exposure=ToolExposure.DEFERRED`, a concise search description, bounded output,
and Pydantic validation inside `execute()`.

Public schemas:

```text
capabilities: {workflow_kind?}
plan:         {workflow_kind, scientific_spec}
prepare:      {workflow_id}
preflight:    {workflow_id}
submit:       {workflow_id, stage?}
status:       {workflow_id}
resume:       {workflow_id}
collect:      {workflow_id}
report:       {workflow_id, report_request}
```

- [ ] Assert submit schema excludes workflow_dir, job_name, input_dir, profile,
  nodes, tasks_per_node, walltime_minutes, approval_ids, and fingerprints.
- [ ] Test nested Pydantic validation catches malformed scientific specs even
  though the generic registry validator is shallow.
- [ ] Test valid collection maps evidence to `ScientificToolResult.evidence`
  and therefore `state_updates`.
- [ ] Test invalid collection returns `data["evidence_gaps"]` and no evidence.
- [ ] Test errors and observations remain bounded and secret-free.

Run:

```bash
uv run pytest -q tests/test_vasp_unified_tools.py tests/test_tool_surface.py tests/test_permissions.py
```

Suggested commit: `feat: expose unified deferred VASP tools`

---

### Task 12: Replace Built-in VASP Registration

**Files:**

- Modify: `src/photomatagent/scientific/applications/vasp/tools.py`
- Modify: `src/photomatagent/scientific/capabilities/registry.py` only if the
  existing factory cannot construct the unified pack cleanly
- Modify: `tests/test_vasp_unified_surface.py`
- Modify: existing registry assertions in VASP tests

**Consumes:** Unified tool pack from Task 11.

Rules:

- `VaspCapabilityPack.tools()` returns only the ten unified tools, including
  `vasp.wait`.
- Remove the legacy `_molecular_tools()` / `_study_tools()` pack-construction
  alternatives; molecular and study execution is reachable only through the
  cached unified composition graph.
- Legacy classes may remain temporarily but are not registered and are not
  `DEFERRED`.
- Remove Task 0 strict-xfail markers.
- Verify both progressive and eager modes.

- [ ] Test registry public VASP names equal the ten-name set.
- [ ] Test search ranks unified tools and never returns legacy names.
- [ ] Test describe and bridged/direct invocation reject legacy names.
- [ ] Test the base runtime still starts when VASP dependencies are absent.

Run:

```bash
PHOTOMATAGENT_MCP_AUTO_CONNECT=0 uv run pytest -q tests/test_vasp_unified_surface.py tests/test_tool_discovery_sprint3.py tests/test_tool_surface.py tests/test_vasp.py
```

Suggested commit: `refactor: register one public VASP tool family`

---

### Task 13: Converge the SCNet MCP Adapters

**Files:**

- Modify: `src/photomatagent/mcp/manager.py`
- Modify: `src/photomatagent/mcp_servers/scnet/server.py`
- Modify: `tests/test_mcp_manager.py`
- Create: `tests/test_vasp_mcp_unified_adapter.py`

Rules:

- Replace partial molecular/study duplicate detection with full VASP-family
  detection when the unified built-in pack is present.
- Remove the environment override that can re-enable duplicate local VASP
  adapters.
- Keep MCP status stubs and non-VASP tools unchanged.
- Bundled MCP VASP aliases call the narrow application service, never locate or
  execute Tool objects.
- Keep `MCPServerManager` as persistent session owner; do not introduce
  throwaway event loops.

- [ ] Test all `scnet_science.vasp_*`, `vasp_molecule_*`, and `vasp_study_*`
  adapters are skipped locally when the unified pack exists.
- [ ] Test NAMD, MAGUS, and status tools remain registered.
- [ ] Test no environment setting re-enables duplicate VASP adapters.
- [ ] Test one external MCP alias call produces one unified service call.
- [ ] Test MCP offline state is typed and does not remove built-in tools.

Run:

```bash
uv run pytest -q tests/test_mcp_manager.py tests/test_mcp_boundary.py tests/test_vasp_mcp_unified_adapter.py
```

Suggested commit: `refactor: make SCNet MCP a unified VASP adapter`

---

### Task 14: Implement Unified Recovery Decisions

**Files:**

- Create: `src/photomatagent/scientific/applications/vasp/unified/recovery.py`
- Modify: `src/photomatagent/scientific/applications/vasp/molecular/recovery.py`
  only to delegate shared classification without changing molecular science
- Create: `tests/test_vasp_unified_recovery.py`

**Produces:**

```python
class RecoveryAction(str, Enum):
    AUTO_RESUME = "AUTO_RESUME"
    RECONCILE = "RECONCILE"
    NEEDS_RESOURCE_CONFIRMATION = "NEEDS_RESOURCE_CONFIRMATION"
    NEEDS_SCIENTIFIC_CONFIRMATION = "NEEDS_SCIENTIFIC_CONFIRMATION"
    STOP = "STOP"

class RecoveryOutcome(BaseModel):
    action: RecoveryAction
    reasons: list[str]
    scientific_changes: list[ScientificChange] = Field(default_factory=list)
    resource_recommendation: ResourceRequest | None = None
```

- [ ] Test SSH/status failure never submits.
- [ ] Test ambiguous submission always reconciles first.
- [ ] Test validated CONTCAR restart with identical scientific intent is
  automatic and records the artifact hash.
- [ ] Parameterize ENCUT, KPOINTS, SOC, functional, smearing, POTIM, IBRION,
  EDIFF, EDIFFG, charge, spin, POTCAR, structure, and stage changes; each must
  produce a pending scientific decision.
- [ ] Test OOM/time-limit escalation produces a resource decision.
- [ ] Test a matching receipt permits only the exact proposed recovery.

Run:

```bash
uv run pytest -q tests/test_vasp_unified_recovery.py tests/test_vasp_molecular_phase43.py tests/test_remote_lifecycle.py
```

Suggested commit: `feat: gate scientific VASP recovery changes`

---

### Task 15: Remove Legacy Public Paths and Migrate Guidance

**Files:**

- Modify: `src/photomatagent/scientific/applications/vasp/tools.py`
- Modify: `src/photomatagent/scientific/applications/vasp/application.py`
- Modify: `src/photomatagent/scientific/applications/vasp/molecular/tool_pack.py`
- Modify: `src/photomatagent/scientific/applications/vasp/study/tools.py`
- Modify: `skills/vasp-hpc-operator/SKILL.md`
- Modify: `skills/molecular-vasp-study/SKILL.md`
- Modify: `docs/scnet_scientific_compute.md`
- Modify: `README.md`
- Modify: `.env.example` only to document verified variable names, not guessed
  cluster limits
- Create: `tests/test_vasp_documented_surface.py`

Rules:

- Remove old Tool registration and direct public submission implementations.
- Preserve internal molecular/study services, parsers, analysis, and supported
  legacy-manifest migration readers.
- Active skill frontmatter and bodies reference only the ten unified tools.
- Legacy names may appear only in an explicitly marked migration section.
- Do not use string-only checks as the sole runtime protection; behavior tests
  from Tasks 0 and 12 remain authoritative.

- [ ] Use static call-site search plus fake-backend tests to prove no public VASP
  path calls `backend.submit_script()` directly.
- [ ] Test every real submission reaches one `SubmitOnceSession` record.
- [ ] Test active skills contain no instructions to call legacy names.

Run:

```bash
uv run pytest -q tests/test_vasp_documented_surface.py tests/test_skills.py tests/test_vasp_hpc_operator_skill.py tests/test_vasp_unified_surface.py
```

Suggested commit: `refactor: remove legacy VASP public paths`

---

### Task 16: End-to-End and Repository-Boundary Verification

**Files:**

- Create: `tests/test_vasp_unified_end_to_end.py`
- Modify production code only through a separately reviewed fix when this task
  exposes a defect
- Update `AGENTS.md` only if repository-wide ownership or invariants actually
  changed; do not add a dated success claim without fresh results

Use fake backends to execute periodic, molecular, and study flows through:

```text
plan -> prepare -> preflight -> submit -> status -> collect -> report
```

Required assertions:

- Only the ten `vasp.*` names are visible in progressive and eager modes.
- Each stage has one stable request ID and repeated submit creates no second job.
- Resource decisions and approval receipt IDs appear in provenance, without
  secrets.
- Scientific changes without a receipt do not submit.
- Runtime allow-all cannot bypass the application-level decision store.
- Scheduler completion plus invalid output creates no evidence.
- Valid output creates structured evidence and state updates.
- Resume after process restart retains manifest and JobRegistry state.
- MCP and built-in entry points return the same bounded service payload.
- Optional VASP configuration failure remains a typed capability diagnostic.

Run in this order:

```bash
PHOTOMATAGENT_MCP_AUTO_CONNECT=0 uv run pytest -q tests/test_vasp_unified_end_to_end.py
PHOTOMATAGENT_MCP_AUTO_CONNECT=0 uv run pytest -q tests/test_vasp*.py tests/test_remote*.py tests/test_mcp*.py tests/test_tool_surface.py tests/test_permissions.py
uv run photomatagent doctor
uv run photomatagent tools surface
uv run pytest -q
uv run mypy src
git diff --check
git diff --stat
git status --short
```

Report exact pass, skip, fail, and mypy counts. Compare with the baseline visible
at implementation start; do not claim a green repository while any failure
remains.

Suggested commit: `test: verify unified VASP paradigm end to end`

---

## Worker Prompt Template

Give the coding agent exactly one task with this prompt:

```text
You are implementing Task N from:
docs/superpowers/plans/2026-08-27-unified-vasp-paradigm.md

Read first:
1. AGENTS.md
2. docs/superpowers/specs/2026-08-27-unified-vasp-paradigm-design.md
3. Task N and every interface it consumes
4. The nearest existing tests and authoritative source files named by Task N

Rules:
- Work only on Task N.
- Start with git status --short and preserve all existing changes.
- Do not modify files outside Task N without stopping and requesting review.
- Do not modify .env, credentials, user_input, or user calculation artifacts.
- Do not connect to SSH/SCNet or submit a real VASP/Slurm job.
- Write the failing test first and run it to confirm the expected failure.
- Implement the smallest change that satisfies the task and AGENTS.md.
- Do not weaken or delete unrelated tests.
- Do not invent scientific defaults or change scientific parameters silently.
- Do not create another ToolRegistry, ResourcePolicy, runtime, or state model.
- Do not auto-commit.

At completion, stop and report:
- files changed;
- failing-test evidence before implementation;
- final test commands and exact results;
- interface and design decisions;
- scientific or authorization behavior affected;
- remaining risks;
- git diff --check, git diff --stat, and git status --short;
- suggested commit message.
```

## Human Review Gate After Every Task

Reject the task if any answer is yes:

- Did it touch files outside the task without prior approval?
- Did it bypass `AgentRuntime` for a model-requested tool?
- Did it add a second registry, resource hard-cap policy, or state authority?
- Did it expose raw Slurm controls, paths, fingerprints, or approval IDs?
- Did it permit allow-all to bypass scientific or HPC gates?
- Did any public VASP path call the backend submit method directly?
- Did it treat scheduler completion as scientific validation?
- Did it silently alter a scientific parameter or default?
- Did it register a legacy VASP or MCP alias as model-visible?
- Did it hide a failure by deleting, weakening, or broadly skipping a test?

## Required Execution Order

```text
0 surface baseline
1 models and fingerprints
2 manifest repository
3 approval receipts
4 user approval command
5 resource orchestration
6 executor contract
7 periodic adapter
8 molecular adapter
9 study adapter
10 router and service
11 unified tools and evidence
12 built-in registration
13 MCP convergence
14 recovery policy
15 legacy removal and guidance
16 end-to-end verification
```

Tasks 3 through 15 modify adjacent authority and execution boundaries. Do not
run them in parallel.
