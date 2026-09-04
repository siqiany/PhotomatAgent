# Unified VASP Paradigm Design

## Purpose

PhotomatAgent currently exposes overlapping periodic, molecular, study, and
SCNet MCP VASP tool families. The model can therefore select a legacy submission
path that bypasses newer lifecycle behavior or exposes raw scheduler controls.
This design replaces those public choices with one stable `vasp.*` facade while
preserving the existing scientific implementations behind internal adapters.

## Public Contract

The only model-visible VASP tools are:

```text
vasp.capabilities
vasp.plan
vasp.prepare
vasp.preflight
vasp.submit
vasp.status
vasp.wait
vasp.resume
vasp.collect
vasp.report
```

All ten tools are `DEFERRED`. They are discoverable through the existing
`tool_search -> tool_describe -> tool_call` path and remain subject to
`AgentRuntime` validation, permissions, observation limits, events, and state
updates.

Legacy names such as `vasp_molecule.*`, `vasp_study.*`,
`vasp.inspect_result`, `vasp.run_workflow`, and
`scnet_science.vasp_*` are not model-visible. During migration they may exist as
internal Python services or `HIDDEN` compatibility tools; after migration they
are removed from registration. `searchable=False` alone is insufficient.

## Target Architecture

```text
AgentRuntime -> ToolRegistry -> unified vasp.* Tool
                              -> UnifiedVaspService
                                 -> ManifestRepository
                                 -> ApprovalReceiptStore
                                 -> ResourceAuthorizationService
                                 -> UnifiedVaspRouter
                                    -> PeriodicVaspExecutor
                                    -> MolecularVaspExecutorAdapter
                                    -> VaspStudyExecutorAdapter
                                       -> MolecularVaspExecutorAdapter
                                 -> SubmitOnceSession.submit_once
                                 -> SCNetBackend
```

`UnifiedVaspService` is an application service, not another runtime or tool
registry. Tool classes, MCP adapters, and CLI approval commands call narrow
methods on this service; none execute another `Tool` instance.

## Workflow Identity and Storage

`vasp.plan` accepts a typed request with a discriminated `workflow_kind` and a
workspace-relative structure reference. It creates a program-generated
`workflow_id`, snapshots the scientific input into a managed workflow directory,
and writes a versioned manifest. Later tools accept `workflow_id`, not arbitrary
workflow directories.

The repository stores manifests under a workspace-owned location and resolves
every path with `Workspace.resolve`. Writes are atomic and revision-checked.
The manifest represents VASP execution state only; it does not replace
`ConversationState`, `ScientificState`, or `ScientificLoopState`.

## Scientific Specifications and Fingerprints

The manifest contains one typed scientific specification:

- `PeriodicScientificSpec`: structure snapshot, named VASP profile, POTCAR
  policy, and explicit scientific overrides.
- `MolecularScientificSpec`: existing `WorkflowSpec`, including explicit total
  charge and spin multiplicity.
- `StudyScientificSpec`: existing `VaspStudyRequest` and resolved study plan.

The caller cannot provide fingerprints. The service computes:

- `scientific_fingerprint` from canonical scientific intent: structure content,
  profile and INCAR/KPOINTS semantics, POTCAR policy, charge, spin, and stage
  topology.
- `execution_fingerprint` from the scientific fingerprint plus effective
  resources and restart-attempt inputs.

Changing resources does not change the scientific fingerprint. Changing ENCUT,
KPOINTS, SOC, functional, smearing, ionic algorithm, convergence thresholds,
charge, spin, pseudopotentials, structure, or stages does.

## Approval Model

Normal runtime permission is necessary but not sufficient for resource or
scientific changes. `/approve -a` must not satisfy an application-level
scientific decision.

When a resource escalation or scientific change is proposed, the service writes
a pending decision with an immutable hash and returns
`NEEDS_RESOURCE_CONFIRMATION` or `NEEDS_SCIENTIFIC_CONFIRMATION`. A user-only CLI
command handled by the central command router records an approval receipt. The
model cannot create a receipt through tool arguments.

Each receipt binds:

- workflow ID;
- decision ID and decision hash;
- approval kind;
- current scientific fingerprint;
- current execution fingerprint when applicable;
- timestamp and approving user/session source.

Any bound change invalidates the receipt. Submit and resume query the receipt
store themselves; their public schemas do not contain `approval_ids` or an
`approved` boolean.

## Resource Decisions

`VaspResourcePlanner` produces a recommendation from the selected profile,
system size, stage, SOC, calibration data, and existing molecular/study resource
models. `ResourceAuthorizationService` is the single orchestrator that combines:

1. the recommendation;
2. the automatic-authorization budget;
3. existing `ResourcePolicy` hard caps and submit feature flag;
4. detected cluster capabilities;
5. a matching approval receipt when confirmation is required.

The service never silently reduces an excessive request. A recommendation above
the automatic budget needs confirmation; a request above `ResourcePolicy` is
denied even with confirmation. Raw nodes, tasks per node, partition, memory, and
walltime are absent from `vasp.submit`.

## Submission Lifecycle

Every periodic, molecular, and study child submission calls
`SubmitOnceSession.submit_once`. Request IDs are stable for the same workflow,
stage, scientific fingerprint, and effective execution decision. Every attempt
uses a unique remote directory. Ambiguous client timeouts enter reconciliation;
unknown or duplicate scheduler states never cause blind resubmission.

The existing `VaspApplication` retains input generation, POTCAR handling, Slurm
rendering, output validation, and parsing. It no longer owns the public job
lifecycle. Study remains an orchestrator and delegates child calculations to the
molecular adapter rather than creating another submission path.

## Recovery Policy

The following operations may run automatically:

- SSH reconnect and bounded status-query retry;
- ambiguous-submission reconciliation;
- scheduler refresh and adoption of a uniquely matched existing job;
- artifact-download retry;
- continuation from a validated `CONTCAR` when scientific inputs are unchanged;
- retry within an already approved resource decision.

The following require resource confirmation:

- more nodes, tasks, memory, or walltime;
- partition changes;
- any change beyond the automatic budget.

The following require scientific confirmation:

- ENCUT, KPOINTS, functional, SOC, smearing, POTIM, IBRION, EDIFF, or EDIFFG;
- total charge, spin multiplicity, ISPIN/NUPDOWN/MAGMOM semantics;
- POTCAR policy or pseudopotential set;
- structure, stage topology, or scientific method changes.

A failed status query never triggers submission. A `CONTCAR` continuation is
automatic only after structural validation and an unchanged scientific intent
fingerprint; the restart artifact hash is recorded in attempt provenance.

## Scientific Results

Slurm `COMPLETED` is scheduler state only. `vasp.collect` downloads artifacts,
runs deterministic application validation, and only then produces
`ScientificEvidence`. Evidence uses the existing contracts and includes source,
method, units, fidelity, provenance, and limitations. Missing or invalid results
are returned as explicit `evidence_gaps` and never promoted to validated claims.

`vasp.report` is the single analysis/report entry point. A typed report request
selects summary, orbital, ESP, binding-energy, or study reporting while internal
molecular and study analysis services remain separate.

## MCP Behavior

MCP remains an adapter governed by the normal runtime when model-visible. When
the built-in unified VASP pack is present, local registration skips every
equivalent `scnet_science.vasp_*`, `vasp_molecule_*`, and `vasp_study_*`
adapter. No environment variable can re-enable duplicate local VASP tools.

The bundled SCNet MCP server may retain transport aliases for external clients,
but aliases call the same narrow unified application service and preserve all
HPC gates. They do not locate or execute Tool objects. NAMD, MAGUS, and unrelated
MCP tools are unchanged. `MCPServerManager` remains the lifecycle owner of its
sessions.

## Success Criteria

- Progressive and eager surfaces expose only the ten unified VASP tools,
  including the bounded `vasp.wait` boundary.
- Search, describe, bridged call, and guessed direct call cannot reach legacy
  names.
- Periodic, molecular, and study workflows use one idempotent lifecycle.
- Repeated submit produces at most one scheduler job for one request ID.
- The model cannot provide raw Slurm controls, fingerprints, paths, or approval
  receipts to `vasp.submit`.
- Allow-all runtime permission cannot bypass application, scientific, resource,
  path, or HPC gates.
- Scheduler completion without deterministic validation produces no evidence.
- Optional configuration failures return typed capability diagnostics.
- The full test suite and mypy are run at completion; exact results are reported
  without hiding pre-existing failures.
