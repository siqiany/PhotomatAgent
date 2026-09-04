# PhotomatAgent Agent Guide

This file is the repository-wide contract for coding agents. It applies to the
entire tree unless a nearer `AGENTS.md` provides more specific instructions.
Read it before changing files, preserve user work, and keep the safety and
scientific-integrity boundaries below.

## Project Mission

PhotomatAgent is a Python 3.12 scientific-agent runtime for materials research,
with emphasis on infrared optoelectronic materials and detectors. It combines
an explicit model/tool loop, structured scientific state and evidence, optional
capability packs, resumable sessions, MCP adapters, and gated SCNet/Slurm
workflows for VASP, Hefei-NAMD, and MAGUS.

The system is designed to make model actions observable, reproducible, and
controllable. Do not trade away permissions, provenance, deterministic checking,
or submission safety for convenience.

## Architecture Map

```text
CLI / experiment runner
  -> AgentRuntime
     -> ContextEngine + ToolSurfacePlanner
     -> ModelProvider
     -> permission + path + argument checks
     -> ToolRegistry -> Tool.execute()
     -> bounded observations + events + state updates

ScientificLoopController
  -> AgentRuntime as Maker
  -> deterministic ScientificEvaluator as Checker
  -> feedback, stagnation, and budget policy

Scientific tools
  -> CapabilityPack or MCPRemoteTool adapter
  -> narrow application service
  -> SCNetBackend -> SSH/Slurm
```

Authoritative areas:

| Area | Primary location |
| --- | --- |
| Runtime and tool execution | `src/photomatagent/runtime/` |
| Tool contracts, registry, and exposure | `src/photomatagent/tools/` |
| Scientific state and evidence contracts | `src/photomatagent/scientific/state.py`, `src/photomatagent/scientific/capabilities/contracts.py` |
| Scientific outer loop | `src/photomatagent/scientific/loop/` |
| VASP/NAMD/MAGUS workflows | `src/photomatagent/scientific/applications/` |
| SSH/Slurm infrastructure | `src/photomatagent/scientific/remote/` |
| MCP lifecycle and adapters | `src/photomatagent/mcp/`, `src/photomatagent/mcp_servers/` |
| CLI assembly and commands | `src/photomatagent/cli/` |
| Model-readable scientific procedures | `skills/` |

## MUST: Repository Invariants

### Runtime and tools

1. **`AgentRuntime` is the only authority for model-requested tool execution.**
   Providers, CLI helpers, scientific evaluators, outer-loop controllers, and
   MCP clients must not execute model-requested tools directly.
2. **Each runtime has one authoritative `ToolRegistry` execution path.** Extend
   the existing registry and factories; do not create a parallel registry or an
   ad-hoc execution path for one subsystem.
3. **Preserve tool exposure semantics.** `DIRECT` tools expose full schemas;
   `DEFERRED` tools use `tool_search -> tool_describe -> tool_call`; `HIDDEN`
   tools are neither model-visible nor executable through model-requested
   runtime calls. Deferred calls still pass all normal runtime checks.
4. **Providers translate streams; they do not own the agent loop.** They do not
   call tools, decide permissions, mutate scientific state, or grade results.

### Scientific integrity

5. **Keep Maker and Checker separate.** The runtime proposes and gathers
   evidence; deterministic evaluators check it. A model does not grade its own
   result, and missing evidence is `UNKNOWN`, not `PASS`.
6. **Keep state responsibilities separate.** `ConversationState` stores the
   durable transcript, `ScientificState` stores scientific knowledge and
   provenance, and `ScientificLoopState` stores search/control state.
7. **Scientific claims require structured evidence.** Use the existing
   `ScientificEvidence`, `ScientificToolResult`, and `ToolResult.state_updates`
   contracts. Preserve source, method, units, fidelity, provenance, limitations,
   and explicit evidence gaps; never promote mock or unvalidated output into a
   validated conclusion.
8. **Optional capabilities fail soft.** Missing scientific dependencies or
   configuration become typed capability diagnostics and must not prevent the
   base runtime from starting.

### MCP and remote execution

9. **MCP is an adapter, not another runtime.** MCP tools enter the normal tool
   registry and keep the same permissions, validation, observation, evidence,
   and event path. Do not register duplicate MCP tools when an equivalent
   built-in scientific tool is authoritative. `MCPServerManager` remains the
   persistent owner of long-lived MCP sessions.
10. **Remote execution stays narrow.** `SCNetBackend` is internal SSH/Slurm
    infrastructure. Expose application operations, never arbitrary remote shell
    execution, to the model.
11. **HPC submission keeps every gate.** Preserve the submit feature flag,
    runtime approval, `ResourcePolicy`, resource limits, readiness checks,
    remote-path validation, and job-ID validation. Submission must use
    `SubmitOnceSession.submit_once` with persistent `JobRegistry`, stable request
    IDs, unique attempt directories, immutable job IDs, and reconciliation after
    ambiguous timeouts. Unknown scheduler state must never cause blind
    resubmission. Scheduler `COMPLETED` is not scientific success; collect and
    validate artifacts first.

### Files, persistence, and user work

12. **Path-based file tools remain workspace-contained.** Use
    `Workspace.resolve`; reject absolute-path and `..` escapes. `BashTool` uses a
    workspace cwd but is host command execution, not an OS sandbox, and must not
    be used to bypass path or sensitive-file controls.
13. **Compaction does not rewrite durable history.** It may change only the
    model-visible working context, never the durable transcript or snapshots.
14. **Preserve existing work.** Treat all pre-existing modified and untracked
    files as user-owned. Do not reset, overwrite, delete, or reformat unrelated
    changes. Never modify `.env`, credentials, `user_input/`, or user-generated
    artifacts unless the user explicitly requests it.

## SHOULD: Implementation Guidance

- Follow the existing Python 3.12, Pydantic, async, typed-event, registry, and
  factory patterns before introducing a new abstraction.
- Keep optional imports lazy and capability failures localized.
- Give every new model-visible tool a schema, exposure classification,
  validation, permission classification, bounded observation, events, and
  focused tests.
- Keep stable instructions and capability descriptions cache-friendly; keep
  live state in dynamic context.
- Keep `src/photomatagent/cli/commands.py` as the slash-command router and reuse
  existing Typer implementations.
- Put user-facing deliverables in `user_output/<task-name>/` and disposable
  intermediates in `tmp/`.
- If a requested change intentionally alters a MUST invariant, state the impact
  before implementation and update the relevant tests and documentation as part
  of the same reviewed change.

## Change Workflow

Before editing:

1. Run `git status --short` and preserve everything already present.
2. Read the nearest tests and the authoritative module for the area.
3. Limit the change to the requested subsystem and agreed design.

While editing:

- Prefer focused changes over opportunistic refactors.
- Do not silently broaden filesystem, shell, network, MCP, or HPC authority.
- Do not silently change scientific defaults, parameters, structures,
  pseudopotentials, charge, spin, or workflow stages.
- Use fake/local backends in tests; never submit a real HPC job as a test.

After editing:

1. Run the narrowest relevant tests first, then every architectural boundary
   touched by the change.
2. Before repository-wide completion claims, run the full suite and `mypy` and
   report exact results. Do not call the build green when known or new failures
   remain.
3. Run `git diff --check`, inspect `git diff --stat`, and re-run
   `git status --short`.

Useful commands:

```bash
uv run photomatagent doctor
uv run photomatagent scientific status
uv run photomatagent tools surface
uv run pytest -q tests/test_loop.py tests/test_permissions.py tests/test_tool_surface.py
uv run pytest -q tests/test_remote.py tests/test_remote_lifecycle.py tests/test_vasp.py
uv run pytest -q
uv run mypy src
git diff --check
```

## Maintaining This Guide

Keep this root file short, stable, and repository-wide. Put subsystem-specific
implementation rules in a nearer `AGENTS.md`, detailed procedures in `skills/`
or architecture documentation, and dated test/status reports in development
notes. Do not record speculative work as completed. Update this guide only when
repository-wide ownership or invariants genuinely change.
