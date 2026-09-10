# Internal Expert Review Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `/expert` inside interactive chat and safely promote current or historical sessions into reviewable, iterable evolution tasks.

**Architecture:** A historical importer materializes a provenance-bound `IMPORTED_SESSION` episode, while a focused CLI wizard selects the source and reuses the existing feedback and compilation flows. The command router intercepts the entire workflow so expert entries never become ordinary model input.

**Tech Stack:** Python 3.12, Pydantic, Typer, prompt_toolkit-compatible prompt protocol, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-internal-expert-mode-design.md`

## Global Constraints

- `AgentRuntime` remains the only authority for model-requested tools.
- Imported sessions never invent scientific evidence, deterministic outcomes, or target constraints.
- A new historical import requires a confirmed `TargetSpec` containing at least one constraint.
- Artifacts remain workspace-contained, immutable, and SHA-256 bound.
- Expert input is intercepted by the slash-command workflow and never sent as ordinary chat.
- Run one focused end-to-end test only, per the user's time constraint.

---

### Task 1: Historical session importer

**Files:**
- Create: `src/photomatagent/scientific/evolution/importer.py`
- Modify: `src/photomatagent/scientific/evolution/models.py`
- Modify: `src/photomatagent/scientific/evolution/service.py`

**Interfaces:**
- Produces: `HistoricalSessionPreview`, `preview_historical_session(...)`, and `HistoricalSessionImporter.import_session(...) -> EvolutionTask`.
- Consumes: `Workspace`, `EvolutionService`, `load_trace`, optional `SessionSnapshot`, confirmed `TargetSpec`, and optional confirmed artifact path.

- [ ] Add a failing focused test fixture that expects an `IMPORTED_SESSION` v001 with source session ID, `summary=None`, canonical artifact hash, and unchanged stored scientific state.
- [ ] Run the test and confirm it fails because the importer and execution mode do not exist.
- [ ] Add `IMPORTED_SESSION` to `ExecutionMode` and allow it only for the initial historical reservation path.
- [ ] Implement preview extraction from the latest trace run and optional snapshot without deriving scientific claims.
- [ ] Implement deterministic, idempotent import with exclusive canonical result creation and content/hash conflict detection.
- [ ] Re-run the focused test until the importer assertions pass.

### Task 2: `/expert` interactive workflow

**Files:**
- Create: `src/photomatagent/cli/expert.py`
- Modify: `src/photomatagent/cli/commands.py`
- Modify: `README.md`
- Test: `tests/test_internal_expert_mode.py`

**Interfaces:**
- Consumes: Task 1 importer, `run_feedback_command`, `run_compile_command`, `ChatCommandRouter` dependencies.
- Produces: `run_expert_mode(...)` and exact slash forms `/expert`, `/expert <session-id>`, `/expert history`.

- [ ] Extend the failing end-to-end test to route `/expert <session-id>`, confirm goal/result/target, complete one rubric form, decline compilation, and assert no expert answer entered `ConversationState`.
- [ ] Run the test and confirm it fails because `/expert` is unknown.
- [ ] Implement bounded argument parsing and history selection in `cli/expert.py`.
- [ ] Implement target-file validation, source preview/confirmation, import-or-reuse behavior, feedback collection, optional compile, and optional iterate prompts.
- [ ] Register `/expert` in `COMMANDS` and route it before generic CLI groups.
- [ ] Document the internal workflow and historical-session requirements in `README.md`.
- [ ] Run `pytest -q tests/test_internal_expert_mode.py` and confirm it passes.

### Task 3: Focused review and handoff

**Files:**
- Review all files changed by Tasks 1–2.

**Interfaces:**
- Consumes: completed implementation and focused test output.
- Produces: a review finding list and final diff summary.

- [ ] Inspect `git diff --check`, `git diff --stat`, and `git status --short`.
- [ ] Review source-session identity, path containment, idempotency, artifact hashing, and no-chat-input guarantees.
- [ ] Fix only issues found in this scoped review.
- [ ] Re-run `pytest -q tests/test_internal_expert_mode.py` once after fixes.
- [ ] Commit the completed feature on `codex/internal-expert-mode`.

