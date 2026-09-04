# DEVSIM Async and Generation Typing Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the DEVSIM async completion hang and make the generation capability pass strict mypy checking without changing scientific behavior.

**Architecture:** Keep DEVSIM execution in its existing restricted, terminable child process, but await the blocking process-management call through a plain concurrent future with event-loop polling so completion does not depend on a lost cross-thread wakeup. Correct generation annotations by preserving distinct path/result metadata names, giving retrieval rows a common iterable type, and using type-checking-only PyTorch imports while retaining optional runtime imports.

**Tech Stack:** Python 3.12, asyncio, concurrent.futures, multiprocessing, pytest, mypy, optional PyTorch.

**Spec:** Root `AGENTS.md` repository invariants and the user request to repair the two previously reported non-VASP failures.

## Global Constraints

- Preserve all existing user and VASP work in the dirty worktree.
- Do not broaden DEVSIM script permissions or execution authority.
- Keep PyTorch optional at package-import time.
- Use local/fake DEVSIM tests; do not run external scientific jobs.

---

### Task 1: Bound and repair DEVSIM async completion

**Files:**
- Modify: `tests/test_device_execution.py`
- Modify: `src/photomatagent/scientific/capabilities/device/__init__.py`

**Interfaces:**
- Consumes: `DeviceRunScriptTool._execute_script(Path, str, float) -> dict[str, Any]`
- Produces: unchanged `DeviceRunScriptTool.execute(dict[str, Any]) -> ScientificToolResult`

- [x] Wrap each real async execution in `asyncio.wait_for` so a lost completion signal fails deterministically instead of hanging the suite.
- [x] Run the success test and confirm it fails with `TimeoutError` before production changes.
- [x] Replace `asyncio.to_thread` with a one-worker `ThreadPoolExecutor` and short async polling of the concurrent future.
- [x] Run both device execution tests and confirm success and timeout behavior pass.

### Task 2: Correct generation typing without behavior changes

**Files:**
- Modify: `src/photomatagent/scientific/capabilities/generation/conditional_vae.py`
- Modify: `src/photomatagent/scientific/capabilities/generation/tools.py`

**Interfaces:**
- Consumes: optional `torch`/`torch.nn`, `VAEFormulaGenerator.generate`, JSON/CSV candidate rows.
- Produces: unchanged conditional-VAE and generation tool runtime APIs.

- [x] Run scoped mypy and retain the exact seven-error baseline.
- [x] Separate optional runtime module aliases from their permissively typed bindings in `conditional_vae.py`.
- [x] Rename asset metadata and generated metadata variables so their types do not collide.
- [x] Annotate retrieval rows with a shared iterable mapping type and keep file closure deterministic.
- [x] Run scoped mypy and generation tests.

### Task 3: Integrated verification

**Files:**
- Verify only; no additional production scope.

**Interfaces:**
- Consumes: repaired device and generation modules.
- Produces: exact test/type-check evidence and an honest report of any remaining failures.

- [x] Run device and generation-focused pytest suites.
- [x] Run `mypy src`.
- [x] Run the full pytest suite, recording the pre-existing literature-model download block separately.
- [x] Run compileall, `git diff --check`, `git diff --stat`, and `git status --short`.
