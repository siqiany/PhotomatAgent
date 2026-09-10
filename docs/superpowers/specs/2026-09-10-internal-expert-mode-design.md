# Internal Expert Review Mode Design

## Goal

Add a dedicated `/expert` workflow inside interactive chat so a user can review
the current result or a historical session without invoking shell-level
`photomatagent evolve ...` commands. Historical sessions become provenance-bound
`v001` evolution episodes and then reuse the existing feedback, compilation,
revision, and iteration lifecycle.

## User interface

- `/expert` reviews the current interactive session.
- `/expert <session-id>` reviews one named historical session.
- `/expert history` lists recent session IDs and asks the user to select one.
- Every prompt is prefixed with `[EXPERT MODE | ...]`; responses are consumed by
  the command router and are never sent to `AgentRuntime.run()` as chat input.
- `/cancel` exits the wizard. Writes occur only after an explicit confirmation.

For a session not already linked to an evolution task, the wizard previews the
detected original goal and final assistant answer, lets the user replace either
with explicit input, and requires a workspace-contained `TargetSpec` JSON file.
The target must contain at least one machine-verifiable constraint. This avoids
inventing constraints from prose and ensures the imported result can safely
continue into a scientific iteration.

After import, the wizard runs the existing 1–5 expert rubric form. It then asks
whether to compile the feedback immediately. If the revision plan is confirmed,
it asks whether to run the next iteration. Answering no leaves a durable
checkpoint that can be resumed later with `/expert <session-id>` or existing
`/evolve` commands.

## Historical import

A new application service reads sessions only from the workspace session root.
It uses `load_trace()` for typed event validation and `load_session_snapshot()`
when a snapshot exists.

- The original goal is copied from the latest `LoopStarted` event or explicit
  user input; it is never inferred from the answer.
- The primary artifact is either a user-confirmed workspace file or the exact
  last recorded final response. It is copied to
  `user_output/<evolution-id>/v001/result.md` using exclusive creation and is
  hashed after writing.
- A resumable session contributes its stored `ScientificState` unchanged.
- A trace-only session contributes an empty `ScientificState` with the confirmed
  goal only. No claims, evidence, or validation status are reconstructed.
- `EpisodeRecord.summary` remains `None`, because the historical run did not
  necessarily execute `ScientificLoopController`.
- The episode uses a new `IMPORTED_SESSION` execution mode and records the
  original runtime session ID and event-log path.

The evolution ID is deterministic from the source session ID. Repeating import
returns the existing task only when its source session, artifact hash, goal, and
target hash match; otherwise it fails without overwriting data.

## Architecture

`src/photomatagent/scientific/evolution/importer.py` owns historical-source
extraction and import orchestration. It composes `EvolutionService` and
`EvolutionStore`; it does not execute model-requested tools. The existing service
gains only the `IMPORTED_SESSION` initial-reservation allowance.

`src/photomatagent/cli/expert.py` owns the interactive wizard. The slash router
resolves the current `PromptSession`, runtime, logger, and workspace, then calls
this module. Feedback and compilation reuse `run_feedback_command()` and
`run_compile_command()` so there is still one authoritative rubric and revision
path.

## Safety and failure behavior

- Absolute paths and `..` escapes are rejected through `Workspace.resolve`.
- Missing final output, invalid target JSON, empty constraints, artifact mismatch,
  or duplicate import with different content aborts before expert feedback.
- Raw expert comments remain immutable and SHA-bound to the imported artifact.
- Trace-only imports are explicitly marked in target metadata and do not create
  scientific evidence.
- A cancelled form does not persist partial expert feedback or a revision plan.
- Existing `/evolve` commands remain compatible.

## Verification

One focused end-to-end test will create a historical session snapshot, enter
`/expert <session-id>` through `ChatCommandRouter`, import it as `v001`, submit a
rubric review, decline compilation, and assert that the result SHA, session
provenance, scientific-state copy, task status, and feedback record are correct.
The test also proves that the expert responses are not appended to the runtime
conversation.

