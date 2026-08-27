# Evidence-Guided Scientific Feedback Loop (design notes)

Status: implemented in `src/photomatagent/scientific/loop/` (P0).

## Problem statement

In the base runtime, `StopPolicy` stops the loop when the model emits a
response with no tool calls (`final_response`). Nothing distinguishes "the
model decided it is done" from "the science is verified". For materials
design tasks this is the core failure mode: a model that proposes one
composition and stops has produced a *candidate*, not a *result*.

The feedback loop enforces one invariant everywhere:

```
model final_response ≠ scientific PASS
```

## Architecture

Two layers, existing inner loop untouched:

```text
Inner loop (unchanged):          AgentRuntime.run(goal)
                                 └─ maker: model stream + tool calls + permissions

Outer loop (new):                ScientificLoopController
     TargetSpec
       → Maker round (AgentRuntime.run(goal + feedback))
       → candidate extraction (structured scientific state only)
       → ScientificEvaluator   (deterministic Checker, Maker-independent)
       → EvaluationReport
       → StagnationDetector    (fingerprints + score deltas + signatures)
       → ScientificLoopPolicy  (SUCCESS / CONTINUE / ESCALATE / STALLED /
                                INCONCLUSIVE / BUDGET_EXHAUSTED)
       → FeedbackSignal        (structured; entered as the next maker turn)
```

- `outer depends on inner`, never the reverse.
- The controller never executes a tool itself; all tool use goes through
  `AgentRuntime._handle_tool_call` (permission policy, approval handlers and
  HPC gating remain authoritative).
- `ScientificState` = what we know scientifically.
  `ScientificLoopState` = where the search over candidates currently is.
  They are kept separate on purpose.

## Deterministic check, independent maker/checker

- `TargetSpec.constraints` are evaluated by `evaluate_constraint()`:
  numeric comparisons (`lt/le/gt/ge/eq/between`) are program decisions.
- `ScientificEvaluator` builds a property → evidence map from
  - `ScientificEvidence` in `ScientificState` (property match, alias table),
  - JSON payloads embedded in `Evidence.content` (e.g. mock tool results),
  - candidate-declared generation-time predictions (always low fidelity).
  Missing value → `UNKNOWN`, never `PASS`.
- Evidence fidelity decides confidence and escalation (`fidelity_rank`):
  `ml_generated < analytical/empirical < continuum/kp/tight_binding <
  ml_potential < dft < experimental`.
- Generator never verifies its own candidate; a VAE proposal stays
  `UNVALIDATED_GENERATED_STRUCTURE` until the evaluator has evidence for the
  target's constraints.

## Feedback

`build_feedback(target, candidate, evaluation, history)` returns a structured
`FeedbackSignal` or `None` (a PASSing candidate needs no feedback). It states
what failed, why, missing evidence, next priorities, and what must not be
repeated. The controller appends the rendered signal to the next maker
instruction (never into the static system prompt, preserving the
cache-friendly trailing-snapshot layout).

## Stagnation and termination

`StagnationDetector` (default `patience=3`, `epsilon=1e-3`):
`HgTe, HgTe, HgTe, HgTe` is one iteration, not four. Identical candidate
fingerprints, identical violation/evidence-gap signatures and below-epsilon
score improvements accumulate toward `STALLED`.

`ScientificLoopPolicy.decide()` terminates deterministically:

| Decision | Condition |
| --- | --- |
| SUCCESS | all HARD constraints pass, no critical evidence gap, confidence ≥ threshold |
| CONTINUE | resolvable violation or evidence gap remains |
| ESCALATE | critical constraints rest on cheap evidence (higher-fidelity needed) |
| STALLED | stagnation detector tripped |
| INCONCLUSIVE | no candidate / no evidence possible (capability unavailable, tool failures) |
| BUDGET_EXHAUSTED | round / candidate caps exceeded |

## Events

New kinds appended to the existing `AnyRuntimeEvent` union and JSONL stream:
`scientific_loop_started`, `candidate_proposed`, `candidate_evaluated`,
`scientific_feedback_generated`, `scientific_loop_decision_made`,
`scientific_loop_completed`, `scientific_loop_stalled`. The JSONL trace can
answer: what was proposed each round, why it failed, what evidence was used,
why the strategy changed, which candidate is best, and why the loop stopped.

## CLI

```bash
uv run photomatagent loop --demo --provider fake --approval auto --max-rounds 6
uv run photomatagent loop --target-json '<TargetSpec JSON>' --goal "..." ...
uv run photomatagent experiments run experiments/scientific-feedback-loop-smoke.json
```

The smoke experiment runs fully offline on the fake provider (no API, no
HPC): candidate generation, deterministic evaluation, structured feedback,
subsequent rounds, and deterministic termination are all exercised.

## P0 limitations (explicit)

- One primary candidate is extracted and evaluated per round; multi-candidate
  list sorting is future work.
- Evidence is matched per property; per-candidate binding uses subject/formula
  when named, with a documented fallback when the evidence names no material.
- Escalation is a decision aid: `ESCALATE_FIDELITY` recommends higher-fidelity
  work but never auto-submits HPC jobs; permissions/approval/HPC gating stay
  authoritative.
- Soft-constraint optimisation is scored but not required for PASS.
- Natural-language → `TargetSpec` compilation is intentionally out of P0
  scope: the CLI requires an explicit target (`--demo` / `--target-json`).