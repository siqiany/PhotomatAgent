# Expert-Feedback Evolution Loop Design

## Purpose

PhotomatAgent already has an evidence-guided scientific loop in which
`AgentRuntime` acts as Maker, `ScientificEvaluator` is the deterministic
Checker, and the optional `ScientificJudge` provides read-only advisory
criticism. That loop improves one scientific run, but it does not yet preserve
external expert reviews, compare successive versions of the same task, or learn
which research strategy should be selected for later runs.

This design adds an asynchronous, expert-in-the-loop evolution layer around the
existing scientific loop. A researcher can run a task, close the CLI, obtain an
expert review later, attach that review to the exact result version, and start a
new episode of the same task. Successive episodes remain reproducible and
auditable. Expert feedback is never confused with normal chat input and never
bypasses deterministic scientific checks, runtime permissions, or HPC gates.

The first implementation establishes the evolution workflow, typed feedback,
versioned reruns, and comparison metrics. Bayesian cross-task strategy learning
is introduced behind a narrow interface after enough real trajectories exist;
the persistence schema must support it from the beginning without pretending
that a small initial dataset is sufficient for model training.

## Terminology

- **Evolution task**: one durable research objective identified by an
  `evolution_id`.
- **Episode**: one independent execution of that objective, identified by a
  monotonic version such as `v001`.
- **Runtime session**: the existing `AgentRuntime`/event-log session used by one
  episode. Every episode has its own runtime session.
- **Expert feedback**: an immutable human review that targets exactly one
  episode and result artifact.
- **Revision plan**: validated, structured changes compiled from one expert
  review for the next episode.
- **Strategy version**: the explicit reasoning, retrieval, evidence-escalation,
  critique, and stopping policy selected for an episode.
- **Experience**: a versioned record derived from an observed trajectory. It is
  not automatically a reusable skill.

## Architectural Placement

```text
CLI: photomatagent evolve / interactive /evolve shortcut
  -> EvolutionService
     -> EvolutionStore
     -> FeedbackCompiler (isolated, structured, no tools)
     -> RevisionPlanner (deterministic contract assembly)
     -> StrategySelector
     -> ScientificLoopController
        -> AgentRuntime (only model-requested tool execution authority)
        -> ScientificEvaluator (authoritative scientific Checker)
        -> ScientificJudge (optional, advisory, read-only)
     -> EpisodeComparator
     -> ExperienceRepository
```

The evolution layer is an application orchestration layer, not a second agent
runtime and not a second tool registry. It may construct and invoke a
`ScientificLoopController`; it may not execute tools, scientific backends, MCP
calls, SSH, Slurm, or HPC submissions directly.

`ScientificJudge` and a human expert have different roles. The Judge reviews
each candidate inside an episode and remains advisory. The human expert reviews
the episode result asynchronously and informs the next episode. Neither may
turn missing evidence or a deterministic hard-constraint failure into PASS.

## H-BEAL Algorithm Boundary

The evolution process has two time scales:

```text
Fast loop within episode:
candidate -> evidence -> deterministic check -> judge -> revise/escalate

Slow loop across episodes:
result -> human review -> feedback decomposition -> strategy revision -> rerun
```

For evolution task `t` and episode `r`:

```text
T_t -> y_(t,r) -> h_(t,r) -> delta_(t,r) -> pi_(r+1) -> y_(t,r+1)
```

`y` is the episode result, `h` is the expert review, `delta` is the typed
feedback, and `pi` is the next strategy. The foundation-model weights are not
modified. The defensible claim is policy-level self-evolution through typed
feedback, evidence-guided reruns, and experience reuse.

The initial strategy selector exposes four fixed, interpretable arms:

- `STATIC`: current fixed scientific-loop behavior;
- `EVIDENCE_FIRST`: close critical evidence gaps before expanding candidates;
- `DIVERSITY_FIRST`: generate distinct candidates before escalating fidelity;
- `UNCERTAINTY_FIRST`: prioritize evidence most likely to change the decision.

The implementation must not claim Bayesian improvement until posterior updates
are enabled and backed by stored observations. Before then, selection is an
explicit configured baseline. A later Bayesian Linear Thompson Sampling
implementation consumes low-dimensional task context, strategy identity,
machine evaluation, expert scores, issue closure, recurrence, and normalized
cost. It must treat repeated episodes of one task as correlated observations,
not independent tasks.

## Public CLI Contract

The asynchronous workflow is a new Typer command group. The existing
`photomatagent loop` remains a single-run scientific loop and keeps its current
behavior.

```text
photomatagent evolve start [--goal ...] (--target-json ... | --target-file ...) [runtime options]
photomatagent evolve list
photomatagent evolve status <evolution-id>
photomatagent evolve feedback <evolution-id> [--version v001] [--file review.json]
photomatagent evolve iterate <evolution-id> [runtime options]
photomatagent evolve history <evolution-id>
photomatagent evolve compare <evolution-id> <left-version> <right-version>
photomatagent evolve accept <evolution-id> [--version ...]
photomatagent evolve stop <evolution-id>
photomatagent evolve reopen <evolution-id>
photomatagent evolve export <evolution-id> [--output ...]
photomatagent evolve evaluate <evolution-id> --fresh [runtime options]
```

`start` creates the durable evolution task before running episode `v001`. If
the episode fails, the task and failed episode remain inspectable. On a
successful episode execution, the state becomes `AWAITING_EXPERT_FEEDBACK` and
the CLI prints the evolution ID, episode version, runtime session ID, result
path, and exact next command.

`--target-file` contains the same validated `TargetSpec` JSON accepted by
`--target-json`; it is not an unchecked natural-language prompt compiler. The
goal may remain natural language, but deterministic scientific convergence
still requires an explicit machine-verifiable target as in the existing loop.

`feedback` records and compiles a review but never starts a model, invokes a
tool, or submits HPC work. `iterate` is the explicit execution boundary. It
loads the latest reviewed episode, constructs the next strategy and revision
context, creates a fresh runtime session, and runs the next episode through the
normal scientific loop and permission path.

`evaluate --fresh` is a controlled evaluation path. It uses the frozen original
task and a chosen general strategy version, but excludes task-specific prior
feedback, prior answers, and task-specific carried evidence. It exists to
measure transfer rather than memorization.

Interactive chat adds only slash-command shortcuts:

```text
/evolve start ...
/evolve list
/evolve status <id>
/evolve feedback <id> [--version ...]
/evolve iterate <id>
/evolve history <id>
```

The central `ChatCommandRouter` intercepts these commands before normal user
text can reach `AgentRuntime.run()`. The shortcuts invoke the same Typer/service
surface as standalone commands; they do not implement another workflow.

## Expert Feedback Entry

Expert feedback may be entered interactively or imported from a JSON file. The
interactive prompt has a visibly distinct prefix:

```text
[EXPERT FEEDBACK | evo_... | v001] scientific_correctness>
[EXPERT FEEDBACK | evo_... | v001] evidence_sufficiency>
[EXPERT FEEDBACK | evo_... | v001] novelty>
[EXPERT FEEDBACK | evo_... | v001] actionability>
[EXPERT FEEDBACK | evo_... | v001] overall>
[EXPERT FEEDBACK | evo_... | v001] comments>
```

Each score is an integer from 1 through 5. The CLI displays the approved rubric
for the current dimension on request. Multiline comments terminate only with
`/submit`; `/cancel` and Ctrl-C write nothing. Before persistence, the CLI shows
the evolution task, targeted episode, result artifact hash, scores, fatal-issue
flag, and comment summary, then asks for confirmation.

Normal chat text, including text beginning with "the expert says", can never
create an expert record or update evolution policy. It remains ordinary user
input. The CLI may suggest `/evolve feedback`, but automatic role inference is
forbidden.

The approved four-dimensional rubric is:

- scientific correctness;
- evidence sufficiency;
- innovation;
- actionability.

An independent overall grade represents readiness for the next research stage.
Free-text fields capture fatal issues, up to three priority corrections,
content worth preserving, and recommended next actions. The exact 1--5 anchors
and hard caps are versioned with the feedback record so later rubric changes do
not reinterpret old reviews.

Rubric version `expert-review-v1` uses these anchors:

| Dimension | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Scientific correctness | Fundamental errors; result unusable | Critical errors likely change conclusions | Core direction plausible; important corrections required | Main conclusions reliable; minor corrections | Robust scientific logic, assumptions, and uncertainty treatment |
| Evidence sufficiency | No traceable evidence or suspected fabrication | Core claims rely on abstracts, secondary sources, or unsupported prediction | Relevant support exists but important gaps remain | Core claims have traceable full-text/primary support and explicit limits | Multi-source, auditable evidence chain with conflicts and gaps handled |
| Innovation | Mature result presented as novel without definition or baseline | Superficial variation without mechanism or comparison | Plausible composition/process novelty hypothesis, incompletely verified | Clear novelty type, baseline, mechanism, and supporting comparison | Systematically checked novelty with quantitative advantage and validation route |
| Actionability | No usable procedure or next step | Direction and a few parameters only | Main steps exist but important inputs, equipment, controls, or checks are missing | Reproducible route with inputs, equipment, ranges, outputs, and quality gates | Execution-ready route with contingencies, failure criteria, safety, and characterization |
| Overall readiness | Reject and rebuild | Major revision | Conditionally usable after named corrections | Ready for the next stage with minor revision | Expert-accepted for the stated task stage |

Hard caps are deterministic metadata, not hidden judgment: fabricated sources
cap evidence and overall at 1; a conclusion-changing scientific error caps
correctness and overall at 2; abstract-only support for core claims caps
evidence at 2; unsupported novelty caps innovation at 2; and a process consisting
only of a route name plus a few parameters caps actionability at 2. The expert
may override an automatically suggested cap only by recording a reason; both
the suggested and confirmed values remain in provenance.

## Persistent Domain Models

### EvolutionTask

Required fields:

- schema version and `evolution_id`;
- immutable original goal and `TargetSpec` snapshot;
- immutable task-input hashes and creation timestamp;
- current lifecycle status;
- current episode version;
- ordered episode, feedback, revision, and strategy references;
- accepted version, if any;
- optimistic revision number for atomic updates;
- creation and last-update timestamps.

### EpisodeRecord

Required fields:

- evolution ID, episode ID, and monotonic version;
- parent episode and applied feedback/revision references;
- runtime session ID and event-log path;
- strategy version;
- execution mode: `NORMAL`, `CARRY_VERIFIED_EVIDENCE`, or `FRESH_EVALUATION`;
- frozen task, target, provider, model, tool-surface, capability, and relevant
  data-source fingerprints;
- start/end timestamps and terminal execution status;
- `ScientificLoopSummary` snapshot;
- result artifact paths and content hashes;
- normalized token, tool-call, wall-time, and optional HPC cost metrics;
- automatic acceptance-test results.

Every completed episode has one program-selected primary result artifact.
Artifact identity is never inferred later by scanning the workspace. When the
Maker writes an explicitly registered deliverable, the episode runner copies or
references that workspace-contained file in the versioned output directory. If
no deliverable was registered, the runner materializes the final assistant
response as `result.md`. The episode stores the chosen path, media type, byte
length, and SHA-256 hash before it can enter `AWAITING_EXPERT_FEEDBACK`.

### ExpertFeedbackRecord

Required fields:

- immutable feedback ID;
- evolution ID, episode version, and result artifact hash;
- rubric version;
- five 1--5 scores;
- fatal-issue flag and free text;
- priority corrections, preserved strengths, and requested next actions;
- original raw input;
- compiler output and compiler provenance;
- user confirmation timestamp;
- supersession reference when a factual entry correction is necessary.

An existing feedback record is never edited. A corrected record supersedes it.
Only one active review may target an episode in the first implementation.

### FeedbackDelta

Each compiled item contains:

- category: task definition, scientific correctness, evidence sufficiency,
  innovation, deliverable completeness, actionability, safety, or other;
- status: correction, query, preference, or positive signal;
- severity;
- responsible module;
- problem statement;
- requested contract delta or required action;
- machine-checkable acceptance test where possible;
- content to preserve;
- parse confidence and source span.

A question remains a `QUERY`; the compiler must not convert uncertainty into a
false factual correction.

### RevisionPlan

Required fields:

- source episode and active feedback IDs;
- ordered contract deltas;
- evidence acquisition/upgrade requirements;
- output-schema requirements;
- preserved facts and evidence IDs;
- prohibited repeats and invalidated claims;
- machine-checkable acceptance tests;
- human-only acceptance items;
- proposed strategy change with rationale;
- compilation warnings and unresolved ambiguities;
- user-confirmed plan status.

## Storage Layout

Runtime control state is stored under the existing managed state area:

```text
.photomatagent/evolutions/<evolution-id>/
├── evolution.json
├── task.json
├── episodes/
│   ├── v001.json
│   └── v002.json
├── feedback/
│   ├── fb_v001_001.json
│   └── fb_v002_001.json
├── revisions/
│   └── rp_v001_to_v002.json
├── strategies/
│   ├── p001.json
│   └── p002.json
├── experience/
│   └── exp_v001_to_v002.json
└── events.jsonl
```

User-facing results are stored separately:

```text
user_output/<evolution-id>/v001/
user_output/<evolution-id>/v002/
```

All paths are constructed by the store from program-generated identifiers and
resolved through `Workspace.resolve`. CLI arguments never select arbitrary
manifest paths. JSON writes are atomic, schema-versioned, revision-checked, and
secret-redacted. Artifact hashes bind reviews to the exact output that the
expert saw.

The evolution store does not replace or merge `ConversationState`,
`ScientificState`, `ScientificLoopState`, or existing session snapshots. An
episode references its runtime session; the runtime session does not own the
evolution task.

## Lifecycle and State Transitions

```text
CREATED
  -> RUNNING
  -> AWAITING_EXPERT_FEEDBACK
  -> FEEDBACK_RECORDED
  -> REVISION_READY
  -> RUNNING
  -> AWAITING_EXPERT_FEEDBACK

Terminal or paused states:
ACCEPTED | STOPPED | BUDGET_EXHAUSTED | BLOCKED
```

Rules:

- feedback requires a completed episode and exact artifact hash;
- a result without an active review cannot be iterated;
- recording feedback does not execute the revision;
- compilation ambiguity prevents `REVISION_READY` until resolved;
- `iterate` creates the next monotonic version before starting execution;
- a failed episode remains recorded and cannot overwrite the previous result;
- concurrent writes use optimistic revision checks and an evolution-task lock;
- `accept` records the selected result but cannot rewrite deterministic
  scientific verdicts;
- `reopen` is explicit and retains the complete prior history;
- model-requested tool execution and HPC work still require all existing
  runtime and application-level approvals.

## Feedback Compilation and Confirmation

The default compiler is an isolated structured LLM call with `tools=[]`, like
the advisory Judge but with a distinct schema and prompt. Provider failure,
non-JSON output, or schema mismatch returns `COMPILATION_UNAVAILABLE`; the raw
review remains safely recorded and the task does not become revision-ready.

The compiler cannot modify state directly. Its output is validated, displayed,
and confirmed by the user before a `RevisionPlan` is activated. A deterministic
fallback allows the user to import an already structured feedback JSON file
without an LLM call.

The revision planner converts confirmed feedback deltas into a bounded dynamic
instruction. It does not rewrite the static system prompt. It distinguishes:

- target/constraint changes;
- evidence gaps and required fidelity;
- report-schema requirements;
- strategy preferences;
- content that must be preserved;
- invalidated claims that must not be repeated.

Raw expert text remains in provenance but is not appended directly to the Maker
conversation. Only the validated `RevisionInstruction` and bounded references
to prior verified state enter the new episode.

## Evidence Carry-Forward

Normal iteration starts a fresh `AgentRuntime` and fresh conversation. It may
carry forward only structured evidence that:

- has explicit provenance and a stable evidence ID;
- was not invalidated by expert feedback;
- is bound to the same candidate/task subject;
- satisfies the current target and data-source fingerprint policy;
- is not merely a model-generated assertion.

Unvalidated candidate predictions, discarded claims, complete prior answers,
raw conversation history, and raw expert prose are not carried forward.
Carried evidence is copied into the new `ScientificState` with provenance that
records its source episode. The new episode still re-evaluates every active hard
constraint.

`FRESH_EVALUATION` carries none of the task-specific evidence or feedback. It
may load a general strategy/experience snapshot frozen before the evaluation.

## Comparison and Learning Signals

For each adjacent episode pair, the comparator reports:

- change in the four expert dimensions and overall grade when both exist;
- closure rate for prior critique items;
- recurrence rate of previously observed issue categories;
- newly introduced issues;
- deterministic constraint/evidence/fidelity changes;
- result-artifact differences;
- token, tool, time, and HPC cost differences;
- unresolved human-only acceptance items.

An issue cannot be marked closed merely because the new report no longer
mentions it. Closure requires its acceptance test to pass or a later expert to
confirm it. Items without machine-checkable tests remain `NEEDS_HUMAN_REVIEW`.

Expert dimensions are normalized with `(score - 1) / 4`. The default expert
utility weights are correctness 0.35, evidence 0.30, innovation 0.15, and
actionability 0.20. Overall grade remains an independent readiness signal. Hard
caps prevent high prose quality from compensating for fabricated evidence,
critical scientific errors, unsupported novelty, or a non-executable process.

The stored learning signal may include utility change, issue closure,
recurrence, new-issue penalty, and normalized cost. Repeated episodes of one
task share a task-group ID. Any later Bayesian estimator must model or group by
that ID and must report cross-task validation separately.

## Experience Lifecycle

Experiences progress through explicit states:

```text
OBSERVATION -> HYPOTHESIS -> VALIDATED_EXPERIENCE -> REUSABLE_SKILL
```

- One expert review creates an `OBSERVATION` only.
- A resolved issue with an acceptance test may become a `HYPOTHESIS`.
- Repeated improvement across distinct tasks may become a
  `VALIDATED_EXPERIENCE`.
- Promotion to `REUSABLE_SKILL` requires explicit evidence thresholds and user
  approval; the initial implementation does not auto-edit repository skills.

Retrieval may use observations as low-confidence context, but it must label
their maturity. Task-specific feedback cannot leak into fresh evaluation tasks.

## Events and Observability

Add typed evolution events to the existing JSONL observability path:

- `evolution_task_created`;
- `evolution_episode_started`;
- `evolution_episode_completed`;
- `expert_feedback_recorded`;
- `expert_feedback_compiled`;
- `revision_plan_confirmed`;
- `evolution_iteration_started`;
- `evolution_comparison_completed`;
- `experience_state_changed`;
- `evolution_task_accepted`;
- `evolution_task_stopped`.

Events include IDs, versions, statuses, hashes, and bounded summaries, not full
expert prose or secrets. Existing inner runtime/scientific-loop events retain
their original run/session identities and gain correlation through episode
metadata rather than mutation of their semantics.

## Error Handling

- Invalid scores or malformed files fail before any state write.
- Ctrl-C during feedback entry leaves no partial record.
- Compiler failure preserves raw feedback and permits retry without duplicate
  active records.
- Revision ambiguity blocks execution and identifies fields requiring user
  correction.
- Provider or tool failure records a failed episode without changing the last
  good episode.
- Artifact-hash mismatch blocks review attachment and iteration.
- Missing prior artifacts produce a typed diagnostic; they are never silently
  substituted.
- Lock contention and revision conflicts fail safely and can be retried.
- Optional Bayesian dependencies fail soft; deterministic configured strategy
  selection remains available.

## Testing Strategy

### Domain and persistence tests

- Pydantic validation for every new model and status transition;
- atomic store create/load/update and optimistic revision conflicts;
- immutable feedback and supersession behavior;
- path containment and generated-ID validation;
- artifact hashing and mismatch rejection;
- schema migration and unknown-version diagnostics.

### Feedback tests

- all rubric bounds and hard caps;
- multiline submit/cancel behavior;
- comments containing slash-like text;
- compiler schema validation, query preservation, source spans, and confidence;
- provider failure and deterministic structured-file fallback;
- raw feedback never entering `ConversationState` or `ScientificState`.

### Orchestration tests

- start -> completed v001 -> awaiting feedback;
- feedback -> recorded without runtime/tool calls;
- iterate rejected before feedback or confirmation;
- iterate creates a fresh runtime session and monotonic v002;
- only eligible verified evidence is carried forward;
- scientific hard failures cannot be overridden by expert scores;
- failed v002 leaves v001 and its artifacts intact;
- accept, stop, and reopen transitions;
- resume after process termination.

### CLI tests

- standalone `evolve` command help and every subcommand;
- `/evolve` routing is intercepted before `AgentRuntime.run()`;
- ordinary text mentioning expert feedback remains ordinary chat;
- visible expert-mode prompts and confirmation summary;
- exact next-command hints after each state transition;
- JSON import/export round trips.

### Experiment and evaluation tests

- deterministic fake-provider end-to-end two-episode evolution fixture;
- issue-closure and recurrence calculations;
- normal carry-forward versus fresh-evaluation isolation;
- task-group-aware metrics that do not count repeated episodes as independent
  tasks;
- event correlation across evolution, episode, runtime session, and run IDs.

After narrow tests, run the existing scientific-loop, session, command-router,
permission, tool-surface, event, and experiment suites. Repository-wide
completion still requires the full test suite, `mypy src`, `git diff --check`,
diff inspection, and final status inspection.

## Delivery Phases

1. **Persistent workflow foundation**: models, store, state machine, artifact
   identity, events, and read-only CLI list/status/history.
2. **Expert input**: rubric, interactive/file entry, immutable feedback, CLI and
   slash-command isolation.
3. **Feedback compilation**: isolated compiler, confirmation, revision plan,
   failure modes, and provenance.
4. **Versioned rerun**: fresh episode creation, bounded revision context,
   verified-evidence carry-forward, result storage, and comparison.
5. **Experience layer**: issue closure/recurrence, experience lifecycle, export,
   and fresh evaluation mode.
6. **Adaptive selector**: fixed-arm baselines first, then Bayesian selection
   only after sufficient trajectories and task-group-aware offline validation.

Each phase must leave the base chat and existing single-run scientific loop
usable when evolution functionality or optional dependencies are unavailable.

## Explicit Non-Goals for the First Implementation

- fine-tuning the foundation model, RLHF, DPO, or reward-model training;
- PSO or genetic mutation over arbitrary prompts;
- automatic editing or creation of repository skills;
- treating repeated runs of one task as independent statistical samples;
- allowing an expert score to override scientific evidence or safety gates;
- automatically submitting HPC work when feedback is recorded;
- inferring expert role from ordinary chat text;
- a graphical frontend or remote multi-user review service;
- automatic publication-level novelty claims.

## Acceptance Criteria

The design is implemented when a user can:

1. create an evolution task and receive a durable ID before episode execution;
2. terminate the CLI, later attach a rubric-scored review to the exact v001
   artifact, and verify that no normal agent turn or tool call occurred;
3. confirm a structured revision plan and explicitly run v002 in a fresh
   runtime session;
4. inspect complete provenance linking task, versions, feedback, strategies,
   scientific summaries, runtime sessions, artifacts, and events;
5. see machine-verifiable issue closure, unresolved human checks, and cost
   changes between versions;
6. prove through tests that ordinary chat cannot mutate evolution state and
   human feedback cannot bypass deterministic scientific checks, permissions,
   or HPC submission gates;
7. run a fresh evaluation that excludes task-specific feedback and carried
   evidence while using an explicitly frozen general strategy snapshot.
