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

`build_feedback(target, candidate, evaluation, history[, judge])` returns a
structured `FeedbackSignal` or `None` (a PASSing candidate with no judge
concerns needs no feedback). It states what failed, why, missing evidence,
next priorities, and what must not be repeated. The controller appends the
rendered signal to the next maker instruction (never into the static system
prompt, preserving the cache-friendly trailing-snapshot layout).

## Advisory LLM Scientific Judge

After the deterministic evaluator, an optional, isolated, structured, read-
only LLM judge (`scientific/loop/judge.py`) reviews the candidate:

- **Isolated**: it uses its own `ModelProvider` and its `ModelRequest` has
  `tools=[]` — it cannot call tools, permissions, or backends by
  construction.
- **Structured**: the prompt demands a schema-validated `JudgeReport` JSON
  object (`scientific_quality`, `issues`, `recommendations`, `rationale`).
- **Read-only**: `assess()` takes an immutable JSON snapshot of target,
  candidate, evaluation and bounded evidence; it never mutates
  `ScientificState`, `ScientificLoopState` or the conversation.
- **Non-authoritative (hard invariant)**: the judge is explicitly told it
  does not decide whether constraints pass or fail. It can never convert a
  deterministic FAIL/UNKNOWN into a PASS, never rescind a hard-constraint
  violation, and cannot manufacture SUCCESS without evidence. Its concerns
  are embedded into `FeedbackSignal` (validation actions, priority lines) and
  can only *hold back* SUCCESS (`judge_min_quality`, `require_judge`).
- **Graceful**: provider failure, non-JSON output or schema mismatch degrade
  to `JudgeReport(status=UNAVAILABLE)`; the deterministic loop keeps working
  and SUCCESS is not blocked unless `require_judge=True`.

```text
Maker -> candidate -> ScientificEvaluator (deterministic, authoritative)
        -> ScientificJudge (advisory, read-only)
        -> EvaluationReport + JudgeReport -> FeedbackSignal -> Policy
```

New event kind: `candidate_judged` (round, candidate_id, status, quality,
issues, summary) joins the JSONL trajectory.

## Stagnation and termination

`StagnationDetector` (default `patience=3`, `epsilon=1e-3`):
`HgTe, HgTe, HgTe, HgTe` is one iteration, not four. Identical candidate
fingerprints, identical violation/evidence-gap signatures and below-epsilon
score improvements accumulate toward `STALLED`.

`ScientificLoopPolicy.decide()` terminates deterministically:

| Decision | Condition |
| --- | --- |
| SUCCESS | all HARD constraints pass, no critical evidence gap, confidence ≥ threshold, and (judge absent, or judge available with quality ≥ threshold, or unavailable without `require_judge`) |
| CONTINUE | resolvable violation or evidence gap remains; or deterministic pass held back by judge concerns |
| ESCALATE | critical constraints rest on cheap evidence (higher-fidelity needed) |
| STALLED | stagnation detector tripped |
| INCONCLUSIVE | no candidate / no evidence possible (capability unavailable, tool failures) |
| BUDGET_EXHAUSTED | round / candidate caps exceeded |

## Events

New kinds appended to the existing `AnyRuntimeEvent` union and JSONL stream:
`scientific_loop_started`, `candidate_proposed`, `candidate_evaluated`,
`candidate_judged`, `scientific_feedback_generated`,
`scientific_loop_decision_made`, `scientific_loop_completed`,
`scientific_loop_stalled`. The JSONL trace can answer: what was proposed each
round, why it failed, what evidence was used, what the advisory judge said,
why the strategy changed, which candidate is best, and why the loop stopped.

## CLI

```bash
uv run photomatagent loop --demo --provider fake --approval auto --max-rounds 6
uv run photomatagent loop --target-json '<TargetSpec JSON>' --goal "..." ...
uv run photomatagent loop --demo --judge-provider openai --judge-model gpt-4o \
  --judge-min-quality 0.6 --require-judge
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
- The LLM judge is advisory only and never auto-submits HPC jobs; it adds
  concerns/validation work and can hold back SUCCESS
  (`require_judge`/`judge_min_quality`), but it cannot create SUCCESS and
  never overrides deterministic hard constraints.
- Escalation is a decision aid: `ESCALATE_FIDELITY` recommends higher-fidelity
  work but never auto-submits HPC jobs; permissions/approval/HPC gating stay
  authoritative.
- Soft-constraint optimisation is scored but not required for PASS.
- Natural-language → `TargetSpec` compilation is intentionally out of P0
  scope: the CLI requires an explicit target (`--demo` / `--target-json`).

## 专家反馈驱动的跨 Episode 演化层

P0 `ScientificLoopController` 是单个 Episode 内的科学闭环。外层 `evolve` 工作流
解决的是不同时间尺度的问题：CLI 可以退出，专家稍后评价某个已固化结果，再由用户
显式启动一个新 Episode。两层不会合并：

```text
Episode 内：Maker → 结构化证据 → 确定性 Checker → 修正/停止
Episode 间：v001 结果 → 专家反馈 → 编译 → 人工确认 → v002 → 比较
```

生产 CLI 必须按状态机顺序执行。`target.json` 必须是可由 `TargetSpec` 验证的 JSON，
并且 `.env` 需要配置能够完成普通生成和严格结构化 JSON 编译的真实 provider。省略
`--provider` 时使用该配置；将尖括号占位符替换为前一步输出的真实 ID：

```bash
uv run photomatagent evolve start --target-file target.json --goal "生成中红外光窗材料候选与工艺"
uv run photomatagent evolve feedback <evolution-id> --version v001
uv run photomatagent evolve compile <evolution-id> --version v001
uv run photomatagent evolve evaluate <evolution-id> --fresh --strategy-id <strategy-id>
uv run photomatagent evolve iterate <evolution-id>
uv run photomatagent evolve compare <evolution-id> v001 v002
```

专家反馈入口与普通聊天严格隔离。反馈先绑定 `v001` 主结果的 SHA-256，再保存为不可变
记录；它不会作为 `runtime.run(raw_feedback)` 的用户消息发送，也不会触发工具、网络或
HPC。反馈保存后必须显式运行 `compile`。无工具 `FeedbackCompiler` 的请求固定为
`tools=[]`，只把反馈编译为结构化 `FeedbackDelta`；命令会预览 RevisionPlan 和
Strategy ID，并要求用户确认。Compiler 失败时原始反馈仍保留，可重跑同一条命令。

确认后任务进入 `REVISION_READY`。`evaluate --fresh` 只接受该状态，并要求显式传入
`--strategy-id`，所以 fresh evaluation 必须先于 `iterate`；它完成后主任务仍保持
`REVISION_READY`。随后 `iterate` 才创建 v002 和新的 runtime session。进入 v002 的
只有有界结构化修订指令和符合 provenance、subject、fidelity 规则的已验证证据；旧
`ConversationState`、旧答案、专家自由文本与未验证预测均不继承。专家评分和可选
Judge 始终是咨询信号，不能把确定性 FAIL/UNKNOWN 改为 PASS，也不能绕过运行时权限
或 HPC 提交门禁。

`evaluate --fresh` 使用冻结策略快照和空白 `ScientificState`，排除当前任务的反馈、
历史答案和继承证据，用于更严格的跨任务评估。离线两轮 smoke 由
`tests/test_evolution_end_to_end.py` 权威执行；
`experiments/expert-feedback-evolution-smoke.json` 只声明输入与期望，不在 experiment
runner 中复制演化服务：

```bash
uv run pytest -q tests/test_evolution_end_to_end.py
```

这是权威、完全离线且可重复的两轮 smoke：测试夹具显式提供符合 schema 的固定 compiler
响应。内置 fake provider 虽可用于局部 runtime/loop 演示，却不会自行生成合法的
`FeedbackCompilation`，因此不能把上面的生产 CLI 链称为 fake 离线全流程。

策略选择在数据不足时安全回退到 deterministic fixed selector。只有累计至少 20 条
observation 且覆盖至少 8 个不同 `task_group_id`，Bayesian Linear Thompson Sampling
才可能启用。即使达到 20/8，只要权威
`reviewed comparison → Experience → StrategyObservation` 链存在缺口，selector 和
status 仍保持 `fixed baseline`；按 status 列出的 evolution/comparison ID 重跑相应
`evolve compare`，补齐链路后再判断门槛。同一任务的 v001/v002 是相关样本，统计时
不能当作两个独立任务。经验从 `OBSERVATION` 开始，满足跨任务证据门槛后才能逐级
晋升；系统不会自动创建或修改 Skill，`REUSABLE_SKILL` 还要求显式用户批准。
