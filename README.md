# PhotomatAgent

PhotomatAgent is a Python scientific-agent runtime for materials research, with a focus on infrared optoelectronic materials and detectors. It provides an explicit model/tool loop, structured scientific state, local workspace tools, optional scientific capability packs, and integrations for MCP and SCNet-based computation.

It is designed to run as a CLI application. The runtime owns context construction, model streaming, tool authorization, tool execution, observation redaction, and event logging.

## Features

- Explicit asynchronous agent loop with streaming model responses and tool calls.
- Provider-neutral model protocol with OpenAI Responses API, Anthropic Messages API, and offline fake-provider adapters.
- Tool registry with direct, deferred, and hidden exposure modes.
- Progressive discovery for deferred tools through `tool_search`, `tool_describe`, and `tool_call`.
- Workspace-scoped file and shell tools with approval policies and sensitive-path blocking.
- Durable conversation history plus a bounded working context with tool-result pruning and optional structured compaction.
- Structured `ScientificState` for claims, evidence, calculations, tasks, questions, and contradictions.
- **Evidence-Guided Scientific Feedback Loop**: a machine-verifiable outer loop
  (`TargetSpec` → candidates → deterministic evaluation → structured feedback →
  convergence policy → stagnation detection) that decides scientific success
  from evidence and constraints — never from the model's own "final answer".
- **专家反馈驱动的跨版本演化**：将异步专家评价绑定到准确的 Episode
  结果哈希，经无工具 Compiler 编译和人工确认后，显式创建全新 runtime session
  运行下一版本；普通聊天不会被当作专家反馈。
- Capability packs for materials databases, literature retrieval, crystal structures, electronic structure, infrared constraints, quantum dots, detector metrics, and more.
- Optional VASP, Hefei-NAMD, MAGUS, Slurm/SCNet, and MCP integrations.
- JSONL execution traces, replay, session analysis, and deterministic experiment runs.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended)

The core package uses Pydantic, Typer, Rich, prompt-toolkit, python-dotenv, OpenAI, Anthropic, pymatgen, and MCP. Scientific integrations are installed as optional extras.

## Installation

```bash
git clone <repository-url>
cd PhomatAgent
uv sync
```

Install only the extras you need:

```bash
# Materials Project
uv sync --extra materials

# Electronic-structure analysis
uv sync --extra electronic

# Local literature indexing and retrieval
uv sync --extra literature

# All optional scientific capability groups
uv sync --extra science
```

## Configuration

Create a workspace-local `.env` file from the supplied template:

```bash
cp .env.example .env
```

For OpenAI, configure at least:

```dotenv
PHOTOMATAGENT_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

For Anthropic:

```dotenv
PHOTOMATAGENT_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
```

Never commit `.env`. It can contain model credentials, Materials Project credentials, and SCNet/HPC settings.

Run the local diagnostic after configuration:

```bash
uv run photomatagent doctor
```

`doctor` checks the Python/runtime configuration, configured provider, available skills directory, writable session directory, and a local fake-provider loop smoke test.

## Quick start

Start an interactive chat:

```bash
uv run photomatagent chat
```

Run a single goal:

```bash
uv run photomatagent chat \
  --goal "Find useful capabilities for infrared detector material screening"
```

Run without a paid model, using the built-in fake provider:

```bash
uv run photomatagent chat \
  --provider fake \
  --approval auto \
  --goal "Inspect this workspace"
```

By default, writes, edits, shell commands, and newly registered scientific tools require approval. `--approval auto` bypasses confirmation and should only be used in a trusted workspace.

## Common commands

```bash
# Inspect registered tools and their exposure mode
uv run photomatagent tools list
uv run photomatagent tools surface
uv run photomatagent tools search "band structure"

# Inspect skills and capability availability
uv run photomatagent skills list
uv run photomatagent scientific status

# Inspect configured MCP servers
uv run photomatagent mcp list
uv run photomatagent mcp status

# Inspect recorded sessions
uv run photomatagent sessions list
uv run photomatagent sessions stats latest
uv run photomatagent sessions replay latest

# Resume a historical session and keep asking on top of it
uv run photomatagent sessions resume latest --goal "Continue the previous task"
uv run photomatagent chat --resume <session-id>

# Run deterministic experiment definitions
uv run photomatagent experiments run experiments/offline-smoke.json
```

During interactive chat, `/help` lists slash commands. The router includes tool, skill, scientific-capability, MCP, session, experiment, configuration, and approval commands.

## How it works

```text
CLI / experiment runner
  → AgentRuntime
  → ContextEngine + ToolSurfacePlanner
  → ModelProvider
  → tool calls
  → safety and permission checks
  → ToolRegistry + Tool.execute()
  → observation redaction and truncation
  → conversation, scientific state, and JSONL events
  → next iteration or completion
```

The main control loop is implemented in `src/photomatagent/runtime/loop.py`. It does not delegate tool execution to a model SDK.

一次 `loop` 只负责一个 Episode 内的“候选—证据—确定性检查”科学闭环；持久化的
`evolve` 位于它的外层，负责跨 Episode 的专家反馈、修订计划、策略版本和结果比较：

```text
EvolutionService（跨 Episode，持久化编排）
  → v001 ScientificLoopController
  → 专家反馈（绑定结果 SHA-256；不进入普通聊天）
  → 无工具 FeedbackCompiler → 人工确认 RevisionPlan
  → 可选且显式的 fresh evaluation
      （仅在 REVISION_READY；独立 evaluation 记录；主线仍为 v001）
  → iterate → v002 全新 AgentRuntime / runtime session
  → v001/v002 确定性比较
```

`EvolutionService` 不直接执行工具、MCP、SSH、Slurm 或 HPC。所有模型请求的工具调用仍
只通过 `AgentRuntime` 的 registry、权限和审批路径执行；专家高分也不能覆盖确定性
Checker 的 FAIL/UNKNOWN。

### Tool exposure

Every tool is registered with one of three exposure modes:

- `DIRECT`: its full schema is supplied to the model each turn.
- `DEFERRED`: it is discoverable through a compact catalog; the model must use `tool_search` → `tool_describe` → `tool_call`.
- `HIDDEN`: it is not model-visible and cannot be executed.

Deferred execution still follows the same sensitive-path checks, permission policy, argument validation, event logging, and observation policy as direct tools.

### Context and state

`ConversationState` retains provider-neutral conversation messages. `ContextEngine` builds a separate model-visible working copy, replaces old large tool outputs with provenance placeholders when needed, and can request a schema-validated compaction summary only when the conversation is safe to compact.

`ScientificState` is maintained independently and can receive claims, evidence, calculation records, and tasks through tool result `state_updates`.

The prompt layout is cache-friendly: the system message stays static for the whole
session (base instructions, skill index, capability manifest). The live scientific
state and the derived investigation ledger are appended as a single trailing
user message each loop iteration, so provider prompt-cache prefixes (system +
conversation history) stay byte-identical and only the final snapshot is
re-processed when state updates.

File layout is enforced in the system prompt: user-facing deliverables go under
a new folder in `user_output/` (e.g. `user_output/<task-name>/`), while all
intermediate/scratch files that the user does not need go under `tmp/`. Both
directories are created automatically in every workspace.

## Scientific capabilities

Capability packs are registered from `src/photomatagent/scientific/capabilities/registry.py`. Optional dependencies are checked at runtime; unavailable packs should report their status rather than prevent the base runtime from starting.

| Capability | Main use |
| --- | --- |
| `materials` | Materials Project search, summaries, and structures. |
| `literature` | arXiv/local-paper search, indexing, passage retrieval, and evidence extraction. |
| `structure` | Structure summary, symmetry, density, neighbours, and format conversion. |
| `electronic` | Band/DOS summaries, plotting, and effective-mass workflows. |
| `ir` | Infrared target and physical-constraint compilation. |
| `quantum_dot` / `alloy` | Brus-model analysis, size/composition scans, and band-gap bowing. |
| `photodetector` | EQE/responsivity conversion and target checks. |
| `defects`, `transport`, `device`, `optics`, `interface`, `kp` | Optional domain-specific calculations and diagnostics. |
| `generation` | Formula retrieval/validation and optional generative workflows. |
| `vasp` | Unified deferred `vasp.*` family: capabilities, plan, prepare, preflight, submit, status, resume, collect, report. |
| `namd`, `magus` | Prepare, submit, monitor, collect, and inspect high-cost workflows. |

Check the capabilities actually available in the current environment:

```bash
uv run photomatagent scientific status
```

## MCP and remote compute

MCP server declarations are loaded from `.photomatagent/mcp.json` or `.photomatagent/mcp.yaml`. Supported transports are stdio and streamable HTTP. MCP tools are registered through the same tool registry and can use deferred exposure.

The VASP, Hefei-NAMD, and MAGUS integrations use a narrow application layer over the SCNet backend. Remote submission is gated by configuration:

```dotenv
PHOTOMATAGENT_ALLOW_HPC_SUBMIT=0
```

Keep this disabled until the target SCNet environment, scheduler configuration, resource policy, and external executables have been verified.

## Sessions and experiments

When event logging is enabled, runtime events are written under:

```text
.photomatagent/sessions/<session-id>/events.jsonl
```

The session tools can list, summarize, inspect working-context metrics, and replay those traces without rerunning a model or tool.

Sessions are also **resumable**: after each run the live state (conversation,
scientific state, and the compaction cursor) is snapshotted next to the trace
as `session_state.json`. `photomatagent chat --resume <id>` (or
`photomatagent sessions resume <id>`) reloads that snapshot into a fresh
runtime and appends follow-up turns to the same session directory, so the
whole conversation stays in one replayable trace. `sessions list` shows which
sessions have a saved resumable state.

Inside a running interactive chat you can jump onto a past task directly with
`/resume <id|directory|latest>`: the historical conversation and scientific
state are restored into the live runtime, the logger switches to that session's
trace, and you keep asking follow-up questions without restarting.

Experiment JSON files define independent tasks and deterministic expectations. The runner creates a fresh runtime and trace for each task, then stores experiment results under `.photomatagent/experiments/`.

两轮专家反馈演化的离线 smoke 输入和期望记录在
`experiments/expert-feedback-evolution-smoke.json`。它是多阶段场景描述，不复制
`EvolutionService`，权威执行入口是：

```bash
uv run pytest -q tests/test_evolution_end_to_end.py
```

该测试只使用固定 `FakeModelProvider` 轨迹和本地测试工具，不访问网络，也不会提交 HPC。

## Evidence-Guided Scientific Feedback Loop

The main `AgentRuntime` remains the **inner execution loop** (the Maker). A
separate **scientific outer loop** (`src/photomatagent/scientific/loop/`)
wraps it so that "the model stopped calling tools" no longer counts as
scientific success. Success is decided by a deterministic Checker, not by the
Maker.

```text
Scientific Goal → TargetSpec → Maker (AgentRuntime + scientific tools)
  → CandidateState → ScientificEvaluator (Checker)
  → EvaluationReport → ScientificLoopPolicy
       ├─ PASS + confidence  → SUCCESS
       ├─ violation / gap    → FeedbackSignal → next round instruction
       ├─ stagnation         → STALLED
       └─ budget exceeded    → BUDGET_EXHAUSTED
```

Design invariants:

- `TargetSpec` expresses hard/soft constraints checked **deterministically**
  (`evaluate_constraint`), never by the LLM: "is 0.21 eV ≤ 0.155 eV" is a
  program decision.
- The Checker is separate from the Maker: the model never grades its own
  candidate (Invariant C). `Generation.vae_formula` output stays
  `UNVALIDATED_GENERATED_STRUCTURE` until it passes the evaluator.
- Missing evidence is `UNKNOWN`, never `PASS` (Invariant B).
- Identical candidates share a deterministic fingerprint; repeats never count
  as progress (Invariant D).
- Feedback is structured (`FeedbackSignal`): what failed, why, missing
  evidence, next priorities, and what must not be repeated. It enters the
  next maker turn as an explicit research instruction appended to the
  conversation (the static system prompt and cache-friendly layout are
  untouched).
- An optional **isolated, structured, read-only LLM Scientific Judge**
  (`--judge-provider`) reviews each candidate **after** the deterministic
  evaluator. Its `JudgeReport` is embedded into the feedback signal and can
  only *hold back* SUCCESS (`--judge-min-quality`, `--require-judge`): it can
  never turn a deterministic FAIL/UNKNOWN into a PASS, never rescinds a hard
  constraint, and never calls tools (its request carries `tools=[]`).
- Expensive tools still run only through the runtime's permission /
  approval / HPC gating (Invariant E); the outer loop never calls a backend
  directly.
- The full trajectory is reconstructible from the JSONL event stream via new
  event kinds (`candidate_proposed`, `candidate_evaluated`, `candidate_judged`,
  `scientific_feedback_generated`, `scientific_loop_decision_made`, ...).

Run a verifiable loop offline (fake provider, built-in 8–14 µm LWIR demo
target):

```bash
uv run photomatagent loop --demo --max-rounds 4 --provider fake --approval auto
```

With the advisory LLM judge enabled (real provider for meaningful output):

```bash
uv run photomatagent loop --demo --judge-provider openai --judge-model gpt-4o \
  --judge-min-quality 0.6 --max-rounds 6
```

Or supply an explicit `TargetSpec` (mode A):

```bash
uv run photomatagent loop \
  --goal "Design an LWIR detector material..." \
  --target-json '{"goal":"...","constraints":[{"property":"band_gap","operator":"le","value":0.155,"unit":"eV","severity":"HARD"}]}' \
  --max-rounds 6
```

Loop experiments run through the normal experiment runner and support the
fake provider for CI:

```bash
uv run photomatagent experiments run experiments/scientific-feedback-loop-smoke.json
```

## 专家反馈驱动的跨 Episode 演化

下面是生产 CLI 的正确状态顺序。先准备可由 `TargetSpec` 验证的 `target.json`，并在
`.env` 中配置能够完成普通生成和严格结构化 JSON 编译的真实 provider；命令省略
`--provider` 时使用该配置。将尖括号占位符替换为前一步输出的真实 ID：

```bash
uv run photomatagent evolve start --target-file target.json --goal "生成中红外光窗材料候选与工艺"
uv run photomatagent evolve feedback <evolution-id> --version v001
uv run photomatagent evolve compile <evolution-id> --version v001
uv run photomatagent evolve evaluate <evolution-id> --fresh --strategy-id <strategy-id>
uv run photomatagent evolve iterate <evolution-id>
uv run photomatagent evolve compare <evolution-id> v001 v002
```

`evolve start` 先持久化任务再运行 `v001`。CLI 退出后仍可用同一个 evolution ID
录入反馈；反馈会绑定到专家实际查看的结果 SHA-256。`feedback` 只走专用专家入口，
不会调用 `runtime.run(raw_feedback)`，也不会自动运行工具或提交 HPC。随后必须显式
执行 `compile`；该命令使用 `tools=[]` 的独立 provider 请求，展示 RevisionPlan 和
Strategy ID，并要求交互确认。只有确认后任务才进入 `REVISION_READY`。编译失败时原始
反馈仍保留，可重跑同一条 `compile` 命令。

`evaluate --fresh` 要求 `REVISION_READY`、`--fresh` 和已确认生成的
`--strategy-id`，因此必须在 `iterate` 之前运行；它保持主任务的修订就绪状态，并使用
空白科学状态排除该任务的历史反馈、答案和继承证据。之后 `iterate` 才创建 v002 与
全新 runtime session。正常迭代只继承满足 provenance 规则的已验证结构化证据；旧
对话、旧答案、专家自由文本和未经验证的模型预测不会进入下一轮。

内置 fake provider 能支持局部 runtime/loop 演示，但不会自行生成符合
`FeedbackCompilation` schema 的编译响应，因此上述生产 CLI 全流程不能标为 fake
离线流程。权威、完全离线且可重复的两轮 smoke 只有：

```bash
uv run pytest -q tests/test_evolution_end_to_end.py
```

策略学习有明确安全边界：少于 20 条 observation 或少于 8 个不同
`task_group_id` 时使用可解释的 fixed selector；门槛满足前不会宣称 Bayesian 学习已经
启用。即使数量达到 20/8，只要任一权威
`reviewed comparison → Experience → StrategyObservation` 链不完整，selector/status
仍显示 `fixed baseline`；按 status 提示重跑相应的 `evolve compare`，补齐链路后才会
重新评估启用条件。同一任务的多个 Episode 是相关样本，不能冒充独立任务。经验默认
从 `OBSERVATION` 开始，系统不会自动把它晋升为 Skill；可复用 Skill 还需要跨任务证据
和显式用户批准。

## Development

Run the test suite:

```bash
uv run pytest
```

Run selected tests while working on a subsystem:

```bash
uv run pytest tests/test_loop.py
uv run pytest tests/test_tool_surface.py
uv run pytest tests/test_context_lifecycle.py
```

## Project layout

```text
src/photomatagent/
├── cli/            # CLI commands, chat UI, rendering, approvals
├── runtime/        # agent loop, context, safety, permissions, events
├── models/         # OpenAI, Anthropic, and fake-provider adapters
├── tools/          # tool protocol, registry, workspace tools, discovery
├── scientific/     # scientific state, capability packs, applications, HPC
├── mcp/            # external MCP gateway
├── mcp_servers/    # bundled SCNet MCP server
├── skills/         # skill discovery and loading
├── logging/        # JSONL event persistence
├── observability/  # trace analysis and replay
└── experiments/    # experiment runner and evaluation

skills/              # domain SOP content
tests/               # pytest suite
experiments/         # runnable experiment specifications and vertical slices
docs/                # supplementary design and domain documentation
user_output/         # agent deliverables handed back to the user (per-task subfolders)
tmp/                 # intermediate/scratch files the user does not need
```

## Safety notes

- `bash` runs commands in the local environment after approval; it is not an OS sandbox.
- Sensitive file paths are blocked by policy, but this is a guardrail rather than a complete filesystem isolation boundary.
- Scientific capabilities may be unavailable, unconfigured, approximate, or dependent on external data and solvers. Inspect their status and returned evidence before relying on results.
- Scheduler completion does not, by itself, establish scientific validity; collect and validate result artifacts.

## License

No repository-level license file is currently included. Confirm the intended license before redistributing the project or its bundled assets.
