# PhotomatAgent V0.5 — Context Lifecycle & Safety

PhotomatAgent 是一个面向材料科学、尤其是红外光电探测材料研究的本地 **Scientific Agent Runtime**。它不是 SDK 自带的 agent，也不是聊天机器人外壳；项目的核心是一个由我们自己控制、可直接阅读和修改的 Agent Loop。

V0.5 保留 V0.4 的 Progressive Tool/Skill Disclosure，并完成 Generic Agent Harness 的 bounded working context、安全 observation 和轻量 trajectory lifecycle。完整事实仍进入 durable session/JSONL；模型每轮只看到由 `ContextEngine` 构建的 working representation，裁剪或压缩不会回写、删改 durable conversation。

## Context lifecycle

```text
Durable Conversation / JSONL (append-only factual history)
                    │
                    ▼
              ContextEngine
         ┌──────────┼───────────┐
         ▼          ▼           ▼
   WorkingLedger  Stage A     Stage B
     (derived)   old result   structured semantic
                  pruning      compaction
         └──────────┼───────────┘
                    ▼
             Working Context
                    ▼
                  Model
```

`ContextEngineConfig` 集中定义 `context_limit_tokens=128000` 的安全 fallback、`prune_trigger_ratio=0.70`、`compact_trigger_ratio=0.82`、`target_ratio=0.60` 与 `protect_recent_turns=2`。调用方知道真实 model context window 时可显式覆盖；未知时不猜 provider billing/tokenizer 行为。

生命周期严格分两级：Stage A 先按体积从旧 observation 开始确定性回收，将 `ToolResultMessage.content` 的工作副本替换为包含 tool、参数摘要、成功/失败和 `session://.../tool-call/...` provenance 的占位符，不虚构语义摘要。只有 Stage A 后仍达到 compact threshold，才用同一普通 provider 发出一次无工具 compaction request，并要求返回 schema-validated `CompactionState`。

`CompactionState` 包含 `goal`、`standing_instructions`、`progress`、`key_findings`、`decisions`、`failed_approaches`、`relevant_resources`、`open_questions` 和 `next_actions`。成功后 working context 变为 initial/system context + structured summary + protected recent turns；失败时不提交 state、不删除 working history，并发出失败事件。

工具协议是 correctness invariant：assistant tool call 与 result 作为 logical atomic group，边界只落在 user-turn 起点；旧 result 被裁剪时仍留下同 call ID 的 placeholder result。`has_inflight_tool_transaction()` 检测尚无 result 的 call，存在时禁止 semantic compaction。因 iteration limit 放弃的 call/result 只在下次 working copy 中成对隐藏，durable history 不被原地重写。

## Progressive capability flow

```text
Registered Tool
      ↓
ToolExposure (DIRECT / DEFERRED / HIDDEN)
      ↓
ToolSurfacePlanner
      ├── direct canonical ToolDefinition[] ──→ Provider Adapter
      ├── deferred ToolCatalog ──→ compact CapabilityManifest
      └── hidden ──→ nowhere model-visible
                              ↓
                 tool_search / tool_describe / tool_call
                              ↓
                  underlying registered Tool
                              ↓
        existing permission → validation → execution → events
                              ↓
                    ObservationPolicy
                              ↓
                  next model-visible turn
```

核心不变量是：

```text
registered capability != model-visible capability
raw tool result != model-visible observation
```

Provider 不理解 exposure、catalog 或 BM25，只接收普通 canonical function specs。`tool_call` 也不是第二套 executor：runtime 先把它解成 underlying tool，再进入原有 PermissionPolicy、ToolRegistry validation、执行、RuntimeEvent 和 tool-call-ID 配对路径。

## Root-cause audit

V0.3 的注入点位于 `AgentRuntime.run()`：每轮直接构造 `ModelRequest(messages=..., tools=self._tools.definitions())`。默认 registry 有 10 个工具，canonical compact JSON 为 2,799 chars，即约 700 estimated tokens/call；OpenAI wire shape 约 3,109 chars，即约 778 estimated tokens/call。`mock.run_calculation`、`edit`、`grep` 是最大的三个 canonical schema。

历史 session `20260809T112843_fbd772` 有 13 model calls、29 tool calls、306,228 provider-reported input tokens。旧 schema 每轮重发的理论累计贡献约为 9,097 canonical estimated tokens，或 10,104 OpenAI-wire estimated tokens，后者约为真实累计 input 的 3.3%。这证明 eager schema 注入存在，但它不是该 session 的主因。

同一 session 的 tool-result text 合计 137,410 chars；在无状态完整对话逐轮重发下，后续请求累计再次暴露约 991,432 chars（约 247,858 estimated tokens）。这只是 `chars/4` 诊断估算，不是 provider billing，也不表示 cached token 计费方式。V0.4 因而同时实现 capability surface 和 observation budget，而不是用 schema 优化掩盖 history/result 膨胀。

当前默认 V0.4 registry 为 14 个工具：10 direct、4 deferred、0 hidden。progressive surface 约 703 schema tokens，manifest 约 101 tokens，并避免约 282 deferred-schema tokens/call。因为当前 deferred catalog 很小，bridge + manifest 开销会让总估算略高于旧 10-tool surface；收益会随 MCP/科学工具数量增长。这个版本首先建立正确边界和可观测性，不宣称当前小 registry 已获得 token 净节省。

## Loop Observatory

```text
Agent Run
   ↓
typed RuntimeEvent
   ↓
JSONL Agent Execution Trace
   ↓
Trace Analyzer ──────→ SessionSummary / anomaly flags
   ↓
Offline Replay
   ↓
Deterministic Experiment Compare
```

依赖方向是单向的：`runtime/` 不导入 observability、experiments 或 CLI。Analyzer 只观察 trace，不干预 StopPolicy；Replay 只读取 JSONL，不会重新调用 LLM 或执行工具。

仍必须区分三个概念：

- **Runtime Completion**：Agent loop 正常到达一个 runtime stop reason。
- **Task Evaluation**：回答和 loop 是否满足 experiment 中的 deterministic expectations。
- **Scientific Correctness**：V0.4 尚未评估；没有 LLM-as-Judge，也没有 scientific quality score。

因此：`runtime completed != task correct != scientifically correct`。

## 架构

```text
                         CLI
                          │
                          ▼
                     Event Stream ──────► JSONL Agent Execution Trace
                          │
                          ▼
                    Agent Runtime
                          │
             ┌────────────┼─────────────┐
             ▼            ▼             ▼
          Context      ModelProvider    ToolRegistry
                          │             │
                  ┌───────┼───────┐     ├─ read / glob / grep
                  ▼               ▼     ├─ write / edit / bash
          OpenAI Responses   Anthropic  └─ scientific mock tools
                  │            Messages
                  └──── canonical stream ────┘
```

`src/photomatagent/runtime/loop.py` 仍然包含完整控制流，不使用 LangChain、LangGraph、CrewAI、AutoGen、OpenAI Agents SDK 或任何自动 tool execution。

## Provider boundary

Runtime 只认识 `models/types.py` 中的 canonical 类型：

- `ModelRequest`
- `SystemMessage` / `UserMessage` / `AssistantMessage` / `ToolResultMessage`
- `ToolCall` / `ToolDefinition`
- `ModelTextDelta`
- `ModelToolCallStarted` / `ModelToolCallArgumentsDelta` / `ModelToolCallCompleted`
- `ModelUsageUpdated`
- `ModelCompleted` / `ModelResponse`

OpenAI 和 Anthropic SDK 类型只存在于各自 adapter 中。Runtime、ConversationState、ToolRegistry、event logger 与 CLI 都不会看到 vendor content block 或 SDK object。

### OpenAI Responses API 映射

```text
response.created                         → ModelStreamStarted
response.output_text.delta               → ModelTextDelta
response.output_item.added(function)     → ModelToolCallStarted
response.function_call_arguments.delta   → ModelToolCallArgumentsDelta
response.function_call_arguments.done    → JSON parse → ModelToolCallCompleted
response.completed                       → usage + ModelCompleted
```

工具参数 delta 只累计字符串；只有收到完成事件才解析 JSON。`call_id` 原样进入 `ToolCall.id`，工具结果再以相同 ID 转成 `function_call_output`。

Adapter 采用无状态续接：每一轮都把完整 canonical 对话作为 `input` 回传（`function_call` 与对应的 `function_call_output` 成对出现且 `call_id` 一致），不使用 `previous_response_id`。这样在官方 Responses API 和只实现 `/v1/responses` 子集、不支持 `previous_response_id` 链式回填的 OpenAI-compatible 服务上都能正确完成多轮工具调用。每轮仍由 ContextBuilder 重新生成 instructions。

PhotomatAgent 内部允许 `mock.run_calculation` 这类 namespaced tool name。OpenAI adapter 会在 provider boundary 将其编码为只包含字母、数字、下划线和连字符的合法名称，并在模型返回 tool call 时可逆地恢复 canonical name；合法名称保持不变，哈希后缀用于避免替换后的名称冲突。

### Anthropic Messages API 映射

```text
message_start                            → ModelStreamStarted + input usage
content_block_delta(text_delta)          → ModelTextDelta
content_block_start(tool_use)            → ModelToolCallStarted
content_block_delta(input_json_delta)    → ModelToolCallArgumentsDelta
content_block_stop                       → JSON parse → ModelToolCallCompleted
message_delta                            → stop reason + output usage
message_stop                             → ModelCompleted
```

Canonical assistant tool calls 被映射为 `tool_use` blocks；连续的 canonical tool results 被组合为下一条 user message 中的 `tool_result` blocks，并保持 `tool_use_id`。

## Tool loop

```text
User input
  → ContextEngine(ContextBuilder + WorkingLedger)
  → Stage A prune → optional Stage B compaction
  → ToolSurfacePlanner / ContextBudget
  → ModelRequest(direct tools + compact manifest)
  → ModelProvider.stream()
  → TextDelta / ToolCall
  → optional tool_call bridge unwrap
  → PermissionPolicy
  → ToolRegistry validation
  → Tool.execute()
  → ObservationPolicy
  → ToolResultMessage(tool_call_id preserved, bounded output)
  → ConversationState
  → ContextEngine
  → 下一次 ModelProvider.stream()
  → Final Response
```

一个模型响应可以包含多个 tool calls；V0.4 按输出顺序串行执行。Assistant tool call message 先进入 conversation，随后每个 tool result 按相同 call ID 写入，下一轮 context 因而能无损回填。bridge call 的 conversation protocol name 保持 `tool_call`，trace execution event 同时记录 `bridge_tool=tool_call` 与真实 `underlying_tool`。

## Exposure、catalog 与 manifest

默认 deterministic policy：

- `read / glob / grep / write / edit / bash` 是高频基础 primitive，标记为 `DIRECT`。
- `calculator / echo / scientific_state_inspect / mock.run_calculation` 代表当前 deferred 示例；未来 MCP、literature、Materials、VASP、HPC 和 plugin tools 默认 `DEFERRED`。
- disabled/unavailable capability 标记为 `HIDDEN`，不会进入 tool specs、manifest 或 search，并拒绝执行。
- `tool_search / tool_describe / tool_call / skill_view` 是稳定 direct helpers；search 不会永久修改 registry 或以后请求的 schemas。

`ToolCatalogEntry` 引用 registry 中原始 `ToolDefinition`，不复制完整 schema。BM25 文档只组合 name、短/长描述、namespace、source、tags 和 parameter names；不索引 JSON Schema body，不使用 embedding、向量库、LLM router 或网络依赖。search 默认返回 5 张 compact cards，硬上限 20；describe 才返回完整调用说明。

CapabilityManifest 有 `manifest_max_tokens=2000` 的 chars/4 预算。小 catalog 展示 `name + one sentence`，中型 catalog 降级为 namespace + names，大型 catalog 只显示 namespace counts；任何形式都不会突破预算。

开发者诊断：

```bash
uv run photomatagent tools list
uv run photomatagent tools surface
uv run photomatagent tools search "submit calculation"
uv run photomatagent tools search "scientific literature" --namespace literature
```

## Progressive skills

初始 system context 只注入 skill index：name、short description、category/tags；不会注入所有 `SKILL.md` 正文，也不会递归读取 `references/`。需要时模型调用：

```text
skill_view(name)                  → primary SKILL.md
skill_view(name, path)            → one path inside that skill directory
```

reference path 在 resolve 后必须仍位于 skill root 内，避免目录逃逸。架构路径是 `index → skill_view → reference`，不是启动时 eager-load everything。

## ObservationPolicy、SensitivePathPolicy 与 ContextBudget

所有成功/失败的 raw tool output 在进入 conversation 与 trace 前经过统一 `ObservationPolicy`。Secret redaction 在这个 model-visible boundary 首先执行，随后才做大小预算；因此 dotenv、已知环境变量密钥、token/password 字段不会先进入模型再仅在日志中脱敏。默认模型可见上限为 12,000 chars；`read` / `bash` 为 16,000，`grep` 为 12,000，`glob` 为 8,000。截断始终显式包含：

```text
truncated=true
original_chars
delivered_chars
redacted
[output truncated: original ... ~... estimated tokens]
```

`bash` 保留 head + marker + tail；`read` 提示用 line range 继续，`grep` 提示缩小 pattern/path，`glob` 保留 result limit。JSONL 保存 model-visible observation 而不是重复保存超大 raw stdout。

每次 `ModelRequestStarted` 还保存薄的 ContextBudget accounting：visible schema、manifest、message/history、tool-result 和 estimated current prompt tokens，以及可选的 model context limit。所有 estimate 都是 `ceil(chars/4)`；真实 usage 仍只采用 provider 回传值，二者不可混用。

`SensitivePathPolicy` 在 permission/execution 前拒绝明显 credential path：`.env`、`.env.*`、`*.pem`、`*.key`、`.git-credentials`、`.netrc`、`.ssh/`、`.aws/`、`credentials*`、`secrets*`。`read(".env")` 默认直接产生 `SensitiveAccessBlocked`；grep/glob 遍历也过滤这些路径。bash 对命令中的显式敏感路径做 lexical blocking，但这不是 shell sandbox；复杂间接访问仍由 workspace/OS 隔离负责。

所有 bash stdout/stderr 在首次进入模型上下文前做 defense-in-depth redaction，覆盖 `*_API_KEY=...`、`*_TOKEN=...`、`*_SECRET=...`、password 和 `Authorization: Bearer ...`。事件日志继续做独立脱敏；redaction 与 sandbox 是两个不同边界。

`WorkingLedger` 不新增持久状态，而是从 durable messages 推导并去重 `searched_queries`、`inspected_paths`、`executed_commands` 与短 `key_observations`，默认总计最多 1,200 chars。它作为短 `Investigation state` 注入，帮助模型发现已经检索/读取过的路径。System prompt 同时加入 evidence-sufficiency 原则，但没有 deterministic StopPolicy 或 planner。

## Streaming

Provider 输出的是 PhotomatAgent canonical stream，而不是 SDK 原始事件。Runtime 将文本 delta 立即转成 `RuntimeEvent.TextDelta`，CLI 逐片打印，不等待完整 response。工具参数 delta 会进入 JSONL trace，但不会在未完成时执行或解析。

## Workspace tools

完整 registry 包含：

- `read`：UTF-8 文本、行范围、输出上限
- `glob`：workspace 内 glob、结果数量上限
- `grep`：正则搜索、path/glob 过滤、结果上限
- `write`：只创建新文件，拒绝覆盖
- `edit`：只允许一次明确的 `old_text → new_text` 替换
- `bash`：cwd 固定为 workspace，timeout，stdout/stderr/exit code，输出上限
- deferred `echo`、`calculator`、`scientific_state_inspect`、`mock.run_calculation`
- direct bridge `tool_search`、`tool_describe`、`tool_call`、`skill_view`

`Workspace.resolve()` 会解析 `..` 和符号链接后的真实路径，普通文件工具拒绝访问 root 之外的路径。

## Permission

默认规则：

```text
read / glob / grep                ALLOW
write / edit / bash               ASK
mock.run_calculation              ASK
其他低风险内置工具                ALLOW
```

CLI 只实现 allow once / deny。Runtime 依赖 `ApprovalHandler` protocol，不导入 Rich 或 prompt_toolkit。

> **Permission is not a security sandbox.** `tool_call` 按 underlying tool name 检查 policy，因此 deferred dangerous tool 不会借 bridge 绕过审批。

权限提示只是 agent harness 的交互控制。V0.4 没有 OS-level sandbox；特别是获批后的 `bash` 进程拥有当前用户本来具备的权限。

## 安装与运行

```bash
uv sync
uv run pytest
uv run photomatagent
uv run photomatagent scientific status
uv run photomatagent skills list --sources
uv run photomatagent tools search "effective mass"
```

第一次执行 `photomatagent` 时，程序会在当前 workspace 创建 `.env`。如果供应商偏好、对应模型名称或 API Key 缺失，终端会逐项询问；API Key 输入不会回显。配置完整后，以后无参数启动会直接进入对话模式。

```dotenv
# PhotomatAgent LLM configuration
PHOTOMATAGENT_PROVIDER='openai'

OPENAI_MODEL='your-openai-model'
OPENAI_BASE_URL='https://api.openai.com/v1'
OPENAI_API_KEY='your-openai-api-key'

ANTHROPIC_MODEL=''
ANTHROPIC_API_KEY=''
```

支持的偏好是 `openai` 和 `anthropic`。程序只要求当前偏好对应的模型名称与密钥；选择 OpenAI SDK 时还会询问 Base URL，直接回车使用官方默认值 `https://api.openai.com/v1`，也可以填写其他 OpenAI-compatible 服务地址。另一家供应商可以留空。`.env` 已加入 `.gitignore`，创建时会尽可能设置为仅当前用户可读写，但它仍是本地明文密钥文件，请勿分享或提交。

也可以只完成或更新配置而不启动对话：

```bash
uv run photomatagent configure
uv run photomatagent configure --provider openai --model your-openai-model
uv run photomatagent configure --provider anthropic --model your-anthropic-model
```

配置优先级为：命令行参数、当前进程环境变量、workspace `.env`。因此 CI 或临时终端仍可用 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANTHROPIC_API_KEY`、`OPENAI_MODEL`、`ANTHROPIC_MODEL` 覆盖文件值，而不会把覆盖后的密钥自动写回文件。

## Scientific Sprint 1 — Infrared Photodetector Capabilities

本轮把 Generic Harness 扩展为一组可真实执行的 scientific capability
packs。所有 scientific 工具保持 `DEFERRED`，通过 `tool_search` / `tool_describe`
/ `tool_call` 渐进暴露，复用现有 ToolRegistry / ToolExecutor /
ObservationPolicy，不引入第二套 executor。

### Capability packs（`src/photomatagent/scientific/capabilities/`）

| Pack | namespace | 依赖 | 工具 |
| --- | --- | --- | --- |
| Materials Project | `materials` | mp-api + API key | `materials.search` / `get_summary` / `get_structure` |
| Literature | `literature` | arxiv, pypdf | `literature.search_arxiv` / `search_local` / `list_papers` / `read_paper` |
| Structure | `structure` | pymatgen | `structure.summary` / `symmetry` / `density` / `neighbors` / `convert` |
| Electronic | `electronic` | sumo, effmass | `electronic.band_summary` / `dos_summary` / `plot_band` / `plot_dos` / `effective_mass` |
| Defects | `defects` | doped | `defects.capabilities` / `generate` / `analyze` |
| Transport | `transport` | amset | `transport.capabilities` / `analyze` |
| Device | `device` | devsim | `device.devsim_capabilities` / `run_script` / `inspect_result` |
| Optics | `optics` | pytaser | `optics.transient_absorption` |
| IR constraints | `ir` | 无（纯 numpy） | `ir.compile_constraints` |

每个 pack 都有 dependency probe，返回 `AVAILABLE` / `MISSING_DEPENDENCY` /
`UNCONFIGURED`；缺失依赖只影响对应工具，绝不阻止 agent 启动。
`photomatagent scientific status` 显示全部状态。

### Third-party Scientific Capabilities（Attribution）

| 名称 | 来源项目 | License | 版本（本环境） |
| --- | --- | --- | --- |
| pymatgen | materialsproject/pymatgen | modified BSD | 2026.5.4 |
| mp-api | materialsproject/mp-api | modified BSD | 0.46.4 |
| sumo | SMTG-Bham/sumo | MIT | 2.4.0.post1 |
| effmass | lucamghini/effmass | MIT | 2.3.1 |
| doped | birnbaum/doped | MIT | 3.2.1（本环境 import 受 coverage/numba 冲突阻断） |
| amset | hackingmaterials/amset | MIT | 未安装 |
| devsim | devsim/DEVSIM | MIT/BSD | 未安装 |
| pytaser | pytaser/pytaser | Apache-2.0 | 2.3.1 |
| arxiv | lukasa/arxiv | BSD-3-Clause | 4.0.1 |
| pypdf | py-pdf/pypdf | BSD-3-Clause | 6.15.0 |
| AtomisticSkills | learningmatter-mit/AtomisticSkills | MIT | 未 clone（配置已就绪） |
| computational-chemistry-agent-skills | jinzhezenggroup/… | — | 未 clone（配置已就绪） |

### Multi-root SkillLoader

`SkillLoader` 现在支持多个 skill 根（`.photomatagent/skills.yaml` 的
`skill_sources`）：native `skills/` + 外部根。每个 skill 带
`SkillDescriptor`（name / description / path / source / license / tags /
priority），同名校验冲突按 priority 解析；缺失根或不符合 SKILL.md 约定的
目录产生 diagnostic 并跳过，不阻断启动。Progressive disclosure 不变：索引
只含 name + description，正文由 `skill_view` 按需加载。

### Native IR skills（`skills/`）

`infrared-photodetector-design`（主领域 skill，驱动 evidence-gap 推理与
multi-fidelity escalation）、`ir-constraint-analysis`、
`infrared-material-screening`、`narrow-gap-electronic-analysis`、
`defect-dark-current-analysis`、`carrier-transport-analysis`、
`optical-response-analysis`、`detector-device-evaluation`。
VASP/QE/phonopy/HPC 的具体执行 SOP 交给外部 skills，IR skills 只负责
“何时需要、为什么需要、什么结果可信”。

### 结果契约与 innovation logging

- `ScientificEvidence`：轻量 provenance carrier（subject / property /
  value / unit / source / source_type / method / summary / limitations /
  provenance），自动进入 `ScientificState`，无 EvidenceGraph。
- `ScientificToolResult`：ToolObservation + evidence[] + artifacts[]，全部
  通过现有 state_updates 通道落地。
- 每次 run 结束发出 `scientific_trace_meta` 事件：skills_loaded /
  scientific_tools_used / evidence_created / evidence_sources /
  evidence_gaps_identified / capability_escalations；
  `photomatagent sessions stats` 直接展示。

### Literature RAG V1（本地检索）

`literature` 能力包在原有 arXiv/本地 PDF 工具之上新增完全本地的
Literature RAG：docling 解析 PDF → HybridChunker 分块 →
sentence-transformers（`intfloat/multilingual-e5-small`）嵌入 →
LanceDB 向量库 → 稠密 + BM25 混合检索（RRF 融合）→
`cross-encoder/ms-marco-MiniLM-L-6-v2` 重排 → 上下文扩展 → 正则数值证据提取。

配置（环境变量，均有默认值）：

- `PHOTOMATAGENT_LITERATURE_DIR`：PDF 根目录，默认 `dataset/paper`（递归扫描）
- `PHOTOMATAGENT_LITERATURE_INDEX_DIR`：索引目录，默认 `output/literature_index`
- `PHOTOMATAGENT_EMBEDDING_MODEL`：默认 `intfloat/multilingual-e5-small`
- `PHOTOMATAGENT_RERANKER_MODEL`：默认 `cross-encoder/ms-marco-MiniLM-L-6-v2`

安装依赖：`uv sync --extra literature`（torch 固定走 CPU 索引，避免 PyPI
CUDA 轮子损坏问题）。首次索引会从 Hugging Face 下载 docling 布局模型与嵌入模型。

工具：`literature.index_papers`（建索引）、`literature.search_passages`
（混合检索，返回带 provenance 的片段）、`literature.read_passage`（按
passage_id 读全文）、`literature.extract_evidence`（提取 responsivity /
detectivity / dark current / wavelength / temperature / bandgap / mobility /
NETD 数值证据，输出 `ScientificEvidence`）。

示例查询：`literature.search_passages` `"HgTe CQD infrared detector
responsivity"`。测试：`uv run pytest tests/test_literature_rag.py`。

### Vertical slices（`experiments/vertical/`）

三个 IR 科研回归任务（scripted 确定性 + Slice 1 的真实 LLM 变体）：

```bash
cd experiments/vertical
uv run python slice_1_lwir_screening.py   # 8-14 um LWIR screening
uv run python slice_2_hgte.py             # HgTe narrow-gap 分析
uv run python slice_3_escalation.py       # 部分证据 → 能力升级 → prerequisite
uv run python slice_1_lwir_llm.py         # 真实 LLM（deepseek-v4-flash）端到端
```

结果写入 `output/vertical/*.json`，含 evidence-gap trajectory 与 trace 路径。

诊断与辅助命令：

```bash
uv run photomatagent doctor
uv run photomatagent tools list
uv run photomatagent sessions list
uv run photomatagent sessions stats latest
uv run photomatagent sessions context latest
```

离线 Fake provider：

```bash
uv run photomatagent chat --provider fake
uv run photomatagent chat --provider fake --goal "investigate material InAs" --approval auto
```

单轮 goal 默认最多迭代 25 次，达到上限会输出 `loop finished: max_iterations`。可通过 `--max-iterations` 调整，例如 `uv run photomatagent chat --max-iterations 50`。交互模式可输入 `/compact` 手动测试结构化 compaction；无可压缩旧轮次时不会改变默认对话体验。达到上限时若模型最后一条回复还带有未执行的工具调用，下次 working context 会成对隐藏该 abandoned transaction，durable conversation 仍完整保留。

交互聊天同时内置斜杠命令，输入 `/help` 可查看完整清单。常用映射为
`/doctor`、`/tools`、`/skills`、`/scientific`、`/mcp`、`/sessions`、
`/experiments`、`/configure` 和 `/compact`；分组命令保留 CLI 的子命令和参数，
例如 `/sessions stats latest`。

权限可在聊天中显式切换：`/approve -o` 仅在当前聊天任务内完全允许所有工具，
`/approve -a` 为当前工作区持久启用完全允许，`/approve -b` 清除两种覆盖并恢复
本次启动所选的初始策略。持久状态保存在被 Git 忽略且权限为 `0600` 的
`.photomatagent/settings.json`。完全允许会跳过工具确认，因此只应在受信工作区启用。

OpenAI（官方 Python SDK + Responses API）：

```bash
uv run photomatagent configure --provider openai --model your-openai-model
uv run photomatagent doctor
uv run photomatagent chat
```

OpenAI-compatible 服务示例配置：

```dotenv
PHOTOMATAGENT_PROVIDER='openai'
OPENAI_MODEL='compatible-model-name'
OPENAI_BASE_URL='https://your-provider.example/v1'
OPENAI_API_KEY='your-provider-api-key'
```

当前 OpenAI adapter 调用的是 Responses API。第三方服务不仅要接受 OpenAI Python SDK，还需要兼容 `/v1/responses`、streaming 和 function calling；只实现 `/v1/chat/completions` 的服务暂不兼容。

Anthropic（官方 Python SDK + Messages API）：

```bash
uv run photomatagent configure --provider anthropic --model your-anthropic-model
uv run photomatagent doctor
uv run photomatagent chat
```

`doctor` 会读取 workspace `.env`，但只显示 API Key 为 `configured` / `missing`，不会输出密钥明文。它验证配置是否存在，不会产生付费模型请求。

## Agent Execution Trace schema

事件写入：

```text
.photomatagent/sessions/<session-id>/events.jsonl
```

所有事件共享 typed envelope：

```text
schema_version   当前为 1.0
kind             event_type discriminator
timestamp        UTC timestamp
session_id       一个 CLI/runtime session
run_id           session 内的一次 AgentRuntime.run()
```

字段按 event 类型出现，不强迫所有行拥有同一组 nullable 字段：

- `LoopStarted`：goal、provider、model、workspace
- `LoopIterationStarted`：iteration
- `ModelRequestStarted`：iteration、provider/model、message_count；registered/direct/deferred/hidden counts；visible schema/manifest chars；estimated schema/manifest/avoided/bridge/history/tool-result/current-prompt tokens
- `ContextPruneStarted/Completed`：before/after tokens/chars/messages、pruned results、protected turns、duration
- `ContextCompactionStarted/Completed/Failed`：before/after、protected turns、duration 或失败原因
- `SensitiveAccessBlocked`：tool/call ID、被阻断的路径显示值，不包含文件内容
- `ModelResponseCompleted`：finish_reason、tool_call_count、usage、duration_ms
- `ToolRequested`：tool_call_id、tool_name、arguments，以及可选 bridge_tool/underlying_tool
- `ToolCompleted` / `ToolFailed` / `ToolPermissionDenied`：tool status、latency、output 或 error/error_type、bridge identity 与 observation truncation metadata
- `ProviderFailed`：provider/model、error/error_type、failed request duration_ms
- `LoopCompleted` / `LoopFailed`：stop reason 或 error/error_type、duration_ms

`kind` 就是 trace schema 的 `event_type`。流式 `TextDelta` 仅保存模型向用户公开的文本；不存在隐藏 chain-of-thought 字段。

Redaction boundary 在写 JSONL 前运行：常见 secret key 字段、Authorization/token/password 字段、已加载的 secret 环境变量、典型 API key 字符串和完整 dotenv 形态文本会被替换。不会保存 Authorization header 或 `.env` 原文；`EventLogger` 仍允许注入轻量自定义 redactor。

```bash
uv run photomatagent sessions list
uv run photomatagent sessions show latest
uv run photomatagent sessions stats latest
uv run photomatagent sessions context latest
uv run photomatagent sessions replay latest
uv run photomatagent sessions replay latest --verbose
```

`SessionSummary` 包括 iterations、model/tool calls、unique tools、failures/denials、provider usage（不可用时为 null）、总/model/tool latency、stop reason，以及以下 loop metrics：

- `repeated_tool_calls`：全 session 内相同 `tool_name + normalized_arguments` 在第一次之后的额外次数。arguments 用 sorted-key compact JSON 规范化，因此 dict key 顺序不影响 identity。
- `consecutive_repeat_count`：相邻 action 与前一个完全相同的次数；`A,A,A` 计 2。
- `tool_failure_rate`：failed actions / requested actions。
- `tools_per_iteration`：tool calls / iterations。
- `model_calls_per_completed_session`：model calls / completed runs；没有 completed run 时为 null。
- Tool Surface：registered/direct/deferred/hidden counts，direct/bridge/manifest/model-visible schema estimated tokens，deferred schemas avoided per call / cumulative。
- Discovery Cost：`tool_search_calls`、`tool_describe_calls`、`tool_call_bridge_calls`。
- Context Lifecycle：last/peak working-context estimate、durable JSONL chars、pruned results、compaction success/failure 与 last before/after。

默认 anomaly diagnostics（只诊断，不停止 runtime）：

- `REPEATED_ACTION`：相同 action 连续出现至少 2 次。
- `TOOL_FAILURE_LOOP`：相同 action 连续失败至少 2 次。
- `MAX_ITERATIONS_REACHED`：stop reason 为 `max_iterations`。
- `HIGH_TOOL_CHURN`：session tool calls 达到默认阈值 20。

Replay 先构建不依赖 Rich/ANSI 的 intermediate model，再由 CLI 渲染 goal、iteration、model、tool arguments、result、final response 和 stop。它不读取或展示 reasoning delta。

## Lightweight experiments

V0.4 使用 JSON，避免为 YAML 新增依赖。variant 新增 `tool_surface: progressive | eager`，可在相同 provider/model/system prompt/tasks 下做控制实验。示例见 `experiments/progressive-tools-v1.json`：

```json
{
  "name": "progressive-tools-v1",
  "variant": {
    "provider": "fake",
    "model": "fake",
    "max_iterations": 10,
    "approval": "auto",
    "tool_surface": "progressive"
  },
  "tasks": [
    {
      "id": "locate-loop",
      "prompt": "找到 Agent Loop 的核心文件。",
      "expect": {
        "answer_contains": ["loop"],
        "tools_used": ["read"],
        "max_tool_calls": 10,
        "max_iterations": 6
      }
    }
  ]
}
```

支持的 optional expectations 只有 `answer_contains`、`answer_not_contains`、`tools_used`、`tools_not_used`、`max_tool_calls`、`max_iterations`。字符串检查大小写不敏感。没有 expectations 的 task 标为 `UNEVALUATED`，不会伪装成 PASS。

```bash
uv run photomatagent experiments run experiments/offline-smoke.json
uv run photomatagent experiments run experiments/baseline-eager-tools.json
uv run photomatagent experiments run experiments/progressive-tools-v1.json
uv run photomatagent experiments compare <experiment-a> <experiment-b>
```

`offline-smoke.json` 固定使用 Fake provider，并真实演练 `tool_search → tool_describe → tool_call → mock.run_calculation`，适合零网络/零费用 E2E。`baseline-eager-tools.json` 与 `progressive-tools-v1.json` 含完全相同的 5 个任务，未固定 provider/model，会使用 workspace `.env` 中的真实配置。eager control 会模拟旧版直接暴露原始工具，不发送 progressive helpers。

Runner 严格顺序执行，每个 task 创建独立 session。variant 未写 provider/model 时读取现有 workspace `.env`；实验模式不交互询问 approval，只支持 `auto` 或 `deny`。每次 experiment 保存：

```text
.photomatagent/experiments/<experiment-id>/
├── config.json      # task config + configuration snapshot
├── runs.json        # per-task runtime/evaluation/SessionSummary
└── summary.json     # aggregate metrics
```

configuration snapshot 记录 provider/model、system prompt SHA-256、StopPolicy、ContextBuilder 和 ToolSurfacePlanner identifier/config。Compare 除原有质量/loop 指标外，展示 model latency、estimated tool-schema tokens/call 及三类 discovery calls 的 B-A delta；不生成综合分数，也不宣称哪个 variant 更好。

V0.5 回归配置 `experiments/context-lifecycle-v05.json` 与 V0.4 progressive 的五个任务逐字一致；summary 额外记录 peak working context、pruned tool results、compaction count/failures。由于 ContextEngine 和 evidence-sufficiency prompt 本身就是 treatment，跨版本结果应作为 harness bundle regression 解读，而不是仅凭单次随机模型样本推断因果。

## ConversationState 与 ScientificState

`ConversationState` 保存 provider-neutral durable conversation；`ScientificState` 独立保存 goal、hypotheses、claims、evidence、calculations、open questions、contradictions 和 pending tasks。`ContextEngine` 决定 working subset 与 lifecycle，`ContextBuilder` 只负责把选中的 messages、scientific state、skill/capability index、ledger 和可选 compaction state 渲染成 provider-neutral context。

## 当前限制

- 未做 OS sandbox；bash 获批后是普通本地 shell
- Semantic compaction 使用模型生成的结构化 state，schema 校验能约束形状但不能证明摘要事实完全无损；provenance reference 也尚不能自动 rehydrate 原 result
- Context token trigger 使用 chars/4 估算；调用方未提供真实 model context limit 时使用 128k fallback，不等同 provider tokenizer
- SensitivePathPolicy 对 bash 是 lexical defense，不是 OS sandbox；复杂 shell 间接访问必须依靠后续真正的 sandbox/容器边界
- 已接 Materials Project（mp-api，需 API key）、arXiv、pymatgen、sumo、
  effmass、pytaser；doped / AMSET / DEVSIM 为 dependency-optional probe；
  MCP server 支持 stdio/HTTP 配置但默认未启用；未接 Slurm，VASP 执行交由
  外部 AtomisticSkills SOP
- 未做 scientific StopPolicy、Evidence Graph、Planner、Multi-Agent、RAG 或 memory system
- Experiment 仅支持 JSON、顺序执行和 deterministic expectations；没有 YAML、并行 runner 或 LLM-as-Judge
- Token usage 依赖 provider；未报告时为 null。Context surface 使用 chars/4 diagnostics estimate，明确不冒充 provider usage
- 文件工具面向文本和单 workspace，不是虚拟文件系统
- 当前配置前端是终端引导，尚未提供桌面/Web 设置页或系统密钥链集成
- 默认 pytest 完全离线；仓库不自动运行付费 live API test

## Generic Harness V1 源码阅读路线

建议依次阅读：

1. `tools/base.py`：Tool 的 exposure / namespace / source / tags 元数据
2. `tools/registry.py`：完整 authorized capability universe 与 canonical definitions
3. `tools/surface.py`：Exposure decision、ToolCatalog、BM25、manifest、surface stats
4. `tools/bridges.py`：compact search cards、describe、call marker 与 skill_view
5. `runtime/context_engine.py`：durable/working boundary、Stage A/B、tool transaction invariants、CompactionState
6. `runtime/ledger.py`：从 durable messages 推导 bounded investigation state
7. `runtime/sensitive.py` 与 `redaction.py`：敏感路径和 model-visible/log output 安全边界
8. `runtime/loop.py`：ContextEngine 接入、bridge unwrap、permission/validation/execution
9. `runtime/observation.py`：raw result 到 insertion-time bounded observation
10. `runtime/events.py` 与 `observability/analyzer.py`：context lifecycle trace、session/experiment 聚合

沿这条路线可以手动跟踪：`Registered Tool → Exposure → ModelVisibleTools → tool_search → tool_describe → tool_call → underlying execution → Observation → next model turn`。

如果只有 30 分钟，优先读 `runtime/context_engine.py` → `runtime/loop.py` → `runtime/sensitive.py`。三者分别回答“模型本轮看到什么”“生命周期如何进入主循环”“敏感 observation 在哪里被阻断”。
