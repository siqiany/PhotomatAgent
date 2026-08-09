# PhotomatAgent V0.2

PhotomatAgent 是一个面向材料科学、尤其是红外光电探测材料研究的本地 **Scientific Agent Runtime**。它不是 SDK 自带的 agent，也不是聊天机器人外壳；项目的核心是一个由我们自己控制、可直接阅读和修改的 Agent Loop。

V0.2 是第一个真实模型 vertical slice：同一个 runtime 可以驱动 Fake、OpenAI Responses API 或 Anthropic Messages API，并由自己的 loop 完成 streaming、工具审批、执行、结果回填和下一轮模型调用。

## 架构

```text
                         CLI
                          │
                          ▼
                     Event Stream ──────► JSONL trace / session stats
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
  → ContextBuilder
  → ModelRequest
  → ModelProvider.stream()
  → TextDelta / ToolCall
  → PermissionPolicy
  → ToolRegistry validation
  → Tool.execute()
  → ToolResultMessage(tool_call_id preserved)
  → ConversationState
  → ContextBuilder
  → 下一次 ModelProvider.stream()
  → Final Response
```

一个模型响应可以包含多个 tool calls；V0.2 按输出顺序串行执行。Assistant tool call message 先进入 conversation，随后每个 tool result 按相同 call ID 写入，下一轮 context 因而能无损回填。

## Streaming

Provider 输出的是 PhotomatAgent canonical stream，而不是 SDK 原始事件。Runtime 将文本 delta 立即转成 `RuntimeEvent.TextDelta`，CLI 逐片打印，不等待完整 response。工具参数 delta 会进入 JSONL trace，但不会在未完成时执行或解析。

## Workspace tools

默认 registry 包含：

- `read`：UTF-8 文本、行范围、输出上限
- `glob`：workspace 内 glob、结果数量上限
- `grep`：正则搜索、path/glob 过滤、结果上限
- `write`：只创建新文件，拒绝覆盖
- `edit`：只允许一次明确的 `old_text → new_text` 替换
- `bash`：cwd 固定为 workspace，timeout，stdout/stderr/exit code，输出上限
- `echo`、`calculator`、`scientific_state_inspect`、`mock.run_calculation`

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

> **Permission is not a security sandbox.**

权限提示只是 agent harness 的交互控制。V0.2 没有 OS-level sandbox；特别是获批后的 `bash` 进程拥有当前用户本来具备的权限。

## 安装与运行

```bash
uv sync
uv run pytest
uv run photomatagent
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

诊断与辅助命令：

```bash
uv run photomatagent doctor
uv run photomatagent tools list
uv run photomatagent sessions list
uv run photomatagent sessions stats latest
```

离线 Fake provider：

```bash
uv run photomatagent chat --provider fake
uv run photomatagent chat --provider fake --goal "investigate material InAs" --approval auto
```

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

## JSONL trace 与 Session Statistics

事件写入：

```text
.photomatagent/sessions/<session-id>/events.jsonl
```

trace 包含 session id、iteration、provider、model、模型流开始/完成、工具请求/成功/失败、permission denial、usage、timestamp 和 duration。默认 redactor 会遮盖常见 API key/header 字段和 key 字符串；`EventLogger` 也允许注入自定义 redaction hook。

```bash
uv run photomatagent sessions list
uv run photomatagent sessions stats latest
```

统计包括 iterations、model calls、tool calls、tool failures、permission denials、input/output tokens 和 duration。这只是 JSONL 派生统计，不是 replay 或 benchmark framework。

## ConversationState 与 ScientificState

`ConversationState` 保存 provider-neutral 的对话协议；`ScientificState` 独立保存 goal、hypotheses、claims、evidence、calculations、open questions、contradictions 和 pending tasks。`ContextBuilder` 是两者合并进入模型上下文的唯一位置。

## 当前限制

- 未做 OS sandbox；bash 获批后是普通本地 shell
- 未做 automatic retry、context compaction 或完整 replay
- 未接 VASP、Slurm、Materials Project、文献 API 或真实 MCP server
- 未做 scientific StopPolicy、Evidence Graph、Planner、Multi-Agent、RAG 或 memory system
- 文件工具面向文本和单 workspace，不是虚拟文件系统
- 当前配置前端是终端引导，尚未提供桌面/Web 设置页或系统密钥链集成
- 默认 pytest 完全离线；仓库不自动运行付费 live API test

## Loop Engineering 阅读入口

建议依次阅读：

1. `runtime/loop.py`：完整控制流；重点看 `run()` 与 `_handle_tool_call()`
2. `models/types.py`：canonical protocol 与 streaming event vocabulary
3. `runtime/context.py`：ConversationState + ScientificState 如何进入 ModelRequest
4. `models/openai.py`：Responses event mapper 与 call ID round-trip
5. `models/anthropic.py`：content block mapper 与 tool_result grouping
6. `runtime/permissions.py`：ALLOW / DENY / ASK 和 ApprovalHandler
7. `tools/registry.py`：工具定义、注册与参数校验
8. `logging/event_logger.py`、`logging/session_stats.py`：可审计 trace 与最小 loop metrics

如果只有 30 分钟，先读 `runtime/loop.py`、`models/types.py`、`runtime/context.py`。
