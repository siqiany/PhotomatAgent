# PhotomatAgent

一个最小、清晰、可测试、可扩展的 **Scientific Agent Runtime** —— 面向材料科学研究（尤其是红外光电探测材料）的本地科学智能体骨架。

> 注意：这是 **Scientific Agent Runtime**，不是聊天机器人。它的核心产物是「受控的 agent loop + 结构化的科学状态 + 可审计的事件流」，而模型、工具、UI 都是可替换的插槽。

## 1. 这是什么

PhotomatAgent 的目标是让你像研究 Claude Code / Codex 的 loop 一样，研究 **Scientific Agent 的 loop engineering**：

- 自然语言科研交互
- 把 VASP、Materials Project、文献检索、结构分析包装成 Tools / MCP Servers
- 用 Skills 保存材料科学 SOP
- 管理 scientific state / evidence / provenance
- 根据结果 verification，根据失败 diagnose → retry → replan
- 计算预算、人工审批

当前迭代**只实现最小骨架**：一个我们自己写的、事件驱动的 agent loop，加上最小但边界清晰的科学状态、工具系统、权限、预算、Skills、MCP 预留接口与 JSONL 事件日志。

## 2. 架构

```text
CLI (Rich + prompt_toolkit)
        ↓ 消费 RuntimeEvent 事件流
Event Stream (Pydantic discriminated union, 可序列化为 JSONL)
        ↓
Agent Runtime (runtime/loop.py —— 全项目唯一的 loop)
   ├─ Model Provider     (models/)         模型抽象，运行时不知道后端是谁
   ├─ Tool Registry      (tools/)          统一 Tool 接口 + schema 校验
   ├─ Context Builder    (runtime/context.py)  融合会话 + 科学状态
   ├─ Permission Policy  (runtime/permissions.py)  allow / deny / ask
   ├─ Stop Policy        (runtime/stop_policy.py)  何时停止
   ├─ Scientific State   (scientific/)      claims / evidence / calculations
   └─ Budget             (runtime/budget.py)  模型调用 / 工具调用计数
```

### 几个容易混淆的概念

| 概念 | 位置 | 是什么 |
| --- | --- | --- |
| Agent Loop | `runtime/loop.py` | 上下文 → 模型 → 工具 → 状态 → 停止的循环，自己实现 |
| ConversationState | `runtime/state.py` | 模型真正看到的消息历史（system/user/assistant/tool） |
| ScientificState | `scientific/state.py` | 科学事实：目标、假说、claim、evidence、calculation、待办任务 |
| Skills | `skills/` + `skills/loader.py` | 科研方法与 SOP 的静态文档，本阶段只加载不选择 |
| Tools | `tools/` | 统一 `Tool` 接口；echo / calculator / 状态检查 / mock 计算 |
| MCP | `mcp/client.py` | 未来把 MCP server 工具包成 `Tool` 的接缝，本阶段仅接口 + TODO |
| Scientific Backend | `scientific/backends/` | 未来 VASP/Slurm 的边界，本阶段只有 mock backend |

## 3. Agent Loop 的执行路径

```text
User Goal
   ↓
Prepare Context (ConversationState + ScientificState → system/user 消息)
   ↓
Model
   ↓
Tool Calls?
 ┌─┴────────────┐
 No             Yes
 ↓               ↓
Finish       Validate Tool Call (schema)
                 ↓
             Permission (allow / deny / ask)
                 ↓
             Execute Tool (async)
                 ↓
             Tool Result → 更新 ScientificState
                 ↓
             Next Iteration
```

每一步都发出一个 `RuntimeEvent`（见 `runtime/events.py`）：`LoopStarted`、`LoopIterationStarted`、`ModelRequestStarted`、`TextDelta`、`ToolRequested`、`ToolApprovalRequired`、`ToolStarted`、`ToolCompleted`、`ToolFailed`、`ScientificStateUpdated`、`BudgetUpdated`、`LoopCompleted`、`LoopFailed` 等。

CLI 只是 `async for event in runtime.run(goal)` 的消费者。未来可以换成 Textual TUI、Web、API、JSONL logger 而不用改 loop。

## 4. 为什么不用 LangChain / CrewAI / LangGraph / AutoGen

这个项目的目标之一就是**自己实现并理解 Agent Loop**。这些框架会替你隐藏 loop 的控制流、状态流转和事件语义。PhotomatAgent 坚持：

- core is explicit：loop 在单个文件里可通读
- control flow is readable：没有 decorator / middleware 迷宫
- state transition is visible：每一步都有事件流出

模型调用只允许通过 `ModelProvider` 协议（`models/base.py`），当前只有 `FakeModelProvider`，未来可加 OpenAI/Anthropic adapter 而不影响 runtime。

## 5. 快速开始

```bash
uv sync --extra dev
uv run pytest
uv run photomatagent doctor
uv run photomatagent tools list
uv run photomatagent skills list
uv run photomatagent chat
```

`photomatagent chat` 使用 fake model（自动模式）：第一轮请求 `mock.run_calculation`，第二轮总结工具结果。默认工具调用需要 y/n 确认（`--approval ask`），可切换 `--approval auto`。

单轮非交互执行：

```bash
uv run photomatagent chat --goal "investigate material GaAs" --approval auto
```

每次会话的事件会追加到 `.photomatagent/sessions/<session-id>/events.jsonl`，可用 `EventLogger.read_events()` 回放。

## 6. 目录结构

```text
photomatagent/
├── pyproject.toml
├── README.md
├── src/photomatagent/
│   ├── cli/            # Typer + Rich + prompt_toolkit（纯事件消费者）
│   ├── runtime/        # loop / events / state / context / budget / permissions / stop_policy
│   ├── models/         # ModelProvider 协议 + FakeModelProvider
│   ├── tools/          # Tool 抽象 + ToolRegistry + 内置工具
│   ├── scientific/     # ScientificState / Evidence / Claim / Calculation / Task / backends
│   ├── skills/         # SkillLoader（扫描 SKILL.md）
│   ├── mcp/            # MCP 集成接缝（TODO）
│   └── logging/        # JSONL 事件日志
├── skills/             # 示例科研 SOP
│   └── electronic-structure-analysis/SKILL.md
└── tests/              # pytest + pytest-asyncio
```

## 7. 当前明确没有实现

Multi-Agent、Subagent、Planner、RAG、向量库、记忆系统、Web UI、Textual TUI、真实 VASP、Slurm、真实 Materials Project、自主材料发现、复杂 MCP 生态、插件市场、数据库服务 —— 全部留到后续迭代。

## 8. 下一阶段建议（3 项）

1. **Loop Engineering 实验台**：用 `events.jsonl` 做 replay + 可视化，观察不同 prompt / stop policy / context 组合下的行为。
2. **OpenAI/Anthropic provider adapter**：在 `models/base.py` 协议后加真实 provider，让 loop 接上真实模型。
3. **scientific StopPolicy**：基于 confidence 阈值、未解决矛盾、信息增益的停止条件（`stop_policy.py` 已留好接口）。
