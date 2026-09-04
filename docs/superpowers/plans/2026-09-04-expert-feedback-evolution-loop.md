# 专家反馈驱动的智能体演化 LOOP 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PhotomatAgent 中实现可异步恢复、专家反馈与普通聊天严格隔离、能够按版本重跑同一科研任务并积累策略经验的 H-BEAL 演化 LOOP。

**Architecture:** 新增 `scientific/evolution` 应用层，以持久化 `EvolutionTask` 串联多个相互独立的 `ScientificLoopController` Episode。专家反馈通过专用 CLI 表单进入，先保存为不可变记录，再由无工具权限的 `FeedbackCompiler` 转成经用户确认的 `RevisionPlan`；只有显式执行 `iterate` 才创建新 Runtime 并重新运行任务。

**Tech Stack:** Python 3.12、Pydantic 2、Typer、Rich、prompt-toolkit、NumPy、pytest/pytest-asyncio、现有 AgentRuntime/ScientificLoopController/EventLogger。

**Spec:** `docs/superpowers/specs/2026-09-04-expert-feedback-evolution-loop-design.md`

## Global Constraints

- 开工前完整阅读根目录 `AGENTS.md`、上述设计文档和本计划；更近层级的 `AGENTS.md` 优先。
- 你不是唯一修改仓库的人。现有已修改、已暂存和未跟踪文件均属于用户；不得 reset、checkout、删除、覆盖或顺手格式化无关内容。
- 每个任务先写失败测试，再写最小实现；每个任务只提交其 `Files` 列出的路径，不得使用 `git add .`。
- `AgentRuntime` 始终是模型请求工具执行的唯一权威入口；EvolutionService、Compiler、Planner、CLI 不得直接执行 Tool、SSH、Slurm 或 HPC。
- `FeedbackCompiler` 和 `ScientificJudge` 都必须使用 `ModelRequest(tools=[])`；人类评分和 Judge 不能把确定性 FAIL/UNKNOWN 改为 PASS。
- 专家原文不进入 `ConversationState`、`ScientificState` 或普通 `runtime.run(feedback_text)`；进入下一 Episode 的只能是 schema 验证并经用户确认的 `RevisionInstruction`。
- 正常 `iterate` 创建全新的 Runtime session；只允许继承满足规则的结构化已验证证据，禁止继承旧对话、旧答案和未验证模型推测。
- 所有路径必须在 Workspace 内，并通过 `Workspace.resolve`；运行状态写入 `.photomatagent/evolutions/`，用户结果写入 `user_output/<evolution-id>/<version>/`。
- 状态 JSON 原子写入、schema 版本化、secret-redacted、带 optimistic revision；反馈和 Episode 记录不可原地覆盖。
- 任何真实 HPC 提交都不属于测试；所有测试使用 FakeModelProvider、fake/local backend 和 `tmp_path`。
- 同一任务的多次 Episode 是相关样本；统计与贝叶斯启用门槛按 `task_group_id` 计算 distinct tasks。
- Python 新代码保持类型标注、Pydantic `extra="forbid"`、lazy optional behavior 和现有命名风格。

## 文件结构与职责

```text
src/photomatagent/scientific/evolution/
├── __init__.py          # 仅导出稳定公共接口
├── models.py            # 持久化领域模型、Literal 状态和 ID
├── rubric.py            # expert-review-v1 量表、硬性封顶和效用计算
├── store.py             # Workspace 内的原子、不可变、revision-checked 存储
├── service.py           # 生命周期和状态转换；不创建模型、不执行工具
├── events.py            # Evolution event 构造/有界摘要（事件类型仍定义在 runtime/events.py）
├── artifacts.py         # Episode 结果固化、SHA-256 和写文件事件收集
├── executor.py          # 用现有 ScientificLoopController 执行一个 Episode
├── feedback.py          # 无工具 LLM Compiler、JSON 提取和失败降级
├── revision.py          # 确定性 RevisionPlan 与动态指令渲染
├── evidence.py          # verified evidence 继承筛选和 provenance 追加
├── comparison.py        # Episode 差异、问题闭合/复发和成本变化
├── experience.py        # Experience 生命周期与学习信号
└── strategy.py          # 固定策略、特征、Bayesian Thompson Sampling

src/photomatagent/cli/evolve.py
    # evolve Typer 子命令、Rich 渲染和 prompt-toolkit 专家表单

tests/test_evolution_models.py
tests/test_evolution_store.py
tests/test_evolution_events.py
tests/test_evolution_service.py
tests/test_evolution_artifacts.py
tests/test_evolution_executor.py
tests/test_evolution_feedback.py
tests/test_evolution_revision.py
tests/test_evolution_evidence.py
tests/test_evolution_comparison.py
tests/test_evolution_strategy.py
tests/test_evolution_cli.py
tests/test_evolution_end_to_end.py
```

测试代码片段中的 fixture（例如 `target`、`service`、`episode_v1`）必须在同一个测试文件
中用 `@pytest.fixture` 明确定义，或从 `tests/conftest.py` 已存在的 fixture 导入；不得依赖
未声明的全局对象。统一的最小 target fixture 使用：

```python
@pytest.fixture
def target() -> TargetSpec:
    return TargetSpec(
        goal="screen a material",
        constraints=[
            ConstraintSpec(
                property="band_gap",
                operator="ge",
                value=2.5,
                unit="eV",
                severity="HARD",
            )
        ],
    )
```

每个 Task 的测试 helper 只构造该 Task 所需的最小合法生产模型；禁止在 helper 中绕过
Store、Service 状态验证或直接伪造最终 PASS。

---

### Task 1: 领域模型、ID 和专家评分量表

**Files:**
- Create: `src/photomatagent/scientific/evolution/__init__.py`
- Create: `src/photomatagent/scientific/evolution/models.py`
- Create: `src/photomatagent/scientific/evolution/rubric.py`
- Create: `tests/test_evolution_models.py`

**Interfaces:**
- Consumes: `TargetSpec`, `ScientificLoopSummary`, Pydantic 2。
- Produces: `EvolutionTask`、`EpisodeRecord`、`ExpertFeedbackRecord`、`FeedbackCompilation`、`RevisionPlan`、`StrategyVersion`、`ArtifactRef`、`ComparisonReport`、`new_evolution_id()`、`new_feedback_id()`、`assess_hard_caps()`、`expert_utility()`。

- [ ] **Step 1: 写模型校验和评分失败测试**

```python
from pydantic import ValidationError
import pytest

from photomatagent.scientific.evolution.models import (
    ExpertFeedbackDraft,
    RubricFlags,
    RubricScores,
    new_evolution_id,
)
from photomatagent.scientific.evolution.rubric import assess_hard_caps, expert_utility


def test_feedback_scores_are_bounded_integers():
    with pytest.raises(ValidationError):
        RubricScores(
            scientific_correctness=6,
            evidence_sufficiency=3,
            novelty=3,
            actionability=3,
            overall=3,
        )


def test_hard_caps_are_suggested_without_rewriting_expert_input():
    scores = RubricScores(
        scientific_correctness=5,
        evidence_sufficiency=5,
        novelty=5,
        actionability=5,
        overall=5,
    )
    result = assess_hard_caps(
        scores,
        RubricFlags(fabricated_source=True),
    )
    assert scores.evidence_sufficiency == 5
    assert result.suggested_scores.evidence_sufficiency == 1
    assert result.suggested_scores.overall == 1
    assert result.reasons


def test_expert_utility_uses_approved_weights():
    scores = RubricScores(
        scientific_correctness=5,
        evidence_sufficiency=1,
        novelty=1,
        actionability=1,
        overall=5,
    )
    assert expert_utility(scores) == pytest.approx(0.35)


def test_generated_evolution_ids_are_path_safe():
    value = new_evolution_id()
    assert value.startswith("evo_")
    assert "/" not in value and ".." not in value
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `uv run pytest -q tests/test_evolution_models.py`

Expected: FAIL，错误包含 `No module named 'photomatagent.scientific.evolution'`。

- [ ] **Step 3: 实现严格领域模型**

在 `models.py` 中定义以下精确类型和字段；所有持久化模型使用
`model_config = ConfigDict(extra="forbid")`：

```python
EvolutionStatus = Literal[
    "CREATED", "RUNNING", "AWAITING_EXPERT_FEEDBACK",
    "FEEDBACK_RECORDED", "REVISION_READY", "ACCEPTED",
    "STOPPED", "BUDGET_EXHAUSTED", "BLOCKED",
]
EpisodeStatus = Literal["RESERVED", "RUNNING", "COMPLETED", "FAILED"]
ExecutionMode = Literal["NORMAL", "CARRY_VERIFIED_EVIDENCE", "FRESH_EVALUATION"]
StrategyArm = Literal["STATIC", "EVIDENCE_FIRST", "DIVERSITY_FIRST", "UNCERTAINTY_FIRST"]
FeedbackItemStatus = Literal["CORRECTION", "QUERY", "PREFERENCE", "POSITIVE_SIGNAL"]
FeedbackSeverity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
CompilationStatus = Literal["PENDING", "AVAILABLE", "UNAVAILABLE"]
AcceptanceStatus = Literal["PENDING", "PASS", "FAIL", "NEEDS_HUMAN_REVIEW"]
ExperienceMaturity = Literal["OBSERVATION", "HYPOTHESIS", "VALIDATED_EXPERIENCE", "REUSABLE_SKILL"]
```

`RubricScores` 的五个字段均为 `int = Field(ge=1, le=5)`；`RubricFlags`
包含 `fabricated_source`、`conclusion_changing_error`、
`abstract_only_core_evidence`、`unsupported_novelty`、
`process_parameters_only` 五个布尔值。`ExpertFeedbackDraft` 包含 scores、flags、
`fatal_issue`、`comments`、`priority_corrections`、`preserved_strengths`、
`recommended_actions`。

`EvolutionTask` 使用 `schema_version=1`、`revision=0`、不可变 goal/target、`task_group_id`、
当前状态、`current_version`、`last_completed_version`、`accepted_version` 及 episode、feedback、
compilation、revision、strategy、comparison、experience 的 ID 列表。`EpisodeRecord` 保存 parent/applied IDs、runtime session、
execution mode、strategy、`scientific_state_path`、summary、artifact、cost 和验收结果。
时间字段统一使用 UTC aware datetime factory。

ID 由程序生成：`evo_<UTC timestamp>_<6 hex>`、`ep_<10 hex>`、
`fb_<10 hex>`、`rp_<10 hex>`、`strategy_<10 hex>`；对 CLI 输入的 ID 使用
`^[A-Za-z0-9_-]+$` 验证函数 `validate_managed_id(value: str) -> str`。

在 `rubric.py` 中实现 `RUBRIC_VERSION = "expert-review-v1"`、中文量表常量、
`HardCapAssessment`、`assess_hard_caps()` 和：

```python
def expert_utility(scores: RubricScores) -> float:
    normalized = lambda value: (value - 1) / 4
    return round(
        0.35 * normalized(scores.scientific_correctness)
        + 0.30 * normalized(scores.evidence_sufficiency)
        + 0.15 * normalized(scores.novelty)
        + 0.20 * normalized(scores.actionability),
        6,
    )
```

- [ ] **Step 4: 运行模型测试和 mypy 窄检查**

Run: `uv run pytest -q tests/test_evolution_models.py`

Expected: PASS。

Run: `uv run mypy src/photomatagent/scientific/evolution/models.py src/photomatagent/scientific/evolution/rubric.py`

Expected: PASS。

- [ ] **Step 5: 只提交 Task 1 文件**

```bash
git add src/photomatagent/scientific/evolution/__init__.py \
  src/photomatagent/scientific/evolution/models.py \
  src/photomatagent/scientific/evolution/rubric.py \
  tests/test_evolution_models.py
git commit -m "feat: add evolution domain models and expert rubric"
```

---

### Task 2: Workspace 内的原子 EvolutionStore

**Files:**
- Create: `src/photomatagent/scientific/evolution/store.py`
- Create: `tests/test_evolution_store.py`
- Modify: `src/photomatagent/scientific/evolution/__init__.py`

**Interfaces:**
- Consumes: Task 1 的全部持久化模型、`Workspace`、`redact_secrets()`。
- Produces: `EvolutionStore(workspace)`，以及 `create_task`、`load_task`、
  `save_task(task: EvolutionTask, expected_revision: int)`、`write_episode`、`write_feedback`、
  `write_revision`、`write_strategy`、`write_scientific_state`、`list_tasks`。

- [ ] **Step 1: 写原子性、不可变和路径逃逸测试**

```python
def test_store_round_trip_and_revision_conflict(tmp_path, target):
    store = EvolutionStore(Workspace(tmp_path))
    task = make_task(target)
    store.create_task(task)
    loaded = store.load_task(task.evolution_id)
    saved = store.save_task(loaded, expected_revision=0)
    assert saved.revision == 1
    with pytest.raises(EvolutionConflictError):
        store.save_task(loaded, expected_revision=0)


def test_store_refuses_to_overwrite_immutable_episode(tmp_path, task, episode):
    store = EvolutionStore(Workspace(tmp_path))
    store.create_task(task)
    store.write_episode(episode)
    with pytest.raises(EvolutionAlreadyExistsError):
        store.write_episode(episode)


@pytest.mark.parametrize("bad", ["../escape", "/tmp/escape", "a/b"])
def test_store_rejects_unmanaged_ids(tmp_path, bad):
    store = EvolutionStore(Workspace(tmp_path))
    with pytest.raises(ValueError):
        store.load_task(bad)
```

同时模拟 `os.replace` 前异常，断言正式 JSON 未出现半文件。

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_store.py`

Expected: FAIL，`EvolutionStore` 尚不存在。

- [ ] **Step 3: 实现 Store**

Store 根目录固定为 `workspace.resolve(".photomatagent/evolutions", must_exist=False)`。
每次写入执行：序列化 → `redact_secrets` → 同目录临时文件 → flush →
`os.fsync` → `os.replace`。不可变记录在目标存在时抛
`EvolutionAlreadyExistsError`。

`save_task` 必须重新读取磁盘版本并比较 `expected_revision`，然后用
`model_copy(update={"revision": current.revision + 1, "updated_at": now})`
写入。任务锁用 `<task-dir>/.lock` 的
`os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)`；5 秒内无法
获得锁则抛 `EvolutionLockError`，finally 必须释放。

科学状态写到 `episodes/v001.scientific.json`，使用 `ScientificState.model_dump(mode="json")`，
读取使用 `ScientificState.model_validate_json()`。

- [ ] **Step 4: 运行 Store 与 Workspace 回归测试**

Run: `uv run pytest -q tests/test_evolution_store.py tests/test_workspace.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/scientific/evolution/store.py \
  src/photomatagent/scientific/evolution/__init__.py \
  tests/test_evolution_store.py
git commit -m "feat: persist versioned evolution tasks atomically"
```

---

### Task 3: 演化事件接入现有 JSONL 协议

**Files:**
- Modify: `src/photomatagent/runtime/events.py`
- Create: `src/photomatagent/scientific/evolution/events.py`
- Create: `tests/test_evolution_events.py`
- Modify: `tests/test_event_logger.py`

**Interfaces:**
- Consumes: `RuntimeEvent`、`EventLogger.log()`。
- Produces: 设计文档列出的 11 个 evolution typed events，并由 `parse_event()` 回读。

- [ ] **Step 1: 写事件 round-trip 与敏感原文排除测试**

```python
def test_evolution_events_round_trip():
    event = ExpertFeedbackRecorded(
        evolution_id="evo_test",
        episode_version="v001",
        feedback_id="fb_test",
        result_sha256="a" * 64,
        scores={"overall": 3},
    )
    parsed = parse_event(event.model_dump(mode="json"))
    assert parsed.kind == "expert_feedback_recorded"
    assert not hasattr(parsed, "raw_comments")


@pytest.mark.asyncio
async def test_evolution_event_is_redacted_in_jsonl(tmp_path):
    logger = EventLogger(tmp_path, session_id="evolution")
    await logger.log(EvolutionTaskCreated(evolution_id="evo_test", goal_summary="safe"))
    assert logger.read_events()[0].kind == "evolution_task_created"
```

- [ ] **Step 2: 运行测试并确认 parse_event 拒绝新 kind**

Run: `uv run pytest -q tests/test_evolution_events.py tests/test_event_logger.py`

Expected: FAIL，discriminated union 中不存在新事件。

- [ ] **Step 3: 定义并注册事件**

在 `runtime/events.py` 定义并加入 `AnyRuntimeEvent` union：

```python
EvolutionTaskCreated
EvolutionEpisodeStarted
EvolutionEpisodeCompleted
ExpertFeedbackRecorded
ExpertFeedbackCompiled
RevisionPlanConfirmed
EvolutionIterationStarted
EvolutionComparisonCompleted
ExperienceStateChanged
EvolutionTaskAccepted
EvolutionTaskStopped
```

每个事件都含 `evolution_id`；涉及 Episode 时含 `episode_version`；只允许有界 summary、
分数和 hash，禁止完整反馈文本。在 evolution/events.py 提供小型构造 helper，统一截断
summary 到 240 字符。

- [ ] **Step 4: 运行事件、日志和可观测性回归测试**

Run: `uv run pytest -q tests/test_evolution_events.py tests/test_event_logger.py tests/test_observability.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/runtime/events.py \
  src/photomatagent/scientific/evolution/events.py \
  tests/test_evolution_events.py tests/test_event_logger.py
git commit -m "feat: add typed evolution lifecycle events"
```

---

### Task 4: EvolutionService 生命周期和严格状态转换

**Files:**
- Create: `src/photomatagent/scientific/evolution/service.py`
- Create: `tests/test_evolution_service.py`
- Modify: `src/photomatagent/scientific/evolution/__init__.py`

**Interfaces:**
- Consumes: `EvolutionStore`、Task 1 模型、evolution event sink。
- Produces: `EvolutionService.create_task()`、`reserve_episode()`、
  `mark_episode_running()`、`complete_episode()`、`fail_episode()`、
  `attach_feedback()`、`confirm_revision()`、`accept()`、`stop()`、`reopen()`。

- [ ] **Step 1: 写状态机失败测试**

```python
def test_lifecycle_requires_feedback_before_next_episode(service, target):
    task = service.create_task(goal="goal", target=target)
    first = service.reserve_episode(task.evolution_id, mode="NORMAL")
    service.mark_episode_running(task.evolution_id, first.version)
    service.complete_episode(task.evolution_id, first.version, result=completed_result())
    assert service.get(task.evolution_id).status == "AWAITING_EXPERT_FEEDBACK"
    with pytest.raises(InvalidEvolutionTransition):
        service.reserve_episode(task.evolution_id, mode="CARRY_VERIFIED_EVIDENCE")


def test_failed_next_episode_never_overwrites_last_good_result(service, completed_task):
    second = service.reserve_episode(
        completed_task.evolution_id,
        mode="CARRY_VERIFIED_EVIDENCE",
    )
    service.fail_episode(completed_task.evolution_id, second.version, "provider failed")
    task = service.get(completed_task.evolution_id)
    assert task.current_version == "v002"
    assert task.last_completed_version == "v001"
```

增加：错误版本 feedback、重复 active feedback、未确认 revision、accepted 后 iterate、
reopen 保留历史、并发 revision 冲突测试。

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_service.py`

Expected: FAIL，service 尚不存在。

- [ ] **Step 3: 实现显式 transition table**

不要散落字符串 if。定义：

```python
ALLOWED_TRANSITIONS = {
    "CREATED": {"RUNNING", "STOPPED"},
    "RUNNING": {"AWAITING_EXPERT_FEEDBACK", "BLOCKED", "BUDGET_EXHAUSTED"},
    "AWAITING_EXPERT_FEEDBACK": {"FEEDBACK_RECORDED", "ACCEPTED", "STOPPED"},
    "FEEDBACK_RECORDED": {"REVISION_READY", "STOPPED"},
    "REVISION_READY": {"RUNNING", "STOPPED"},
    "ACCEPTED": {"AWAITING_EXPERT_FEEDBACK"},
    "STOPPED": {"AWAITING_EXPERT_FEEDBACK"},
    "BUDGET_EXHAUSTED": {"AWAITING_EXPERT_FEEDBACK", "STOPPED"},
    "BLOCKED": {"AWAITING_EXPERT_FEEDBACK", "STOPPED"},
}
```

所有 public mutation 都：加载 → 验证目标版本/哈希/状态 → 写 immutable record →
用 expected revision 更新 task → 发有界事件。事件 sink 为可选
`Callable[[RuntimeEvent], Awaitable[None] | None]`；同步 service 方法只生成事件并返回，
由调用方统一落盘，避免在 Store 中隐藏 event loop。

内层 `ScientificLoopSummary.status` 即使是 STALLED、INCONCLUSIVE 或 BUDGET_EXHAUSTED，只要
产生了可审阅主结果，Episode 仍可标记 COMPLETED，EvolutionTask 进入
AWAITING_EXPERT_FEEDBACK。EvolutionTask 自身的 BUDGET_EXHAUSTED 只表示跨 Episode 的演化
预算耗尽，不能与内层 scientific verdict 混用。

- [ ] **Step 4: 运行状态机、Store 和事件测试**

Run: `uv run pytest -q tests/test_evolution_service.py tests/test_evolution_store.py tests/test_evolution_events.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/scientific/evolution/service.py \
  src/photomatagent/scientific/evolution/__init__.py \
  tests/test_evolution_service.py
git commit -m "feat: enforce evolution task lifecycle"
```

---

### Task 5: Episode 结果固化和科学闭环执行器

**Files:**
- Create: `src/photomatagent/scientific/evolution/artifacts.py`
- Create: `src/photomatagent/scientific/evolution/executor.py`
- Create: `tests/test_evolution_artifacts.py`
- Create: `tests/test_evolution_executor.py`

**Interfaces:**
- Consumes: 已 reserve 的 `EpisodeRecord`、`AgentRuntime`、`ScientificLoopController`、
  `ScientificLoopConfig`、可选 `ScientificJudge`、`EvolutionStore`。
- Produces: `EpisodeArtifactCollector.observe(event)`、`materialize_primary_result()`、
  `ScientificEpisodeExecutor.execute(*, task: EvolutionTask, episode: EpisodeRecord,
  runtime: AgentRuntime, config: ScientificLoopConfig, revision: RevisionPlan | None = None,
  judge: ScientificJudge | None = None, on_event: EventSink | None = None)
  -> EpisodeExecutionResult`。

- [ ] **Step 1: 写主结果与独立 runtime 测试**

```python
def test_fallback_result_is_last_nonempty_assistant_text(tmp_path):
    conversation = ConversationState(messages=[
        UserMessage(content="goal"),
        AssistantMessage(text="final report"),
    ])
    artifact = materialize_primary_result(
        workspace=Workspace(tmp_path),
        evolution_id="evo_test",
        version="v001",
        conversation=conversation,
        collector=EpisodeArtifactCollector(),
    )
    path = Workspace(tmp_path).resolve(artifact.path)
    assert path.read_text(encoding="utf-8") == "final report\n"
    assert artifact.sha256 == sha256_file(path)


@pytest.mark.asyncio
async def test_executor_only_runs_tools_through_agent_runtime(tmp_path, fake_runtime):
    result = await executor.execute(
        task=task,
        episode=episode,
        runtime=fake_runtime,
        config=ScientificLoopConfig(max_rounds=1),
    )
    assert result.runtime_session_id == fake_runtime.session_id
    assert result.artifact.path.endswith("v001/result.md")
    assert result.scientific_summary is not None
```

补充测试：预分配目录之外的 write 事件不能成为主结果；失败执行不生成 COMPLETED；
同一 `result.md` 不得被覆盖。

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_artifacts.py tests/test_evolution_executor.py`

Expected: FAIL，artifact collector/executor 尚不存在。

- [ ] **Step 3: 实现执行器**

`EpisodeArtifactCollector` 观察 controller 转发的全部事件，用 tool_call_id 关联
`ToolCallCompleted(name in {"write", "edit"})` 与成功 `ToolCompleted`。只接受
`user_output/<evolution-id>/<version>/` 下通过 Workspace.resolve 的路径。

主结果选择顺序固定为：

1. 成功写入的预定 `user_output/<evolution-id>/<version>/result.md`；
2. 最后一个非空 AssistantMessage 固化为该 `result.md`；
3. 两者都不存在则抛 `MissingEpisodeResultError`。

`ScientificEpisodeExecutor.execute` 必须构造现有 `ScientificLoopController`，把 logger
加入 controller sinks，迭代 `controller.run(goal=instruction)`，同时把每个事件交给
collector 和 CLI renderer callback。执行结束保存 ScientificState 快照、Budget snapshot、
summary 和 ArtifactRef。执行器不能调用 `Tool.execute()`。

- [ ] **Step 4: 运行执行器和原 scientific loop 回归测试**

Run: `uv run pytest -q tests/test_evolution_artifacts.py tests/test_evolution_executor.py tests/test_scientific_loop_controller.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/scientific/evolution/artifacts.py \
  src/photomatagent/scientific/evolution/executor.py \
  tests/test_evolution_artifacts.py tests/test_evolution_executor.py
git commit -m "feat: execute and materialize evolution episodes"
```

---

### Task 6: `evolve start/list/status/history` CLI

**Files:**
- Create: `src/photomatagent/cli/evolve.py`
- Modify: `src/photomatagent/cli/app.py` at Typer sub-app registration and imports
- Create: `tests/test_evolution_cli.py`

**Interfaces:**
- Consumes: EvolutionService/Store、ScientificEpisodeExecutor、现有 `build_runtime()`、
  `resolve_loop_target()`、`_build_judge()`、Rich Console。
- Produces: `evolve_app` 及 start/list/status/history 子命令。

- [ ] **Step 1: 写 CLI 帮助、创建失败保留和只读命令测试**

```python
def test_evolve_help_is_registered(cli_runner):
    result = cli_runner.invoke(app, ["evolve", "--help"])
    assert result.exit_code == 0
    assert "start" in result.stdout
    assert "feedback" in result.stdout
    assert "iterate" in result.stdout


def test_start_requires_machine_verifiable_target(cli_runner, tmp_path):
    result = cli_runner.invoke(app, ["evolve", "start", "--goal", "design material",
                                     "--workspace", str(tmp_path)])
    assert result.exit_code != 0
    assert "target" in result.stdout.lower()


def test_status_prints_exact_next_command(cli_runner, stored_awaiting_task, tmp_path):
    result = cli_runner.invoke(app, ["evolve", "status", stored_awaiting_task.evolution_id,
                                     "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "AWAITING_EXPERT_FEEDBACK" in result.stdout
    assert f"evolve feedback {stored_awaiting_task.evolution_id}" in result.stdout
```

- [ ] **Step 2: 运行测试并确认命令不存在**

Run: `uv run pytest -q tests/test_evolution_cli.py`

Expected: FAIL，Typer 无 `evolve` group。

- [ ] **Step 3: 实现 CLI 薄层**

`evolve.py` 定义
`evolve_app = typer.Typer(help="Run persistent expert-feedback evolution tasks.")`。
`start` 解析 target 后先调用
service.create_task/reserve，再创建 runtime 并调用 executor；异常路径调用
`fail_episode`，绝不删除 task。provider/model/approval/max_rounds/patience/
min_confidence/judge 参数与现有 `loop` 对齐。

`list/status/history` 只读 Store，不创建 provider。Rich 表格必须显示 ID、状态、当前版本、
最后成功版本、反馈/修订数量和 next command。

在 `app.py` 只做：

```python
from photomatagent.cli.evolve import evolve_app
app.add_typer(evolve_app, name="evolve")
```

若产生循环导入，把对 `build_runtime`/`_build_judge` 的 import 放在命令函数内部，不移动
现有 runtime 工厂。

- [ ] **Step 4: 运行 CLI 和原命令回归测试**

Run: `uv run pytest -q tests/test_evolution_cli.py tests/test_chat_commands.py tests/test_loop.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/cli/evolve.py src/photomatagent/cli/app.py \
  tests/test_evolution_cli.py
git commit -m "feat: expose persistent evolution tasks in CLI"
```

---

### Task 7: 专家表单、JSON 导入和反馈不可变记录

**Files:**
- Modify: `src/photomatagent/cli/evolve.py`
- Modify: `src/photomatagent/scientific/evolution/service.py`
- Create: `tests/test_evolution_feedback_entry.py`
- Modify: `tests/test_evolution_cli.py`

**Interfaces:**
- Consumes: `PromptSession`、`ExpertFeedbackDraft`、rubric、Episode ArtifactRef。
- Produces: `collect_expert_feedback()`、`load_feedback_file()`、`feedback` CLI 命令。

- [ ] **Step 1: 写输入隔离与取消测试**

```python
@pytest.mark.asyncio
async def test_feedback_form_uses_distinct_prompt_and_submit(scripted_prompt, console):
    scripted_prompt.feed("4", "2", "3", "2", "2", "n", "证据不足", "/submit", "y")
    draft = await collect_expert_feedback(
        session=scripted_prompt,
        console=console,
        evolution_id="evo_test",
        version="v001",
    )
    assert draft.scores.evidence_sufficiency == 2
    assert draft.comments == "证据不足"
    assert "EXPERT FEEDBACK" in scripted_prompt.prompt_history[0]


@pytest.mark.asyncio
async def test_cancel_writes_nothing(scripted_prompt, service, awaiting_task):
    scripted_prompt.feed("/cancel")
    result = await run_feedback_flow(
        session=scripted_prompt,
        console=console,
        service=service,
        evolution_id=awaiting_task.evolution_id,
        version="v001",
    )
    assert result is None
    assert service.get(awaiting_task.evolution_id).feedback_ids == []


def test_import_rejects_feedback_for_changed_artifact(service, review_file):
    with pytest.raises(ArtifactMismatchError):
        service.attach_feedback(
            evolution_id="evo_test",
            version="v001",
            draft=load_feedback_file(review_file),
            result_sha256="0" * 64,
        )
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_feedback_entry.py tests/test_evolution_cli.py`

Expected: FAIL，表单与 feedback 命令不存在。

- [ ] **Step 3: 实现反馈录入**

`collect_expert_feedback` 接收已有 PromptSession，逐项显示 `expert-review-v1` 中文锚点；
分数循环直到得到 1–5；随后逐项询问五个 RubricFlags（默认 `n`），再询问 fatal issue；
评论支持多行 `/submit`；任意位置 `/cancel` 抛内部
`FeedbackEntryCancelled`，调用方安静退出且不写文件。

JSON 导入必须 `ExpertFeedbackDraft.model_validate_json()`，`extra=forbid`。确认画面显示
task/version/hash/五分/flags/评论长度。只有输入 `y` 才调用 `service.attach_feedback`。

Service 验证 Episode COMPLETED、任务处于 AWAITING、hash 当前仍匹配，然后先写 immutable
feedback，再把 task 置为 `FEEDBACK_RECORDED`。此步骤不得拥有 Runtime 或 ModelProvider。

- [ ] **Step 4: 运行反馈、Service 和 CLI 测试**

Run: `uv run pytest -q tests/test_evolution_feedback_entry.py tests/test_evolution_service.py tests/test_evolution_cli.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/cli/evolve.py \
  src/photomatagent/scientific/evolution/service.py \
  tests/test_evolution_feedback_entry.py tests/test_evolution_cli.py
git commit -m "feat: record isolated expert feedback"
```

---

### Task 8: 无工具 FeedbackCompiler 与可重试 `evolve compile`

**Files:**
- Create: `src/photomatagent/scientific/evolution/feedback.py`
- Modify: `src/photomatagent/scientific/evolution/models.py`
- Modify: `src/photomatagent/scientific/evolution/service.py`
- Modify: `src/photomatagent/cli/evolve.py`
- Create: `tests/test_evolution_feedback.py`

**Interfaces:**
- Consumes: `ModelProvider`、`ModelRequest`、原始 feedback、TargetSpec、最多 12,000 字符的结果文本。
- Produces: `FeedbackCompiler.compile(*, task: EvolutionTask, episode: EpisodeRecord,
  feedback: ExpertFeedbackRecord, result_text: str) -> FeedbackCompilation`、`compile` CLI 命令。

- [ ] **Step 1: 写工具隔离、QUERY 保留和失败降级测试**

```python
@pytest.mark.asyncio
async def test_compiler_has_no_tools_and_preserves_query(fake_model, feedback_record):
    fake_model.set_responses([FakeResponse(text=json.dumps({
        "status": "AVAILABLE",
        "items": [{
            "category": "EVIDENCE_SUFFICIENCY",
            "status": "QUERY",
            "severity": "HIGH",
            "responsible_module": "retrieval_planner",
            "problem": "摘要是否足够",
            "requested_actions": ["读取正文"],
            "acceptance_test": "核心结论绑定全文证据",
            "preserve": [],
            "confidence": 0.9,
            "source_span": "目前检索没有本地文献，完全依靠 arXiv 摘要是否足够，正文信息呢？"
        }],
        "warnings": []
    }, ensure_ascii=False))])
    result = await FeedbackCompiler(fake_model).compile(
        task=task,
        episode=episode,
        feedback=feedback_record,
        result_text="report text",
    )
    assert fake_model.requests[0].tools == []
    assert result.items[0].status == "QUERY"


@pytest.mark.asyncio
async def test_invalid_json_degrades_without_losing_raw_feedback(fake_model):
    fake_model.set_responses([FakeResponse(text="not json")])
    result = await FeedbackCompiler(fake_model).compile(
        task=task,
        episode=episode,
        feedback=feedback_record,
        result_text="report text",
    )
    assert result.status == "UNAVAILABLE"
    assert "schema" in result.error or "JSON" in result.error
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_feedback.py`

Expected: FAIL，FeedbackCompiler 尚不存在。

- [ ] **Step 3: 实现 Compiler 和重试语义**

仿照 `scientific/loop/judge.py` 的隔离模式，但不要让 Compiler 导入或修改 Judge。
定义专用 system prompt，明确：只分类专家意见，不判断硬约束 PASS，不把问题改成事实，
严格输出 `FeedbackCompilation` JSON。

请求固定 `tools=[]`。捕获 provider、空文本、JSON 提取和 Pydantic 校验错误，返回
`status="UNAVAILABLE"`，不抛出破坏 workflow 的异常。输出记录 provider/model/error。

`service.save_compilation` 只给既有 immutable feedback 增加独立 compilation 文件引用，
不重写 feedback 原文文件。`evolve compile` 找到 active feedback 并重试；成功编译不得
创建第二条 feedback。编译成功后仍保持 `FEEDBACK_RECORDED`，等待 Task 9 的 plan 确认。

- [ ] **Step 4: 运行 Compiler、Judge 隔离和 CLI 测试**

Run: `uv run pytest -q tests/test_evolution_feedback.py tests/test_scientific_loop_judge.py tests/test_evolution_cli.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/scientific/evolution/feedback.py \
  src/photomatagent/scientific/evolution/models.py \
  src/photomatagent/scientific/evolution/service.py \
  src/photomatagent/cli/evolve.py tests/test_evolution_feedback.py
git commit -m "feat: compile expert feedback without tools"
```

---

### Task 9: RevisionPlan、用户确认和固定策略选择

**Files:**
- Create: `src/photomatagent/scientific/evolution/revision.py`
- Create: `src/photomatagent/scientific/evolution/strategy.py`
- Modify: `src/photomatagent/scientific/evolution/service.py`
- Modify: `src/photomatagent/cli/evolve.py`
- Create: `tests/test_evolution_revision.py`
- Create: `tests/test_evolution_strategy.py`

**Interfaces:**
- Consumes: AVAILABLE FeedbackCompilation、原 TargetSpec、上一 ScientificLoopSummary。
- Produces: `build_revision_plan()`、`format_revision_instruction()`、
  `FixedStrategySelector.select()`、`feedback/compile` 完成后的 plan 预览确认。

- [ ] **Step 1: 写计划确定性和原文隔离测试**

```python
def test_revision_plan_routes_feedback_by_module(compilation, feedback):
    plan = build_revision_plan(feedback=feedback, compilation=compilation)
    assert plan.evidence_requirements == ["读取正文"]
    assert plan.machine_acceptance_tests == ["核心结论绑定全文证据"]
    assert plan.human_acceptance_tests


def test_revision_instruction_contains_no_raw_expert_prose(plan, feedback):
    text = format_revision_instruction(plan, strategy=StrategyArm.EVIDENCE_FIRST)
    assert feedback.raw_input not in text
    assert "Revision requirements" in text
    assert "Do not override deterministic constraints" in text


def test_fixed_selector_prefers_evidence_first_for_high_evidence_issue(plan):
    selected = FixedStrategySelector().select(task, plan)
    assert selected.arm == "EVIDENCE_FIRST"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_revision.py tests/test_evolution_strategy.py`

Expected: FAIL，Planner/Selector 不存在。

- [ ] **Step 3: 实现 Planner 和首版选择规则**

`build_revision_plan` 按 responsible_module/category 分流到 contract deltas、evidence、
output schema、preserve、prohibited repeats、machine/human tests。`QUERY` 默认进入 evidence
或 human acceptance，不能写成已确认事实。存在 CRITICAL 但没有动作/验收条件的 item 时，
plan 标记 `has_blocking_ambiguity=True`。

固定选择优先级：

1. HIGH/CRITICAL evidence issue → `EVIDENCE_FIRST`；
2. HIGH innovation/diversity issue → `DIVERSITY_FIRST`；
3. QUERY 或 uncertainty issue → `UNCERTAINTY_FIRST`；
4. 无有效负向 item → `STATIC`。

同优先级按 CRITICAL 数、HIGH 数、原始顺序确定，禁止随机。

CLI 展示结构化 plan、策略变化、保留内容、禁止项和验收项。用户确认后
`service.confirm_revision` 写 immutable plan/strategy 并把状态置 `REVISION_READY`。
拒绝确认不改变状态。

- [ ] **Step 4: 运行 Planner/Selector/Service 测试**

Run: `uv run pytest -q tests/test_evolution_revision.py tests/test_evolution_strategy.py tests/test_evolution_service.py tests/test_evolution_cli.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/scientific/evolution/revision.py \
  src/photomatagent/scientific/evolution/strategy.py \
  src/photomatagent/scientific/evolution/service.py \
  src/photomatagent/cli/evolve.py \
  tests/test_evolution_revision.py tests/test_evolution_strategy.py
git commit -m "feat: plan confirmed expert-guided revisions"
```

---

### Task 10: Verified evidence 继承和 `evolve iterate`

**Files:**
- Create: `src/photomatagent/scientific/evolution/evidence.py`
- Modify: `src/photomatagent/cli/chat.py` in `build_runtime`
- Modify: `src/photomatagent/scientific/evolution/executor.py`
- Modify: `src/photomatagent/scientific/evolution/service.py`
- Modify: `src/photomatagent/cli/evolve.py`
- Create: `tests/test_evolution_evidence.py`
- Modify: `tests/test_evolution_executor.py`
- Modify: `tests/test_evolution_cli.py`

**Interfaces:**
- Consumes: 上一 Episode ScientificState、RevisionPlan.invalidated_evidence_ids、Task 9 strategy。
- Produces: `select_carry_forward_evidence()`、`build_inherited_scientific_state()`、
  `build_runtime(scientific_state: ScientificState | None = None)`、`iterate` CLI。

- [ ] **Step 1: 写证据过滤和 fresh runtime 测试**

```python
def test_only_verified_structured_evidence_is_carried(previous_state):
    inherited, decisions = build_inherited_scientific_state(
        previous_state,
        source_episode="v001",
        invalidated_evidence_ids={"sev_invalid"},
    )
    ids = {item.id for item in inherited.evidence}
    assert "sev_dft" in ids
    assert "sev_model_prediction" not in ids
    assert "sev_invalid" not in ids
    assert all(item.provenance["inherited_from_episode"] == "v001"
               for item in inherited.evidence)


def test_build_runtime_can_bind_inherited_scientific_state(tmp_path):
    inherited = ScientificState(goal="same task")
    runtime, _ = build_runtime(
        provider="fake", workspace_root=tmp_path, approval="deny",
        scientific_state=inherited,
    )
    assert runtime.scientific_state is inherited
```

增加 iterate 创建不同 runtime session ID、旧 ConversationState 不出现、原 feedback prose
不出现、硬约束重新求值、HPC approval 未被继承的测试。

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_evidence.py tests/test_evolution_executor.py tests/test_evolution_cli.py`

Expected: FAIL，carry-forward 和 iterate 尚不存在。

- [ ] **Step 3: 实现保守继承和 iterate**

允许继承：

- `ScientificEvidence` 且 source_type 不为 `model`/`generative_model`、fidelity 不为
  `ml_generated`；
- 普通 `Evidence` 仅当 `provenance["validated"] is True`；
- ID 不在 RevisionPlan.invalidated_evidence_ids；
- subject 兼容检查通过。

复制 evidence 时保留 ID，并在 provenance 加 `inherited_from_episode` 和
`inherited_at`。Claims、pending tasks、旧 conversation 和未验证 hypotheses 不继承。

给 `build_runtime` 增加可选参数：

```python
scientific_state: ScientificState | None = None
```

并使用 `scientific = scientific_state or ScientificState()`；Registry 必须绑定同一个实例。

`iterate` 只接受 `REVISION_READY`，先 reserve vNNN，再构造 inherited state、新 runtime、
bounded RevisionInstruction，最后交给 executor。任何权限和应用级审批重新走现有路径。

- [ ] **Step 4: 运行迭代与边界回归测试**

Run: `uv run pytest -q tests/test_evolution_evidence.py tests/test_evolution_executor.py tests/test_evolution_cli.py tests/test_permissions.py tests/test_scientific_loop_policy.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/scientific/evolution/evidence.py \
  src/photomatagent/cli/chat.py \
  src/photomatagent/scientific/evolution/executor.py \
  src/photomatagent/scientific/evolution/service.py \
  src/photomatagent/cli/evolve.py \
  tests/test_evolution_evidence.py tests/test_evolution_executor.py tests/test_evolution_cli.py
git commit -m "feat: iterate with verified evidence carry-forward"
```

---

### Task 11: Episode 比较、问题闭合和 Experience 生命周期

**Files:**
- Create: `src/photomatagent/scientific/evolution/comparison.py`
- Create: `src/photomatagent/scientific/evolution/experience.py`
- Modify: `src/photomatagent/scientific/evolution/store.py`
- Modify: `src/photomatagent/scientific/evolution/service.py`
- Modify: `src/photomatagent/cli/evolve.py`
- Create: `tests/test_evolution_comparison.py`

**Interfaces:**
- Consumes: 相邻 Episode、上一 RevisionPlan、两轮 feedback（右侧可为空）、cost snapshots。
- Produces: `compare_episodes()`、`compute_learning_signal()`、`promote_experience()`、
  compare CLI 和 immutable comparison/experience 文件。

- [ ] **Step 1: 写闭合不能靠“没再提”判定的测试**

```python
def test_missing_repeated_comment_does_not_auto_close_human_check():
    report = compare_episodes(
        previous=episode_v1,
        current=episode_v2,
        previous_plan=plan_with_human_only_check,
        previous_feedback=feedback_v1,
        current_feedback=None,
    )
    assert report.acceptance_results[0].status == "NEEDS_HUMAN_REVIEW"
    assert report.closure_rate is None


def test_machine_acceptance_pass_closes_issue():
    report = compare_episodes(
        previous=episode_v1,
        current=episode_v2,
        previous_plan=plan_with_machine_check,
        previous_feedback=feedback_v1,
        current_feedback=None,
        machine_results={"process_steps_complete": True},
    )
    assert report.closed_issue_ids == ["issue_process"]


def test_one_review_creates_observation_not_skill():
    exp = create_experience(comparison)
    assert exp.maturity == "OBSERVATION"
    with pytest.raises(ExperiencePromotionError):
        promote_experience(exp, to="REUSABLE_SKILL", evidence=[])
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_comparison.py`

Expected: FAIL，comparison/experience 不存在。

- [ ] **Step 3: 实现比较与学习信号**

机器 test 有 PASS/FAIL 才进入 closure 分母；human-only 是 NEEDS_HUMAN_REVIEW。
复发按同 category/responsible_module 且后续专家再次提出计算；新问题是当前反馈中未在上一
反馈出现的 signature。

学习信号固定为：

```python
reward = clip(
    0.45 * expert_utility_delta
    + 0.25 * closure_rate
    - 0.15 * recurrence_rate
    - 0.10 * new_issue_rate
    - 0.05 * normalized_cost_increase,
    -1.0,
    1.0,
)
```

某项未知时不填 0，而是将可用权重重新归一化，并记录 `components_used`。

同时生成 `module_credit: dict[str, float]`：按 FeedbackDelta.responsible_module 分组，
对每组使用该组问题的 closure、recurrence、new-issue 和 severity 计算 `[-1, 1]` 信号。
模块信号用于论文中的 credit-assignment 分析和 Task 14 的特征，不得虚增 observation 数；
同一 Episode 对仍然只是一条 task-group observation。

Experience 晋升规则：一次比较为 OBSERVATION；至少两个 distinct task group 命中且均有
正 reward 才可 HYPOTHESIS；至少五个 distinct tasks、平均 reward > 0、无 safety/fabrication
失败才能 VALIDATED_EXPERIENCE；REUSABLE_SKILL 还要求显式用户批准，本任务只记录批准，
不得修改 Skill 文件。

- [ ] **Step 4: 运行比较、Store 和 Service 测试**

Run: `uv run pytest -q tests/test_evolution_comparison.py tests/test_evolution_store.py tests/test_evolution_service.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/scientific/evolution/comparison.py \
  src/photomatagent/scientific/evolution/experience.py \
  src/photomatagent/scientific/evolution/store.py \
  src/photomatagent/scientific/evolution/service.py \
  src/photomatagent/cli/evolve.py tests/test_evolution_comparison.py
git commit -m "feat: compare episodes and mature experiences"
```

---

### Task 12: Fresh evaluation、导出和接受/停止/重开

**Files:**
- Modify: `src/photomatagent/cli/evolve.py`
- Modify: `src/photomatagent/scientific/evolution/service.py`
- Modify: `src/photomatagent/scientific/evolution/executor.py`
- Modify: `src/photomatagent/scientific/evolution/store.py`
- Modify: `tests/test_evolution_cli.py`
- Create: `tests/test_evolution_fresh_evaluation.py`

**Interfaces:**
- Consumes: 冻结的原始 task、显式 strategy snapshot、空白 ScientificState。
- Produces: `run_fresh_evaluation(*, service: EvolutionService, task: EvolutionTask,
  strategy_id: str, executor: ScientificEpisodeExecutor, runtime_factory: RuntimeFactory)`、
  `evaluate --fresh`、`export`、`accept`、`stop`、`reopen`。

- [ ] **Step 1: 写泄漏隔离和导出测试**

```python
@pytest.mark.asyncio
async def test_fresh_evaluation_excludes_task_specific_history(
    executor, evolved_task, service, recording_runtime_factory
):
    result = await run_fresh_evaluation(
        service=service,
        task=evolved_task,
        strategy_id="strategy_baseline",
        executor=executor,
        runtime_factory=recording_runtime_factory,
    )
    assert result.episode.execution_mode == "FRESH_EVALUATION"
    assert recording_runtime_factory.initial_evidence == []
    assert "expert raw comment" not in serialized_requests(
        recording_runtime_factory.model
    )


def test_export_contains_provenance_but_redacts_secrets(service, evolved_task, tmp_path):
    path = service.export_evolution(
        evolved_task.evolution_id,
        output=tmp_path / "export.json",
        include_content=False,
    )
    payload = json.loads(path.read_text())
    assert payload["task"]["evolution_id"] == evolved_task.evolution_id
    assert "api-key" not in path.read_text().lower()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_fresh_evaluation.py tests/test_evolution_cli.py`

Expected: FAIL，fresh/export 命令不存在。

- [ ] **Step 3: 实现控制命令**

Fresh evaluation 创建独立 Episode，mode 为 FRESH_EVALUATION，parent 可记录用于比较但
`applied_feedback_id=None`、`revision_plan_id=None`，ScientificState 全新。只允许指定在
评估开始前已经存在的 StrategyVersion；把 strategy hash 和 cutoff timestamp 固化到 Episode。

Export 聚合 task、episodes、feedback metadata、compilations、revision、comparisons、
experiences、事件引用和 artifact hashes；默认不内嵌完整结果和原始专家文本，使用
`--include-content` 才包含，并继续 secret-redact。

accept 必须指向 COMPLETED artifact；stop 保留当前状态；reopen 将 ACCEPTED/STOPPED/
BLOCKED/BUDGET_EXHAUSTED 恢复为与最后完成 Episode 相符的等待状态。

- [ ] **Step 4: 运行 fresh、CLI、session 隔离测试**

Run: `uv run pytest -q tests/test_evolution_fresh_evaluation.py tests/test_evolution_cli.py tests/test_chat_commands.py tests/test_observability.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/cli/evolve.py \
  src/photomatagent/scientific/evolution/service.py \
  src/photomatagent/scientific/evolution/executor.py \
  src/photomatagent/scientific/evolution/store.py \
  tests/test_evolution_cli.py tests/test_evolution_fresh_evaluation.py
git commit -m "feat: add fresh evaluation and evolution controls"
```

---

### Task 13: `/evolve` 聊天快捷入口且不污染普通输入

**Files:**
- Modify: `src/photomatagent/cli/commands.py` in COMMANDS, `_CLI_GROUPS`, `execute`
- Modify: `src/photomatagent/cli/chat.py` in `run_interactive_chat`
- Modify: `src/photomatagent/cli/evolve.py`
- Modify: `tests/test_chat_commands.py`
- Create: `tests/test_evolution_chat_boundary.py`

**Interfaces:**
- Consumes: 当前 PromptSession、`ChatCommandRouter`、同一个 evolve service/CLI handlers。
- Produces: `/evolve` list/status/history/start/feedback/compile/iterate 快捷入口。

- [ ] **Step 1: 写普通文本与 slash command 边界测试**

```python
@pytest.mark.asyncio
async def test_evolve_slash_command_never_calls_runtime_run(router, monkeypatch):
    called = False
    async def forbidden_run(*args, **kwargs):
        nonlocal called
        called = True
        if False:
            yield None
    monkeypatch.setattr(router.runtime, "run", forbidden_run)
    await router.execute("/evolve status evo_test")
    assert called is False


@pytest.mark.asyncio
async def test_normal_expert_sentence_remains_normal_goal(runtime, chat_driver):
    await chat_driver.send("专家说证据不足")
    user_messages = [m.content for m in runtime.conversation_state.messages
                     if isinstance(m, UserMessage)]
    assert user_messages[-1] == "专家说证据不足"
```

增加 `/help` 显示 `/evolve`、feedback 复用当前 PromptSession、`/cancel` 不保存测试。

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_chat_boundary.py tests/test_chat_commands.py`

Expected: FAIL，router 未注册 evolve。

- [ ] **Step 3: 实现快捷路由**

给 `ChatCommandRouter.__init__` 增加可选 `prompt_session`，由
`run_interactive_chat` 传入当前 session。非交互 evolve 子命令仍可复用 `_run_cli`；
feedback/compile 的交互流程直接 await `cli.evolve` 中的共享 async handler，避免
`CliRunner` 在后台线程等待 stdin。

路由判定仍以首 token `/evolve` 为准。禁止扫描普通文本中的“专家”“反馈”等关键词。
`COMMANDS` 加入专用说明。

- [ ] **Step 4: 运行聊天、CLI、resume 回归测试**

Run: `uv run pytest -q tests/test_evolution_chat_boundary.py tests/test_chat_commands.py tests/test_evolution_cli.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/cli/commands.py src/photomatagent/cli/chat.py \
  src/photomatagent/cli/evolve.py tests/test_chat_commands.py \
  tests/test_evolution_chat_boundary.py
git commit -m "feat: isolate evolution shortcuts from normal chat"
```

---

### Task 14: 有启用门槛的 Bayesian Linear Thompson Sampling

**Files:**
- Modify: `src/photomatagent/scientific/evolution/strategy.py`
- Modify: `src/photomatagent/scientific/evolution/experience.py`
- Modify: `src/photomatagent/scientific/evolution/store.py`
- Modify: `src/photomatagent/cli/evolve.py`
- Modify: `tests/test_evolution_strategy.py`
- Create: `tests/test_evolution_bayesian_selector.py`

**Interfaces:**
- Consumes: immutable StrategyObservation、task_group_id、TaskContext、Task 11 reward。
- Produces: `TaskContext.from_target()`、`feature_vector()`、
  `BayesianLinearStrategySelector.fit/select()`、posterior snapshot 和诊断 CLI。

- [ ] **Step 1: 写冷启动、distinct-task 门槛和可复现测试**

```python
def test_bayesian_selector_stays_disabled_below_data_gate():
    selector = BayesianLinearStrategySelector(seed=7)
    selector.fit(observations_from_same_task(count=25))
    assert selector.enabled is False
    assert selector.diagnostics.distinct_tasks == 1


def test_selector_enables_at_twenty_observations_and_eight_tasks():
    selector = BayesianLinearStrategySelector(seed=7)
    selector.fit(observations(count=20, distinct_tasks=8))
    assert selector.enabled is True


def test_thompson_sampling_is_reproducible_for_fixed_seed():
    left = fitted_selector(seed=23)
    right = fitted_selector(seed=23)
    assert left.select(context).arm == right.select(context).arm
```

增加 posterior covariance 对称正定、非法 NaN reward 拒绝、未启用时退回 FixedSelector、
高 reward arm 在重复种子样本中更常被选中测试。

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run pytest -q tests/test_evolution_strategy.py tests/test_evolution_bayesian_selector.py`

Expected: FAIL，Bayesian selector 尚不存在。

- [ ] **Step 3: 用现有 NumPy 实现后验**

TaskContext 使用六个有界数值特征：intercept、hard constraint count/10、soft constraint
count/10、objective count/10、operating-condition count/10、上一 Episode critical-gap
count/10，全部 clip 到 `[0, 1]`。Feature vector 拼接 context、4 维 arm one-hot 和
`context[1:] * arm_one_hot` interactions；生成顺序写成常量并存入 posterior schema。

使用 Bayesian ridge 形式：

```python
precision = prior_precision * np.eye(d) + (X.T @ X) / noise_variance
covariance = np.linalg.inv(precision)
mean = covariance @ ((X.T @ y) / noise_variance)
sample = rng.multivariate_normal(mean, covariance)
```

默认 `prior_precision=4.0`、`noise_variance=0.25`。只有 observation >= 20 且
distinct task_group_id >= 8 时 `enabled=True`；否则调用 FixedStrategySelector。

Posterior snapshot 保存 feature schema、mean、covariance、超参数、样本数、distinct tasks、
训练 observations hashes 和生成时间。CLI status 显示“fixed baseline”或“Bayesian enabled”，
不得在 disabled 时显示“正在学习完成”。

- [ ] **Step 4: 运行策略、比较和 NumPy 回归测试**

Run: `uv run pytest -q tests/test_evolution_strategy.py tests/test_evolution_bayesian_selector.py tests/test_evolution_comparison.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/photomatagent/scientific/evolution/strategy.py \
  src/photomatagent/scientific/evolution/experience.py \
  src/photomatagent/scientific/evolution/store.py \
  src/photomatagent/cli/evolve.py \
  tests/test_evolution_strategy.py tests/test_evolution_bayesian_selector.py
git commit -m "feat: select evolution strategies with gated Bayesian updates"
```

---

### Task 15: 两轮端到端实验、文档和最终验证

**Files:**
- Create: `tests/test_evolution_end_to_end.py`
- Create: `experiments/expert-feedback-evolution-smoke.json`
- Modify: `README.md` in CLI usage, architecture, and experiments sections
- Modify: `docs/scientific_feedback_loop.md` after P0 limitations
- Modify: `src/photomatagent/scientific/evolution/__init__.py`

**Interfaces:**
- Consumes: Tasks 1–14 的完整公共接口。
- Produces: 离线可重复的 v001→feedback→v002 smoke、用户操作文档和最终验证证据。

- [ ] **Step 1: 写真正跨进程语义的端到端测试**

```python
@pytest.mark.asyncio
async def test_async_expert_feedback_iteration_round_trip(tmp_path):
    first_process = build_test_harness(tmp_path, responses=episode_v1_responses())
    task = await first_process.start(target=target, goal="design window material")
    assert task.status == "AWAITING_EXPERT_FEEDBACK"
    v1_session = first_process.store.load_episode(task.evolution_id, "v001").runtime_session_id

    second_process = build_test_harness(tmp_path, responses=compiler_response())
    await second_process.record_and_compile_feedback(task.evolution_id, expert_review())
    second_process.confirm_revision(task.evolution_id)

    third_process = build_test_harness(tmp_path, responses=episode_v2_responses())
    await third_process.iterate(task.evolution_id)
    reloaded = third_process.store.load_task(task.evolution_id)
    v2 = third_process.store.load_episode(task.evolution_id, "v002")

    assert reloaded.status == "AWAITING_EXPERT_FEEDBACK"
    assert v2.runtime_session_id != v1_session
    assert v2.parent_version == "v001"
    assert v2.applied_feedback_id is not None
    assert v2.artifact.sha256 != ""
    assert "专家原始自由文本" not in serialized_runtime_conversation(v2)
```

本测试文件定义本地 `EvolutionHarness`，构造函数签名为
`EvolutionHarness(workspace: Path, responses: list[FakeResponse])`，并显式持有
`store`、`service`、`model` 和 `executor`。它的 `start`、
`record_and_compile_feedback`、`confirm_revision`、`iterate` 只能组合生产接口，不能在
测试 helper 中直接写 manifest 或更改状态字段。`episode_v1_responses()` 和
`episode_v2_responses()` 返回可被 FakeModelProvider 消费的确定性响应；
`compiler_response()` 返回符合 FeedbackCompilation schema 的 JSON 响应；
`expert_review()` 返回合法 ExpertFeedbackDraft。

同一文件增加：反馈命令不产生 ModelRequest/ToolRequested、普通聊天不更新 store、
deterministic FAIL 不被 5 分覆盖、fresh evaluation 无泄漏、事件全链路可解析测试。

- [ ] **Step 2: 运行端到端测试并修正所有真实失败**

Run: `uv run pytest -q tests/test_evolution_end_to_end.py`

Expected: PASS。不得通过放宽断言、跳过测试或 mock 掉核心 Service/Store 边界来获得 PASS。

- [ ] **Step 3: 增加 smoke 配置和中文使用文档**

README 必须包含以下可复制流程：

```bash
uv run photomatagent evolve start --target-file target.json --goal "生成中红外光窗材料候选与工艺" --provider fake
uv run photomatagent evolve feedback <evolution-id> --version v001
uv run photomatagent evolve iterate <evolution-id> --provider fake
uv run photomatagent evolve compare <evolution-id> v001 v002
uv run photomatagent evolve evaluate <evolution-id> --fresh --provider fake
```

解释 `loop` 是 Episode 内部科学闭环、`evolve` 是跨 Episode 的人类反馈演化层；解释专家
反馈不会作为普通聊天发送。更新 P0 limitations，明确 Bayesian gate、同任务样本相关和
尚未自动晋升 Skill。

Smoke 必须完全离线、无 HPC、固定 FakeModelProvider 轨迹，检查版本、状态、闭合指标和
fresh isolation；如果现有 experiment schema 不适合多阶段流程，使用 Python 测试夹具作为
权威 smoke，并让 JSON 文件只描述输入与期望，不要在 experiment runner 中复制 EvolutionService。

- [ ] **Step 4: 执行分层验证**

先运行 evolution 全集：

```bash
uv run pytest -q tests/test_evolution_models.py tests/test_evolution_store.py \
  tests/test_evolution_events.py tests/test_evolution_service.py \
  tests/test_evolution_artifacts.py tests/test_evolution_executor.py \
  tests/test_evolution_feedback_entry.py tests/test_evolution_feedback.py \
  tests/test_evolution_revision.py tests/test_evolution_evidence.py \
  tests/test_evolution_comparison.py tests/test_evolution_strategy.py \
  tests/test_evolution_bayesian_selector.py tests/test_evolution_cli.py \
  tests/test_evolution_chat_boundary.py tests/test_evolution_fresh_evaluation.py \
  tests/test_evolution_end_to_end.py
```

再运行边界回归：

```bash
uv run pytest -q tests/test_scientific_loop_controller.py \
  tests/test_scientific_loop_judge.py tests/test_scientific_loop_policy.py \
  tests/test_chat_commands.py tests/test_permissions.py tests/test_tool_surface.py \
  tests/test_event_logger.py tests/test_observability.py tests/test_experiments.py
```

最后执行仓库级验证：

```bash
uv run pytest -q
uv run mypy src
git diff --check
git diff --stat
git status --short
```

必须保存并汇报每条命令的准确通过/失败数量。若存在已知的预先失败，分别列出基线失败和
本次新增失败；有新增失败时禁止声称完成。

- [ ] **Step 5: 最终代码审查后提交 Task 15**

检查：没有 `runtime.run(raw_feedback)`；没有 EvolutionService 直接 Tool.execute；没有
对 `.photomatagent/evolutions` 的路径逃逸；没有把同 task episodes 当独立样本；没有
真实网络/HPC 测试；没有自动编辑 Skill。

```bash
git add tests/test_evolution_end_to_end.py \
  experiments/expert-feedback-evolution-smoke.json \
  README.md docs/scientific_feedback_loop.md \
  src/photomatagent/scientific/evolution/__init__.py
git commit -m "docs: complete expert feedback evolution workflow"
```

## DeepSeek-Flash 执行方式

把下面这段连同本计划路径交给执行子智能体：

```text
请先完整阅读 /home/shiqiany/AIagent/PhomatAgent/AGENTS.md、
/home/shiqiany/AIagent/PhomatAgent/docs/superpowers/specs/2026-09-04-expert-feedback-evolution-loop-design.md
和 /home/shiqiany/AIagent/PhomatAgent/docs/superpowers/plans/2026-09-04-expert-feedback-evolution-loop.md。

按计划从 Task 1 开始严格顺序执行，一次只完成一个 Task。每个 Task 都必须先写失败测试、
运行确认失败、实现最小代码、运行指定测试、检查 diff，再只提交该 Task 的文件。你不是
唯一修改仓库的人，不得 reset、checkout、删除、覆盖、重新格式化或提交任何不属于当前
Task 的既有修改；必须适配工作区中的并行变化。禁止真实网络和 HPC 操作。每完成一个
Task 后停下，汇报提交 hash、测试结果、实际改动和任何与计划的偏差，等待主智能体审查
后再继续下一个 Task。
```
