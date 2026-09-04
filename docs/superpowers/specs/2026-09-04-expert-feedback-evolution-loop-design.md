# 专家反馈驱动的智能体演化 LOOP 设计

## 1. 设计目标

PhotomatAgent 已经具备证据引导的科学闭环：`AgentRuntime` 作为 Maker
生成候选并调用工具，`ScientificEvaluator` 作为确定性 Checker 检查科学约束，
可选的 `ScientificJudge` 以只读方式提供模型评审意见。

现有闭环能够改进一次科学任务的内部执行，但还不能完成以下工作：

- 保存外部专家在任务结束后给出的评价；
- 将专家评价精确绑定到某一版结果；
- 在 CLI 已经退出后继续同一个任务；
- 比较同一任务的多个演化版本；
- 从多轮反馈中积累经验，并学习下一次应选择哪种科研策略。

本设计在现有科学闭环外增加一个异步的 Human-in-the-LOOP 演化层。用户可以先运行
任务并关闭 CLI，之后获得专家评价，再把评价录入到对应的结果版本，然后显式启动
下一轮。所有版本都可复现、可审计，并且专家反馈不会与普通聊天输入混淆，也不能
绕过确定性科学检查、运行时权限或 HPC 提交门禁。

第一阶段先实现演化工作流、结构化专家反馈、版本化重跑和结果比较。跨任务贝叶斯
策略学习通过独立接口预留，等积累了足够真实轨迹后再启用。不能在样本不足时宣称
系统已经完成了可靠的贝叶斯学习或模型训练。

## 2. 核心术语

- **演化任务（EvolutionTask）**：一个持久化的科研目标，由 `evolution_id` 唯一标识。
- **执行回合（Episode）**：该目标的一次独立运行，版本号单调递增，如 `v001`。
- **运行时会话（Runtime session）**：一个 Episode 使用的现有 `AgentRuntime` 和事件日志
  会话；每个 Episode 都创建独立会话。
- **专家反馈（ExpertFeedbackRecord）**：指向某个具体 Episode 和结果文件的不可变人工评价。
- **修订计划（RevisionPlan）**：由一份专家反馈编译得到、供下一 Episode 使用的结构化要求。
- **策略版本（StrategyVersion）**：某次 Episode 使用的推理、检索、证据升级、评审和停止策略。
- **经验（Experience）**：由实际轨迹产生的版本化记录；一条经验不能自动升级为 Skill。

## 3. 系统架构位置

```text
CLI：photomatagent evolve / 交互式 /evolve 快捷命令
  → EvolutionService
     ├─ EvolutionStore
     ├─ FeedbackCompiler（隔离、结构化、无工具权限）
     ├─ RevisionPlanner（确定性组装修订合同）
     ├─ StrategySelector
     ├─ ScientificLoopController
     │  ├─ AgentRuntime（模型请求工具执行的唯一权威入口）
     │  ├─ ScientificEvaluator（权威科学 Checker）
     │  └─ ScientificJudge（可选、只读、咨询性质）
     ├─ EpisodeComparator
     └─ ExperienceRepository
```

演化层是应用编排层，不是第二套 Agent Runtime，也不是第二个 ToolRegistry。它可以构造
并调用 `ScientificLoopController`，但不能直接执行工具、科学后端、MCP、SSH、Slurm 或
HPC 提交。

`ScientificJudge` 与人类专家职责不同：

- Judge 在单个 Episode 内评审候选，只提供建议；
- 人类专家在 Episode 完成后异步评审最终结果，并影响下一 Episode；
- 两者都不能把缺少证据或违反硬约束的结果改判为 PASS。

## 4. H-BEAL 算法边界

整个演化过程包含两个时间尺度：

```text
Episode 内部快循环：
候选 → 证据 → 确定性检查 → Judge → 修正或升级

Episode 之间慢循环：
结果 → 人类评价 → 反馈分解 → 策略修订 → 重新运行
```

对演化任务 `t` 的第 `r` 个 Episode：

```text
T_t → y_(t,r) → h_(t,r) → delta_(t,r) → pi_(r+1) → y_(t,r+1)
```

- `y`：本轮结果；
- `h`：专家评价；
- `delta`：从评价编译出的结构化反馈；
- `pi`：下一轮采用的策略。

算法不修改基础模型参数。论文中可以准确声称的是：通过结构化反馈、证据引导的重跑
和经验复用，实现 **policy-level self-evolution**，而不是模型权重自我训练。

首版策略选择器只提供四个固定且可解释的策略臂：

- `STATIC`：当前固定科学闭环行为，作为基线；
- `EVIDENCE_FIRST`：先关闭关键证据缺口，再扩展候选；
- `DIVERSITY_FIRST`：先生成具有差异性的候选，再提高验证保真度；
- `UNCERTAINTY_FIRST`：优先获取最可能改变最终决策的证据。

在启用并验证后验更新之前，策略选择必须明确标记为配置基线，不能宣称已经产生贝叶斯
提升。后续的 Bayesian Linear Thompson Sampling 可以使用低维任务上下文、策略身份、
机器评价、专家分数、问题闭合率、复发率和归一化成本。相同任务的重复 Episode 是相关
样本，不能当作相互独立的科研任务。

## 5. CLI 对外契约

新增 Typer 命令组 `evolve`。现有 `photomatagent loop` 继续表示一次性的自动科学闭环，
行为不变。

```text
photomatagent evolve start [--goal ...] (--target-json ... | --target-file ...) [运行参数]
photomatagent evolve list
photomatagent evolve status <evolution-id>
photomatagent evolve feedback <evolution-id> [--version v001] [--file review.json]
photomatagent evolve compile <evolution-id> [--version v001]
photomatagent evolve iterate <evolution-id> [运行参数]
photomatagent evolve history <evolution-id>
photomatagent evolve compare <evolution-id> <left-version> <right-version>
photomatagent evolve accept <evolution-id> [--version ...]
photomatagent evolve stop <evolution-id>
photomatagent evolve reopen <evolution-id>
photomatagent evolve export <evolution-id> [--output ...]
photomatagent evolve evaluate <evolution-id> --fresh [运行参数]
```

### 5.1 `start`

`start` 必须先创建持久化 EvolutionTask，再运行 `v001`。即使运行失败，任务和失败记录
也必须保留，不能因异常而消失。

Episode 成功执行后，状态变为 `AWAITING_EXPERT_FEEDBACK`。CLI 输出：

- evolution ID；
- Episode 版本；
- runtime session ID；
- 主结果路径；
- 下一条应执行的准确命令。

`--target-file` 与 `--target-json` 一样，必须包含可验证的 `TargetSpec` JSON，不是未经检查
的自然语言目标编译器。`goal` 可以是自然语言，但科学收敛仍然需要显式、机器可验证的
TargetSpec，与现有 scientific loop 保持一致。

### 5.2 `feedback`

`feedback` 只录入、验证并编译专家评价。它绝不启动 Maker、执行工具或提交 HPC 任务。

### 5.3 `iterate`

`iterate` 是显式执行边界。它会：

1. 加载最近一版已经获得专家评价的 Episode；
2. 加载已确认的 RevisionPlan；
3. 计算下一策略版本；
4. 创建全新的 runtime session；
5. 通过正常权限路径运行下一 Episode；
6. 保存新结果和相邻版本比较。

如果原始反馈已经保存，但 Compiler 因 provider、JSON 或 schema 错误而失败，使用
`evolve compile` 重试同一条不可变反馈。该命令不得创建第二份 active review，也不得
运行科学工具；编译成功后仍需经过 RevisionPlan 预览和用户确认。

### 5.4 `evaluate --fresh`

该命令用于受控论文评估。它加载冻结的原始任务和指定的通用策略版本，但排除：

- 该任务历史反馈；
- 该任务以前的答案；
- 从该任务以前 Episode 继承的证据。

它用于判断跨任务迁移能力，避免把“记住一道题”误写成算法泛化。

### 5.5 聊天内快捷命令

交互聊天只增加以下快捷入口：

```text
/evolve start ...
/evolve list
/evolve status <id>
/evolve feedback <id> [--version ...]
/evolve compile <id> [--version ...]
/evolve iterate <id>
/evolve history <id>
```

中央 `ChatCommandRouter` 必须在普通文本到达 `AgentRuntime.run()` 前截获这些命令。
快捷命令调用与独立 CLI 相同的 Typer/Service 接口，不得再实现一套工作流。

## 6. 专家反馈输入设计

专家反馈支持两种方式：

1. CLI 交互式表单；
2. 导入结构化 JSON 文件。

进入交互式表单后必须使用明显不同的提示符：

```text
[EXPERT FEEDBACK | evo_... | v001] scientific_correctness>
[EXPERT FEEDBACK | evo_... | v001] evidence_sufficiency>
[EXPERT FEEDBACK | evo_... | v001] novelty>
[EXPERT FEEDBACK | evo_... | v001] actionability>
[EXPERT FEEDBACK | evo_... | v001] overall>
[EXPERT FEEDBACK | evo_... | v001] comments>
```

每个分数必须是 1–5 的整数。用户可以在当前维度查看评分说明。多行评论只有输入
`/submit` 才结束；`/cancel` 或 Ctrl-C 不写入任何内容。

持久化之前，CLI 必须展示并要求用户确认以下信息：

- 演化任务；
- 被评价的 Episode；
- 专家实际查看的结果文件哈希；
- 五项分数；
- 是否存在致命问题；
- 评论摘要。

普通聊天中的任何自然语言，包括“专家说……”“导师认为……”，都不能创建专家反馈记录
或更新演化策略。系统可以提示使用 `/evolve feedback`，但禁止自动推断说话人的专家角色。

### 6.1 专家评分维度

固定四个分项：

- 科学正确性；
- 证据充分性；
- 创新性；
- 可执行性。

另外设置一个独立的总体等级，用于表示结果是否可以进入下一科研阶段。自由文本字段包括：

- 致命问题；
- 最多三条优先修改项；
- 应保留的正确内容；
- 建议的下一步动作。

评分量表必须带版本号。以后量表发生变化时，旧反馈仍按录入时的原始量表解释。

### 6.2 `expert-review-v1` 量表

| 维度 | 1 分 | 2 分 | 3 分 | 4 分 | 5 分 |
| --- | --- | --- | --- | --- | --- |
| 科学正确性 | 存在根本错误，结果不可使用 | 关键错误可能改变结论，需要大修 | 核心方向合理，但需要重要修正 | 主要结论可靠，仅有局部问题 | 科学逻辑、假设和不确定性处理稳健 |
| 证据充分性 | 无可追溯证据或疑似伪造 | 核心结论依赖摘要、二手资料或无依据预测 | 有相关支持，但仍有重要证据缺口 | 核心结论有全文或一手证据，局限明确 | 多源可审计证据链，妥善处理冲突和缺口 |
| 创新性 | 把成熟结果包装成创新，无定义和基线 | 只有表面变化，无机制或比较 | 组分或工艺创新假设合理，但验证不完整 | 创新类型、基线、机制和比较清晰 | 系统检索后仍成立，并有定量优势与验证路线 |
| 可执行性 | 没有可用流程或下一步 | 只有方向和少数参数 | 有主要步骤，但缺重要原料、设备、控制或质检 | 路线基本可复现，输入、设备、参数和质控齐全 | 接近执行级，包含备选、失败判据、安全和表征 |
| 总体等级 | 拒绝并重做 | 大修 | 完成明确修改后可用 | 小修后可进入下一阶段 | 专家认可，可进入当前任务的下一阶段 |

硬性封顶规则由确定性元数据实现，不能藏在模型判断中：

- 存在伪造来源：证据充分性和总体等级最高 1 分；
- 存在会改变结论的科学错误：科学正确性和总体等级最高 2 分；
- 核心结论只有摘要支持：证据充分性最高 2 分；
- 创新性没有定义、基线或证据：创新性最高 2 分；
- 工艺只有路线名称和少数参数：可执行性最高 2 分。

系统可以建议封顶，但最终仍由专家确认。专家若覆盖系统建议，必须填写原因；建议值、
确认值和原因都保留在 provenance 中。

## 7. 持久化领域模型

### 7.1 `EvolutionTask`

必须包含：

- schema version 和 `evolution_id`；
- 不可变的原始 goal 与 TargetSpec 快照；
- 不可变的任务输入哈希和创建时间；
- 当前生命周期状态；
- 当前 Episode 版本；
- Episode、Feedback、Revision 和 Strategy 的有序引用；
- 已接受的结果版本（如果存在）；
- 用于原子更新的乐观 revision number；
- 创建时间和最后更新时间。

### 7.2 `EpisodeRecord`

必须包含：

- evolution ID、episode ID 和单调递增版本；
- 父 Episode、所应用 Feedback 和 Revision 的引用；
- runtime session ID 和事件日志路径；
- 策略版本；
- 执行模式：`NORMAL`、`CARRY_VERIFIED_EVIDENCE` 或 `FRESH_EVALUATION`；
- 冻结的任务、Target、provider、model、工具表面、能力和相关数据源指纹；
- 起止时间和终态；
- `ScientificLoopSummary` 快照；
- 结果文件路径和内容哈希；
- Token、工具调用、运行时间和可选 HPC 成本；
- 自动验收结果。

每个完成的 Episode 必须有一个由程序确定的主结果文件，不能事后扫描工作区猜测。

- 如果 Maker 明确注册了交付物，Episode runner 将该工作区内文件复制或引用到版本化
  输出目录；
- 如果没有注册交付物，则把最终 assistant response 固化为 `result.md`；
- 进入 `AWAITING_EXPERT_FEEDBACK` 前，保存主文件路径、媒体类型、字节数和 SHA-256。

### 7.3 `ExpertFeedbackRecord`

必须包含：

- 不可变 feedback ID；
- evolution ID、Episode 版本和结果文件哈希；
- rubric version；
- 五个 1–5 分数；
- 致命问题标记和自由文本；
- 优先修改项、应保留内容和下一步动作；
- 原始输入；
- compiler 输出及其 provenance；
- 用户确认时间；
- 发生录入纠错时的 supersession 引用。

已经存在的反馈记录永远不能原地修改。纠错时创建新记录并 supersede 旧记录。第一阶段
每个 Episode 只允许一份 active review。

### 7.4 `FeedbackDelta`

每一条编译后的反馈必须包含：

- 类别：任务定义、科学正确性、证据充分性、创新性、交付完整性、可执行性、安全或其他；
- 状态：`CORRECTION`、`QUERY`、`PREFERENCE` 或 `POSITIVE_SIGNAL`；
- 严重程度；
- 责任模块；
- 问题描述；
- 要求的合同变化或动作；
- 能够机器检查时的 acceptance test；
- 必须保留的内容；
- 解析置信度和原文位置。

专家提出的疑问必须保留为 `QUERY`。Compiler 不能把不确定性擅自改写成事实性错误。

### 7.5 `RevisionPlan`

必须包含：

- 源 Episode 和 active feedback ID；
- 有序的任务/输出合同变化；
- 证据获取和升级要求；
- 输出 schema 要求；
- 需要保留的事实及 evidence ID；
- 禁止重复的失败行为和已失效结论；
- 可机器检查的验收项；
- 只能由人类验收的项目；
- 策略变化及理由；
- 编译警告和未解决歧义；
- 用户是否已经确认计划。

## 8. 存储结构

运行控制状态保存在工作区现有的托管状态区域：

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

用户可读交付物单独保存在：

```text
user_output/<evolution-id>/v001/
user_output/<evolution-id>/v002/
```

所有路径都由 Store 根据程序生成的 ID 构造，并通过 `Workspace.resolve` 校验。CLI 参数
不能任意指定 manifest 路径。JSON 写入必须满足：

- 原子写入；
- schema 版本化；
- revision 冲突检查；
- secret redaction。

专家评价通过 SHA-256 绑定到其实际查看的结果文件。

EvolutionStore 不替代或合并 `ConversationState`、`ScientificState`、
`ScientificLoopState` 或现有 session snapshot。Episode 引用 runtime session，runtime
session 不拥有 EvolutionTask。

## 9. 生命周期和状态转换

```text
CREATED
  → RUNNING
  → AWAITING_EXPERT_FEEDBACK
  → FEEDBACK_RECORDED
  → REVISION_READY
  → RUNNING
  → AWAITING_EXPERT_FEEDBACK

终止或暂停状态：
ACCEPTED | STOPPED | BUDGET_EXHAUSTED | BLOCKED
```

强制规则：

- 只有完成的 Episode 才能接收反馈；
- 反馈必须绑定准确的结果文件哈希；
- 没有 active review 的结果不能 `iterate`；
- 录入反馈不能自动执行修订；
- 存在编译歧义时，必须先解决歧义才能进入 `REVISION_READY`；
- `iterate` 必须先创建下一个单调版本，再启动运行；
- 失败的 Episode 需要保留，不能覆盖上一版成功结果；
- 并发写入使用 EvolutionTask 锁和乐观 revision 检查；
- `accept` 选择用户认可的结果，但不能改写确定性科学 verdict；
- 已终止任务只能通过显式 `reopen` 继续；
- 模型工具调用和 HPC 仍然执行已有 runtime 权限与应用级审批。

## 10. 反馈编译与确认

默认 `FeedbackCompiler` 是一个隔离的结构化 LLM 调用，`tools=[]`，其安全形态类似
ScientificJudge，但使用独立 schema 和 prompt。

以下情况返回 `COMPILATION_UNAVAILABLE`：

- provider 调用失败；
- 没有 JSON；
- JSON 不满足 schema。

此时原始反馈仍然安全保存，任务不会进入 `REVISION_READY`，用户可以重试。已经写好的
结构化反馈 JSON 可作为不依赖 LLM 的确定性 fallback。

Compiler 不能直接修改状态。其输出必须经过 schema 验证、CLI 预览和用户确认，之后
才能激活 RevisionPlan。

RevisionPlanner 把确认后的 FeedbackDelta 转成有界的动态指令，不能改写静态 system
prompt。它必须区分：

- Target/Constraint 变化；
- 证据缺口与所需保真度；
- 报告 schema 要求；
- 策略偏好；
- 必须保留的正确内容；
- 禁止重复的无效结论。

专家原始文本只保留在 provenance 中，不能直接追加到 Maker conversation。只有通过验证
的 `RevisionInstruction` 和经过筛选的历史状态引用才能进入下一 Episode。

## 11. 证据继承规则

正常迭代创建全新的 `AgentRuntime` 和 ConversationState。只有满足以下条件的结构化证据
才能继承：

- 具有明确 provenance 和稳定 evidence ID；
- 没有被专家反馈判为失效；
- 与当前候选或任务 subject 正确绑定；
- 满足当前 Target 和数据源指纹策略；
- 不是模型自己生成的未经验证的断言。

不得继承：

- 未验证的候选性能预测；
- 已经被否定的 claim；
- 上一版完整答案；
- 原始 ConversationState；
- 原始专家自由文本。

继承的证据复制到新 `ScientificState` 时，provenance 必须记录来源 Episode。新 Episode
仍要重新检查所有有效硬约束。

`FRESH_EVALUATION` 不继承任何任务专属证据或反馈，只能加载在评估开始前冻结的通用
策略或经验快照。

## 12. 版本比较与学习信号

每对相邻 Episode 需要比较：

- 两轮都有专家评分时，四个分项和总体等级的变化；
- 上一轮问题闭合率；
- 已出现问题类型的复发率；
- 新引入的问题；
- 确定性约束、证据和保真度变化；
- 主结果文件差异；
- Token、工具、时间和 HPC 成本变化；
- 仍需人类判断的验收项。

新报告不再提到某个问题，不代表该问题已经关闭。只有以下情况可以标记为 CLOSED：

- 对应的机器验收条件已经通过；或
- 后续专家明确确认问题已解决。

无法机器检查的问题保持 `NEEDS_HUMAN_REVIEW`。

专家分项归一化方法：

```text
normalized_score = (score - 1) / 4
```

默认专家效用权重：

```text
科学正确性 0.35
证据充分性 0.30
创新性     0.15
可执行性   0.20
```

总体等级是独立的科研阶段可用性信号，不参加上述加权平均。硬性封顶用于防止语言质量
抵消伪造证据、关键科学错误、无依据创新或不可执行工艺。

学习信号可以包含：

- 专家效用变化；
- 问题闭合率；
- 问题复发率；
- 新问题惩罚；
- 归一化执行成本。

同一任务的所有 Episode 使用同一个 `task_group_id`。后续贝叶斯估计必须按任务分组或
显式建模相关性，论文中必须单独报告跨任务验证结果。

## 13. 经验生命周期

```text
OBSERVATION
  → HYPOTHESIS
  → VALIDATED_EXPERIENCE
  → REUSABLE_SKILL
```

- 一次专家评价只能产生 `OBSERVATION`；
- 某问题通过验收后，可以形成 `HYPOTHESIS`；
- 同一策略在不同任务中反复改善结果，才能成为 `VALIDATED_EXPERIENCE`；
- 升级为 `REUSABLE_SKILL` 需要明确的证据阈值和用户批准；
- 第一阶段禁止自动修改仓库中的 Skill 文件。

检索模块可以把 Observation 作为低置信度上下文，但必须展示其成熟度。任务专属反馈
不能泄漏进 fresh evaluation。

## 14. 事件和可观测性

在现有 JSONL 事件体系中增加以下 typed events：

- `evolution_task_created`；
- `evolution_episode_started`；
- `evolution_episode_completed`；
- `expert_feedback_recorded`；
- `expert_feedback_compiled`；
- `revision_plan_confirmed`；
- `evolution_iteration_started`；
- `evolution_comparison_completed`；
- `experience_state_changed`；
- `evolution_task_accepted`；
- `evolution_task_stopped`。

事件只记录 ID、版本、状态、哈希和有界摘要，不能记录完整专家原文或敏感信息。

已有的 runtime/scientific-loop 事件继续使用原来的 run/session identity。演化层通过
Episode metadata 建立关联，不修改已有事件语义。

## 15. 错误处理

- 非法分数或格式错误的导入文件必须在任何写入前失败；
- 录入过程中 Ctrl-C 不留下半条反馈；
- Compiler 失败时保留原始反馈，并允许无重复地重试；
- RevisionPlan 存在歧义时阻止执行并指出需要用户确认的字段；
- provider 或工具失败时记录失败 Episode，但不改变上一版成功 Episode；
- 结果文件哈希不匹配时拒绝绑定反馈和继续迭代；
- 历史结果文件丢失时返回 typed diagnostic，禁止静默替换；
- 锁冲突和 revision 冲突安全失败，并允许用户重试；
- 贝叶斯可选依赖缺失时软失败，仍可使用固定策略。

## 16. 测试策略

### 16.1 领域模型与持久化

- 所有新 Pydantic 模型及状态转换验证；
- Store 的创建、读取、原子更新和乐观 revision 冲突；
- Feedback 不可变性与 supersession；
- 路径包含检查和程序生成 ID 校验；
- 主结果文件哈希和 mismatch 拒绝；
- schema migration 和未知版本诊断。

### 16.2 专家反馈

- 所有评分上下界及硬性封顶；
- 多行 `/submit`、`/cancel` 和 Ctrl-C；
- 评论正文包含类似 slash command 的文本；
- Compiler schema、QUERY 保留、source span 和 confidence；
- provider 失败及结构化文件 fallback；
- 原始反馈永远不进入 `ConversationState` 或 `ScientificState`。

### 16.3 演化编排

- start → v001 完成 → 等待专家反馈；
- feedback → 只记录，不产生 runtime/tool call；
- 缺少反馈或确认时拒绝 iterate；
- iterate 创建新的 runtime session 和单调 v002；
- 只继承合格的 verified evidence；
- 专家高分不能覆盖科学硬约束失败；
- v002 失败后 v001 及其文件保持完整；
- accept、stop、reopen 状态转换；
- 进程退出后重新加载并继续。

### 16.4 CLI

- 独立 `evolve` 命令帮助和所有子命令；
- `/evolve` 在 `AgentRuntime.run()` 前被截获；
- 普通文本提到专家反馈时仍为普通聊天；
- 专家模式提示符与提交前确认摘要；
- 每次状态转换后的准确 next-command 提示；
- JSON 导入导出 round trip。

### 16.5 实验和评估

- fake provider 下的两 Episode 确定性端到端演化；
- 问题闭合率和复发率计算；
- 正常证据继承与 fresh evaluation 隔离；
- 按 task group 统计，禁止把重复 Episode 当作独立任务；
- evolution、episode、runtime session 和 run ID 的事件关联。

完成窄测试后，还要运行现有 scientific-loop、session、command-router、permission、
tool-surface、event 和 experiment 测试。宣称仓库整体通过前必须执行：

```bash
uv run pytest -q
uv run mypy src
git diff --check
git diff --stat
git status --short
```

## 17. 分阶段交付

### 阶段 1：持久化工作流基础

实现 models、store、状态机、结果身份、事件，以及只读的 CLI list/status/history。

### 阶段 2：专家反馈输入

实现量表、交互式/文件录入、不可变反馈、CLI 与 slash command 隔离。

### 阶段 3：反馈编译

实现隔离 Compiler、用户确认、RevisionPlan、失败降级和 provenance。

### 阶段 4：版本化重跑

实现新 Episode、受控 RevisionInstruction、verified evidence 继承、结果固化和版本比较。

### 阶段 5：经验层

实现问题闭合/复发、经验生命周期、导出和 fresh evaluation。

### 阶段 6：自适应策略选择

先保留固定策略基线；积累足够轨迹并完成按 task group 的离线验证后，再启用贝叶斯选择。

每个阶段都必须保证：即使 evolution 功能或可选依赖不可用，基础 chat 和现有单次
scientific loop 仍能正常启动和使用。

## 18. 第一版明确不做的内容

- 基础模型微调、RLHF、DPO 或奖励模型训练；
- 对任意 Prompt 进行 PSO 或遗传变异；
- 自动编辑或创建仓库 Skill；
- 把同一任务的重复运行当成独立统计样本；
- 用专家分数覆盖科学证据或安全门禁；
- 录入反馈时自动提交 HPC；
- 从普通聊天文本自动判断专家身份；
- 图形化前端或远程多人评审系统；
- 自动声称达到论文级材料新颖性。

## 19. 最终验收标准

实现完成后，用户必须能够：

1. 创建演化任务，并在 Episode 执行前获得持久化 evolution ID；
2. 退出 CLI，之后把带量表分数的专家评价绑定到准确的 v001 结果文件，同时证明没有
   产生普通 Agent turn 或工具调用；
3. 确认结构化 RevisionPlan，并显式启动使用全新 runtime session 的 v002；
4. 查看任务、版本、反馈、策略、科学摘要、runtime session、结果文件和事件之间的完整
   provenance；
5. 查看版本之间可机器验证的问题闭合、待人工验收项和成本变化；
6. 通过测试证明普通聊天不能修改 evolution state，专家反馈也不能绕过确定性科学检查、
   权限系统或 HPC 提交门禁；
7. 执行一次 fresh evaluation：排除任务专属反馈和继承证据，只使用显式冻结的通用策略
   快照。
