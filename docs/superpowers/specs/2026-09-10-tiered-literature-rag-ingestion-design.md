# 分层文献 RAG 与两阶段导入设计

- 状态：待用户最终审查
- 日期：2026-09-10
- 数据范围：1,424 篇 PDF、100,116 条独立摘要
- 部署方式：WSL 中运行 PhotomatAgent，Docker Qdrant 绑定宿主回环地址

## 1. 背景

PhotomatAgent 已使用 Qdrant 保存 PDF 文档与检索片段，并支持本地或显式配置的外部
embedding/reranker provider。当前摄取边界只接受 PDF，检索也没有区分全文与摘要。

用户的数据包含两个互不关联的知识源：

- `Photoelectric detection/dataset/paper/pdf`：1,424 篇 PDF 全文；
- `Photoelectric detection/dataset/paper/abstract/abstracts.sqlite3`：100,116 条摘要。

摘要不是 PDF 的附属元数据，也不应通过 DOI、标题或文件名与 PDF 强制关联。它们的
作用是扩充本地知识覆盖面。检索证据优先级必须保持为：本地 PDF 全文、本地摘要、
arXiv 在线搜索。

## 2. 已确认决策

1. PDF 与摘要作为独立 document 保存，不建立跨语料关联。
2. 两类数据使用同一 Qdrant documents/passages generation，通过
   `source_kind=fulltext|abstract` 隔离。
3. 检索时分别召回全文和摘要，不把全部候选直接混排。
4. 智能体先检索全文；全文没有直接支撑或存在知识缺口时检索摘要；本地摘要仍不足时
   才调用现有 `literature.search_arxiv`。
5. arXiv 结果仅进入当前会话，不自动下载或写入 Qdrant。
6. 提供一个支持阶段选择、可见进度、主动分段、幂等重跑和断点续跑的 WSL Bash
   驱动脚本。
7. 首期只导入 PDF 和摘要；器件、材料、工艺及其他结构化 CSV 暂不处理。
8. 实现验证只运行最小必要测试，不运行真实全量导入、真实模型下载、真实 arXiv 请求或
   仓库全量测试。

## 3. 目标与非目标

### 3.1 目标

- 保留现有 PDF 的分页、章节和邻接 passage 来源追踪。
- 从 SQLite 流式摄取摘要，不生成十万个临时文件，也不一次性把全部摘要载入内存。
- 对每条摘要保留稳定来源标识、书目信息和 `abstract_only` 限制。
- 每个阶段按小批次持久化进度；终端中断、WSL 重启或脚本退出后可以继续。
- 相同语料和模型的重复运行保持幂等，只嵌入新增或变更记录。
- 本地与外部模型沿用同一 provider 配置和显式外发确认边界。
- 新 generation 在两个阶段完成并通过显式确认前不替换 current aliases。

### 3.2 非目标

- 不自动建立 PDF 与摘要的 DOI、标题或语义关联。
- 不把摘要解释为全文证据。
- 不自动持久化 arXiv 搜索结果。
- 不实现 GraphRAG、知识图谱、图片向量、PDF 自动下载或器件数据导入。
- 不在此次任务中重新调参或运行大规模检索质量评测。

## 4. 总体架构

```text
用户运行 WSL 脚本
  -> photomatagent rag import pdf|abstracts
     -> LiteratureIngestionService
        -> PDF parser 或 SQLite abstract reader
        -> configured EmbeddingProvider
        -> QdrantLiteratureStore
           -> documents generation
           -> passages generation

智能体检索
  -> literature.search_passages(source_kind="fulltext")
     -> 无直接支撑或仍有证据缺口
  -> literature.search_passages(source_kind="abstract")
     -> 仍有证据缺口且网络权限允许
  -> literature.search_arxiv
```

Qdrant 仍是内部应用后端。模型请求继续通过
`AgentRuntime -> ToolRegistry -> Tool.execute()`，不得让模型直接操作 Qdrant、SQLite
或网络客户端。CLI 是用户直接操作入口，但必须复用同一应用服务和数据合同。

## 5. Qdrant 存储合同

### 5.1 Collection 布局

继续使用一对 generation-scoped collections 和一对稳定 aliases：

```text
<prefix>_documents_<generation>
<prefix>_passages_<generation>
<prefix>_documents_current
<prefix>_passages_current
```

全文与摘要使用相同 embedding 模型和向量维度。新增 payload 字段和分层检索语义要求
提升 schema/chunk version，从而创建新的物理 generation；旧 current generation 在新
generation 完成前保持可用。

### 5.2 公共来源字段

document 和 passage payload 增加以下向后兼容字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `source_kind` | keyword | `fulltext` 或 `abstract` |
| `source_record_id` | keyword | 摘要为 `paper_key`；全文为空 |
| `doi` | keyword | 摘要来源提供的 DOI，可为空 |
| `pmid` | keyword | 可为空 |
| `pmcid` | keyword | 可为空 |
| `journal` | text | 可为空 |
| `relevance_tier` | keyword | 摘要数据集的相关性等级，可为空 |

至少为 `source_kind`、`source_record_id`、`doi` 和 `relevance_tier` 创建 payload
index。现有 `workspace_id`、`ingest_state`、`document_id`、`year` 等索引继续保留。

### 5.3 全文表示

- 一个 PDF 对应一个 document 和多个 passage。
- `source_kind="fulltext"`，`source_record_id=""`。
- `relative_source_path` 仍为工作区相对 PDF 路径。
- 页码、章节、标题层级和相邻 passage 指针保持现有语义。
- 内容 revision 仍由 PDF 文件内容 SHA-256 决定。

### 5.4 摘要表示

- SQLite 中一条有效记录对应一个 document 和一个 passage。
- document ID 由 `workspace_id`、SQLite 工作区相对路径和 `paper_key` 共同确定；不同
  记录即使标题相同也不会冲突。
- `relative_source_path` 指向 `abstracts.sqlite3`，`source_record_id` 精确定位源行。
- passage 文本采用规范化的 `Title: <title>\nAbstract: <abstract>`，改善语义与 BM25
  召回；payload 中仍分别保存标题和作者等字段。
- `section` 与 `heading_path` 均为 `Abstract`，页码和邻接指针为空。
- `limitations` 必须包含 `abstract_only` 和“未检查全文”的等价说明。
- revision 由影响检索与来源显示的规范化行字段生成 SHA-256，而不是 SQLite 文件整体
  哈希。因此单条摘要更新只重建该记录。
- 缺少摘要正文的行不建立向量，计入 `skipped_empty`；有摘要但缺 DOI/PMID/PMCID
  的行仍可使用稳定 `paper_key` 入库。

## 6. 两阶段摄取与断点恢复

### 6.1 PDF 阶段

PDF 阶段复用现有完整目录枚举、文件哈希、解析、分块、staged passage、计数核验、
ready 切换和删除同步逻辑。增加逐批进度事件和单次运行上限，不改变每批的安全上限。

### 6.2 摘要阶段

摘要 reader 以 SQLite 只读模式打开数据库，并使用稳定 keyset pagination：

```sql
SELECT ... FROM papers
WHERE paper_key > ?
ORDER BY paper_key
LIMIT ?
```

每批只保留当前行、规范化记录和向量。对本批 document IDs 执行有界 manifest 查询，
内容 revision 未变化的记录计为 `unchanged`，不重复调用 embedding provider。

摘要导入不默认执行删除同步。本任务是知识扩充，源库误删不应自动删除已入库知识；未来
如需要镜像语义，应通过单独、显式、完整扫描后的 `--prune` 设计实现。

### 6.3 运行状态

每个阶段有独立 `run_id` 和 ingestion-run control record，至少保存：

- workspace ID、generation fingerprint、source kind；
- 源路径和摘要数据库身份信息；
- 当前稳定 cursor；
- discovered、processed、indexed、unchanged、failed、skipped、passages；
- 状态 `running|paused|retryable|complete`；
- 最多 20 条截断、脱敏错误。

每个批次成功提交后才推进 cursor。`Ctrl-C`、进程退出或 WSL 重启不会丢失已提交批次。
恢复时必须验证 workspace、generation、source kind 和源路径；embedding 模型或 schema
改变时拒绝在旧 run 上续跑。

SQLite 在一次运行开始时记录文件身份信息。恢复时若源数据库发生变化，脚本明确要求
开始新的 run，避免在两个数据库快照之间拼接一个不一致的导入结果。每条记录仍通过
revision 实现跨 run 的增量幂等。

## 7. 用户脚本与进度输出

新增 `scripts/import_literature_qdrant.sh`，默认使用当前数据位置，同时允许命令行覆盖
workspace、PDF 目录和摘要 SQLite 路径。脚本提供：

```bash
./scripts/import_literature_qdrant.sh pdf --stop-after 100
./scripts/import_literature_qdrant.sh pdf --resume
./scripts/import_literature_qdrant.sh abstracts --stop-after 5000
./scripts/import_literature_qdrant.sh abstracts --resume
./scripts/import_literature_qdrant.sh status
./scripts/import_literature_qdrant.sh activate
```

规则：

- `--stop-after N` 在完成不超过 N 个源记录后安全暂停并返回成功；最后不足一批时允许
  少于 N。
- `--resume` 读取对应阶段最近的未完成 run；没有可恢复状态时给出明确错误。
- 直接运行阶段命令且未指定 `--resume` 时创建新 run；幂等 manifest 检查避免重复嵌入。
- 每批打印单行 JSON 或稳定表格，包含总量、累计处理量、indexed、unchanged、failed、
  skipped、passages、速率、ETA、cursor 和 run ID。
- `status` 只读显示两个阶段和当前 aliases 状态。
- `activate` 要求两个阶段均存在成功完成记录，并再次请求显式确认；它使用现有原子 alias
  切换，不在脚本中直接调用 Qdrant REST API。
- 外部 embedding/reranker 配置沿用现有环境变量。检测到外部 provider 时，阶段命令必须
  显示数据外发警告并要求交互确认；只有用户传入 `--yes` 才可非交互继续。

脚本只负责参数、可读输出和调用 CLI，不复制业务逻辑，也不持有 API key。

## 8. 分层检索策略

`literature.search_passages` 增加必需或有明确默认值的 `source_kind` 参数：

- `fulltext`：只查询 PDF passage；
- `abstract`：只查询摘要 passage。

为了保持现有调用兼容，省略参数时默认 `fulltext`。store 的 dense、sparse 和 hybrid
查询都必须在 Qdrant 端应用 `workspace_id + source_kind + ingest_state=ready` 过滤，不能
先取混合候选再在 Python 中过滤。

稳定能力说明和工具描述明确要求智能体遵循以下策略：

1. 默认先查 `fulltext`。
2. 返回结果与问题没有直接关系、只部分回答问题或仍有明确证据缺口时，再查
   `abstract`。
3. 本地摘要仍不能覆盖问题，或用户明确要求最新研究时，才调用
   `literature.search_arxiv`。
4. 不使用未校准的 Qdrant/RRF 分数作为跨来源证据等级；来源等级由
   `fulltext > abstract > arXiv metadata` 决定。
5. 输出必须保留来源类型和限制。摘要与 arXiv 结果不能被表述为已检查全文。

arXiv 工具保持独立网络权限和有界结果数量。其返回值只进入会话观察与结构化证据，
不触发 Qdrant upsert、PDF 下载或本地文件写入。

## 9. 故障与一致性处理

- PDF 文件在计划后或解析中变化：保留现有 fail-closed 行为，不推进失败 cursor。
- SQLite 无法以只读模式打开、schema 缺失或查询失败：停止摘要阶段并保存 retryable
  状态。
- embedding/Qdrant 临时失败：当前记录不标记 ready，不推进 cursor，恢复后重试。
- malformed 行：若有稳定 `paper_key` 和非空摘要，则缺失字段按空值保存；缺少稳定 key
  或摘要为空时跳过并计数，不伪造标识或正文。
- 批次 upsert 后必须核验 staged 数量，再将 passage/document 标记 ready。
- 日志和 control records 不保存密钥、绝对敏感路径、完整摘要错误副本或无限错误列表。
- 尚有 running/retryable run 时拒绝 `activate`；任何一阶段未完成也由脚本前置检查拒绝。

## 10. 最小验证策略

根据用户要求，只运行与改动直接相关的最小测试：

1. 使用临时 SQLite 三至五行数据验证 keyset pagination、字段规范化、空摘要跳过和稳定
   revision。
2. 使用 fake Qdrant store/provider 验证摘要 upsert、未变化记录不重复 embedding、失败
   不推进 cursor，以及使用相同 run ID 恢复。
3. 验证 Qdrant 查询将 `source_kind` 作为服务端过滤条件；摘要不能出现在全文查询中。
4. 验证公开检索结果包含 `source_kind` 和 `abstract_only` 限制。
5. 验证 CLI 的 `--stop-after`、`--resume`、进度输出、外部 provider 确认和激活前置检查。
6. 对 Bash 脚本执行语法检查、`--help` 和 dry-run 参数测试。
7. 运行上述相关 pytest 文件、修改模块的定向 mypy，以及 `git diff --check`。

测试不得读取真实 2.9 GB PDF 目录、遍历完整十万条摘要、下载模型、访问 arXiv、调用
外部 embedding API 或运行仓库全量测试。最终交付必须明确报告实际运行的测试及未运行
的全量验证。

## 11. 预期改动边界

实现应集中在：

- 文献 payload/model 合同；
- Qdrant source-kind payload index、批量 manifest 查询和过滤召回；
- SQLite 摘要 reader 与摘要摄取编排；
- RAG CLI 的分段、恢复、状态和进度输出；
- 文献能力说明与检索工具 schema；
- 一个薄 WSL Bash 驱动脚本；
- 对应的小型 fake/临时数据库测试与操作文档。

不重构 AgentRuntime、ToolRegistry、科学状态、其他能力包、HPC 工作流或 Qdrant Docker
部署。实现不得建立第二套工具注册表或绕过现有权限、工作区和 provider 安全门。

## 12. 验收标准

- 脚本能够分别启动 PDF 和摘要阶段，并支持安全暂停和恢复。
- 摘要摄取的常驻内存不随 100,116 条记录线性增长。
- 同一记录重复导入不重复生成向量；变更记录产生新 revision。
- Qdrant 能按 `source_kind` 在服务端隔离全文和摘要召回。
- 智能体工具说明形成“全文 -> 摘要 -> arXiv”的明确回退策略。
- arXiv 结果不自动持久化。
- 两阶段未完成或存在 retryable 运行时不能通过脚本激活新 generation。
- 最小相关测试通过，且交付报告不声称未执行的全量测试已经通过。
