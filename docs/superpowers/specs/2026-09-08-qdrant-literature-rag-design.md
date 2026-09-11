# 基于 Qdrant 的文献 RAG 改造设计

- 状态：待用户审查
- 日期：2026-09-08
- 目标规模：约 10,000 篇 PDF 论文
- 首期部署：单机 Docker，自托管 Qdrant

## 1. 背景与问题

PhotomatAgent 当前的文献 RAG 使用 LanceDB 保存片段和 dense 向量，并在
Python 进程中构建自定义 BM25。现场检查发现：

- 默认论文根目录 `dataset/paper` 不存在；
- `output/literature_index` 中残留 3 篇文档、363 个片段，约 904 KB；
- LanceDB 的 passages/documents 表均未创建向量索引或全文索引；
- dense 查询因此没有显式 ANN 索引；
- lexical 查询通过 `all_passages()` 将全部片段载入 Python 内存，再逐条计算
  BM25；
- 当前工具缺少面向 10,000 篇论文的持久摄取进度、批次边界、快照恢复和模型迁移
  协议。

这意味着主要风险不是“本地数据占用磁盘”，而是检索和摄取生命周期与单个
PhotomatAgent 进程耦合。数据增长后，查询内存、启动时间、全量扫描和失败恢复都会
成为瓶颈。

## 2. 已确认决策

1. 使用自托管 Qdrant，首期由 Docker Compose 在当前工作站运行。
2. 设计容量覆盖约 10,000 篇论文，预计约 500,000–2,000,000 个结构化片段。
3. 从运行时代码、依赖、配置和测试中彻底移除 LanceDB，不保留 LanceDB fallback。
4. dense embedding 和 reranker 均通过可替换 provider 接口提供：
   - 默认完全本地运行；
   - 可显式切换到外部模型 API；
   - 绝不在本地失败后静默切换到外部服务。
5. Qdrant 只作为内部应用后端，不直接暴露为模型可调用工具。
6. 保持 `AgentRuntime -> ToolRegistry -> Tool.execute()` 为模型请求工具执行的唯一
   权威路径。
7. 保留现有 `literature.index_papers`、`literature.search_passages` 和
   `literature.read_passage` 工具名及其有界输出语义。

## 3. 目标

### 3.1 功能目标

- 对工作区内 PDF 执行结构化解析、分块、dense embedding、sparse BM25 和增量摄取。
- 在 Qdrant 内执行 dense+sparse 候选召回和 RRF 融合，不把全库载入 Python。
- 支持本地与外部 dense embedding provider。
- 支持本地与外部 reranker provider，并允许关闭 reranking。
- 为每个返回片段保留论文、文件、页码、章节、标题层级、邻接片段和内容版本来源。
- 支持中断后恢复、单文档失败隔离、源文件删除同步、模型/schema 蓝绿迁移和快照恢复。
- 提供可操作的 CLI 状态、索引、检索和备份入口。

### 3.2 非功能目标

- 基础运行时在 Qdrant、Docling 或本地模型缺失时仍能启动，能力包以结构化状态软失败。
- 查询路径的内存使用不随语料总片段数线性增长。
- 所有路径继续由 `Workspace.resolve` 约束在工作区内。
- 外部模型调用必须显式启用，密钥不写入日志、事件、Qdrant payload 或工具输出。
- 不将缺失、部分写入、mock 或未经验证的结果升级为有效科学证据。

### 3.3 非目标

- 首期不部署 Qdrant 分布式集群、Kubernetes 或 Qdrant Cloud。
- 首期不实现 GraphRAG、知识图谱、多模态图像向量或自动论文下载。
- 首期不增加文件系统实时 watcher；摄取由显式工具或 CLI 命令触发。
- 首期不自动调度周期快照；只提供可重复执行的快照命令。
- 首期不删除或自动迁移已有 `output/literature_index` 用户数据目录。

## 4. 总体架构

```text
模型请求
  -> AgentRuntime
     -> 权限、参数、路径、预算检查
     -> ToolRegistry
        -> literature.* Tool（有界模型可见契约）
           -> LiteratureApplicationService
              ├─ PdfDiscovery + DoclingParser + Chunker
              ├─ EmbeddingProvider
              │  ├─ LocalSentenceTransformerProvider（默认）
              │  └─ OpenAICompatibleEmbeddingProvider（显式外部模式）
              ├─ RerankerProvider
              │  ├─ LocalCrossEncoderProvider（默认）
              │  ├─ CohereCompatibleRerankerProvider（显式外部模式）
              │  └─ DisabledReranker
              ├─ QdrantLiteratureStore
              ├─ LiteratureIngestionService
              └─ LiteratureRetriever
                    -> Qdrant Docker（HTTP/gRPC，仅宿主机回环地址）
```

Provider、检索器、CLI 和能力探针不能绕过 `AgentRuntime` 直接执行模型请求的工具。
CLI 是用户直接操作入口，可以调用相同的应用服务，但不能构造第二套工具注册表或改变
模型侧权限语义。

## 5. 模块边界

建议将文献目录拆分为以下单一职责模块：

```text
src/photomatagent/scientific/capabilities/literature/
  __init__.py             # CapabilityPack 和薄 Tool 适配器
  models.py               # Paper/Passage/ingestion/retrieval 数据合同
  parser.py               # PDF -> 结构化片段；不接触数据库或模型 API
  evidence.py             # 现有确定性科学证据抽取
  providers/
    base.py               # EmbeddingProvider/RerankerProvider 协议和结果类型
    local.py              # SentenceTransformer/CrossEncoder；lazy import
    external.py           # OpenAI-compatible embedding、Cohere-compatible rerank
    factory.py            # 配置验证与 provider 构造
  qdrant_store.py         # collection、alias、point、filter、snapshot 适配
  ingestion.py            # 批次、恢复、幂等更新、删除同步、进度统计
  retrieval.py            # hybrid query、降级、去重、rerank、上下文扩展
```

边界规则：

- `parser.py` 不生成向量，不访问 Qdrant。
- provider 不访问 Qdrant、工作区或科学状态。
- `qdrant_store.py` 不解析 PDF，不调用模型，不生成 `ScientificEvidence`。
- `ingestion.py` 编排解析、provider 和 store，但不暴露为模型工具。
- Tool 只验证输入、调用应用服务、限制输出并构造 `ScientificToolResult`。
- `evidence.py` 继续以来源明确的片段为输入，缺失数值仍返回空证据而非猜测。

## 6. Docker 与网络设计

新增根目录 `compose.qdrant.yaml`，只包含一个 Qdrant 服务：

- 镜像固定为 `qdrant/qdrant:v1.18.2`，禁止使用 `latest`；
- HTTP 仅发布到 `127.0.0.1:6333`；
- gRPC 仅发布到 `127.0.0.1:6334`；
- 使用 Docker named volume，而不是 Windows/WSL bind mount，避免宿主文件系统兼容性
  风险；
- `restart: unless-stopped`；
- Qdrant 数据、WAL、segment、索引和服务端快照保存在 named volume；
- PDF 原件不复制到容器，也不保存在 Qdrant volume 中；
- 应用通过宿主机 URL 访问 Qdrant，容器不获得工作区或 `.env` 挂载权限。

本地默认不启用 Qdrant API key，因为端口只绑定回环地址。只要 URL 使用非回环主机，
配置校验就必须要求 API key，并由 `rag status` 明确报告 TLS 状态；首期不代替反向代理
实现 TLS。

Python 文献 extra 调整为：

```text
保留：arxiv、pypdf、docling、sentence-transformers、CPU torch
新增：qdrant-client>=1.19,<2
移除：lancedb、pylance
```

Qdrant server/client 的组合必须由 integration test 验证。升级时先创建快照，再升级一个
minor 版本并重跑 integration tests；不能无验证跳跃多个 server minor 版本。

## 7. Qdrant 数据模型

### 7.1 Collection 对

每个可检索 generation 包含两个物理 collection：

```text
<prefix>_documents_<generation>
<prefix>_passages_<generation>
```

对外只访问两个稳定 alias：

```text
<prefix>_documents_current
<prefix>_passages_current
```

`generation` 由 collection schema version 和 embedding fingerprint 的短哈希组成。
更换 dense provider、dense model、vector dimension、document/query prefix、归一化方式、
sparse model 或分块 schema 时必须创建新 generation。只更换 reranker 不要求重建向量库。

完成全量重建和检索验收后，使用一个 alias update 请求切换 documents/passages 两个
alias。旧 collection 在保留期内只读保留，不能在 alias 切换操作中立即删除。

### 7.2 Documents collection

Documents collection 使用 vectorless point；它是摄取控制面，不参加相似度检索。每篇
论文一个 `record_type=document` point，摄取任务使用
`record_type=ingestion_run` point。Qdrant 支持无向量、仅含 payload 的 point。

文档 payload：

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| `schema_version` | keyword | payload 合同版本 |
| `record_type` | keyword | 固定为 `document` |
| `workspace_id` | keyword | 工作区隔离，所有操作强制过滤 |
| `document_id` | keyword/UUID | `UUIDv5(workspace_id, relative_source_path)` |
| `relative_source_path` | keyword | 工作区相对路径，不存绝对路径 |
| `file_name` | keyword | 显示用途 |
| `content_sha256` | keyword | 内容版本与增量判断 |
| `status` | keyword | `pending/staged/ready/failed/deleted` |
| `title` | text | 论文标题 |
| `authors` | keyword array | 作者 |
| `year` | integer | 年份过滤 |
| `num_pages` | integer | 来源统计 |
| `chunk_count` | integer | 完整性核验 |
| `model_fingerprint` | keyword | 所属 generation |
| `indexed_at` | datetime | 最近成功时间 |
| `last_error` | text | 截断、脱敏的失败诊断 |

摄取任务 payload 至少包含 `run_id`、`workspace_id`、`generation`、`root`、`cursor`、
`status`、累计计数和最多 20 条截断错误。它不能包含 API key、PDF 正文或无限增长的
错误列表。

### 7.3 Passages collection

向量字段：

- `dense`：配置维度，Cosine distance，HNSW；
- `sparse_bm25`：Qdrant BM25 sparse vector，启用 IDF modifier。

每个 passage point 的 ID 为：

```text
UUIDv5(document_id, content_sha256 + ":" + chunk_index)
```

因此同一内容的重试是幂等的，内容变化产生新 revision point，旧 revision 可独立清理。

passage payload：

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| `schema_version` | keyword | payload 合同版本 |
| `record_type` | keyword | 固定为 `passage` |
| `workspace_id` | keyword | 工作区隔离 |
| `document_id` | keyword | 文档关联与删除过滤 |
| `document_revision` | keyword | PDF 内容 SHA-256 |
| `ingest_state` | keyword | `staged/ready/superseded` |
| `passage_id` | keyword/UUID | 模型可见的 opaque ID |
| `chunk_index` | integer | 文档内顺序 |
| `text` | text | 片段原文 |
| `title` | text | 标题 |
| `authors` | keyword array | 作者 |
| `year` | integer | metadata filter |
| `section` | text | 当前章节 |
| `heading_path` | text | 标题层级 |
| `page_start/page_end` | integer | 来源定位 |
| `previous_passage_id` | keyword | 邻接上下文 |
| `next_passage_id` | keyword | 邻接上下文 |
| `relative_source_path` | keyword | 来源文件 |
| `model_fingerprint` | keyword | 防止混库 |
| `limitations` | text array | 解析或来源限制 |

至少为 `workspace_id`、`record_type`、`document_id`、`document_revision`、
`ingest_state`、`passage_id`、`relative_source_path` 和 `year` 创建 payload index。
Strict mode 禁止对未索引字段执行高成本过滤。

### 7.4 存储参数

面向 500,000–2,000,000 个 384 维向量，首期默认：

- 单 shard、单 replica；
- dense 原始向量 on-disk；
- payload on-disk；
- HNSW 使用 Qdrant 稳定默认值作为基线，查询 `hnsw_ef` 和是否启用 scalar
  quantization 由基准测试决定，不在无测量时擅自压缩；
- 初次大批量导入期间降低或暂停 HNSW 构建，上传完成后恢复索引阈值并等待 optimizer
  收敛；
- 批量上传默认每批 128 个片段，可由配置在 16–512 范围内调整；
- 应用级候选数量严格有界，默认 dense 50、sparse 50、融合后 50、最终返回 5。

容量规划以实际分块统计为准。部署前建议至少准备 8 CPU、16–32 GB RAM、SSD/NVMe
和 100 GB 可用 Qdrant 空间；PDF 原件空间另算。`rag status` 必须报告实际 point 数、
indexed vector 数、collection 状态和 Qdrant volume 容量告警所需信息，不能把估算当成
精确容量。

## 8. Provider 设计

### 8.1 Dense embedding 协议

```text
EmbeddingProvider
  model_id: str
  dimension: int
  fingerprint_material(): mapping
  embed_documents(texts: Sequence[str]) -> list[list[float]]
  embed_query(text: str) -> list[float]
```

共同约束：

- 文档和查询前缀由 provider 明确声明；
- 返回数量必须与输入数量一致；
- 每个向量必须是有限浮点数且维度一致；
- 写入前验证 provider 维度与 collection schema；
- 禁止自动截断后静默继续；截断策略必须进入 fingerprint；
- 本地同步模型调用通过受控 worker thread 执行，不阻塞 async runtime event loop；
- 外部请求使用 bounded timeout、指数退避和最多 3 次重试，只重试可判定的瞬时错误。

默认 `LocalSentenceTransformerProvider` 继续使用
`intfloat/multilingual-e5-small`、384 维、归一化 embedding 和 E5 的
`passage:`/`query:` 前缀，避免在改数据库的同时无证据改变检索模型。

外部 `OpenAICompatibleEmbeddingProvider` 使用独立于 Agent LLM 的 base URL、模型和
密钥环境变量。它支持 OpenAI `/embeddings` 兼容接口，但不复用 Agent provider 对象，
防止聊天模型配置意外改变索引语义。

### 8.2 Sparse BM25

Sparse 向量固定使用 Qdrant `qdrant/bm25`，在自托管 Qdrant 内生成和评分。文档和查询
使用同一 tokenizer/model 配置。任何 sparse model 或 tokenizer 配置变化都进入 collection
fingerprint，并触发新 generation。

### 8.3 Reranker 协议

```text
RerankerProvider
  model_id: str
  rerank(query, passages, top_n) -> ordered scores
```

- 默认 `LocalCrossEncoderProvider` 使用现有
  `cross-encoder/ms-marco-MiniLM-L-6-v2`；
- `CohereCompatibleRerankerProvider` 支持明确配置的 `/rerank` 兼容端点；
- `DisabledReranker` 直接保留 Qdrant 融合排序；
- reranker 只接收融合后最多 50 个候选；
- reranker 失败时返回 Qdrant 融合结果，并在结果 limitations/diagnostics 中标记
  `reranker_unavailable`，不得把失败伪装为正常精排；
- reranker 不影响存储向量，不进入 collection fingerprint。

### 8.4 外部数据边界

新增 `PHOTOMATAGENT_RAG_ALLOW_EXTERNAL=0`，默认禁止将论文内容发送到外部服务。
只有同时满足以下条件，外部 provider 才可构造：

1. `PHOTOMATAGENT_RAG_ALLOW_EXTERNAL=1`；
2. provider 类型显式设置为外部；
3. base URL、model 和专用 API key 环境变量有效；
4. 能力状态和 CLI 明确显示“论文片段将发送到外部服务”。

本地 provider 失败时不得自动切换外部 provider。外部 embedding 与 reranker 使用不同
配置和密钥；embedding 会发送全部待索引片段，reranker 只发送有限候选，两者风险不能
混为一谈。

## 9. 配置合同

`.env.example` 增加以下非秘密默认值和注释；真实密钥仍只放用户 `.env` 或进程环境：

```dotenv
# --- Literature RAG / Qdrant ----------------------------------------------
PHOTOMATAGENT_LITERATURE_DIR=dataset/paper
PHOTOMATAGENT_QDRANT_URL=http://127.0.0.1:6333
PHOTOMATAGENT_QDRANT_API_KEY_ENV=QDRANT_API_KEY
PHOTOMATAGENT_QDRANT_COLLECTION_PREFIX=photomat_literature
PHOTOMATAGENT_QDRANT_TIMEOUT_SECONDS=20

PHOTOMATAGENT_RAG_ALLOW_EXTERNAL=0
PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER=local
PHOTOMATAGENT_RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small
PHOTOMATAGENT_RAG_EMBEDDING_VECTOR_DIM=384
# PHOTOMATAGENT_RAG_EMBEDDING_BASE_URL=
# PHOTOMATAGENT_RAG_EMBEDDING_API_KEY_ENV=RAG_EMBEDDING_API_KEY

PHOTOMATAGENT_RAG_RERANK_PROVIDER=local
PHOTOMATAGENT_RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# PHOTOMATAGENT_RAG_RERANK_BASE_URL=
# PHOTOMATAGENT_RAG_RERANK_API_KEY_ENV=RAG_RERANK_API_KEY

PHOTOMATAGENT_RAG_BATCH_SIZE=128
PHOTOMATAGENT_LITERATURE_TOP_K=5
PHOTOMATAGENT_LITERATURE_PASSAGE_CHARS=600
```

删除 `PHOTOMATAGENT_LITERATURE_INDEX_DIR`。配置解析不能读取或打印密钥值，只保存密钥
环境变量名并在真正调用时解析。

配置校验：

- 数值范围必须在构造时验证，而不是静默回退到危险值；
- local 模型维度从模型探针获取并与配置对比；
- external 模型维度在首次探针返回后锁定并与配置对比；
- alias 指向 fingerprint 不匹配的 collection 时拒绝写入和 dense 查询；
- 旧 schema 不能自动原地升级，必须创建新 generation。

## 10. 摄取流程

### 10.1 计划与批次

一次摄取先执行只读计划：

1. 使用 `Workspace.resolve` 解析论文根目录；
2. 根目录不存在时立即失败，不删除 Qdrant 中任何文档；
3. 递归枚举 PDF 并按工作区相对路径排序；
4. 与 documents collection 中的 SHA-256/status 比较；
5. 分类为 unchanged/new/changed/deleted/retry_failed；
6. 创建或恢复 `ingestion_run` 控制记录；
7. 每个模型工具调用默认最多处理 20 篇变化论文，并返回 `run_id`、`next_cursor` 和
   `complete`，避免单次 Agent tool call 无边界运行数小时；
8. CLI `rag index` 可以在用户直接控制下循环执行批次直到完成。

目录扫描成功并不等于索引成功。统计分别报告 discovered、unchanged、indexed、failed、
deleted、chunks、staged_cleanup 和 retryable。

### 10.2 单文档两阶段更新

对每篇 new/changed PDF：

1. 读取并计算 SHA-256；
2. Docling 完成解析和分块；
3. 对整篇文档的片段执行 bounded batch dense embedding；
4. 验证全部向量数量、维度和有限值；
5. 以 `ingest_state=staged` upsert revision-specific passage points；
6. `wait=true` 后按 revision 精确计数，必须等于预期 chunk count；
7. 将新 revision 标记为 `ready`；
8. 将旧 revision 标记为 `superseded` 并删除；
9. 更新 document point 为 `ready` 和新 SHA/chunk_count；
10. 返回受限进度信息。

查询只过滤 `ingest_state=ready`。如果进程在第 7 步前退出，staged points 不可检索并在
下次恢复时清理。如果在新 revision ready 后、旧 revision 删除前退出，查询层按
`document_id + normalized text hash` 去重，下次恢复完成旧版本清理；这允许短暂重复，
但不允许出现无结果窗口。

只有当前文档的解析和全部 embedding 成功后才能开始写入。一个坏 PDF 不停止整个批次，
但其 document point 必须记录 `failed`、脱敏错误和旧 ready revision（如存在）；旧可用
内容继续服务，不能被失败更新覆盖。

### 10.3 删除同步

只有在论文根目录成功解析且完整枚举完成后，才处理 deleted 文档：

1. document point 标记 `deleted`；
2. 按 `workspace_id + document_id` 删除所有 passage revision；
3. `wait=true` 确认操作被接受；
4. 保留最小 tombstone，记录删除时间和最近内容哈希，支持审计。

根目录缺失、权限错误或扫描中止时不执行任何删除。

### 10.4 初次全量导入

初次导入 10,000 篇论文使用 CLI，而不是要求模型循环数百次调用工具。流程为：

```text
docker compose -f compose.qdrant.yaml up -d
uv run photomatagent rag status
uv run photomatagent rag plan
uv run photomatagent rag index
uv run photomatagent rag evaluate
uv run photomatagent rag snapshot
```

如果使用外部 embedding，`rag plan/index` 在首次发送内容前显示 provider、模型、待处理
文档数和“将发送全文片段到外部服务”的提示；交互式 CLI 要求确认，非交互模式必须显式
传入 `--yes`。计划不声称能在解析前精确估算 token 或费用。

## 11. 检索流程

`literature.search_passages` 执行：

1. 验证 Qdrant、alias、schema、fingerprint 和 provider 就绪；
2. 生成 query dense embedding；
3. 通过 Qdrant Query API 同时 prefetch：
   - dense top 50；
   - sparse BM25 top 50；
4. 两路都强制过滤 `workspace_id`、`record_type=passage`、
   `ingest_state=ready`；
5. 在 Qdrant 内使用 RRF 融合并取最多 50 个候选；
6. 按 document/text hash 去除恢复窗口内的重复 revision；
7. 可选 reranker 对有限候选精排；
8. 选出 `top_k`（仍限制 1–10）；
9. 通过 `previous_passage_id`/`next_passage_id` 批量 retrieve 邻接上下文；
10. 返回受限 passage、score、source、page、section、heading、context 和 diagnostics。

降级策略：

- dense provider 失败但 sparse 可用：返回 sparse-only 结果并标记 degraded；
- sparse 查询失败但 dense 可用：返回 dense-only 结果并标记 degraded；
- reranker 失败：返回 RRF 结果并标记 degraded；
- dense 与 sparse 均失败：返回 typed error，不生成空洞成功结果；
- Qdrant 不可达或 schema/fingerprint 不匹配：返回明确修复指导，不创建空 collection；
- 不允许任何降级路径切换到未显式授权的外部 provider。

`literature.read_passage` 只按 `workspace_id + passage_id` 精确 retrieve，并要求
`ingest_state=ready`。模型返回仍受现有字符上限约束；完整 payload 只存在结构化 data，
也不能无限注入模型上下文。

## 12. 工具、权限与 CLI

### 12.1 模型工具

- `literature.index_papers`
  - 保留名称；
  - `cost_class=EXPENSIVE`；
  - 新增 `max_documents`、`run_id` 和 `resume_cursor`；
  - 默认每次最多处理 20 篇变化文档；
  - 继续经过默认 ASK 权限路径；
  - 外部 provider 还受 `PHOTOMATAGENT_RAG_ALLOW_EXTERNAL` 硬门控制。
- `literature.search_passages`
  - 保留名称、top_k 上限和 provenance 输出；
  - 内部切换为 Qdrant hybrid retrieval。
- `literature.read_passage`
  - 保留 opaque passage_id 合同。
- `literature.extract_evidence`
  - 保持确定性抽取行为，只替换 passage 读取后端。

这些工具仍注册为 `DEFERRED`。Qdrant client、collection 管理、snapshot 或任意 query JSON
不能作为通用模型工具暴露。

### 12.2 用户 CLI

新增 Typer `rag` 命令组：

```text
photomatagent rag status
photomatagent rag plan [--directory PATH]
photomatagent rag index [--directory PATH] [--run-id ID] [--yes]
photomatagent rag search QUERY [--top-k N]
photomatagent rag read PASSAGE_ID
photomatagent rag evaluate
photomatagent rag snapshot [--output PATH]
```

- CLI 调用与 Tool 共用应用服务，不能复制检索或摄取实现；
- `rag status` 是只读探针，输出 Qdrant/version/alias/schema/provider/source root 状态且
  隐藏密钥；
- `rag plan` 不解析全文、不生成 embedding、不写 Qdrant 数据，只创建内存计划；
- `rag index` 是用户直接启动的有界批次循环；Ctrl-C 保存 run cursor，不损坏 ready 数据；
- `rag evaluate` 运行冻结检索评测集；
- `rag snapshot` 调用 Qdrant snapshot API 并把下载副本、SHA-256 和恢复清单写入工作区
  内显式输出目录。

聊天命令路由增加 `/rag`，只复用上述 Typer 命令。`src/photomatagent/cli/commands.py`
继续是 slash-command 中央路由，不新增旁路。

## 13. 能力探针与错误合同

`LiteratureProbe` 分层检查：

1. 缺少 `qdrant-client`、Docling、pypdf 或本地 provider 依赖：
   `MISSING_DEPENDENCY`；
2. Qdrant URL 缺失、外部 provider 未获授权或配置不完整：`UNCONFIGURED`；
3. Qdrant 配置存在但不可达、alias/schema/fingerprint 异常：`ERROR`；
4. Qdrant、provider、source root 和当前 alias 全部就绪：`AVAILABLE`。

基础 runtime 不因 literature probe 失败而退出。探针本身不得下载模型、创建 collection、
写入 Qdrant、修改 `.env` 或执行长时间 embedding。

应用内部使用稳定错误码，例如：

```text
qdrant_unreachable
qdrant_auth_failed
collection_missing
schema_mismatch
model_fingerprint_mismatch
source_root_missing
embedding_unavailable
embedding_dimension_mismatch
external_provider_not_allowed
ingestion_incomplete
passage_not_found
snapshot_failed
```

工具输出可包含错误码、短诊断和下一步指导，但不包含 traceback、密钥、绝对敏感路径或
未截断远端响应。

## 14. 快照、恢复与升级

`rag snapshot` 必须：

1. 解析两个 current alias 的实际 physical collection；
2. 分别创建 Qdrant collection snapshot；
3. 下载到 `.photomatagent/rag/backups/<UTC timestamp>/` 或用户指定的工作区内目录；
4. 生成 `manifest.json`，记录 server version、collection、alias、schema、fingerprint、
   point count、文件大小和 SHA-256；
5. 使用临时文件后原子 rename，部分下载不冒充有效备份。

恢复是显式运维操作，不暴露给模型。恢复步骤先导入到非 current collection，验证 schema、
fingerprint、point count 和冻结检索评测，再切 alias。未知状态不得删除当前 collection。

旧 generation 的默认保留策略为“至少保留最近 1 个已验证 generation”；删除旧 collection
必须是单独、显式、带目标名称的用户操作，不作为 index、evaluate 或 alias 切换的副作用。

## 15. LanceDB 移除与遗留数据策略

实现完成时：

- 删除 `literature/index.py` 的 LanceDB 实现；
- 删除自定义全库内存 `_Bm25`；
- 删除 `lancedb`、`pylance` 依赖和相关 imports；
- 删除 `PHOTOMATAGENT_LITERATURE_INDEX_DIR` 配置；
- 将 README、能力描述、probe 和测试全部改为 Qdrant；
- 全仓搜索不得残留运行时 LanceDB 引用，历史设计文档中的文字记录除外。

现有 `output/literature_index` 是用户生成数据。虽然代码彻底移除 LanceDB，实施过程不会
自动删除该目录，也不会把其中 3 篇/363 个残留片段提升为新的权威来源，因为配置的原始
PDF 根目录当前不存在。`rag status` 将其报告为 legacy artifact，并提供人工清理指引。
用户在 Qdrant 重建、评测和快照完成后可以另行显式删除；该删除不属于本改造的自动步骤。

不提供 LanceDB runtime fallback，也不保留为可选 extra。

## 16. 测试策略

### 16.1 单元测试

不依赖 Docker、网络或真实模型：

- provider 工厂、fingerprint 和维度校验；
- relative path、workspace_id、document_id、passage_id 稳定性；
- 文档计划的 new/changed/unchanged/deleted 分类；
- staged/ready/superseded 状态机；
- 部分 upsert、超时、重试和恢复；
- root missing 不删除；
- 外部 provider 未授权时拒绝构造；
- local 失败不静默外送；
- hybrid RRF、去重、reranker 降级和邻接扩展；
- typed error 和工具输出上限；
- evidence/provenance 字段保持；
- capability probe 软失败；
- slash CLI 仍复用 Typer 路由。

使用 fake Qdrant store 和 fake provider；不以 Qdrant Python local mode 冒充 Docker
integration 行为。

### 16.2 Docker integration tests

由 `PHOTOMATAGENT_RUN_QDRANT_INTEGRATION=1` 显式启用，使用实际 Compose 服务：

- 创建两个 generation collection 和 alias；
- upsert vectorless document point；
- dense+sparse hybrid query；
- payload filter/strict mode；
- 单文档修改和删除；
- staged 数据不可见；
- alias 蓝绿切换；
- 服务重启后数据持久；
- snapshot 创建、下载和恢复到非 current collection；
- qdrant-client 1.19.x 与 server 1.18.2 合同兼容。

测试只使用生成的小型文本/PDF和假向量；不得上传真实论文或调用付费外部模型。

### 16.3 冻结检索评测

在仓库保存小型、许可明确的 fixture 与至少 20 个 query/relevant-passage 标注，覆盖：

- 材料化学式、器件名称、波段和缩写的精确匹配；
- 中英文语义查询；
- 数值/单位检索；
- 同义表达；
- metadata year/source filter；
- 不相关查询。

报告 Recall@5、MRR@10、无结果率、重复率和 provenance 完整率。首期验收要求：

- fixture Recall@5 >= 0.90；
- provenance 完整率 100%；
- ready 结果重复率 0%；
- 外部 provider 关闭时没有任何外部 HTTP 请求；
- dense、sparse、reranker 的降级路径均有明确 diagnostics。

阈值只代表该冻结 fixture，不可宣称代表全部材料文献质量。

### 16.4 性能与容量测试

提供非 CI 的可重复 benchmark，使用预生成向量和合成 payload，至少覆盖 100k 和 1M
points。记录硬件、Qdrant 配置、collection 状态和冷/热查询：

- Qdrant hybrid candidate retrieval p95 目标小于 500 ms；
- 端到端本地检索（含 query embedding 和本地 reranker）p95 目标小于 3 s；
- 查询进程 RSS 不随总 points 数线性加载全部文本；
- 一次 tool call 的返回数量、文本长度和候选数始终受上限约束；
- 中断摄取后可从 run cursor 恢复且不重复 ready 数据。

性能目标在目标工作站实测后判定；未运行 1M benchmark 时不得声称已经验证 10,000 篇
规模性能。

## 17. 文档与操作体验

README 增加完整本地启动路径：

1. 安装 `literature` extra；
2. 启动固定版本 Qdrant Compose；
3. 创建 `dataset/paper` 或配置工作区内其他目录；
4. 运行 `rag status`；
5. 运行 `rag plan/index/evaluate/snapshot`；
6. 在聊天中使用 `literature.search_passages`；
7. 排查 Docker、模型缓存、外部 provider 和 schema/fingerprint 错误；
8. 执行快照恢复和版本升级。

文档必须明确：

- Qdrant index 与 PDF 原件用途不同，快照不能代替 PDF 原件备份；
- external embedding 会发送全部待索引片段；
- external reranker 只发送有限候选；
- 更换 embedding 模型要求重建 generation；
- 更换 reranker 不要求重建；
- 删除 PDF 会在一次成功完整扫描后同步删除检索片段；
- legacy LanceDB 目录不会被自动删除。

## 18. 交付范围与预计修改位置

新增：

- `compose.qdrant.yaml`
- `src/photomatagent/scientific/capabilities/literature/providers/*`
- `src/photomatagent/scientific/capabilities/literature/qdrant_store.py`
- `src/photomatagent/scientific/capabilities/literature/ingestion.py`
- `src/photomatagent/cli/rag.py`
- Qdrant 单元、CLI、集成和检索评测测试
- RAG 运维与恢复文档

修改：

- `pyproject.toml` / `uv.lock`
- `.env.example`
- `README.md`
- `src/photomatagent/scientific/capabilities/config.py`
- `src/photomatagent/scientific/capabilities/literature/__init__.py`
- `src/photomatagent/scientific/capabilities/literature/models.py`
- `src/photomatagent/scientific/capabilities/literature/retrieval.py`
- `src/photomatagent/cli/app.py`
- `src/photomatagent/cli/commands.py`
- `tests/test_literature_rag.py` 及相关 probe/CLI 测试

删除：

- LanceDB `index.py` 实现；如果文件名被 Qdrant store 复用，必须先删除全部 LanceDB
  内容和命名，不能留下兼容分支；
- 自定义内存 BM25；
- LanceDB/pylance 依赖与专属测试。

不得修改 `.env`、真实 PDF、现有 `output/literature_index`、用户产物或无关科学能力。

## 19. 完成定义

只有同时满足以下条件，改造才可宣布完成：

1. Qdrant Compose 可按文档启动并通过 `rag status`；
2. 从空库完成小型 fixture 的 index/search/read/evidence 全链路；
3. 修改、删除、坏 PDF、中断恢复和 staged cleanup 测试通过；
4. 本地 embedding/reranker 模式通过；
5. 外部 embedding/reranker 使用 mock HTTP 合同通过，未授权外送被拒绝；
6. dense+sparse RRF 和降级模式测试通过；
7. alias 蓝绿迁移和 snapshot restore integration test 通过；
8. 冻结检索评测达到第 16.3 节门槛；
9. 全仓运行时代码和依赖不再引用 LanceDB/pylance；
10. legacy index 未被自动删除；
11. `uv run pytest -q` 全部通过；
12. `uv run mypy src` 通过；
13. `git diff --check` 通过并人工检查 `git diff --stat`、`git status --short`；
14. 若未运行 1M 性能 benchmark，最终报告必须明确标记为未验证项。

## 20. 主要风险与控制

| 风险 | 控制 |
| --- | --- |
| 10,000 篇初次解析耗时很长 | CLI 批次、run cursor、单文档隔离、可恢复 |
| 外部 embedding 费用或数据泄露 | 默认关闭、独立硬门、显式 provider、无静默 fallback |
| embedding 维度/语义混库 | fingerprint + versioned collection + alias 切换 |
| 更新中途产生半成品 | staged 不可检索、计数核验、ready promotion、恢复清理 |
| 改动 PDF 后短暂重复 revision | ready-first、查询去重、恢复后清理；不接受空窗 |
| Qdrant 数据损坏或误操作 | named volume、显式 snapshot、校验和、旧 generation 保留 |
| Qdrant 不可用拖垮基础 runtime | lazy import、typed probe、能力软失败 |
| Docker/WSL 挂载问题 | named volume，不把 Qdrant storage bind-mount 到 Windows 路径 |
| 检索质量因数据库迁移退化 | 冻结 fixture、Recall/MRR/provenance 门禁 |
| 模型直接获得数据库权力 | 只暴露 narrow literature tools，Qdrant adapter 内部化 |

## 21. 参考依据

- Qdrant 本地 Docker：<https://qdrant.tech/documentation/quick-start/>
- Qdrant dense+sparse 混合查询与 RRF：
  <https://qdrant.tech/documentation/search/hybrid-queries/>
- Qdrant BM25/full-text：
  <https://qdrant.tech/documentation/search/text-search/full-text-search/>
- Qdrant 自托管与客户端推理边界：
  <https://qdrant.tech/documentation/inference/>
- Qdrant named vectors 和 vectorless point：
  <https://qdrant.tech/documentation/concepts/points/>
- Qdrant collection、HNSW 和 payload index：
  <https://qdrant.tech/documentation/manage-data/collections/>
- Qdrant snapshot 与迁移恢复：
  <https://qdrant.tech/documentation/migration-recovery-options/>
- Qdrant server v1.18.2：<https://github.com/qdrant/qdrant/releases>
- qdrant-client 1.19.0：<https://pypi.org/project/qdrant-client/>
