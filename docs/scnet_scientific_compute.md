# SCNet Scientific Compute (Sprint 3)

## 架构

```text
PhotoMatAgent
      │  MCP stdio (official mcp SDK client, mcp>=1.24)
      ▼
Local SCNet MCP server  (photomatagent-mcp-scnet, official FastMCP SDK)
      │  system OpenSSH (BatchMode, argv lists, quoting, timeouts)
      ▼
SCNet login node
      │  Slurm (sbatch / squeue / sacct / scancel)
      ▼
Scientific applications (VASP, Hefei-NAMD, MAGUS)
```

SCNet 只作为 remote compute backend；MCP server 运行在用户本地
（WSL），不需要在 SCNet 登录节点常驻任何服务。

## 代码分层

| 层 | 模块 | 职责 |
|---|---|---|
| Backend | `scientific/remote/` | generic SSH + Slurm：`SCNetBackend`、`RemoteJobRef/Spec`、`ResourcePolicy`、Slurm 状态映射、artifact ref |
| Application | `scientific/applications/vasp|namd|magus/` | 应用知识：VASP profiles/INCAR/validation、NAMD trajectory/WAVECAR bridge、MAGUS probe/prepare |
| Tools | 各 application 的 `tools.py` | 暴露给 Agent 的 deferred 工具（`vasp.*`、`namd.*`、`magus.*`） |
| MCP | `mcp_servers/scnet/` | FastMCP stdio server，把 application 工具注册为 MCP tools |
| Gateway | `mcp/manager.py`（既有） | 启动 server、initialize、list_tools、call_tool；失败降级为 `<namespace>.status` stub |

调用链（一个 VASP 作业）：

```text
LLM → tool_search("DFT band structure") → vasp.prepare (本地输入生成, 无提交)
    → vasp.submit (policy 校验 + SSH 上传 + sbatch) → RemoteJobRef (detached)
    → vasp.status (squeue/sacct) → vasp.collect (下载 + vasprun 校验 + parse)
    → ScientificEvidence(source_type=dft_calculation) → Agent
```

## 安全边界（不可协商）

* 不暴露 generic remote shell：模型只能看到 `vasp.*` / `namd.*` /
  `magus.*` 应用工具，永远没有 `scnet.run_shell(command)`。
* HPC 提交默认关闭：需要 `PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1` **并且**通过
  deterministic `ResourcePolicy`（max_nodes / max_tasks_per_node /
  max_walltime / allowed_partitions）。LLM 无法绕过资源上限。
* SSH 使用 argv list（无 `shell=True`）、remote path 一律
  `shlex.quote` + 字符白名单、job id 严格数字校验、每次调用有 timeout、
  stdout/stderr 有界截断。
* private key path 不进入 model context / 日志 / status 输出
  （`RemoteServerConfig.public_dict()` 永远移除）。
* POTCAR 内容绝不生成、绝不提交入库、绝不记录：只解析
  `PMG_VASP_PSP_DIR`（本地）或远端赝势位置。
* Slurm COMPLETED ≠ 科学有效：`vasp.collect` / `validate_output` 必须
  通过 vasprun.xml 契约（well-formed、SCF 收敛标记、relax 的离子收敛）
  才产生 `dft_calculation` evidence。

## 常用命令

```bash
uv run photomatagent scientific status            # Capability/Execution Mode/Status/Backend
uv run photomatagent scientific scnet-doctor      # SSH + Slurm + VASP/NAMD/MAGUS probe
photomatagent-mcp-scnet --doctor                  # MCP server 的诊断模式
```

## 离线测试策略

默认 `uv run pytest` 不触碰 SCNet：`FakeSCNetBackend`（内存文件系统 +
scripted state progression）覆盖 submit/status/download/failure/timeout/
OOM/cancel。Live 测试全部 gated：

```text
PHOTOMATAGENT_RUN_LIVE_SCIENCE=1        # 允许 connection probe（无费用）
+ PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1      # 才允许真实作业提交
```

## 配置

`.photomatagent/mcp.json` 的 `scnet` server 条目（见
`.photomatagent/mcp.json.example`）通过 `SCNET_HOST` / `SCNET_USERNAME` /
`SCNET_PRIVATE_KEY_PATH` 等环境变量（含 workspace `.env` 兜底）配置；
server 在 SCNet 未配置时也能启动，工具返回 typed
`missing_prerequisites`。
