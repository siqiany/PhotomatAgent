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

### SCNet `.env` 最小配置

先在商城开通对应软件，再从商品的“命令行”页面确认登录地址、软件模块名和
启动命令。队列不能照抄文档示例；用 MCP 的 `scnet_partitions`（远端优先
执行 `whichpartition`）读取当前中心可用队列。

```dotenv
SCNET_HOST=登录节点地址
SCNET_USERNAME=用户名
SCNET_PORT=22
SCNET_PRIVATE_KEY_PATH=/绝对路径/id_scnet
SCNET_REMOTE_ROOT=~/photomatagent
SCNET_PARTITION=通过 scnet_partitions 确认的队列

# SCNet VASP 6.4.2 商品文档中的模块；若商品终端显示不同名称，以终端为准
SCNET_VASP_MODULE=vasp-6.4.2-intelmpi2017_ioptcell
# 若商品自带 case/vasp.slurm 使用 source env.sh，则配置其远端绝对路径
# SCNET_VASP_ENV_SCRIPT=/public/home/USER/apprepo/vasp/VERSION/scripts/env.sh
# 二选一：本地赝势库，或 SCNet 上的远端赝势库（都包含 potpaw_PBE.64/）
PMG_VASP_PSP_DIR=/本地/vasp_psp
# SCNET_VASP_PSP_DIR=~/path/to/vasp_psp

# Hefei-NAMD 必须填写商品/自编译环境的真实模块名和可执行文件名
SCNET_NAMD_MODULE=hefei-namd/实际版本
SCNET_NAMD_EXECUTABLE=namd
# 若商品案例使用 source env.sh，填写其远端绝对路径
# SCNET_NAMD_ENV_SCRIPT=/public/home/USER/apprepo/hefei_namd/VERSION/scripts/env.sh

# WAVECAR 树通常很大，传输超时独立于 SSH 连接超时
SCNET_CONNECT_TIMEOUT_SECONDS=20
SCNET_TRANSFER_TIMEOUT_SECONDS=3600

# 真实提交必须显式开启，并限制最大资源与允许队列
PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1
PHOTOMATAGENT_HPC_MAX_NODES=1
PHOTOMATAGENT_HPC_MAX_TASKS_PER_NODE=64
PHOTOMATAGENT_HPC_MAX_WALLTIME_MINUTES=720
PHOTOMATAGENT_HPC_ALLOWED_PARTITIONS=你的队列
```

配置后先执行只读诊断，不会提交或计费：

```bash
uv run photomatagent-mcp-scnet --load-dotenv --doctor
```

报告中应同时满足：SSH `connected=true`、`slurm_ready=true`、目标队列出现在
`available_partitions`，且 VASP/NAMD 的 `software.available=true`。SCNet 文档
明确要求作业脚本与输入文件在同一目录，通过 `sbatch` 提交，并用 `squeue` /
`sacct` 查询；本实现遵循该流程，且不会在登录节点直接运行计算。

若 SSH 返回 `Permission denied (publickey...)`，按 SCNet 的连接文档回到
E-Shell 的“SSH连接”重新选择有效期并下载密钥，同时复制该密钥对应的主机、
端口和用户名；这四项必须来自同一次连接信息。旧密钥、另一中心的用户名或
过期有效期都会在进入 Slurm 前被拒绝。

### VASP MCP 运行链

1. `vasp_capabilities`：确认 SSH、Slurm、队列、VASP 模块与赝势策略。
2. `vasp_prepare`：生成 POSCAR/INCAR/KPOINTS 和工作流。
3. `vasp_submit`：显式传入 `scnet_partitions` 返回的队列；脚本执行
   `module purge`、`module load`、`srun --mpi=pmi2 vasp_std|vasp_ncl`，并设置
   `ulimit -s unlimited`。
4. `vasp_status`，结束后 `vasp_collect`；调度状态 COMPLETED 不替代科学结果校验。

### Hefei-NAMD MCP 运行链

Hefei-NAMD 官方 VASP 工作流需要 `inp`、`INICON` 和保持层级的
`run/0001.../WAVECAR`。`namd_prepare` 会复制完整快照树，避免同名 WAVECAR
被覆盖；运行输入可用两种方式提供：

- 传入版本匹配的 `inp_path` 与 `inicon_path`；
- 传入完整 `parameters`（BMIN/BMAX/NBANDS/NSW/POTIM/TEMP/NSAMPLE/
  NAMDTIME/NELM/NTRAJ/LHOLE）和 `initial_conditions=[[起始步, 能带], ...]`。

随后调用 `namd_submit`、`namd_status`、`namd_collect`。提交前会校验
`NSAMPLE`、能带范围以及 `起始步 + NAMDTIME <= NSW`；远端脚本加载配置的
Hefei-NAMD 模块后直接运行 `namd`。结果收集覆盖官方输出 NATXT、EIGTXT、
COUPCAR、PSICT.* 与 SHPROP.*。

### MAGUS MCP 运行链（Sprint 4）

MAGUS 是安装在 SCNet 的进化结构搜索程序（当前实测版本 **2.1.0**，conda
环境，可执行文件 `<SCNET_MAGUS_ROOT>/bin/magus`）。PhotoMatAgent 把它当作
SCNet Scientific MCP 下的 application capability，绝不在本地 fallback。

```text
LLM → tool_search("structure search") → magus.capabilities (真实远程 probe)
    → magus.prepare_generate | magus.prepare_search (本地确定性 renderer)
    → magus.submit (policy + POTCAR 前置检查 + SSH 上传 + sbatch)
    → magus.status → magus.collect (有界 artifacts) → magus.inspect_results
    → CandidateRecord / 有界 ScientificEvidence → Agent
```

#### MAGUS `.env` 配置

```dotenv
# MAGUS 安装根目录（conda 环境，含 bin/magus）
SCNET_MAGUS_ROOT=~/magus
# 可选：显式可执行文件绝对路径（否则按 <root>/bin/magus -> <root>/magus
# -> root 内 maxdepth 4 bounded find 探测）
# SCNET_MAGUS_EXECUTABLE=/public/home/USER/magus/bin/magus
# 可选：激活 MAGUS python 环境的脚本（默认自动设置 PATH=<root>/bin）
# SCNET_MAGUS_ENV_SCRIPT=/path/to/magus_env.sh
# 可选：MAGUS+VASP 时激活 VASP 工具链的 env.sh（不设置则复用
# SCNET_VASP_ENV_SCRIPT；两者都无则 environment 报 MISSING）。这只是
# “VASP 环境初始化脚本”，不是 ASE 的执行命令（见
# SCNET_MAGUS_ASE_VASP_COMMAND）。
# SCNET_MAGUS_VASP_SCRIPT=/public/home/USER/apprepo/vasp/VERSION/scripts/env.sh

# ASE VASP_PP_PATH 语义：指向“包含 potpaw_PBE/ 的父目录”，ASE 自行拼
# potpaw_PBE/<setup>/POTCAR。与 SCNET_VASP_PSP_DIR（精确库目录，
# $SCNET_VASP_PSP_DIR/<setup>/POTCAR）语义不同，不要混淆。
SCNET_MAGUS_VASP_PP_PATH=/public/home/USER

# ASE 在 MAGUS 内实际执行 VASP 的命令（如 "srun --mpi=pmi2 vasp_std"）。
# 必须先在真实 SCNet 环境验证，否则 MAGUS+VASP readiness 报 PARTIAL；
# 未配置时绝不自动提交 VASP。
# SCNET_MAGUS_ASE_VASP_COMMAND=srun --mpi=pmi2 vasp_std
```

#### 已验证的 MAGUS 2.1.0 事实（Sprint 4 远程探测）

* `magus -v` → `2.1.0`；子命令含 `search/summary/clean/prepare/calculate/
  generate/checkpack/test/update/tool/mutate`（无 `parmhelp`）。
* `magus generate -i input.yaml -o gen.traj -n N`、`magus search -i input.yaml`
  为实际 CLI 语义（`magus generate -h` / `magus search -h` 实测）。
* `checkpack calculators` 实测可用：`emt espresso gulp KIM lj lammps mtp*
  vasp vaspc abacus castep confine dftb siesta`；torch 系（deepmd/mace/
  m3gnet/nep/quip/tblite/xtb/hotpp）因缺 torch/matgl 报 plugin 失败，不影响
  VASP calculator。
* `JOB_SYSTEM=SLURM` 时 MAGUS 会嵌套 sbatch（`#SBATCH --partition=...`），
  因此在 PhotoMatAgent 的 Slurm 分配内只支持 `execution_mode=serial`
  （`MainCalculator.mode: serial`，进程内跑 calculator，无嵌套提交）。
* input.yaml 键名以安装包官方 example 为准（generate: `01--1-B12`；VASP
  search: `03--2-Al-fix-VASP`；cluster: `05--1-LJ26`；surface: `06--1-*`）：
  `formulaType / structureType / symbols / formula / min_n_atoms /
  max_n_atoms / spacegroup / d_ratio / volume_ratio / pressure / initSize /
  popSize / numGen / saveGood / rand_ratio / add_sym / MainCalculator`.

#### MAGUS 工具与错误契约

```text
magus.capabilities        只读远程 probe（root/executable/version/commands/
                          calculators/structure types/JOB_SYSTEM/VASP readiness）
magus.prepare_generate    生成 input.yaml + magus.slurm + manifest（不提交）
magus.prepare_search      生成 input.yaml + inputFold/VASP/INCAR + Slurm + manifest
magus.submit              上传 + sbatch（需 ALLOW_HPC_SUBMIT=1 + resource policy；
                          VASP search 先做远程 POTCAR 前置检查）
magus.status / collect / inspect_results
```

错误契约：`UNCONFIGURED`（SCNet 未配）、`MISSING_DEPENDENCY`（root/
executable 不存在）、`MISSING_PREREQUISITE`（缺 slab 配置、缺 VASP launcher）、
`MISSING_PSEUDOPOTENTIALS`（缺 POTCAR setup）、`SUBMISSION_BLOCKED`（策略
拒绝）、`EXECUTION_FAILED`。**从不**用 LLM 猜测替代结果。

#### 赝势布局（Sprint 4 修复）

旧实现硬编码 `potpaw_PBE.64`；现改为三布局确定性探测（`vasp/psp.py`）：

```text
direct:       <configured>/<setup>/POTCAR           （SCNET_VASP_PSP_DIR 精确库目录）
potpaw_PBE:   <configured>/potpaw_PBE/<setup>/POTCAR （ASE VASP_PP_PATH 父目录）
potpaw_PBE.64:<configured>/potpaw_PBE.64/<setup>/POTCAR（legacy）
```

VASP Slurm 脚本在作业内按上述顺序自动定位库目录并组装 POTCAR；MAGUS
VASP search 提交前用 `test -f` 检查每个所需 setup，缺则返回
`MISSING_PSEUDOPOTENTIALS`（绝不先提交再失败）。POTCAR 内容永不读写到
模型上下文、日志或 git。
