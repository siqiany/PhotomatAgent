# Third-Party Scientific Tools (Sprint 2)

PhotoMatAgent 不复制外部科学软件代码；通过 `native package`、隔离
`subprocess` 或 `MCP server` 调用。下表记录当前实际使用的第三方工具。

| Tool | Project | Version | License | Execution Mode | Citation |
|---|---|---|---|---|---|
| `mp-api` (native `materials.*` + official MCP `mpmcp`) | Materials Project | 0.46.4 | modified BSD | native + MCP stdio | Ong et al., Comput. Mater. Sci. 161, 211 (2019); Materials Project API |
| `mcp` Python SDK | modelcontextprotocol/python-sdk | 1.29.0 | MIT | MCP client transport | MCP specification (Anthropic, 2025) |
| `fastmcp` | jlowin/fastmcp | 2.14.1 | MIT | MCP server runtime (via `mpmcp`) | — |
| `pymatgen` | materialsproject/pymatgen | 2026.5.4 | modified BSD | native | Ong et al., Comput. Mater. Sci. 49, 2295 (2013) |
| `sumo` / `effmass` | SMTG-Bham | 2.4.0.post1 / 2.1.0 | MIT / BSD | native (post-processing only) | Ganose et al., JOSS 3, 717 (2018); Whalley et al., JOSS 3, 797 (2018) |
| `doped` (defects) | birbette/doped | 3.2.1 | MIT | native (analysis only; requires vasprun.xml) | Mosquera-Lois & Kavanagh, npj Comput. Mater. 10, 105 (2024) |
| `pytaser` (optics) | pytaser | 2.3.1 | GPL-3.0 | native package (algorithm unchanged) | Shterengas & Belenky (2016) |
| `kdotpy` | kdotpy collaboration (Univ. Würzburg) | 1.4.1 | GPL-3.0 | **isolated subprocess** (`.venvs/kdotpy`) — GPL code never imported into PhotoMatAgent | kdotpy docs (2026); cite per project guidance |
| `mp-api[mcp]` server deps (emmet-core, ...) | Materials Project | — | BSD-style | transitively loaded by `mpmcp` | — |
| `arXiv` / `pypdf` | arxiv.org SDK | 4.x / 6.x | MIT / BSD | native (literature search) | — |
| VASP | VASP Software GmbH | 6.x (SCNet module) | **commercial license; never redistributed** | remote Slurm job via SCNet MCP (`vasp.*`); input generation only locally; POTCAR resolved from the user's `PMG_VASP_PSP_DIR` or remote location | Kresse & Furthmüller, Phys. Rev. B 54, 11169 (1996) |
| Hefei-NAMD | Qijing Zheng, USTC (official repo: QijingZheng/Hefei-NAMD) | SCNet module (probed) | free for academic use per project page | remote Slurm job via SCNet MCP (`namd.*`); adapter only prepares/validates the VASP trajectory + WAVECAR bridge | Zheng et al., Comput. Phys. Commun. 288, 108745 (2023) |
| MAGUS | Xia et al. (magus-software) | SCNet module or local binary (probed) | open source (project page) | remote/local execution (`magus.*`); probe-gated, UNCONFIGURED when absent | Xia et al., Comput. Phys. Commun. 295, 109021 (2024) |
| MatterGen | Microsoft Research AI4Science | `dft_band_gap` / `chemical_system` checkpoints | MIT (code), model weights per HF hub terms | **isolated subprocess** (`generation.mattergen`); never loaded into the main venv | Zeni et al., Nature 626, 345 (2024) |
| NIST JARVIS-DFT data | NIST | 3D v11 / 2D v8 snapshots | CC BY 4.0 | committed training snapshots and derived VAE/index assets (`generation.vae_formula`, `generation.vae_retrieve`) | Choudhary et al., npj Comput. Mater. 6, 173 (2020); Figshare DOIs `10.6084/m9.figshare.6815699.v11`, `10.6084/m9.figshare.6815705.v8` |
| CHGNet | Materials Virtual Lab (UC Berkeley) | 0.3.0 | BSD-3-Clause | optional isolated backend (cheap screening; never a DFT substitute) | Deng et al., Nat. Mach. Intell. 5, 1031 (2023) |
| Meep | MIT (Joannopoulos group) | 1.x | GPL-2.0 | **isolated environment** (`optics.meep_thinfilm`); 1D thin-film only | Oskooi et al., Comput. Phys. Commun. 181, 687 (2010) |

## 许可注意事项

* **GPL 工具（kdotpy、pytaser）只通过子进程/独立包调用**，不把其源码
  合入 PhotoMatAgent；PhotoMatAgent 自身保持其既有许可。
* `amset`（transport）与 `devsim`（device）未安装（见
  `photomatagent scientific status`）；恢复时建议采用隔离环境 + MCP/子进程
  模式，避免把重依赖压入主环境。
* VASP / Hefei-NAMD / MAGUS 通过 SCNet MCP 以远程 Slurm 作业方式调用；
  PhotoMatAgent 只做输入生成、提交、状态轮询、下载与结果验证，**不复制
  任何商业/受限软件代码**（POTCAR 内容绝不入库、绝不记录）。
* MatterGen / Meep 采用隔离环境（conda/venv），与主环境完全隔离。
* JARVIS 原始归档、派生候选表、索引和 VAE 权重随仓库分发，遵循
  CC BY 4.0；再分发和发表结果时必须保留 NIST/JARVIS 归属与上述 DOI。
* 未来接入 GPAW / nextnano 时同样遵循：不复制代码、以子进程或 MCP 调用、
  记录许可证与引用。
