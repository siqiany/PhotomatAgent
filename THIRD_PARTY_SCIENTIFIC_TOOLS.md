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

## 许可注意事项

* **GPL 工具（kdotpy、pytaser）只通过子进程/独立包调用**，不把其源码
  合入 PhotoMatAgent；PhotoMatAgent 自身保持其既有许可。
* `amset`（transport）与 `devsim`（device）未安装（见
  `photomatagent scientific status`）；恢复时建议采用隔离环境 + MCP/子进程
  模式，避免把重依赖压入主环境。
* 未来接入 GPAW / nextnano / VASP 时同样遵循：不复制代码、以子进程或 MCP
  调用、记录许可证与引用。
