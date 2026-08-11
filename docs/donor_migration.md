# Donor Migration Report (Sprint 3)

来源：`github.com/siqiany/photoelectric-detection`
（`Agent/src/photoelectric_agent/...`）。

原则：`extract → simplify → adapt → test`；不复制整个旧项目；不迁移
LangGraph graph / nodes / 旧 manager / CLI / AgentState（当前 runtime
已替代这些）。

## Donor → Target 映射

| Donor File | Target Module | Reuse % | Modification |
|---|---|---|---|
| `tools/vasp_cloud.py` | `scientific/remote/scnet.py` (SCNetBackend) + `scientific/applications/vasp/application.py` (VaspApplication) | ~55% (backend) / ~40% (app) | SSH/Slurm 抽成 generic backend（async subprocess、argv list、无 shell=True、路径白名单、job id 校验、redaction、timeout、bounded output）；提交默认关闭 + ResourcePolicy |
| `tools/vasp_input_generator.py` | `scientific/applications/vasp/inputs.py` + `profiles.py` | ~60% | INCAR 分支改为 4 个带来源注释的 profile（standard_semiconductor / narrow_gap_soc / optics / namd_preparation）；SOC 设置按 VASP wiki 重写（vasp_ncl、GGA_COMPAT=.FALSE.、LASPH、ISYM=-1、LREAL=.FALSE.）；KPOINTS 密度公式确定性化；POTCAR 只写 policy 清单，绝不生成/提交内容 |
| `tools/vasp_workflow.py` | `scientific/applications/vasp/workflow.py` | ~70% | 相同依赖传播（CONTCAR→POSCAR、CHGCAR 拷贝）与逐阶段验证；轮询改为 async；仍保留 approval policy 语义（改由 ResourcePolicy 承担） |
| `tools/meep_adapter.py` | `scientific/capabilities/optics/meep_thinfilm.py` | ~75% | 改名 `optics.meep_thinfilm`；scope 明确为 1D thin-film（非 3D 器件、非 transport、非 EQE 仿真）；n/k 转换保留 |
| `tools/optical_data.py` | 同上（`optical_point_from_vasprun` / `dielectric_to_nk` / `interpolate_dielectric`） | ~80% | 保持；返回 dict 契约，附来源 |
| `models/vae_formula.py` | `scientific/capabilities/generation/formulas.py` | ~65% | 删除默认 forbidden_elements=[Hg,Cd] 与 prefer_lower_atomic_number（§46）；torch 解码 dependency-optional（decoder 注入），缺失时 typed missing_prerequisites；integerization/氧化态/电中性/novelty 纯 Python 保留 |
| `models/vae_generator.py` | `generation/tools.py` (`generation.vae_retrieve`) | ~25% | 只保留检索语义（轻量 CSV 索引）；heavy inverse-retrieval 结构归档不迁移（无数据资产） |
| `models/mattergen.py` | `scientific/capabilities/generation/mattergen.py` | ~60% | 保留 isolated subprocess provider 与 manifest 解析；新增 formula consistency 契约（vae_proposed_formula / mattergen_generated_formula / formula_preserved / composition_distance，§49）+ CandidateLineage（§50） |
| `evaluation/verdict.py` | 未迁移 | — | verdict 逻辑属于旧 LangGraph workflow；当前 Agent Loop 动态决定证据缺口，不需要固定 verdict 管道（discarded，理由：§1 禁止固定 workflow） |
| `graph.py` / `nodes.py` | 未迁移 | — | LangGraph orchestration 已由当前 runtime 替代（discarded） |
| `tools/chgnet_adapter.py` | 未迁移（Phase I optional） | — | 当前无 CHGNet 需求；若需要，按 `ml_interatomic_potential` evidence 语义迁移，且不把 CHGNet 能量当 DFT evidence |
| `schemas.py` (Candidate/SimulationResult) | 未迁移 | — | 当前用 `ScientificEvidence` + `RemoteJobRef`/`ScientificToolResult` 契约替代 |

## 科学原因摘要

* Anderson 带对齐改为真空能级约定（`Ec=-χ, Ev=-(χ+Eg)`），修正旧实现符号
  反转（Sprint 3 §5）。
* Brus 反演区分数学解与科学有效解（SOLVED / NO_MATHEMATICAL_SOLUTION /
  OUTSIDE_MODEL_VALIDITY / AMBIGUOUS_BRANCH，§8）。
* 介电常数类型化（static/optical/high_frequency/unknown），不兼容时返回
  INCOMPATIBLE_SCIENTIFIC_PARAMETER（§7）。
* 候选生成 ≠ 性能验证：一切生成候选默认
  `UNVALIDATED_GENERATED_STRUCTURE`（§51/§83）。
