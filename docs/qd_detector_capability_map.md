# Quantum-Dot IR Detector Capability Map (Sprint 2)

目标任务（DoD 参照）：

> 设计量子点的尺寸、组分、能带及能级结构，在硅基芯片上覆盖 2–5 μm 波段，
> 并达到峰值响应度 R ≥ 0.8 A/W、量子效率 EQE ≥ 20%。

本表诚实列出当前能力与缺口。**任何未标 "available" 的行都不得由 LLM
自行推断数值**；必须返回 `missing_prerequisites` / `insufficient_evidence`。

| Requirement | Current Tool | Fidelity | Available? | Missing Inputs | Next Higher-Fidelity Capability |
|---|---|---|---|---|---|
| spectral constraint (2–5 μm → 光子能量/热限/理想响应度) | `ir.compile_constraints` | analytical | ✅ | — | 实验光谱/吸收测量 |
| size → confinement energy | `qd.brus_transition_energy` | L1 analytical (Brus/EMA) | ✅ | 半径、体带隙、有效质量、介电常数（带来源） | k·p / TB / DFT 限域求解 |
| size 反演（目标波长 → 半径） | `qd.solve_size_for_transition` | L1 analytical | ✅ | 同 Brus | 同上 |
| size 扫描 | `qd.size_sweep` | L1 analytical | ✅ | 同 Brus | 同上 |
| 激子玻尔半径 / 限域 regime | `qd.excitonic_regime` | analytical | ✅ | 有效质量、介电常数 | 实验激子数据 |
| composition → band gap | `alloy.bandgap_bowing` | empirical (quadratic) | ✅ | 端点带隙、bowing 参数（带来源） | 完整组成依赖 k·p/DFT |
| size + composition 联合筛选 | `qd.screen_size_composition` | L1 analytical | ✅ | 全部材料参数 | k·p/TB 逐点验证 |
| 材料参数来源 | `qd.parameter_lookup` | curated examples | ✅ (example-only) | 设计级参数需文献/数据库验证 | RAG 子系统（独立线程） |
| bulk/heterostructure k·p | `kp.run_kdotpy` + `kp.capabilities` | k.p (external) | ✅ (subprocess, isolated venv) | 合法 kdotpy 参数/配置、HOME 可写目录 | 完整 k·p 工作流 |
| 3D QD 电子态（0D 限域能级） | — | — | ❌ NOT YET HIGH-FIDELITY | kdotpy 不支持 0D；需 TB/nextnano/DFT 求解器 | 隔离求解器 MCP/子进程 |
| Si 集成 / QD-Si 界面 | `interface.anderson_band_alignment` | LOW FIDELITY | ✅ (diagnostic only) | 界面偶极/钝化/费米钉扎证据 | 界面 DFT / 实验 XPS |
| R ↔ EQE 转换 | `photodetector.responsivity_from_eqe` / `eqe_from_responsivity` | analytical | ✅ | 波长 + EQE 或 R（含 gain） | 实验光谱响应 |
| R/EQE 目标一致性 | `photodetector.check_targets` | analytical | ✅ | 波段、目标 R、目标 EQE | 器件仿真 |
| defects | `defects.analyze` (doped) | DFT-based | ✅ (env available) | vasprun.xml | 完整缺陷工作流 |
| transport | `transport.analyze` (amset) | — | ❌ MISSING_DEPENDENCY | amset 未安装（boltztrap2 构建缺 cmake） | 隔离 transport 环境/MCP |
| device simulation | `device.run_script` (devsim) | — | ❌ MISSING_DEPENDENCY | devsim 未安装 | 隔离 device 环境/MCP |
| Materials Project 数据库 | `materials.*` (native) + `materials_mcp.*` (MCP) | database | ✅ | API key | 高保真计算数据交叉验证 |
| 文献证据 | `literature.search_arxiv` | literature | ✅ (search only) | — | RAG（独立线程） |

## 无幻觉不变量

1. 工具层以 typed 错误返回 `missing_prerequisites` / `insufficient_evidence`，
   禁止 LLM 用自身知识填充参数后"假装计算"。
2. 所有确定性结果携带 `fidelity`（analytical / empirical / kp / tight_binding /
   dft / experimental）与 `assumptions` / `limitations`。
3. Brus 等 L1 结果永远不是设计值；HgTe/PbTe 等窄带隙或反转带系统会输出
   强 warning。
4. MCP 外部结果以 `source_type=database` + `trust_level` 标注，不冒充本地计算。

## 端到端推荐链路（当前可用部分）

```text
ir.compile_constraints(2–5 μm)
  → qd.parameter_lookup / materials.* (数据库带隙)
  → alloy.bandgap_bowing (组分)
  → qd.screen_size_composition (组分×尺寸候选)
  → qd.solve_size_for_transition (反演)
  → photodetector.check_targets (R/EQE 一致性)
  → kp.run_kdotpy (k·p 佐证, 若适用)
  → 明确 missing：3D QD 高保真电子态、Si 界面、吸收、transport、device
```
