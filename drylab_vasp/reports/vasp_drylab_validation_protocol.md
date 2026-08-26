# VASP 干实验验证协议 —— Pb₁₋ₓSnₓSe 量子点玻璃 MWIR 探测器
## 最可行方案的选择与第一性原理验证设计

**版本**: v1.0 ｜ **关联报告**: `reports/mwir_uncooled_glass_detector_design.md`
**状态**: 输入已全部离线准备完成（5 体系 × 4 阶段 = 20 个 VASP 计算）；后端 SCNet 当前连接断开（ssh 超时），**提交待连接恢复**，本协议即"干实验设计+判据"。

---

## 0. 最可行方案（从上一报告中遴选）

从上一报告的五维方案中，选出**可行性最高、VASP 可验证性最强**的路线：

> **方案 A（主选）：Pb₁₋ₓSnₓSe 量子点玻璃陶瓷光电导探测器，合金窗口 x(Sn) ∈ [0.08, 0.20]**
> 吸收体 = 岩盐结构 Pb₁₋ₓSnₓSe（x≈0.12–0.20 为设计靶点，MP 已有近基态有序化合物 SnPb4Se5, x=0.2, e_hull=0.0024 eV/atom）；
> 基体 = 宽隙 Ge-As-Se 玻璃（以 GeSe2/As2Se3 结晶类似物为 DFT 代理，MP e_hull=0, gap≈1.44 eV）；
> 器件 = 光电导叉指电极结构（室温、无制冷）。

**选择理由**：
1. **结构可计算性**：岩盐 Pb-Se 与有序 Pb-Sn-Se 合金已有 MP 结构（小晶胞 2–10 原子），可直接做高保真 DFT；
2. **证据完备度最高**：PbSe 实验带隙 0.28 eV（RT）已知、SnPb4Se5 近基态稳定、合金化趋势在 MP 数据中已有体现（PbSe 0.43 → SnPb4Se5 0.278 → SnTePbSe 0.221 eV，PBE）；
3. **验证闭环**：VASP 可直接回答"合金带隙是否落入 0.25–0.31 eV 设计窗 + 基体是否透明 + 能带/有效质量是否支持输运"，这是器件性能预测的前提；
4. 备选（HgTe QD、Ge-As-Te 玻璃自光电导）因毒性/法规或 DFT 半金属风险（Sb₂Te₃ gap=0）列为 B/C 路线，不进入本轮干实验。

---

## 1. VASP 计算矩阵（已生成，20 个作业）

| # | 体系 | MP id | 结构 | Profile | 物理问题 | 晶格/k点 |
|---|---|---|---|---|---|---|
| S1 | **PbSe** | mp-2201 | 岩盐，2 原子 | `narrow_gap_soc` (vasp_ncl) | **SOC 参考端点**：PBE+SOC 带隙、L 点直接带隙、CBM/VBM 有效质量 | 7×7×7 |
| S2 | **SnPb4Se5** (x=0.2) | mp-1218958 | 岩盐衍生有序，10 原子 | `narrow_gap_soc` (vasp_ncl) | **目标合金**：带隙是否落入设计窗、与 PbSe 的 ΔEg、能带色散 | 4×4×4 |
| S3 | **SnTePbSe** (x=0.5) | mp-1218892 | 4 原子（PbTe-SnSe 混盐） | `narrow_gap_soc` (vasp_ncl) | **高 Sn 端点**：带隙坍塌趋势 → bowing 曲线的 DFT 锚点 | 6×6×6 |
| S4 | **GeSe2** | mp-540625 | 层状，48 原子 | `standard_semiconductor` (vasp_std) | **宽隙基体代理**：基体透明性（Eg ≥ 1.3 eV PBE） | 4×2×1 |
| S5 | **As2Se3** | mp-909 | 层状，20 原子 | `standard_semiconductor` (vasp_std) | **基体备选**：同上，交叉验证基体设计 | 6×2×2 |

每体系 4 阶段流水线：`01_relax`（ISIF=3 全弛豫）→ `02_static` → `03_band`（ICHARG=11）→ `04_dos`（NEDOS=2000）。
公共设置：PREC=Accurate、ENCUT=450 eV、EDIFF=1e-5、ISMEAR=0/SIGMA=0.05（窄隙半导体，需在 OUTCAR 检查占据展宽）、LORBIT=11。
SOC 体系：LNONCOLLINEAR=T、LSORBIT=T、GGA_COMPAT=F、LASPH=T、LMAXMIX=4（d 电子）、LREAL=Auto（relax）/False（静态）、ISYM=-1（非共线必需）。
POTCAR：提交时从 PMG_VASP_PSP_DIR 解析（policy-only，不入库）。

**资源评估**: cost_class=EXPENSIVE；SOC 体系（S1–S3, 12 个作业）占绝大部分成本；S4/S5（8 个作业）较轻。预算紧张时可先跑 S2（目标合金）的 relax+static+band，再补齐 S1 作 ΔEg 基准。

---

## 2. 验证问题与成功判据（Gates）

### G1 — 合金带隙进入设计窗（核心门，否决性）
- **问题**: SnPb4Se5 (x=0.2) 的 PBE+SOC 带隙是否显著低于 PbSe，且剪刀差校正后落在 0.18–0.27 eV？
- **方法（关键，诚实处理 PBE 低估）**: 用**相对位移 + 实验锚定**，不用绝对 PBE 值：
  - ΔEg_DFT(x) = Eg_DFT(SnPb4Se5) − Eg_DFT(PbSe)（同方法同收敛参数，系统误差相消）；
  - 预测实验尺度带隙: Eg_pred(x) = Eg_exp(PbSe, RT=0.28 eV) + ΔEg_DFT(x)。
- **判据**: Eg_pred(x=0.2) ∈ [0.18, 0.27] eV ⇔ λc ∈ [4.6, 6.9] μm；且 Eg_pred(x=0.2) < Eg_pred(x=0)（单调下降趋势与 bowing 预注册方向一致）。
- **预注册（经验 bowing, RT 端点 PbSe=0.28 / 岩盐 SnSe=−0.15 / b=0.1）**: x=0.08→0.238 eV (5.20 μm)；x=0.12→0.218 eV (5.69 μm)；x=0.20→0.178 eV (6.97 μm)。**DFT 结果用于检验此趋势，而不是背书此拟合**。

### G2 — 基体透明性（必需门）
- **判据**: GeSe2 与 As2Se3 的 PBE 带隙 ≥ 1.3 eV（PBE 低估 → 真实 ≥ 1.5 eV），确保 3–5 μm（≤0.41 eV）光子不被基体吸收、基体不成为漏电通道。MP 参考：1.44/1.44 eV。

### G3 — 带边有效质量支持输运（性能门）
- **问题**: CBM/VBM 处 L 点直接带隙的带边有效质量（电/空穴）是否轻（≤0.1 m₀ 量级）→ 迁移率代理。
- **判据**: m* ≤ 0.15 m₀（各向异性需报方向）；同时确认 **L 点直接带隙**（PbSe 族特征，直接吸收 → 高 α）。
- **输出**: electronic.effective_mass + electronic.band_summary（band 阶段产物）。

### G4 — 稳定性（支持门）
- **判据**: relax 后能量与 MP 一致性、无虚频性检查（ISIF=3 收敛）；SnPb4Se5 的 e_hull=0.0024 eV/atom（MP）佐证近基态可合成；形成能趋势：E(SnPb4Se5) ≈ 组分平均 + 有序稳定化。

### G5 — DOS 与费米能级（支持门）
- **判据**: dos_summary 的带隙与 band_summary 一致；费米能级位于带隙内（本征）或靠近带边（按掺杂设计）；价带顶 DOS 峰位（Te/Se 主导）与文献一致。

---

## 3. 结果解读协议（防误用清单）

1. **绝不用 PBE 绝对带隙当设计值**: PbSe PBE+SOC 预期显著低于实验 0.28 eV（PBE 已知低估、SOC 进一步压隙）；一切设计结论走 ΔEg 相对量 + 实验锚定（G1 方法）。
2. **合金有序性局限**: SnPb4Se5 是有序超胞（x=0.2 规则排列），真实玻璃中 QD 为无序合金（含 Sn 团簇/偏析）——DFT 值代表"理想有序"上限，需结合合金无序修正（SQS/大超胞）作为后续扩展；报告中应标注此偏差。
3. **体相 ≠ QD**: VASP 计算的是体相合金带隙；QD 限域蓝移需另行建模（kp.run_kdotpy 或大超胞 QD），Brus L1 已提示体相 m*=0.04 m₀ 下 R=6 nm 限域达 0.52 eV（L1 对窄隙体系是悲观上界）——设计取 R≥12–20 nm（d≈24–40 nm）或降低体相 Eg（x↑）以补偿。
4. **计算≠器件**: 本干实验验证材料层面（带隙/基体/质量/稳定性），不验证 D*/R（需输运+噪声+器件模拟，属 G2–G4 实验阶段）。

---

## 4. 后续扩展（预算允许时的优先级）

| 优先级 | 计算 | 目的 |
|---|---|---|
| P1 | HSE06(+SOC) 单点（S1、S2 静态态上） | 更准带隙，校验 PBE+ΔEg 锚定法 |
| P2 | optics profile（SOC 手动叠加）于 S2 | 吸收谱 α(λ)，连接 η/EQE |
| P3 | SQS 无序超胞 x=0.12 | 真实合金无序修正 |
| P4 | QD 大超胞（>500 原子）或 k·p | 限域蓝移定量，锁 QD 尺寸窗口 |
| P5 | 缺陷计算（Se 空位、Sn 反位） | 暗电流/陷阱物理（与 defect 技能对接） |

---

## 5. 执行状态与操作手册

- ✅ 已完成: 5 个 MP 结构 CIF（溯源记录）→ 5 套 VASP 输入（20 个作业, relax/static/band/dos, SOC 正确配置）→ 判据协议（本文档）。
- ⏳ 待办: SCNet 连接恢复后按 `vasp.run_workflow` 或 vasp 命名空间逐作业提交；`PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1` 时可用 bounded 便捷接口；产物用 `electronic.band_summary / dos_summary / effective_mass` 解析。
- 目录: `drylab_vasp/structures/*.cif`；`drylab_vasp/prep/{体系}_{profile}/{01..04}_stage/`。

**一句话**: 用 5 体系 20 作业的 PBE+SOC/standard VASP 矩阵，验证"Sn 合金化把 PbSe 带隙压入 0.25–0.31 eV 设计窗 + Ge-As-Se 基体透明 + L 点直接带隙与轻有效质量"，通过相对位移+实验锚定法规避 PBE 低估，为量子点玻璃 MWIR 光电导探测器提供第一性原理依据。
