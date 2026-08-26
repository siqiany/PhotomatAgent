"""Final study report (report.md) with the mandated 14 sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from photomatagent.scientific.applications.vasp.study.models import (
    StudyTaskState,
    VaspStudySpec,
)


ASSUMED_STRUCTURES_WARNING = (
    "以下数值适用于本研究构造的假设模型，不应直接解释为真实聚合物"
    "网络的唯一数值。"
)


def render_report(
    spec: VaspStudySpec,
    results: dict[str, Any],
) -> str:
    """Render report.md for one completed/partial study."""
    lines: list[str] = []
    add = lines.append
    request = spec.request
    matrix = spec.calculation_matrix
    systems = results.get("systems", [])
    binding_rows = results.get("binding_energies", [])
    assumptions = results.get("structure_assumptions", [])

    add("# VASP 研究报告")
    add("")
    add(f"- study_id: `{spec.study_id}`")
    add(f"- 报告语言: {request.report_options.report_language}")
    add(f"- 日期: {spec.study_dir.stat().st_mtime:.0f} (study_dir mtime)")

    # 1. 用户原始需求
    add("")
    add("## 1. 用户原始需求")
    add("")
    add(f"> {request.original_request or '(空)'}")

    # 2. 计算任务矩阵
    add("")
    add("## 2. 计算任务矩阵")
    add("")
    add(
        "| task | 体系 | 电荷 | 自旋 | 可靠性 | 状态 | 结构来源 |"
    )
    add("|---|---|---|---|---|---|---|")
    for row in systems:
        add(
            f"| {row['task_id']} | {row['system']} | {row['charge']:+d} "
            f"| {row['spin_multiplicity']} | {row['reliability']} "
            f"| {row['state']} | {row['structure_status']} |"
        )
    add("")
    add(f"- 去重后的唯一计算体系数: {results['summary']['unique_calculations']}")
    add(f"- 已通过验证: {results['summary']['validated']}")
    add(f"- 总核时估算: {matrix.total_core_hours:g} core-h; "
        f"预算: {request.resource_budget.max_core_hours:g} core-h")

    # 3. 计算方法
    add("")
    add("## 3. 计算方法")
    add("")
    method = results.get("method", {})
    add(f"- 泛函: {method.get('functional')}")
    add(f"- ENCUT: {method.get('encut_ev'):g} eV")
    add(f"- 固定真空盒: {method.get('box_ang'):g} Å (Γ-only 1×1×1)")
    add("- 每个唯一体系复用一条 typed 分子工作流 "
        "(relax → static_preconverge → corrected_static → 需求阶段)")
    add("- POTCAR: PAW-PBE，元素顺序与 POSCAR 一致；内容不落日志")
    add(f"- 修正策略: {method.get('corrections')}")

    # 4. 每个结构的来源
    add("")
    add("## 4. 结构来源")
    add("")
    for row in systems:
        add(
            f"- {row['system']}: {row['structure_status']} "
            f"(可靠性 {row['reliability']})"
        )

    # 5. 所有结构假设
    add("")
    add("## 5. 结构假设")
    add("")
    if assumptions:
        for item in assumptions:
            add(f"- **{item['system']}** (可靠性 {item['reliability']}, "
                f"来源: {item['source']}, 置信度 {item['confidence']}):")
            for assumption in item["assumptions"]:
                add(f"  - {assumption}")
        add("")
        add(f"> {ASSUMED_STRUCTURES_WARNING}")
    else:
        add("- 无 (所有结构为用户提供或明确 SMILES 生成)")

    # 6. 构象筛选方法
    add("")
    add("## 6. 构象筛选方法")
    add("")
    add(
        "- RDKit ETKDG，固定随机种子并记录；每个体系生成多个构象；"
        "MMFF94 优先、UFF 兜底；严重碰撞剔除；按力场能量排序；"
        "复合物按配位位点(O/N/F/S)与多方向采样。"
    )
    add(
        f"- 每体系候选数: {request.structure_policy.max_candidates_per_system}; "
        f"失败时按预生成构象重试。"
    )

    # 7. SCNet/VASP 作业信息
    add("")
    add("## 7. SCNet/VASP 作业信息")
    add("")
    add("- 提交策略: submit-once + 唯一远程目录 + 注册表状态机")
    add("- 每个体系的工作流目录:")
    for row in systems:
        if row.get("workflow_dir"):
            add(f"  - `{row['system']}`: `{row['workflow_dir']}`")
    add("- 作业状态记录在 SQLite 注册表与各工作流 task_state.json")

    # 8. 收敛与验证状态
    add("")
    add("## 8. 收敛与验证状态")
    add("")
    for row in systems:
        e0 = row.get("e0_ev")
        converged = row.get("scf_converged")
        add(
            f"- {row['system']}: state={row['state']}, "
            f"E0={f'{e0:.8f} eV' if e0 is not None else 'n/a'}, "
            f"SCF={converged if converged is not None else 'n/a'}"
            + (f", error: {row['error']}" if row.get("error") else "")
        )

    # 9. HOMO/LUMO 表格和图
    add("")
    add("## 9. HOMO/LUMO")
    add("")
    add(
        "| 体系 | HOMO raw (eV) | LUMO raw (eV) | HOMO 真空对齐 (eV) "
        "| LUMO 真空对齐 (eV) | KS gap (eV) |"
    )
    add("|---|---|---|---|---|---|")
    orbital_systems = 0
    for row in systems:
        if row.get("homo_raw_ev") is None and row.get("homo_aligned_ev") is None:
            continue
        orbital_systems += 1
        fmt = lambda value: f"{value:.4f}" if value is not None else "—"
        add(
            f"| {row['system']} | {fmt(row.get('homo_raw_ev'))} "
            f"| {fmt(row.get('lumo_raw_ev'))} "
            f"| {fmt(row.get('homo_aligned_ev'))} "
            f"| {fmt(row.get('lumo_aligned_ev'))} "
            f"| {fmt(row.get('ks_gap_ev'))} |"
        )
    add("")
    add(
        "- 原始本征值携带晶胞势参考，**禁止跨分子比较**；跨分子比较只使用 "
        "LOCPOT 真空平台对齐后的数值。"
    )
    add("- 图: `figures/orbital_levels.png`, `figures/homo_isosurface_*.png`, "
        "`figures/lumo_isosurface_*.png` (PARCHG 等值面 + 分子骨架)")
    if not orbital_systems:
        add("- (本研究中没有已验证的轨道结果)")

    # 10. 结合能表格
    add("")
    add("## 10. 结合能")
    add("")
    if binding_rows:
        add(
            "| 复合物 | 片段 (formula, charge, 可靠性) | ΔE (eV) | ΔΔE (eV) "
            "| 状态 |"
        )
        add("|---|---|---|---|---|")
        for row in binding_rows:
            fragments = "; ".join(
                f"{item['name']}({item['formula']}, {item['charge']:+d}, "
                f"{item['reliability']})"
                for item in row["fragments"]
            )
            delta = (
                f"{row['delta_e_ev']:.4f}"
                if row.get("delta_e_ev") is not None
                else "—"
            )
            delta_delta = (
                f"{row['delta_delta_e_ev']:.4f}"
                if row.get("delta_delta_e_ev") is not None
                else "—"
            )
            add(
                f"| {row['complex']} | {fragments} | {delta} | {delta_delta} "
                f"| {row['state']} |"
            )
        add("")
        add("- E_binding = E(complex) − ΣE(fragments)；复合物电荷 = 片段电荷之和；"
            "同 functional/ENCUT/盒/修正策略（参数一致性由 binding 检查强制）。")
        add("- **电子结合能**：不含振动、热力学或溶剂自由能。")
        add("- 裸离子参考 (Li+) 存在真空参考误差；ΔΔE / 替代参考可降低该误差。")
        for row in binding_rows:
            if row.get("state") == "VALIDATED" and row.get("complex_reliability") in {
                "C", "D",
            }:
                add(f"> {ASSUMED_STRUCTURES_WARNING} (涉及 {row['complex']})")
    else:
        add("- 本研究未请求结合能。")

    # 11. ESP 图
    add("")
    add("## 11. ESP")
    add("")
    esp_systems = [
        row for row in systems if row.get("esp_has_locpot") is not None
    ]
    if esp_systems:
        for row in esp_systems:
            add(
                f"- {row['system']}: LOCPOT 已收集 "
                f"(LVHAR 离子+Hartree 势); 图: "
                f"`figures/esp_surface_{row['system']}.png`"
            )
        add("- 色标单位: eV；方法: vdW+1.4 Å 分子表面代理采样 "
            "(真实 CHGCAR 等值面待 SCNet 实测验证)。")
    else:
        add("- 本研究未请求 ESP 或无已验证结果。")

    # 12. 失败或未完成任务
    add("")
    add("## 12. 失败或未完成任务")
    add("")
    incomplete = [
        row for row in systems
        if row["state"]
        not in {StudyTaskState.VALIDATED.value, "PLANNED"}
    ]
    if not incomplete:
        add("- 无")
    else:
        for row in incomplete:
            add(
                f"- {row['system']} ({row['state']}): "
                f"{row.get('error') or '无错误信息'}"
            )

    # 13. 方法限制
    add("")
    add("## 13. 方法限制")
    add("")
    for limitation in results.get("limitations", []):
        add(f"- {limitation}")

    # 14. 可复现文件路径
    add("")
    add("## 14. 可复现文件路径")
    add("")
    add(f"- `{spec.study_dir / 'study_request.json'}`")
    add(f"- `{spec.study_dir / 'structure_manifest.json'}`")
    add(f"- `{spec.study_dir / 'calculation_matrix.json'}`")
    add(f"- `{spec.study_dir / 'study_state.json'}`")
    add(f"- `{spec.study_dir / 'results.json'}`")
    add(f"- `{spec.study_dir / 'results.csv'}`")
    add(f"- `{spec.study_dir / 'report.md'}`")
    add(f"- `{spec.study_dir / 'figures' / '**'}`")
    add(f"- 工作流目录: `{spec.study_dir / 'workflows' / '**'}`")
    return "\n".join(lines) + "\n"
