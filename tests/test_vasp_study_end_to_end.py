"""Full natural-language end-to-end test of the study layer (offline).

The request equals the acceptance natural language; every unique calculation
is executed through the vasp_molecule.* machinery on FakeSCNetBackend. No
SSH, no sbatch, no VASP.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", message=".*explicit Hs.*")

from test_vasp_study import (  # noqa: E402  (helpers live in the sibling file)
    NATURAL_LANGUAGE_REQUEST,
    make_psp,
    make_runtime,
    study_request,
)

from photomatagent.scientific.applications.vasp.study.executor import (
    StudyExecutor,
)
from photomatagent.scientific.applications.vasp.study.planner import (
    load_planned_study,
    plan_study,
)


async def test_nl_end_to_end_full_study(tmp_path):
    psp = make_psp(tmp_path)
    runtime = make_runtime(tmp_path, psp=psp)
    backend = runtime.backend

    # plan: the full natural-language request
    request = study_request()
    spec = plan_study(request, tmp_path)
    assert spec.calculation_matrix.total_jobs > 0
    assert len(spec.calculation_matrix.tasks) == 13

    # execute
    executor = StudyExecutor(spec, runtime)
    report = await executor.execute()
    assert report["authorized"] is True
    assert report["failed"] == [], report["failed"]
    task_map = spec.calculation_matrix.task_map()
    for task in spec.calculation_matrix.tasks:
        assert task.state == "VALIDATED", (task.task_id, task.error[:120])
    jobs = len(backend.submitted_scripts)
    assert jobs > 0
    assert len(runtime.session.registry.list()) == jobs
    assert all(
        group.state == "VALIDATED"
        for group in spec.calculation_matrix.binding_groups
    )
    # Bindings are electronic and consistent with the seeded E0s: a complex
    # with two real fragments (TVM-TFSI-) is not 0 by construction.
    assert all(
        group.delta_e_ev is not None
        for group in spec.calculation_matrix.binding_groups
    )
    assert task_map["li|q+1|s1"].error.startswith("zero-electron")

    # resume after "process exit": nothing is resubmitted.
    spec2 = load_planned_study(spec.study_dir)
    executor2 = StudyExecutor(spec2, runtime)
    report2 = await executor2.execute()
    assert report2["failed"] == []
    assert len(runtime.session.registry.list()) == jobs

    # report artifacts
    from photomatagent.scientific.applications.vasp.study.tools import (
        VaspStudyReportTool,
    )

    report_result = await VaspStudyReportTool(runtime).execute(
        {"study_id": spec.study_id, "study_dir": str(spec.study_dir)}
    )
    assert report_result.data["ok"] is True
    study_dir = spec.study_dir
    results = json.loads((study_dir / "results.json").read_text(encoding="utf-8"))
    assert results["summary"]["validated"] == 13
    assert results["summary"]["binding_groups_computed"] == 3
    assert (study_dir / "results.csv").is_file()
    report_text = (study_dir / "report.md").read_text(encoding="utf-8")
    for section in (
        "## 1. 用户原始需求",
        "## 2. 计算任务矩阵",
        "## 3. 计算方法",
        "## 4. 结构来源",
        "## 5. 结构假设",
        "## 6. 构象筛选方法",
        "## 7. SCNet/VASP 作业信息",
        "## 8. 收敛与验证状态",
        "## 9. HOMO/LUMO",
        "## 10. 结合能",
        "## 11. ESP",
        "## 12. 失败或未完成任务",
        "## 13. 方法限制",
        "## 14. 可复现文件路径",
    ):
        assert section in report_text
    assert "假设模型" in report_text  # mandated C/D warning
    assert "电子结合能" in report_text
    assert "零" in results["limitations"][-1] or "bare-ion" in " ".join(
        results["limitations"]
    )
    assert NATURAL_LANGUAGE_REQUEST.split("；")[0] in report_text
    figures = {path.name for path in (study_dir / "figures").glob("*.png")}
    assert "orbital_levels.png" in figures
    assert "binding_energies.png" in figures
    assert any(name.startswith("homo_isosurface_") for name in figures)
    assert any(name.startswith("esp_surface_") for name in figures)
