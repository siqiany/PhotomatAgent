"""Tests for the vasp-hpc-operator skill (routing + behavior constraints).

The skill must route concrete VASP job operations ("继续 VASP 作业",
"Slurm 完成但未收敛") and must NOT attract generic materials or
literature-only requests. Tests assert behavior constraints (which
failures are restartable, which artifacts are small/large, the
COMPLETED != scientific rule), not fixed wording.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.skills.loader import SkillLoader


SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills" / "vasp-hpc-operator"


def _skill_entry():
    loader = SkillLoader()
    entries = {entry.name: entry for entry in loader.load_index()}
    assert "vasp-hpc-operator" in entries
    return entries["vasp-hpc-operator"]


def _skill_body() -> str:
    loader = SkillLoader()
    body, resolved = loader.view("vasp-hpc-operator")
    assert resolved == "SKILL.md"
    return body


def test_skill_discoverable_and_frontmatter_valid():
    entry = _skill_entry()
    # Frontmatter legality: name matches directory, description present.
    assert entry.name == "vasp-hpc-operator"
    assert entry.description.strip()


def test_routing_attracts_concrete_job_operations():
    description = _skill_entry().description
    assert "继续/重试/查看作业" in description
    assert "Slurm status checks" in description
    assert "CONTCAR restarts" in description
    assert "convergence diagnosis" in description
    assert "completed jobs that still need scientific validation" in description


def test_routing_does_not_attract_literature_or_generic_qa():
    description = _skill_entry().description
    # Negative clause is part of the routing contract.
    assert "Do not use for generic materials questions" in description
    assert "literature-only requests" in description
    # The description never claims generic chemistry/material knowledge.
    assert "材料问答" not in description
    assert "文献综述" not in description


def test_reference_files_exist_and_behavior_constraints_present():
    convergence = SKILL_ROOT / "references" / "convergence-and-recovery.md"
    artifacts = SKILL_ROOT / "references" / "artifacts-and-postprocessing.md"
    assert convergence.is_file()
    assert artifacts.is_file()
    convergence_text = convergence.read_text(encoding="utf-8")
    artifacts_text = artifacts.read_text(encoding="utf-8")

    # Every failure class of the deterministic decision table is covered.
    for failure in (
        "NSW_EXHAUSTED",
        "FORCE_PLATEAU",
        "OOM",
        "WALLTIME",
        "SCF_NOT_CONVERGED",
        "LINE_SEARCH_EXCURSION",
        "AMBIGUOUS_SUBMISSION",
        "STATUS_QUERY_FAILED",
        "STATUS_UNKNOWN",
    ):
        assert failure in convergence_text, failure
    # Every class states whether it may auto-retry.
    assert "Can auto-retry?" in convergence_text
    # Practical convergence must always be provenance-marked, never masked
    # as the original threshold.
    assert "practical_convergence" in convergence_text
    assert "original threshold" in convergence_text
    # NSW restarts come from CONTCAR, never from the initial POSCAR.
    assert "never initial POSCAR" in convergence_text
    # Status failures never resubmit.
    assert "never resubmit" in convergence_text.lower()

    # Artifact policy: small/large split and the proxy-surface rule.
    assert "SMALL" in artifacts_text and "LARGE" in artifacts_text
    assert "WAVECAR" in artifacts_text
    assert "CHGCAR" in artifacts_text
    assert "PARCHG" in artifacts_text
    assert "proxy" in artifacts_text
    assert "never" in artifacts_text.lower()
    assert "x-fastest" in artifacts_text


def test_skill_body_contains_non_negotiable_behavior_rules():
    body = _skill_body()
    behaviors = [
        "capabilities/doctor",      # rule 1
        "PHOTOMATAGENT_ALLOW_HPC_SUBMIT",  # rule 2: gate explicit
        "vasp.*",                 # rule 3: unified public entry points
        "total_charge",             # rule 4: explicit charge
        "preflight",                # rule 5: chain order
        "status query",             # rule 6: no resubmit on query failure
        "reconciled",               # rule 7: ambiguity handling
        "NOT scientific completion",  # rule 8: COMPLETED != done
        "maximum atomic force",     # rule 9: force criterion
        "EDIFFG",                   # rule 9
        "initial POSCAR",           # rule 10: never restart from POSCAR
        "calibration",              # rule 11: production calibration
        "vacuum",                   # rule 12/13: alignment + ESP
        "LVHAR",                    # rule 13
        "model context",            # rule 14: large-file policy
        "warning",                  # rule 15: VM/TVM applicability warning
        "Static DFT",               # rule 16: no transport/SEI claims
        "provenance",               # rule 17
        "job_id",                   # rule 18: return identifiers
        "remote directory",
    ]
    for behavior in behaviors:
        assert behavior in body, behavior
    # The skill references its own decision table.
    assert "convergence-and-recovery.md" in body
    assert "artifacts-and-postprocessing.md" in body
