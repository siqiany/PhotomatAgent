# CHGNet Screening and MatterGen Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a usable CHGNet screening/relaxation capability and complete the isolated MatterGen execution/configuration path for infrared-material candidate workflows.

**Architecture:** CHGNet is a new optional deferred capability pack using lazy in-process imports and workspace-contained paths. MatterGen remains an isolated executable, gains a packaged deterministic runner and structured configuration, and feeds generated structures into CHGNet before costly DFT.

**Tech Stack:** Python 3.12, Pydantic, pymatgen, ASE, CHGNet 0.4.x, MatterGen 1.0.3 isolated CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-chgnet-mattergen-design.md`

## Global Constraints

- Preserve `AgentRuntime` and the one authoritative `ToolRegistry` execution path.
- Every new model-visible tool is `DEFERRED`, schema-bounded, workspace-contained, permission-checked by the normal runtime, and returns bounded observations.
- CHGNet evidence uses `source_type=ml_interatomic_potential` and `fidelity=ml_potential`; it never claims DFT or thermodynamic validation.
- Raw CHGNet energies are ranked only within identical reduced-composition groups.
- MatterGen remains outside the main Python environment and all outputs remain `UNVALIDATED_GENERATED_STRUCTURE`.
- MatterGen supports exactly `dft_band_gap` and `chemical_system`; it never claims joint conditioning.
- Missing optional dependencies/configuration fail soft and never block base runtime startup.
- Model-visible paths stay within the workspace via `Workspace.resolve`.
- Do not modify credentials or existing API-key values.
- Verification is focused: no repository-wide suite and no large MatterGen batch.

---

### Task 1: CHGNet screening and relaxation capability

**Files:**
- Create: `src/photomatagent/scientific/capabilities/chgnet/__init__.py`
- Modify: `src/photomatagent/scientific/capabilities/config.py`
- Modify: `src/photomatagent/scientific/capabilities/registry.py`
- Modify: `src/photomatagent/scientific/capabilities/status.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `tests/test_chgnet.py`
- Modify: `tests/test_capability_probe.py`

**Interfaces:**
- Produces: `chgnet_pack(config: ScientificConfig, workspace: Workspace) -> CapabilityPack`.
- Produces: deferred tools named `chgnet.screen` and `chgnet.relax`.
- Produces: `ScientificConfig` fields `chgnet_model_name`, `chgnet_device`, `chgnet_max_structures`, `chgnet_relax_fmax`, and `chgnet_relax_steps`.
- Consumes: `Workspace.resolve`, `ScientificToolResult`, `ScientificEvidence`, and existing capability registry/status patterns.

- [ ] **Step 1: Write focused failing tests**

Add tests proving that the pack probes missing dependencies without raising,
both tools are deferred and registered, all input paths are workspace-contained,
screen results use ML-potential evidence, cross-composition ranking is omitted,
same-composition ranking is deterministic, and relax writes below
`user_output/chgnet/`. Inject small fake model/optimizer objects for unit tests.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q tests/test_chgnet.py tests/test_capability_probe.py -k 'chgnet'
```

Expected: collection/import or assertions fail because the capability does not exist.

- [ ] **Step 3: Implement the capability and bounded configuration**

Implement lazy CHGNet loading, normalized scalar/list conversion, maximum-force
calculation, same-composition grouping, bounded summaries, and relaxed CIF output.
Validate all arguments before loading the model. Add the pack to both registry
assembly paths. Add `chgnet>=0.4.1,<0.5` as an optional extra and include it in
`science`; refresh the lock without upgrading unrelated dependencies.

- [ ] **Step 4: Run focused GREEN tests**

Run the command from Step 2 and additionally:

```bash
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q tests/test_retrieval.py tests/test_tool_discovery_sprint3.py
```

- [ ] **Step 5: Commit and report**

Commit only Task 1 files with subject `feat: add CHGNet screening capability`.
Write the exact RED/GREEN evidence and changed-file list to the assigned report.

---

### Task 2: MatterGen isolated runner, configuration, and skills

**Files:**
- Create: `src/photomatagent/scientific/capabilities/generation/mattergen_runner.py`
- Modify: `src/photomatagent/scientific/capabilities/generation/mattergen.py`
- Modify: `src/photomatagent/scientific/capabilities/generation/tools.py`
- Modify: `src/photomatagent/scientific/capabilities/config.py`
- Modify: `skills/material-composition-generation/SKILL.md`
- Modify: `skills/infrared-material-screening/SKILL.md`
- Create: `skills/ml-potential-screening/SKILL.md`
- Create: `skills/ml-potential-screening/agents/openai.yaml`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_skills.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 tools `chgnet.screen` and `chgnet.relax`.
- Produces: immutable `MatterGenRunSpec` fields `output_dir: Path`,
  `pretrained_name: Literal["dft_band_gap", "chemical_system"]`,
  `candidate_count: int`, `target_band_gap_eV: float | None`,
  `chemical_system: str | None`, `guidance_factor: float`, and `seed: int`.
- Produces: `MatterGenRunner.build_command(spec: MatterGenRunSpec) -> list[str]`
  and `MatterGenRunner.run(spec: MatterGenRunSpec) -> Path`, where the latter
  returns a workspace-contained manifest path.
- Produces: `ScientificConfig` fields `mattergen_executable`,
  `mattergen_hf_home`, `mattergen_pretrained_name`,
  `mattergen_candidate_limit`, `mattergen_timeout_seconds`,
  `mattergen_guidance_factor`, and `mattergen_seed`.
- Produces: `generation.mattergen` inputs `pretrained_name` and `seed` while keeping existing lineage fields.

- [ ] **Step 1: Write focused failing tests**

Add tests for command construction without shell interpolation, `dft_band_gap`
and `chemical_system` mode validation, deterministic archive extraction and
manifest reuse, workspace path rejection, configuration/probe reporting, and
skill routing through MatterGen then CHGNet then VASP. Use a tiny ZIP fixture;
do not invoke the real model in unit tests.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q tests/test_generation.py tests/test_skills.py -k 'mattergen or chgnet or ml_potential'
```

Expected: the new runner/config/skill assertions fail before implementation.

- [ ] **Step 3: Implement the runner and workflow guidance**

Build argument arrays directly, call only the configured executable, set
`HF_HOME`/`MPLCONFIGDIR`, bound captured failure text, extract non-empty CIFs
with safe deterministic names, and atomically replace `manifest.json`. Select
the conditioning property strictly from `pretrained_name`; a chemical-system
run must not pretend it was band-gap conditioned. Place outputs below
`user_output/mattergen/` and reuse matching manifests.

Update the two existing skills and add `ml-potential-screening` with the exact
evidence hierarchy and funnel in the spec. Update README capability/configuration
documentation.

- [ ] **Step 4: Run focused GREEN tests and E2E smoke**

Run the command from Step 2, then:

```bash
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q tests/test_generation.py tests/test_chgnet.py tests/test_skills.py tests/test_tool_discovery_sprint3.py
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m photomatagent.cli.app scientific status
```

Run a MatterGen dry-run against the configured executable and a one-structure
real CHGNet screen if the optional dependency installs. Do not run an actual
large MatterGen batch.

- [ ] **Step 5: Commit and report**

Commit only Task 2 files with subject `feat: configure MatterGen candidate funnel`.
Write the exact RED/GREEN and smoke-test evidence to the assigned report.
