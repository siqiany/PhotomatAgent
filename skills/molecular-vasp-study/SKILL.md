---
name: molecular-vasp-study
description: Convert natural-language VASP computation requests into typed VASP study plans, resolve/generate structures with recorded provenance, execute the deduplicated calculation matrix through the unified vasp.* service, and produce figures + final reports. Use when the user asks to compute HOMO/LUMO, binding energies, ESP or full VASP studies for molecules/complexes/polymer proxies (Chinese requests like 计算 HOMO/LUMO、结合能、ESP、VASP 计算计划 also trigger this skill).
---

# Molecular VASP study skill

This skill turns a natural-language VASP request into a structured study.
It does **not** re-implement VASP submission, POTCAR, Slurm, monitoring or
collection: everything runs through the unified `vasp.*` service.
It does **not** read PPT/PPTX (a generic document capability).

## Orchestration order

For a new study use the planning steps below. For an existing workflow, continue
from its recorded state using `vasp.status`, `vasp.wait`, `vasp.resume` or
`vasp.collect` as appropriate; do not recreate the plan or resubmit completed jobs.
A request for a plan or an interpretation does not itself authorize execution.

1. Parse the user request into typed parameters (never call an LLM inside a
   tool):
   - systems (system_id / display_name / smiles / structure_path /
     total_charge / spin_multiplicity / role / properties);
   - property_requests (homo_lumo / binding_energy / esp);
   - method preferences (functional, encut_ev, box_ang);
   - structure policy (allow_assumed_structures, max_candidates_per_system);
   - resource budget (max_core_hours);
   - report language.
2. Call `vasp.plan` with those parameters plus `original_request`.
3. If the plan reports missing/assumed structures, note the reliability
   grades (C/D) and the mandated warning for the final report.
4. Validate the calculation matrix and budget; check charges:
   - obtain every explicit charge and spin from the supplied identity record;
     ambiguous names or formula labels are not charge assignments;
   - complex charge must equal the fragment charge sum.
5. Execute only when the user asked for real computation AND
   `PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1`: submit the planned study through `vasp.submit`/`vasp.resume` with
   `user_requested_computation=True`. Study-level authorization replaces
   per-stage prompting.
6. After submission prefer `vasp.wait` for bounded waiting. Use `vasp.status`
   for a status request and `vasp.resume` for a supported transition or recovery;
   never re-sbatch on a status query failure or resubmit a VALIDATED task.
7. Collect and validate with `vasp.collect`.
8. Generate the report with `vasp.report`.
9. Return the report path, a result summary and every assumed-structure
   warning to the user. Do not ask the user to call low-level molecule tools manually.

## Rules

- Charges are always explicit; names are never decoded (TFSI is not assumed
  to be -1 without the identity record).
- Hidden assumptions are forbidden: everything goes into provenance and the
  report (reliability grades A/B/C/D).
- C/D results must carry: "以下数值适用于本研究构造的假设模型，不应直接
  解释为真实聚合物网络的唯一数值。"
- HOMO/LUMO cross-molecule comparisons use vacuum-aligned values only;
  raw eigenvalues are never compared.
- Binding energies are electronic only (no vibrational/thermal/solvation);
  bare-ion references (Li+) use the declared zero-electron reference model
  (E = 0 convention, ΔΔE recommended).
- Missing VM/TVM connectivity produces ASSUMED_REPRESENTATIVE oligomers
  or an explicit ASSUMED_PROXY only when the study's structure policy permits
  assumed structures. Otherwise identify the missing input and progress only
  independent, fully specified systems. A proxy is never called the real polymer.
- Over-budget studies stop starting new jobs and still produce a partial
  report.
