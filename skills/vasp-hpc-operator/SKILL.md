---
name: vasp-hpc-operator
description: Operate, monitor, validate, resume, and recover concrete VASP jobs on the configured HPC backend. Use for real or prepared VASP workflows, Slurm status checks, “继续/重试/查看作业”, CONTCAR restarts, convergence diagnosis, HOMO/LUMO vacuum alignment, ESP postprocessing, or completed jobs that still need scientific validation. Do not use for generic materials questions or literature-only requests.
---

# VASP HPC operator skill

This skill operates jobs that are ALREADY concretized: submission,
monitoring, collection, scientific acceptance, restarts and recovery on the
configured SCNet backend. It deliberately does NOT plan full molecular
studies from natural language (that is `molecular-vasp-study`), does NOT
generate novel structures, and never re-implements SSH/Slurm/POTCAR handling
(all of that stays inside `vasp_molecule.*` / `vasp_study.*`).

## Scope of one operator session

1. capabilities/doctor check (read-only);
2. status / resume / collect / validate;
3. convergence diagnosis and typed recovery decisions;
4. restarting a relax from CONTCAR (or a historical best snapshot);
5. HOMO/LUMO vacuum alignment and ESP postprocessing;
6. reporting `job_id`, remote directory, scientific state and next step.

## Non-negotiable rules

1. ALWAYS start with the VASP capabilities/doctor probe
   (`vasp_molecule.capabilities` or `photomatagent scientific
   scnet-doctor`) and confirm configuration, partition, VASP version and
   the pseudopotential strategy before touching any job.
2. Submit a REAL calculation only when the user explicitly asked for real
   computation AND the HPC submit gate is open
   (`PHOTOMATAGENT_ALLOW_HPC_SUBMIT=1` + `ResourcePolicy` caps); never
   bypass the gate.
3. Use `vasp_study.*` or `vasp_molecule.*` tools; never hand-assemble
   lower-level lifecycle calls for the user.
4. `total_charge` must always be explicit; it is never guessed from a
   molecule name, file name or formula.
5. Follow the fixed chain: prepare → preflight → submit → status/resume →
   collect → validate → analyze/report.
6. A failed status query NEVER triggers a resubmission (refresh/reconcile
   only; UNKNOWN stays UNKNOWN until proven otherwise).
7. An ambiguous sbatch result MUST be reconciled (registry + marker +
   squeue/sacct) before any further action.
8. Slurm COMPLETED is NOT scientific completion: results must pass
   validation before any evidence, binding value or orbital claim.
9. For a relax stage, check the maximum atomic force, EDIFFG, NSW and
   VASP's formal "reached required accuracy" marker; adjacent-step dE is
   never a convergence criterion.
10. When a relax is not converged, resume from CONTCAR (or the historical
    lowest-force snapshot); never restart from the initial POSCAR.
11. Production parameters require a matching resource calibration record
    before submission; never run production on guessed resources.
12. HOMO/LUMO values may only be compared across systems after vacuum
    alignment (LVHAR LOCPOT); raw eigenvalues are never compared.
13. ESP uses the LVHAR LOCPOT; large LOCPOT files are processed with the
    streaming/remote postprocessing path, never loaded whole into context.
14. WAVECAR/CHGCAR/LOCPOT follow the large-file artifact policy: they stay
    remote (or referenced by hash/size/remote path) and never enter the
    model context.
15. VM/TVM model structures always carry their applicability warning; they
    describe the constructed model, never the real network.
16. Static DFT never proves transport, SEI or macroscopic electrochemistry;
    such claims are refused without dedicated simulations.
17. Every recovery, threshold change and method change is written into
    provenance (attempt id, old/new values, reason, practical-convergence
    markers).
18. Every response returns job_id, remote directory, current scientific
    state and the next step — never just "已完成".

## Worked flows

### Continue / retry a submitted job ("继续/重试/查看作业")

1. `vasp_study.status` / `vasp_molecule.status` for the recorded request.
2. Never re-sbatch on a status failure; reconcile ambiguous submissions.
3. `vasp_study.collect` (or `vasp_molecule.collect`) to download and
   validate; only VALIDATED results generate evidence.

### Slurm completed but not converged

1. Read the convergence report (max force, EDIFFG, NSW consumed, formal
   marker, detected VASP errors).
2. Apply the deterministic decision table (see
   `references/convergence-and-recovery.md`): CONTCAR restarts for
   NSW_EXHAUSTED/WALLTIME, POTIM/IBRION changes for FORCE_PLATEAU,
   historical snapshot for LINE_SEARCH_EXCURSION; OOM and unknown modes
   STOP without repeating identical resources.
3. Every attempt gets a NEW attempt_id and remote directory; auto-retry
   counts are capped; relaxed EDIFFG is recorded as practical convergence
   with old/new values and reason, never presented as the original
   threshold.

### HOMO/LUMO and ESP

1. Vacuum-align from the LVHAR LOCPOT (streaming planar averages; six-face
   boundary layers); raw eigenvalues stay incomparable.
2. ESP figures use PARCHG densities for orbitals and the LVHAR LOCPOT for
   the electrostatic potential; proxy surfaces are never labeled as real
   CHGCAR isosurfaces (see `references/artifacts-and-postprocessing.md`).

## Boundaries

- Planning new studies, structure generation and full-study reporting →
  `molecular-vasp-study` / `vasp_study.*`.
- Generic materials questions, literature-only requests → NOT this skill.

