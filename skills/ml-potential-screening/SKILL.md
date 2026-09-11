---
name: ml-potential-screening
description: Screen generated or known crystal structures with CHGNet as a bounded ML-potential funnel before gated DFT; preserve evidence fidelity and composition-scoped ranking.
category: materials
tags: [CHGNet, machine learning potential, structure screening, relaxation]
license: MIT
---

# ML Potential Screening

Use this skill when a small set of crystal structures needs inexpensive
single-point screening or pre-relaxation before DFT. CHGNet is an ML
interatomic potential, not a thermodynamic oracle.

## Evidence hierarchy and funnel

1. Compile the target and collect database/literature evidence. If a structure
   is needed, use `generation.mattergen` with exactly one supported checkpoint
   mode (`dft_band_gap` or `chemical_system`). MatterGen outputs remain
   `UNVALIDATED_GENERATED_STRUCTURE`.
2. Call `chgnet.screen` on at most a small candidate batch. Preserve each
   structure path, model name, units, provenance, and limitations. CHGNet
   evidence must use `source_type=ml_interatomic_potential` and
   `fidelity=ml_potential`.
3. Compare raw CHGNet energies only among structures with the same reduced
   composition. Do not rank mixed-composition structures by raw energy, and do
   not call it DFT formation energy, energy above hull, thermodynamic stability,
   synthesizability, or detector performance.
4. Call `chgnet.relax` only for finalists. Record convergence and before/after
   summaries; a relaxed ML structure is still not DFT-validated.
5. Forward a much smaller, user-approved set to gated VASP validation. Verify
   DFT artifacts and convergence before making stability or electronic claims.

## Failure and scope rules

- Report missing CHGNet, malformed or unsupported structures, out-of-domain
  behavior, and incomplete evidence explicitly. Never silently promote an
  unavailable or failed result.
- Keep generated-structure lineage separate from CHGNet observations and later
  VASP evidence. Preserve `formula_preserved` and `composition_distance` when
  MatterGen follows a VAE proposal.
- Do not submit HPC work from this skill without the normal VASP approval and
  resource gates.
