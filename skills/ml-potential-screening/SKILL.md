---
name: ml-potential-screening
description: Screen or pre-relax crystal structures with CHGNet when ML-potential estimates answer the task. Keep composition-scoped ranking and evidence fidelity; DFT is optional.
license: MIT
---

# ML Potential Screening

Use this skill when a small set of crystal structures needs ML-potential
single-point estimates or pre-relaxation. CHGNet is an ML
interatomic potential, not a thermodynamic oracle.

## Select the requested analysis

1. Identify the requested comparison and reuse available structures and evidence.
   Retrieve a suitable missing structure if supported; for requested model
   generation use `generation.mattergen` with one supported checkpoint
   mode (`dft_band_gap` or `chemical_system`). MatterGen outputs remain
   `UNVALIDATED_GENERATED_STRUCTURE`.
2. For single-point estimates call `chgnet.screen` on a small candidate batch. Preserve each
   structure path, model name, units, provenance, and limitations. CHGNet
   evidence must use `source_type=ml_interatomic_potential` and
   `fidelity=ml_potential`.
3. Compare raw CHGNet energies only among structures with the same reduced
   composition. Do not rank mixed-composition structures by raw energy, and do
   not call it DFT formation energy, energy above hull, thermodynamic stability,
   synthesizability, or detector performance.
4. For requested pre-relaxation call `chgnet.relax` on the selected structures;
   screening is not a mandatory preceding step. Record convergence and before/after
   summaries; a relaxed ML structure is still not DFT-validated.
5. Stop at the requested ML result when that fulfills the task. If the decision
   requires new DFT evidence, forward an authorized subset to gated VASP validation.
   Verify DFT artifacts and convergence before making stability or electronic claims.

## Failure and scope rules

- Report missing CHGNet, malformed or unsupported structures, out-of-domain
  behavior, and incomplete evidence explicitly. Never silently promote an
  unavailable or failed result.
- Keep generated-structure lineage separate from CHGNet observations and later
  VASP evidence. Preserve `formula_preserved` and `composition_distance` when
  MatterGen follows a VAE proposal.
- Do not submit HPC work from this skill without the normal VASP approval and
  resource gates.
