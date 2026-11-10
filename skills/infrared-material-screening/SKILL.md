---
name: infrared-material-screening
description: Shortlist infrared materials using explicit constraints and evidence. Choose retrieval, hypotheses or requested generation; ML screening and DFT are conditional follow-ups.
category: ir
tags: [infrared, screening, materials project, band gap]
license: MIT
---

# Infrared Material Screening

## When to use

A spectral band and a need to shortlist candidate materials.

## Select a candidate source

- Explicit mechanism request: load `mechanism-guided-material-design`, register
  hypotheses, and keep them unvalidated until evidence exists.
- Explicit VAE request: load `material-composition-generation` and call the VAE
  route; do not represent mechanism reasoning as model sampling.
- Known-material request: search materials/literature and label results as
  known priors, never as new candidates or validation.

## Select evidence needed for this shortlist

Use the cheapest evidence that can answer the current question, and preserve
the source and fidelity of every result:

1. Compile or reuse `ir.compile_constraints` when numeric targets need cutoff
   or thermal bounds. Use relevant supplied data, `materials` and `literature`
   for known candidates; do not repeat searches already supported by current
   evidence. A database match is a prior, not validation.
2. For requested inverse composition proposals use `generation.vae_formula`;
   mechanism-based proposals use hypothesis registration. Neither route is
   required when the task asks for known materials or an existing-data shortlist.
3. Generate structures only when the requested analysis needs unavailable
   structures; prefer a suitable supplied or retrieved structure when it exists.
   For requested model generation, route to `generation.mattergen` with
   `pretrained_name=dft_band_gap` for a gap/cutoff target or
   `pretrained_name=chemical_system` for an explicit element system. The two
   checkpoints are not joint-condition models. All outputs are
   `UNVALIDATED_GENERATED_STRUCTURE`.
4. When a structure comparison needs ML-potential estimates, use `chgnet.screen`
   on small batches and rank energies only within the same
   reduced composition. The tool recomputes the input file SHA-256 and
   canonical structure hash. Label this evidence
   `source_type=ml_interatomic_potential`, `fidelity=ml_potential`; it is not
   DFT formation energy, energy above hull, stability, or detector evidence.
5. Use `chgnet.relax` only if finalist geometry refinement serves the current
   question. A changed geometry receives a new
   structure candidate ID and must be rechecked by scope before downstream
   evidence is attached. If DFT is needed and computation is authorized, send
   a much smaller set through
   the existing VASP prepare/validate/`SubmitOnceSession` gates; scheduler
   completion or command success alone is not scientific success.
6. For each candidate, list known versus missing transport, defects, optical
   absorption, thermal behavior, and device results. Rank by evidence
   completeness and decision relevance, not gap match alone.

## Rules

- A database match is not a validated detector.
- Finish when the requested shortlist is supported. A literature comparison,
  formula proposal or existing-structure comparison need not traverse all the
  steps above. Cheap sufficient evidence takes priority over added computation.
- Explicitly mark candidates whose gap only partially covers the band.
- Report which missing evidence is most decision-relevant.
- Report unsupported or out-of-domain structures; never silently promote a
  generated or ML-potential result to a validated material or detector claim.
- Structure evidence must carry both `candidate_id` and `structure_hash` from
  the bytes actually read. Reject model-supplied identity claims that disagree
  with those hashes. File renames preserve identity, while different ordered
  structures of one composition remain separately checkable. Energy/force
  evidence with a valid chemical-formula subject is projected into that
  structure candidate. Changed CHGNet/VASP results retain the attested input
  candidate as parent; `cif_hash` is the trusted file-byte SHA-256, not the
  canonical geometry hash.
- A fixture, fake CHGNet result, x=0.25 ratio check, or local VASP simulation
  validates software boundaries only. It does not predict performance or
  establish DFT, experimental, stability, or synthesizability conclusions.
