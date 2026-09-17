---
name: infrared-material-screening
description: Use when screening infrared candidates and choosing among known-material retrieval, mechanism hypotheses, or explicit model generation before a bounded MatterGen → CHGNet → VASP evidence funnel.
category: ir
tags: [infrared, screening, materials project, band gap]
license: MIT
---

# Infrared Material Screening

## When to use

A spectral band and a need to shortlist candidate materials.

## Route before the funnel

- Explicit mechanism request: load `mechanism-guided-material-design`, register
  hypotheses, and keep them unvalidated until evidence exists.
- Explicit VAE request: load `material-composition-generation` and call the VAE
  route; do not represent mechanism reasoning as model sampling.
- Known-material request: search materials/literature and label results as
  known priors, never as new candidates or validation.

## Evidence hierarchy and funnel

Use the cheapest evidence that can answer the current question, and preserve
the source and fidelity of every result:

1. Compile constraints with `ir.compile_constraints` (cutoff gap, thermal
   limits), then search `materials` (Materials Project) and `literature` for
   known candidates. A database match is a prior, not validation.
2. Retrieve or generate candidate formulas. Use `generation.vae_formula` for
   inverse composition proposals; do not use retrieval as a substitute for
   generation.
3. Generate structures only when needed. Route to `generation.mattergen` with
   `pretrained_name=dft_band_gap` for a gap/cutoff target or
   `pretrained_name=chemical_system` for an explicit element system. The two
   checkpoints are not joint-condition models. All outputs are
   `UNVALIDATED_GENERATED_STRUCTURE`.
4. Run `chgnet.screen` on small batches and rank energies only within the same
   reduced composition. Label this evidence
   `source_type=ml_interatomic_potential`, `fidelity=ml_potential`; it is not
   DFT formation energy, energy above hull, stability, or detector evidence.
5. Run `chgnet.relax` only on finalists. Send a much smaller, explicitly
   approved set through gated VASP validation; scheduler completion alone is
   not scientific success.
6. For each candidate, list known versus missing transport, defects, optical
   absorption, thermal behavior, and device results. Rank by evidence
   completeness and decision relevance, not gap match alone.

## Rules

- A database match is not a validated detector.
- Cheap evidence first: database/literature -> formula -> MatterGen structure
  (when required) -> CHGNet screen -> CHGNet finalist relaxation -> gated VASP.
- Explicitly mark candidates whose gap only partially covers the band.
- Report which missing evidence is most decision-relevant.
- Report unsupported or out-of-domain structures; never silently promote a
  generated or ML-potential result to a validated material or detector claim.
