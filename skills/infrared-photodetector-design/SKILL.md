---
name: infrared-photodetector-design
description: >-
  Interpret infrared photodetector targets, material and device evidence, and
  select relevant specialist analyses. Use for infrared questions, not general orchestration.
category: ir
tags: [infrared, photodetector, lwir, mwir, evidence-gap]
license: MIT
---

# Infrared Photodetector Design

## Purpose

Supply infrared-specific physics and evidence interpretation within the general
`scientific-research-workflow`. Apply these constraints to infrared questions;
do not impose wavelength targets or material-generation workflows on other fields.
A conceptual explanation can finish without compiling numeric constraints.

When the user asks for mechanism-based candidate design, route to
`mechanism-guided-material-design` and register hypotheses before validation.
When the user explicitly asks for VAE sampling, route to the VAE composition
skill. When the user asks for known materials, use retrieval. Keep these three
sources distinct in the report.

## Choose the relevant specialist

- Numeric spectral or device bounds: `ir-constraint-analysis`; reuse compatible
  results and preserve supplied temperatures and other conditions.
- Shortlisting materials: `infrared-material-screening`; finish at a supported
  shortlist unless further computation is needed for the requested result.
- Existing electronic data: `electronic-structure-analysis` or
  `narrow-gap-electronic-analysis`; need the actual calculation outputs.
- Transport, defect, optical or device gaps: load the corresponding analysis
  skill. Check its current implementation and input requirements; registered
  transport/defect analysis wrappers do not yet run complete solvers.
- New supported DFT computation: `vasp-hpc-operator` (or
  `molecular-vasp-study` for molecular study planning), within the normal gates.

Select only what can change the current decision. Existing experimental evidence
may suffice without generation, ML screening, DFT or device simulation.

## Check applicability before transferring a method

Use `method_search` if the current detector lies outside a specialist's
material, device, temperature or spectral range. For film processing, ligand
exchange and interface engineering, trace method settings to specific sample
conditions and record transfer assumptions with the proposed process. Load
`materials-process-design` for the process contract.

## Core rules

- A database match is NOT a validated detector. Band-gap matching alone never
  justifies "promising detector" conclusions.
- Apply band-gap cutoff criteria to the stated absorption mechanism. A
  band-to-band assumption must not silently exclude thermal or other detector
  mechanisms; record the mechanism and the bounds' applicability.
- Use `ir.compile_constraints` when numeric targets need deterministic physics constraints
  (photon energies, cutoff energy, thermal limits, responsivity, BLIP). When
  the goal specifies wavelength, EQE, active dimensions, detectivity, or speed,
  pass those typed targets to the same tool before estimating a device design.
  Treat its ideal responsivity, total-noise allowance, time constant, and
  Johnson R0/R0A values as `NECESSARY_CONSTRAINT`, never measured performance.
- For a stated EQE lower bound, call `photodetector.check_targets` with
  `eqe_relation="minimum"` for each requested pointwise wavelength. A receipt
  for one point cannot certify another. Keep the resulting joint lower bound separate
  from the responsivity at an exact EQE and from any demonstrated device value.
  In tables, use the scoped Checker joint lower bound, including the user's
  original EQE floor. Keep the same wavelength, gain and operating conditions.
  A lifetime or noise threshold inferred from a prior device model stays tied
  to that model. Verify detectivity with condition-matched responsivity and
  noise evidence before claiming a target is met.
- Read `background_reference_conditions` before discussing BLIP: the tool's
  current temperature input also sets its background reference temperature.
  Do not replace a supplied 250 K with the 300 K default. Its hemispherical
  exitance is not a measured detector field of view. The geometric-mean
  reference wavelength is not a spectral average. Noise budgets are conditional
  on the assumed target responsivity; recompute when the working point changes.
- When numeric device claims are supplied, use `claimed_metrics` to compare
  values and exact units with those bounds. `ARITHMETICALLY_COMPATIBLE` is only
  a formula check; source identity and empirical performance remain unknown.
- Check the target against evidence for the same device, material phase,
  operating temperature, wavelength, bias, and physical current path. Missing
  or mismatched evidence leaves empirical performance `UNKNOWN`; do not infer
  a measured bound from a formula or a different device.
- For public primary HTML or abstract evidence, use deferred
  `literature.fetch_source` when available. Record blocked access and abstract
  coverage explicitly; an abstract does not establish full-text details or
  same-wavelength device performance. Use existing local PDF tools for local
  papers, and do not use arbitrary shell or web access to bypass source limits.
- Report missing prerequisites explicitly; never fabricate DFT/experimental
  values.
- Record every observation as evidence with source and method.
- The unified native execution services own supported calculations and HPC
  lifecycle gates. Skills guide when to use them and how to interpret outputs;
  an external skill is usable only when present in the index and compatible
  with the registered tool surface.

Deliver the requested explanation, comparison, shortlist or analysis with evidence
scope and the most decision-relevant remaining gap. Further investigation is a
conditional next step, not a mandatory continuation of every IR request.

For evidence-linked noise acquisition and sample identity, load the detector-noise method card under `materials-process-design`.
