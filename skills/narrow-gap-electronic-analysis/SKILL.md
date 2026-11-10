---
name: narrow-gap-electronic-analysis
description: >-
  Analyze narrow-gap semiconductors (HgTe, HgCdTe-like) for IR photodetection:
  band structure, effective mass, DOS near Fermi level; band-gap match alone
  is never sufficient.
category: ir
tags: [narrow-gap, hgte, band structure, effective mass]
license: MIT
---

# Narrow-Gap Electronic Analysis

For narrow-gap candidates (e.g. HgTe, HgCdTe, PbSnTe, InAsSb):

Select the operations needed for the current question and existing inputs;
an explanation or comparison of reported gaps need not run all four analyses.

1. Confirm the band gap and its character with `electronic.band_summary`
   (direct/indirect, CBM/VBM) when a band structure exists.
2. Check DOS near the band edges with `electronic.dos_summary` (carrier
   density implications).
3. Estimate band-edge effective masses with `electronic.effective_mass`
   (mobility implications for transport).
4. Compare against a compatible `ir.compile_constraints` cutoff when the task
   concerns an infrared spectral target.

## Interpretation traps

- Gap matches cutoff -> necessary, not sufficient.
- When thermal generation matters, compare the gap and kT at the supplied
  operating temperature; do not invent a temperature for a numeric conclusion.
- Semimetallic or gapless behavior (e.g. HgTe) needs explicit treatment of
  the zero-gap/negative-gap regime before calling it a photodetector.
- DFT gaps of narrow-gap systems are unreliable: require experimental or
  corrected values, or report the uncertainty.

## Execution delegation

This skill consumes existing band-structure outputs. For new supported VASP
calculations discover `vasp.*` and load the indexed `vasp-hpc-operator` skill;
retain the normal submission and validation gates. External SOPs apply only
when indexed and compatible with the configured execution backend.

