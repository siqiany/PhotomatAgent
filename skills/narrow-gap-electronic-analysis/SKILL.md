---
name: narrow-gap-electronic-analysis
description: Analyze narrow-gap semiconductors (HgTe, HgCdTe-like) for IR photodetection: band structure, effective mass, DOS near Fermi level; band-gap match alone is never sufficient.
category: ir
tags: [narrow-gap, hgte, band structure, effective mass]
license: MIT
---

# Narrow-Gap Electronic Analysis

For narrow-gap candidates (e.g. HgTe, HgCdTe, PbSnTe, InAsSb):

1. Confirm the band gap and its character with `electronic.band_summary`
   (direct/indirect, CBM/VBM) when a band structure exists.
2. Check DOS near the band edges with `electronic.dos_summary` (carrier
   density implications).
3. Estimate band-edge effective masses with `electronic.effective_mass`
   (mobility implications for transport).
4. Compare against the `ir.compile_constraints` cutoff.

## Interpretation traps

- Gap matches cutoff -> necessary, not sufficient.
- Narrow-gap systems have strong thermal generation; always state the kT
  comparison.
- Semimetallic or gapless behavior (e.g. HgTe) needs explicit treatment of
  the zero-gap/negative-gap regime before calling it a photodetector.
- DFT gaps of narrow-gap systems are unreliable: require experimental or
  corrected values, or report the uncertainty.

## Execution delegation

Band-structure DFT execution belongs to external skills (AtomisticSkills VASP
band structure SOP, QE SOP); this skill consumes their outputs.

