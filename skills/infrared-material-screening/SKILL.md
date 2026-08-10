---
name: infrared-material-screening
description: Screen candidate materials for an IR band using database matches, band-gap constraints, and explicit evidence gaps; never equate database matches with validated detectors.
category: ir
tags: [infrared, screening, materials project, band gap]
license: MIT
---

# Infrared Material Screening

## When to use

A spectral band and a need to shortlist candidate materials.

## Procedure

1. Compile constraints with `ir.compile_constraints` (cutoff gap, thermal
   limits).
2. Search `materials` (Materials Project) and `literature` for candidates in
   the band.
3. Filter: gap near/below cutoff, stability (energy_above_hull), known IR
   literature.
4. For each candidate, list what is known vs missing (transport, defects,
   optical absorption, device results).
5. Rank by evidence completeness, not by gap match alone.

## Rules

- A database match is not a validated detector.
- Cheap evidence first: database -> structure -> electronic analysis.
- Explicitly mark candidates whose gap only partially covers the band.
- Report which missing evidence is most decision-relevant.

