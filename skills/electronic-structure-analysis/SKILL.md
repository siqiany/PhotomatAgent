---
name: electronic-structure-analysis
description: >-
  SOP for analyzing electronic structure of infrared photodetection materials
  from real DFT outputs (vasprun.xml): band gap, DOS, effective mass; never
  substitutes mock or fabricated numbers for DFT results.
category: ir
tags: [electronic, band gap, dos, effective mass, vasprun]
license: MIT
---

# Electronic Structure Analysis

## Scope

Infrared photodetection materials (e.g. III-V semiconductors, narrow-gap
compounds).

## Prerequisite

A completed VASP calculation is required first: a `vasprun.xml` file inside
the workspace. This skill analyzes existing DFT output; it never runs DFT
itself. If no `vasprun.xml` exists, report the missing prerequisite and point
to an external execution SOP (e.g. AtomisticSkills VASP/QE) instead of
fabricating band structure data.

## Procedure

1. Define the material and the target property (band gap, DOS near Fermi
   level, effective mass).
2. `electronic.band_summary` on the vasprun.xml for gap, CBM/VBM, and
   direct/indirect character.
3. `electronic.dos_summary` for the Fermi level and edge DOS.
4. `electronic.effective_mass` (effmass) when band-edge segments are needed;
   note the segment type and direction in the evidence.
5. `electronic.plot_band` / `electronic.plot_dos` (Sumo) only when a figure
   artifact is actually needed.
6. Record every result as Evidence with source (`vasprun.xml` path, method,
   code version) and provenance; promote to a Claim only when supported.
7. Mark open questions explicitly (e.g. SOC, temperature, alloy disorder).

## Rules

- Never report band structure numbers without a real vasprun.xml; the
  `mock.run_calculation` tool is test-only and excluded from tool_search.
- Prefer `electronic.*` native tools for stable structured calls; delegate
  DFT execution to external skills.
