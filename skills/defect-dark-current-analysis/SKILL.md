---
name: defect-dark-current-analysis
description: Link defect physics to dark current for IR detectors: formation energies, charge states, trap levels; requires doped and DFT defect data, never fabricated.
category: ir
tags: [defects, dark current, traps, formation energy]
license: MIT
---

# Defect / Dark Current Analysis

## Role of this skill

Decide when defect evidence is needed (typically when dark current or
recombination lifetime limits detector performance) and what counts as
trustworthy defect evidence.

## Evidence requirements

- Formation energies and thermodynamic transition levels for relevant charge
  states (`defects.analyze` via doped).
- Chemical potential range consistent with growth conditions.
- Trap depth relative to the band edges (deep vs shallow), because deep traps
  dominate generation-recombination dark current in narrow-gap detectors.

## Rules

- Never report formation energies without DFT inputs; use
  `defects.capabilities` and `defects.analyze` prerequisite messages.
- Relate defect evidence to dark current: SRH generation rate scales with
  trap density and inverse lifetime; state the mechanism explicitly.
- If doped or DFT data are missing, report the missing prerequisite as the
  next step.

