---
name: defect-dark-current-analysis
description: >-
  Interpret defect energies, charge states and traps for infrared dark current.
  The current thermodynamics wrapper checks prerequisites only; distinguish this from computed evidence.
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

- Actual formation energies and thermodynamic transition levels for relevant
  charge states from a complete calculation or matching primary-source evidence.
- Chemical potential range consistent with growth conditions.
- Trap depth relative to the band edges (deep vs shallow), because deep traps
  dominate generation-recombination dark current in narrow-gap detectors.

## Rules

- Do not compute formation energies without the required DFT inputs; use
  attested computed results or identify values explicitly as literature evidence.
  `defects.capabilities` reports the supported operations. `defects.generate`
  can generate candidate inputs when its dependency and structure inputs are
  available; generated defects are not formation-energy evidence.
- `defects.analyze` currently checks prerequisites and does not implement full
  thermodynamics. Installing doped or supplying the two accepted files will not
  complete that wrapper. Use a supported checked method or a fully specified
  task-local analysis with independent verification when available; otherwise
  report the implementation gap. Never replace missing charged-defect energies,
  chemical potentials or corrections with guessed values.
- Relate defect evidence to dark current: SRH generation rate scales with
  trap density and inverse lifetime; state the mechanism explicitly.
- Select defect analysis only when it addresses the task; not every dark-current
  question requires generating new defects or starting DFT. Report the most
  relevant missing evidence or capability.

