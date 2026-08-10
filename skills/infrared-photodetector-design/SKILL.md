---
name: infrared-photodetector-design
description: >-
  Main domain skill for infrared photodetector materials research: turn a
  detector goal into physical constraints, find existing evidence, identify
  the evidence gap, select the cheapest sufficient capability, and iterate.
category: ir
tags: [infrared, photodetector, lwir, mwir, evidence-gap]
license: MIT
---

# Infrared Photodetector Design (main domain skill)

## Purpose

Drive a research goal through: Goal -> IR physical constraints -> existing
evidence -> evidence gap -> tool/external-skill selection -> new evidence ->
re-evaluation -> answer / next investigation.

## Evidence-gap reasoning policy (apply after every evidence round)

Answer these four questions explicitly:

1. What is known?
2. What is uncertain?
3. What evidence is still missing?
4. Which available capability can reduce the most important uncertainty?

Do not start the next tool call until question 4 has a concrete answer.

## Multi-fidelity escalation (cheap evidence first)

Prefer, in order, only when justified by the current gap:

1. database / literature (materials, literature)
2. structure / cheap analysis (structure)
3. electronic / transport / defect analysis (electronic, transport, defects)
4. DFT (delegate execution to external skills such as AtomisticSkills VASP/QE SOPs)
5. device simulation (device)

This is not a fixed workflow. The evidence gap decides the next step.
Never escalate to expensive evidence when cheap evidence can close the gap.

## Core rules

- A database match is NOT a validated detector. Band-gap matching alone never
  justifies "promising detector" conclusions.
- Use `ir.compile_constraints` first to get deterministic physics constraints
  (photon energies, cutoff energy, thermal limits, responsivity, BLIP).
- Report missing prerequisites explicitly; never fabricate DFT/experimental
  values.
- Record every observation as evidence with source and method.
- External skills (AtomisticSkills, computational-chemistry-agent-skills)
  own the *execution* SOPs for VASP/QE/phonopy/HPC. This skill only decides
  when they are needed, why, and what counts as trustworthy output.

## Typical loop

1. `skill_view` this skill and `ir.compile_constraints` for the band.
2. Search materials database + literature for candidate evidence.
3. Cross-check candidates with structure/electronic analysis when inputs exist.
4. State known / uncertain / missing; pick the next capability.
5. Produce an evidence-grounded recommendation with explicit gaps.

