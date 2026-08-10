---
name: carrier-transport-analysis
description: Assess carrier transport evidence for IR detector materials: mobility, lifetime, diffusion length; delegate DFT transport to AMSET when data exists.
category: ir
tags: [transport, mobility, lifetime, amset]
license: MIT
---

# Carrier Transport Analysis

## When transport evidence matters

Collection efficiency and response speed depend on mobility and lifetime.
Short diffusion lengths or low mobilities can invalidate an otherwise
gap-appropriate candidate.

## Evidence ladder

1. Effective mass (cheap, `electronic.effective_mass`) as a mobility proxy.
2. Full transport analysis (`transport.analyze` via AMSET) when DFT
   deformation potentials, dielectric, and elastic data exist.
3. Experimental mobility/lifetime from literature (search via `literature`).

## Rules

- Report prerequisites when AMSET input data are missing; never invent
  mobility numbers.
- Distinguish phonon-limited mobility (intrinsic) from defect-limited
  mobility (extrinsic).
- For photoconductors, gain and lifetime are part of the transport story.

