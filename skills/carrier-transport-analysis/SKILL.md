---
name: carrier-transport-analysis
description: >-
  Assess mobility, lifetime and diffusion-length evidence for carrier transport.
  The AMSET wrapper currently checks prerequisites only, without computing transport.
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

- Existing experimental mobility/lifetime may directly answer the task when
  material identity and measurement conditions match.
- `electronic.effective_mass` can analyze suitable band-structure inputs, but
  mass alone does not determine mobility or scattering time.
- `transport.capabilities` describes the current wrapper and AMSET prerequisites.
  `transport.analyze` currently checks dependency and input presence; it does not
  execute the full AMSET solver even if those checks succeed. Do not repeatedly
  supply data or install packages expecting this wrapper to produce mobility.
- When computation is required, discover a compatible checked method or use a
  task-local AMSET script only if its complete scientific inputs, runtime,
  permissions and validation are supported. Otherwise report the implementation
  or input gap. Load `capability-and-script-workflow` for that execution route.

Choose among these evidence sources by the question; they are not a required order.

## Rules

- Report prerequisites when AMSET input data are missing; never invent
  mobility numbers.
- Distinguish phonon-limited mobility (intrinsic) from defect-limited
  mobility (extrinsic).
- For photoconductors, gain and lifetime are part of the transport story.

