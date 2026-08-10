---
name: optical-response-analysis
description: Evaluate optical response evidence for IR candidates: absorption coefficient at target wavelengths, quantum efficiency; absorption onset from band gap is necessary but not sufficient.
category: ir
tags: [optical, absorption, quantum efficiency, spectra]
license: MIT
---

# Optical Response Analysis

## Evidence to collect

- Absorption coefficient alpha(lambda) across the spectral band (orders of
  magnitude matter: e.g. >1e3 /cm for thin-film detectors).
- Quantum efficiency or external quantum efficiency at the target
  wavelengths.
- Optical transitions: direct vs indirect gap determines absorption strength.
- Optional transient absorption analysis via `optics.transient_absorption`
  when spectra exist.

## Rules

- Band gap onset is necessary, not sufficient: a candidate can absorb weakly
  or have a poor collection efficiency.
- Report the absorption depth vs device thickness trade-off.
- When only the gap is known, state optical response as a missing evidence
  item.

