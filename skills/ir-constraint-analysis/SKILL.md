---
name: ir-constraint-analysis
description: Use ir.compile_constraints to derive detector physics constraints (cutoff energy, thermal limits, responsivity, BLIP detectivity) from a spectral band.
category: ir
tags: [infrared, constraints, band gap, detectivity]
license: MIT
---

# IR Constraint Analysis

Always start an IR task with `ir.compile_constraints` (namespace `ir`).

Key derived quantities and their meaning:

- `photon_energy_range_eV`: photon energies inside the band; candidates must
  absorb at the lowest energy end.
- `cutoff_energy_requirement_eV` (= band gap upper bound): Eg must be <= hc /
  lambda_max to reach the longest wavelength.
- `thermal.kBT_eV` and `thermal_suppression_ratio_at_cutoff`: how strongly
  thermal carriers are suppressed at the cutoff gap; small ratios mean a
  thermally limited detector (dark current risk).
- `ideal_responsivity_A_per_W`: upper bound at unity quantum efficiency.
- `blackbody_photon_flux_photons_s_m2` and `blip_detectivity_cm_Hz_W`:
  background-limited ideal performance; realistic D* is below BLIP.

Interpretation guidance:

- Eg slightly below cutoff is necessary but far from sufficient: transport,
  defects, and device architecture decide the actual performance.
- When the band gap requirement is below ~4 kT, expect strong thermal dark
  current and cooling requirements.
- Compare candidate evidence (band gap from database or DFT) against the
  cutoff bound and report the margin.

