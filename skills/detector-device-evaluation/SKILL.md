---
name: detector-device-evaluation
description: Evaluate device-level evidence for IR detectors: dark current, responsivity, detectivity, NETD; use DEVSIM device simulation only with material parameters.
category: ir
tags: [device, dark current, detectivity, devsim]
license: MIT
---

# Detector Device Evaluation

## Evidence needed at device level

- Dark current vs bias and temperature (generation-recombination,
  diffusion, tunneling, surface leakage).
- Responsivity and detectivity measurements or device simulations.
- NETD for thermal-imaging targets.
- Operating temperature and cooling requirement.

## Tooling

- `device.devsim_capabilities` lists DEVSIM capabilities and required inputs.
- `device.run_script` executes workspace DEVSIM scripts only (restricted).
- `device.inspect_result` summarizes saved device results.

## Rules

- Device simulation requires material parameters (gap, mobilities, lifetimes,
  doping); report prerequisites when absent.
- Compare device dark current against `ir.compile_constraints` targets and
  the thermal guideline.
- Never present a device simulation as a measurement.

