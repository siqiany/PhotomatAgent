---
name: detector-device-evaluation
description: >-
  Evaluate device-level evidence for IR detectors: dark current, responsivity,
  detectivity, NETD; use DEVSIM device simulation only with material parameters.
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

Select only the operation needed: interpreting measurements or existing results
does not require a new device simulation. `device.run_script` uses the agent's
host Python dependency and does not accept a `script_run` environment ID. Installing
DEVSIM in a new script environment does not install it into that host. For script
discovery or environment repair load `capability-and-script-workflow`; use the
execution route supported by the actual schema and permissions.

## Rules

- Device simulation requires material parameters (gap, mobilities, lifetimes,
  doping); report prerequisites when absent.
- Compare device dark current against `ir.compile_constraints` targets and
  the thermal guideline.
- Never present a device simulation as a measurement.
- Verify model equations, geometry, boundary conditions, parameter provenance,
  numerical behavior and the requested observables. A completed script alone
  does not validate the device model or its physical predictions.

