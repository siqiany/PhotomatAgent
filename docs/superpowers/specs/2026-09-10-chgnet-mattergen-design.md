# CHGNet Screening and MatterGen Runtime Design

## Goal

Make CHGNet and MatterGen usable through PhotomatAgent's normal deferred-tool
runtime, and teach the native scientific skills to use them as a bounded
candidate funnel for infrared-material work.

## Scientific roles

MatterGen is a generative model. Its outputs remain
`UNVALIDATED_GENERATED_STRUCTURE` and are never evidence of stability,
synthesizability, band gap, or detector performance. For the repository's
main infrared workflows, the supported modes are:

- `dft_band_gap` when a target band gap or cutoff wavelength is the primary
  condition;
- `chemical_system` when an explicit element set or a VAE proposal supplies
  the chemical system.

The supplied MatterGen checkpoints do not jointly condition on band gap and
chemical system. If both constraints matter, generate small, separately
conditioned batches and apply downstream CHGNet and DFT checks. A VAE formula
is lineage, not an exact-composition guarantee; `formula_preserved` and
`composition_distance` remain explicit.

CHGNet is a cheap ML interatomic-potential screen. It may provide single-point
energy, forces, stress, magnetic moments, and pre-relaxed structures. Its
evidence is always `source_type=ml_interatomic_potential` and
`fidelity=ml_potential`. Raw CHGNet energies may rank structures only within
the same reduced composition. They must not be compared across compositions
or described as DFT formation energy, energy above hull, thermodynamic
stability, synthesizability, or detector validation.

## Architecture

### CHGNet capability pack

Add a `chgnet` capability pack registered through the existing scientific
capability registry and status reporter. The pack owns two deferred tools:

- `chgnet.screen`: accept one to 32 workspace-contained structure paths, run
  pretrained CHGNet single-point inference, and return bounded per-structure
  energy/force/stress/magnetic-moment summaries. It may rank only within
  same-composition groups.
- `chgnet.relax`: pre-relax one workspace-contained structure with bounded
  `fmax`, `steps`, and optional cell relaxation, then write a CIF below
  `user_output/chgnet/` and return convergence and before/after summaries.

Imports stay lazy. Missing CHGNet becomes a typed capability diagnostic and
never prevents the base runtime from starting. Configuration comes from
`ScientificConfig`: checkpoint `0.3.0`, device `cpu`, screen limit 32,
relaxation force threshold `0.1 eV/angstrom`, and at most 200 steps by default.
Environment overrides remain bounded and non-secret.

### MatterGen execution

Keep MatterGen outside the PhotomatAgent Python 3.12 environment. Replace the
missing-script-only path with a packaged runner that invokes a configured
`mattergen-generate` executable, extracts `generated_crystals_cif.zip`, and
writes a deterministic `manifest.json`. Retain `MATTERGEN_SKILL_SCRIPT` as a
legacy override, but normal operation must not require it.

`ScientificConfig` resolves the executable, optional `HF_HOME`, default
checkpoint, candidate limit, timeout, guidance factor, and seed. The tool
uses workspace-owned output under `user_output/mattergen/`, never accepts a
model-supplied executable, and reuses a completed manifest for identical
parameters. Supported checkpoint values are exactly `dft_band_gap` and
`chemical_system`.

The current workstation will be configured with the existing MatterGen 1.0.3
environment at `/home/shiqiany/AIagent/Photoelectric detection/external/mattergen/.venv`
and its existing Hugging Face cache. These machine-specific values do not
become portable source defaults.

## Dependency and packaging policy

Add a `chgnet` optional dependency group with `chgnet>=0.4.1,<0.5` and include
it in the aggregate `science` extra. MatterGen is not a project dependency
because its pinned torch/PyG stack conflicts with the lightweight main
runtime; it remains an isolated executable integration.

## Skill behavior

Update `material-composition-generation` and `infrared-material-screening`,
and add a focused `ml-potential-screening` skill. The prescribed funnel is:

1. compile targets and retrieve or generate candidate formulas;
2. generate structures with MatterGen only when structures are needed;
3. run `chgnet.screen` on small candidate batches;
4. run `chgnet.relax` only for finalists;
5. send a much smaller set to gated VASP validation.

Every skill must preserve the evidence boundary and must report unsupported or
out-of-domain structures instead of silently promoting them.

## Error handling and safety

- All model-visible structure and output paths are resolved through
  `Workspace.resolve`; absolute or `..` escapes are rejected.
- Candidate count, relaxation steps, force tolerance, guidance factor, and
  timeouts are bounded.
- External-process stdout/stderr is summarized and never returned without a
  size bound.
- Missing executables, dependencies, checkpoints, manifests, archives, and
  malformed CIFs become typed tool errors.
- No HPC submission or remote shell surface is added.

## Verification scope

Use TDD with focused tests only. Run the new CHGNet unit tests, affected
generation/skill/registry tests, one tool-registry E2E, one real small CHGNet
inference when installation succeeds, and a MatterGen dry-run/manifest E2E.
Do not run the repository-wide suite or a large MatterGen generation batch.

