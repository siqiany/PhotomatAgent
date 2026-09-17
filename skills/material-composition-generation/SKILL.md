---
name: material-composition-generation
description: Use when the user explicitly requests VAE-based inverse generation of inorganic compositions or chemical formulas from trained material properties, with optional MatterGen structure generation; use for 成分生成、组分生成、化学式生成 and do not substitute retrieval or mechanism reasoning for the requested model route.
---

# Material Composition Generation

Generate new formula candidates only when the user explicitly requests the
deployed multi-property
`generation.vae_formula` model tool. This is inverse generation, not nearest
neighbour, database retrieval, or mechanism reasoning. Preserve model
provenance and keep composition proposal, structure generation, and property
validation as separate stages. For mechanism-based requests, use
`generation.register_hypothesis` through the mechanism-guided skill.

## Workflow

1. Map every requested, trained material property into `target_properties`.
   Supported canonical fields are `gap_selected_eV`,
   `cutoff_wavelength_um_from_gap`, `formation_energy_eV_per_atom`,
   `energy_above_hull_eV_per_atom`, `density_g_cm3`, `dielectric_mean`,
   `avg_electron_mass_m0`, `avg_hole_mass_m0`, `bulk_modulus_GPa`,
   `shear_modulus_GPa`, `exfoliation_energy_meV_per_atom`,
   `max_IR_mode_cm-1`, `min_IR_mode_cm-1`, and `spillage`. Omit unspecified
   fields; do not fill them with guessed values. The legacy
   `target_band_gap_eV` and `target_wavelength_um` parameters remain valid for
   single-condition requests. If the user supplies a wavelength band, use the
   longest-wavelength cutoff and state that choice.
2. Discover the deferred generation tools with `tool_search` when they are not
   already visible. Inspect `generation.capabilities` if model availability is
   unknown.
3. Call `generation.vae_formula` with all supplied material conditions in one
   request. Keep
   `require_charge_neutral=true` and `require_novel=true` unless the user asks
   for an exploratory relaxation. Pass `forbidden_elements` only when the user
   explicitly supplies that constraint; never exclude Hg, Cd, Pb, Bi, Te, or
   Sb by default.
4. Report the normalized `target_properties`, unspecified conditions, any
   clipped out-of-distribution conditions, and each proposed formula with its
   chemical system, integer atom counts,
   composition discretization error, charge-neutrality result, novelty flag,
   target conditions, checkpoint provenance, novelty reference count and
   definition, model scope, and rejection counts.
5. Do not call `generation.vae_retrieve` for inverse-generation requests. It
   searches existing records and cannot replace VAE sampling. Use it only when
   the user separately and explicitly asks for known database analogues.
6. If the user requests crystal structures, select the MatterGen checkpoint
   explicitly: use `pretrained_name=dft_band_gap` for a band-gap/cutoff target
   or `pretrained_name=chemical_system` for an element system. The supplied
   checkpoints do not jointly condition on both fields; run separate small
   batches when both constraints matter. Pass a selected proposal's `formula`
   as `proposed_formula` and call `generation.mattergen`. Preserve `vae_proposed_formula`,
   `mattergen_generated_formula`, `formula_preserved`, and
   `composition_distance`. The normal path uses the configured isolated
   `mattergen-generate` executable. `MATTERGEN_SKILL_SCRIPT` is retained only
   as a legacy compatibility override; never imply that a structure was
   generated when the executable, checkpoint, archive, or manifest is missing.
7. After structures exist, run `chgnet.screen` on a small batch. Compare raw
   CHGNet energies only within identical reduced compositions; label all
   CHGNet values `source_type=ml_interatomic_potential` and
   `fidelity=ml_potential`. Run `chgnet.relax` only for finalists, then send a
   much smaller set through the gated VASP workflow for DFT validation.
8. Treat every generated formula or structure as
   `UNVALIDATED_GENERATED_STRUCTURE`. Use database, electronic-structure,
   stability, transport, defect, optical, and device capabilities in later
   stages according to the evidence gap. A CHGNet result is not DFT energy,
   energy above hull, synthesizability, or detector validation.

## Failure handling

- If `generation.vae_formula` reports missing prerequisites, show the missing
  asset or dependency exactly. Do not replace the VAE result with invented
  formulas. The model and novelty metadata normally ship inside PhotomatAgent;
  `PHOTOMATAGENT_VAE_ASSET_ROOT`, `VAE_CHECKPOINT_PATH`, and
  `VAE_METADATA_PATH` are optional retrained-model overrides only.
- If novelty metadata is unavailable, only set `require_novel=false` after
  telling the user that novelty can no longer be established against the
  training set.
- If no proposals survive, report the rejection counts. Relax at most one
  explicit constraint at a time and label the rerun exploratory.
- Do not send responsivity, EQE, detectivity, dark current, response time, or
  other device-level properties to the VAE: its training pairs contain the 14
  material-level fields above, not device labels. Route device validation to
  the corresponding scientific capabilities after a composition/structure
  exists.
- If band gap and cutoff wavelength are both supplied, they must agree with
  `wavelength_um = 1.239841984 / gap_eV`; otherwise ask the user to resolve the
  conflicting targets.

## Reproducibility requests

When the user asks to audit or retrain the VAE, use the project-contained
`data/photoelectric_vae/README.md` rebuild sequence and verify all committed
inputs/assets with `scripts/vae/verify_assets.py`. Do not look for model files
in sibling repositories.

## Output contract

Return three clearly separated parts:

- model inputs and configuration;
- generated/retrieved candidates with provenance;
- validation status, limitations, and the next evidence-producing step.
