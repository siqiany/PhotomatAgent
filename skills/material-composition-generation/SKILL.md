---
name: material-composition-generation
description: Generate inorganic material composition and chemical-formula candidates with the deployed conditional VAE for a target band gap or wavelength, optionally retrieve known formula matches and hand selected compositions to MatterGen for crystal generation. Use for 成分生成、组分生成、化学式生成、VAE 逆向设计、候选材料生成、target-band-gap composition proposals, and wavelength-conditioned semiconductor discovery; do not use it to claim device performance or validated structures.
---

# Material Composition Generation

Generate formula candidates with the real `generation.vae_formula` model tool,
preserve model provenance, and keep composition proposal, structure generation,
and property validation as separate scientific stages.

## Workflow

1. Convert the request into exactly one conditioning input:
   `target_band_gap_eV` or `target_wavelength_um`. If the user supplies a
   wavelength band, use the longest-wavelength cutoff for a gap upper bound and
   state that choice.
2. Discover the deferred generation tools with `tool_search` when they are not
   already visible. Inspect `generation.capabilities` if model availability is
   unknown.
3. Call `generation.vae_formula` with the conditioning input. Keep
   `require_charge_neutral=true` and `require_novel=true` unless the user asks
   for an exploratory relaxation. Pass `forbidden_elements` only when the user
   explicitly supplies that constraint; never exclude Hg, Cd, Pb, Bi, Te, or
   Sb by default.
4. Report each proposed formula with its chemical system, integer atom counts,
   composition discretization error, charge-neutrality result, novelty flag,
   target conditions, checkpoint provenance, novelty reference count and
   definition, model scope, and rejection counts.
5. If the user wants known database analogues rather than new formulas, call
   `generation.vae_retrieve` separately. Label its rows as retrieval results,
   not VAE-generated formulas.
6. If the user requests crystal structures, pass a selected proposal's
   `formula` as `proposed_formula` and its `chemical_system` to
   `generation.mattergen`. Preserve `vae_proposed_formula`,
   `mattergen_generated_formula`, `formula_preserved`, and
   `composition_distance`; never silently treat the MatterGen stoichiometry as
   identical to the VAE proposal. If MatterGen is unconfigured, report the
   missing `MATTERGEN_SKILL_SCRIPT` prerequisite instead of implying that a
   structure was generated.
7. Treat every generated formula or structure as
   `UNVALIDATED_GENERATED_STRUCTURE`. Use database, electronic-structure,
   stability, transport, defect, optical, and device capabilities in later
   stages according to the evidence gap.

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
  other device properties to the VAE. Route those to the corresponding
  scientific capabilities after a composition/structure exists.

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
