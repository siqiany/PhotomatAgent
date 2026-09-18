---
name: mechanism-guided-material-design
description: Use when a materials task asks for mechanism-based hypotheses, explicit compositional or process constraints, candidate design operations, or a validation plan without asking a generative model to sample formulas.
category: materials-discovery
tags: [mechanism, hypotheses, materials, proposal, validation]
license: MIT
---

# Mechanism-Guided Material Design

Use this skill to produce a small, auditable set of material hypotheses. A
proposal is a design hypothesis and validation plan; it is not a measured
property, a structure calculation, or a scientific PASS.

## Select one explicit route

1. **Mechanism route** — the user asks to reason from a mechanism, substitution,
   ordering, distortion, prototype transfer, or synthesis constraint. Preserve
   the original goal and register each actionable composition with
   `generation.register_hypothesis` before requesting validation. This route
   does not sample a VAE or call a database retrieval tool as a substitute for
   reasoning.
2. **Model-generation route** — the user explicitly asks for VAE formula
   generation or model sampling. Use the dedicated generation skill and report
   the model, conditions, seed, and missing dependencies. Never claim that a
   model proposal is measured or validated.
3. **Known-material route** — the user asks for known materials, analogues, or
   database records. Use search/retrieval tools and label every result as a
   known candidate or prior. Retrieval is not new generation and does not prove
   novelty or performance.

Do not infer a route from a candidate answer. If the request is ambiguous, ask
which source path the user authorizes and state the choice in the report.

## Goal and constraint contract

Keep the exact user goal. Store explicit required or forbidden elements,
alloying/doping permissions, wavelength, temperature, mechanism, material form,
and process limits as user constraints. Unspecified fields remain unspecified;
do not silently exclude heavy infrared elements. Separate user HARD constraints
from model suggestions. Proposal tasks may have no numeric constraints. A
validation task needs explicit, checkable constraints and cannot pass with an
empty target.

## Hypothesis card

For each candidate record: normalized formula, design operation, mechanism
statement, supporting basis and the finite claim it supports, assumptions,
counter-hypothesis, synthesis notes with source or exploration label, and
validation questions. If no external basis exists, label `NO_EXTERNAL_BASIS`
and call it an exploratory hypothesis. Register through
`generation.register_hypothesis`; registration records an unvalidated lineage
and performs no calculation.

Use citations only for claims the cited source actually supports. A known
analogue does not establish the proposed composition; charge balance does not
establish phase stability; a target band gap does not establish responsivity.
Any DFT, experiment, transport, defect, stability, or device statement must
point to an actual corresponding tool result with source, method, conditions,
units, and limitations. Generated formulas and structures remain
`UNVALIDATED_GENERATED_STRUCTURE` until independent evidence is accepted by
the Checker.

## Validation hand-off

Prioritize the smallest evidence-producing step that can change the decision:
composition legality, parent structure/site feasibility, structure or phase
competition, electronic/optical properties, defects/transport, then process or
device measurements. Keep unknowns as unknowns and report the next capability;
do not turn a guessed threshold, analogy, mock result, or generated structure
into evidence.

When a registered hypothesis has a trusted structure derivation, carry the
structure hash and workspace artifact path into the next tool call. A structure
candidate is identified by its canonical geometry hash, so renaming a CIF does
not create a new candidate; two geometries with the same composition remain
separate. A derived parent may be claimed only when the runtime finds the
input-file hash in a previously registered structure artifact. An external
mother structure can support the composition hypothesis, but it is not a
structure parent unless its artifact is registered.

The CHGNet funnel is a same-composition ML-potential screen or pre-relaxation.
Its energy is never an `E_hull`, formation-energy, stability, or synthesizability
claim. CHGNet relaxation output with changed geometry is a new structure
identity. VASP evidence must come from the existing prepare, validation, and
`SubmitOnceSession` gates; a successful command or Slurm `COMPLETED` state alone
does not establish a scientific observation.

## Case boundary

For a sulfide request that allows isovalent alloying and forbids selected
elements, expand explicit compositions and treat a value such as `x=0.25` as a
sampled point, never as an optimum. Keep formal valence checks separate from
stability. Recognize incompatible simple band-to-band assumptions (for example
conflicting wavelength and gap claims) and list alternative mechanisms as
hypotheses needing evidence. No real HPC job or successful material may be
reported without independent artifacts and validation.

The x=0.25 example is only a fixed-ratio composition test. Test fixtures and
fake backends demonstrate provenance and boundary behavior; they are not
predictions, DFT results, or experimental evidence. A whole composition family
may be rejected when the parent-structure or evidence assumptions fail.
