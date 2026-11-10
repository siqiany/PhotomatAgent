---
name: materials-process-design
description: Record an evidence-linked material processing proposal with parameters, specimen conditions, controls and unresolved inputs.
license: MIT
---

# Propose a material process

Start with the intended comparison, available material and operating constraints.
Expand only the details that determine the decision or would change reproducibility.
Use a coherent baseline from Methods, equipment documentation or observations.
State the transfer assumption when changing the phase, scale, surface or equipment.

Deliver the candidate, constraints and discriminating validation in the requested
answer. Use `process.register_plan` when a durable plan is requested or needed
for subsequent tool actions. Tool schema discovery uses `tool_describe`; only
register the actual candidate. If registration fails, report its status and deliver
the usable proposal and evidence gaps. Give registered steps and samples
stable IDs. Put characterization before the corresponding pass/fail branches.
Record REPORTED intervals exactly and bind them to the sample and conditions
attested from the original text. TRANSFERRED parameters need a target sample, a
transfer assumption, and a control experiment. PROPOSED values remain hypotheses.
Use `previous_plan_id` and `revision_basis` when changing a plan after feedback.

For a deliverable route, add a `proposal` to the plan. Describe the baseline,
change, mechanism or exploration rationale, design variables and a validation
test with its control, metric, criterion and next action. For a conditional
route, give branches and actionable unknowns. Use PROPOSED for an untested
step or parameter without direct literature. Keep execution prerequisites and
scientific validation separate from whether the proposal can be delivered.
When a protocol is missing, keep constrained design variables or conditional
branches. A safety data sheet informs handling limits. Use equipment and process
records to establish operating settings or identify the pretests needed to set them.
Use the actual operation name and typed values for quantities, material choices,
timelines, endpoints and pending variables. State the dose basis and distinguish
an exploration window from a setpoint. Compare preparation risk alongside
performance potential. Connect risks and results to the steps, specimens or
assumptions they depend on. For numerical outcomes, carry their identity,
conditions, range basis and limitations into summaries.

For a process with sequential chemical steps, trace one coherent conditional
baseline from input through the final specimen. At each handoff state the
material form and medium, the condition for entering the next step, and what to
test if compatibility is unknown. A list of independent reagent choices is not
a baseline. For a timed temperature change, start at the actual reaction
temperature and state the endpoint selection rule, measured trajectory T(t),
and observable that demonstrates reaction arrest. A bath or coolant temperature
is separate from the specimen temperature. Identify external heat exchange
versus direct addition. A directly added medium changes specimen composition;
define the numerator and denominator of any dilution ratio.
To test arrest, compare the specimen immediately after quenching with the same
quenched condition after a hold. A hot continuation control checks whether the
chosen observable can detect continued reaction over that hold.

For scale-dependent quantities, preserve the user's total and the definition of
the normalization basis throughout the calculation, table and operating steps.
State the units of a total, its per-area or per-mass value, and the conversion
between them. Alternate geometry is a separate scenario. For comparisons with a
coefficient of variation, normalize the effect using a declared reference mean,
or compare in physical units with an uncertainty in those same units.
Changing a route invalidates assumptions tied to the old material, geometry or
device. Report previous thresholds as historical model scenarios until the new
route's dependencies are checked.

Before delivery, reconcile totals and subtotals across the narrative, tables
and operating steps. Separate independent specimens from repeat readings on
one specimen. Record intentionally changed conditions as interventions and
match the other relevant conditions in the control. For a brief consultation,
give the tradeoff and smallest discriminating comparison without a full plan.

Use `decision.record` with `research_stop` and the registered plan IDs when
new searches would leave the decision unchanged or a test must resolve the gap.
For a hard conflict, record the conflicting constraint and the smallest change
that would permit reconsideration. If the runtime closes search, deliver the
bounded findings and the next discriminating action from the available record.

Load only the relevant method card with `skill_view`:

- [Ligand exchange and transport](references/ligand-transport.md)
- [Core–shell interdiffusion](references/core-shell-interdiffusion.md)
- [Film formation and interface](references/film-interface.md)
- [Electrodeposition metrology](references/electrodeposition.md)
- [Detector noise and responsivity](references/detector-noise.md)
- [Batch repeatability](references/batch-repeatability.md)

Keep the proposal's missing inputs and scientific validation state visible. A
numerical plan or deterministic unit check does not establish a recipe's yield or
device performance.
