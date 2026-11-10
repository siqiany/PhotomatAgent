---
name: paper-code-reproduction
description: Prepare and check a bounded paper reference calculation using commit-pinned code and an isolated environment.
license: MIT
---

# Reproduce a reference calculation

Identify the paper version, target figure or metric and its stated inputs. Check
whether the implementation is the authors' code, a third-party example or a
reimplementation. Record the code commit, license source, input conditions and
reference tolerance. Treat a license label from repository metadata as a lead.
Bind each downloaded data/weight input through `spec.assets` with its URL,
version, license, retrieval time, byte length and SHA-256. Bind a seed to the
actual frozen argv. Record interface repairs and scientific adaptations
separately and freeze the patch as an input.

`reproduction.fetch_script` can retrieve one Python file from a public GitHub
commit with a supplied SHA-256. For a multi-file case, use
`reproduction.fetch_archive` with the pinned archive digest and an explicit
entry script and file list. Pass the returned bundle ID, source root and source
files to `reproduction.prepare`. An operator-supplied local tree can also be
frozen by naming its task-local source root and files. Use a task-local Python path with
`reproduction.prepare` to bind source, script, environment, arguments, timeout
and optional numerical reference. Use a registered object and exact source
sentence for a managed reference. A source-reported absolute tolerance needs
its own matching anchor. A supplied value or tolerance can support an
engineering comparison, with paper reproduction remaining UNKNOWN.
Obtain the environment ID from an approved
Podman environment. `reproduction.execute` takes the returned plan ID and
hash and follows the runtime's approval policy. Inspect the stored result with
`reproduction.inspect` before attempting recovery. An authorized repeat of `reproduction.execute`
with the same plan can recover a durable Podman receipt. When the receipt is
missing, an existing container may be reported as running or terminal with an
unknown scientific outcome. That request never starts a second container.

Report execution status, reference comparison and current-task applicability
separately. Numerical output must provide the named metric and its unit in JSON
for the built-in comparison. If the required backend, CUDA profile, inputs or
reference is unavailable, record the specific missing prerequisite. Avoid
host execution of untrusted code. A changed source or parameter set requires a
new plan and its own check.

For reusable Workbench methods, bind task-local example data with
`reproduction.prepare.spec.input_refs`. These files are staged separately from
the read-only source tree under `PHOTOMAT_INPUT_DIR`. Keep the numerical reference
comparison separate from qualification. The existing Workbench capability
learning action can qualify executions with a registered host suite (currently
`rc_network`, `report_tables`, or `fresnel_film`) and archive the frozen method version. Each
suite requires independent regression, transfer and negative checks. A new
research sample or operating condition requires its own applicability check.
