---
name: capability-and-script-workflow
description: Find tools or reusable methods, write task-local scripts, select execution environments, and resolve Python dependencies through reproducible plans. Use for capability discovery and execution repair.
license: MIT
---

# Tools, scripts and environments

Choose the smallest supported route for the current gap. Respect runtime
permissions and inspect the actual tool schema before using a deferred capability.

## Find an existing capability

- Call directly exposed tools with their supplied schemas. For deferred tools,
  use `tool_search`, inspect a match with `tool_describe`, then invoke it with
  `tool_call`. Do not try to describe a direct tool through the deferred bridge.
  An installed Python package
  supplies script imports; it does not create a registered runtime tool.
- For a reusable checked method, call `skill_search` directly when exposed and use the returned
  exact skill ID, scope, current input keys, and checker contract with `skill_run`.
  Prior validation does not validate a new dataset or operating condition.
- `skill_view` loads procedures and references; its skill name is not an
  executable `skill_id`. If a search misses, refine it using the operation or
  namespace before concluding that the capability is unavailable.

Begin unfamiliar work with `method_search` to compare the local catalog and
verified archive. Supply known inputs, incompatible regimes and explicit
exclusions. Use `method_search_external` with selected sources when a public
query is needed and the runtime authorizes it. The result separates method
cards, software leads, literature metadata and per-source diagnostics. Use
`method_inspect` with the exact candidate identifier before deciding how to
verify a method for the current inputs.

For third-party paper code, load `paper-code-reproduction`. Acquisition,
environment preparation and execution need their own permissions. The
reproduction workflow uses a frozen Podman environment. A local development
environment is suitable only for its separately authorized task-local scripts.

## Write and run a script

Use `write` or `edit` for a task-local Python script. In workbench runs use the
provided `tmp/workbench/<task_id>/` directory and copy required inputs there.
In other runs follow the script tool's path contract. Use workspace-relative
paths and discover `script_run`; a shell is not required for script creation.
Keep final deliverables under `user_output/<task-name>/` using the permitted
file tools after inspecting execution artifacts.

If no usable environment ID is known, call `script_run` with `script_path`
and omit `environment_id` when its current schema permits. This returns READY
environments without executing. Frozen evaluation may require a prebound ID;
use that binding and do not discover/install a replacement or edit frozen scripts.
Compare each environment's `packages` and versions against the script's
requirements. Check Python compatibility from plan metadata when available;
do not infer it from the environment ID. Import names may differ from distribution names.
`packages_status` other than `verified` or nonzero `packages_omitted` or
`environments_omitted` leaves the inventory incomplete or unknown. READY alone
does not prove compatibility. Use an exact returned ID; `local_dev`, `python`,
`default`, and `local` are not substitute IDs.

## Create an environment when dependencies are missing

1. Identify all direct third-party requirements, including packages already
   present in the old environment. Prefer standard-library code when it
   satisfies the task without losing required functionality.
2. Discover `capability_search` for each exact PyPI package name or
   `name==version`. Keep returned candidate IDs; use `capability_inspect` when
   compatibility or package identity is unclear.
3. Pass all selected candidate IDs together to `capability_plan(candidate_ids=...)`.
   Review direct requirements, jointly resolved dependencies, backend and plan
   hash. Resolve conflicts by a compatible version choice within the task's
   constraints; do not silently relax a required version or scientific method.
4. Call `capability_install` with the exact `plan_id` and `plan_sha256`, then
   run the script with its returned READY environment ID. The new environment
   does not inherit packages from the old one. Preserve existing environments.

This route installs wheel-based Python dependencies. A missing model checkpoint,
API credential, system executable, HPC configuration or registered tool needs
its own supported configuration path; repeated package installation cannot fix it.

An environment created for `script_run` does not change the Python environment
hosting the agent. Existing domain tools that import packages in the host, including
`device.run_script`, do not acquire those dependencies through `capability_install`.
Use an authorized host configuration path, or an appropriate task-local script
when supported; preserve the scientific method and validation requirements. A
registered prerequisite-only wrapper is not a complete solver. Check capability
metadata and diagnostics before attempting repair. Read the general research
skill's learning reference when capturing or qualifying a method; running a script
does not by itself authorize publication or promotion.

## Recover and verify

Use the actual diagnostic to distinguish an unavailable backend, unresolved
dependency, permission denial, missing input or script error. Change only the
relevant cause before retrying. Do not guess identifiers, repeat unchanged
failed calls, or bypass the registry with shell installation. Report any remaining
blocker and useful partial result. Verify output existence and content; successful
installation or exit status does not establish scientific validation.

Podman uses an existing pinned image and network-disabled execution. `local_dev`
executes on the host without OS isolation. Both retain hashed wheel plans and
normal runtime permissions.
