---
name: scientific-research-workflow
description: Coordinate multi-step research, reproducible analysis and reuse or learning of checked methods across domains. Choose evidence and capabilities by the task; load specialists when relevant.
license: MIT
---

# Scientific execution and capability learning

Use this skill for research tasks that require choosing methods, gathering evidence,
executing an analysis or reusing a learned capability. A direct explanation or a
small file edit need not become a research campaign.

## Define what the current task needs

Identify the requested answer or artifact, supplied inputs, scientific constraints,
and the evidence needed to support that answer. Distinguish explanation, literature
review, data processing, hypothesis proposal, simulation and method qualification.
Do not turn one into another without a task-relevant reason. Preserve explicit units,
conditions, exclusions and required methods; identify assumptions separately.

Use the current request to interpret the durable goal. Read relevant attachments and
reuse attested results before repeating work. If prior project work could supply an
input or method, use exposed `research_history_search` / `research_history_read`
when available. Historical summaries are leads: inspect the source and its scope
before treating a conclusion as evidence or an artifact as a current input.

## Choose an action, rather than a fixed pipeline

Pick the smallest action that resolves an uncertainty important to the result:

| Current need | Appropriate route |
| --- | --- |
| Explain a concept with sufficient existing information | Answer with the relevant assumptions and limits |
| Establish a published fact or compare reported results | Retrieve relevant sources; retain source coverage and experimental conditions |
| Reuse a known computation | Find a checked local method and compare its input contract and scope with this task |
| Transform or analyze supplied data | Use a suitable registered tool or a task-local script with verified dependencies |
| Propose a new scientific explanation or design | State its mechanism, basis, alternatives and questions for validation |
| Produce new simulation evidence | Verify inputs, supported implementation, resource needs and permissions before execution |
| Improve a reusable method | Use the supported host learning workflow and independent checks within the approved budget |

These routes may be combined when justified. A literature result does not require
new simulation, and a generated candidate does not automatically authorize screening
or an HPC submission. Stop when the requested result is supported; report important
remaining gaps without silently expanding the assignment.

When the method is unclear, call `method_search` with the current need and any
known inputs. Use `decision.record` when the question, alternatives, assumptions,
or a method gap with a stated basis should survive the current turn. For a known
paper, locate its version before reading the target objects. For wider discovery,
select the relevant sources with `method_search_external` under runtime approval.
Check each source status before treating an empty result as a search miss.
Declare unavailable-but-obtainable inputs separately from inputs the task
forbids obtaining or varying. For a conditional choice, record how to measure
or scan the missing input. Use a validation proposal when further searching
would leave the decision unchanged. A search stop bounds the inspected sources
and can yield a proposal or a partial answer with explicit gaps.

Load `capability-and-script-workflow` for tool/method discovery, writing scripts,
environment selection or dependency recovery. Direct tools use their exposed schemas;
deferred tools use `tool_search -> tool_describe -> tool_call`. Tool registration,
package availability and a complete solver implementation are separate facts.

For infrared detector questions load `infrared-photodetector-design`, then the
specialist analysis needed for the current gap. Other subjects use relevant indexed
skills and available tools without applying infrared targets or material-design
workflows. A broad research remit does not imply every domain has an executable
backend, experimental interface or trusted checker.

## Verify and deliver

Check actual artifacts and provenance, not just stdout or exit codes. Use checks
appropriate to the result: input/output schema, units, conservation or limiting cases,
known reference cases, independent numerical checks or accepted scientific validators.
Distinguish a script's own assertions from an independent trusted checker. Where a
checker is unavailable, describe the checks performed and keep unsupported claims
UNKNOWN; never invent a PASS or substitute a model's confidence for validation.

Retain original input identity, method and dependency versions when available,
parameters, environment/execution IDs, output paths and limitations. Do not change
scientific assumptions or thresholds merely to make a check pass. Explain conflicts
in evidence and match claims to the same sample, dataset, structure and conditions.
Store final artifacts under `user_output/<task-name>/` and scratch under `tmp/`,
subject to the active mode's narrower path restrictions. Report achieved results,
validation scope and any blocker in the user's language.

For learning, qualifying or promoting methods, load
`skill_view(name="scientific-research-workflow", path="references/capability-learning.md")`.
Ordinary execution does not need that reference. Runtime checks, explicit approvals,
frozen evaluation bindings and HPC submission gates remain authoritative in every route.
