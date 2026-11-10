# Learning and method qualification

Read when the task asks to capture a method, improve a reusable capability or assess
what an existing learned skill proves. Keep these distinct:

| Record | What it permits or supports |
| --- | --- |
| Static or approved procedural guidance, loaded by `skill_view` | Instructions with provenance; approval of text does not establish execution or scientific truth |
| Executable local method, found with `skill_search` | A frozen script/environment/checker contract admitted for replay within its scope |
| Result of `skill_run` on current inputs | Actual execution and the returned checker outcome; inspect artifacts and limitations |
| Scientific evidence | Claims limited to the supported subject, method and conditions; software checks alone do not establish experimental performance |

Use an exposed `skill_search` directly. When it is absent, search the deferred tool
surface only if that mode supports discovery; do not assume CLI, workbench and frozen
evaluation have identical tools. Use the exact returned `skill_id`, scope, input keys
and current input paths for `skill_run`. A method's earlier PASS does not cover new
data or a changed operating regime. Skill names in the static index are not replay IDs.

Preserve a reproducible execution before proposing capture: script, explicit input
contract, environment/plan identity, parameters and actual output artifacts. Checker
requirements must reflect the intended method, not merely echo its output. The host
owns capture, trusted validation and promotion; the agent may propose improvements
through available tools but cannot approve itself or edit stores/manifests to publish.
Honor frozen scripts, checker contracts, independent cases, budgets and exact-bundle
approval requirements where the active host workflow requires them.

Current automatic workbench learning is enabled only for supported source executions
and checker families. Its source classifier currently recognizes RC-network analysis
(`rc_network`) and report-table extraction (`report_tables`) with their required input
and `result.json` contracts. Do not reshape unrelated scientific work into those
contracts or promise that any successful script can be automatically learned. This
limit describes automatic learning; other existing manual/local/QD workflows retain
their own admission rules and do not become authorized by loading this skill.

When learning is unsupported, deliver the useful analysis and, when requested, a
candidate method description with explicit gaps. Do not call it validated, published
or reusable until the applicable host workflow returns that status. Regression,
transfer and negative cases improve method qualification; their success still does
not turn a local artifact check into a material or experimental validation claim.
