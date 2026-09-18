# Mechanism-guided material discovery pilot

This P0 pilot evaluates the workflow mechanics, rather than claiming an
improvement in material quality. The fixed task set is in
`experiments/mechanism-discovery-pilot.json`: it contains twelve tasks, three
planned offline repetitions per task, fixed fixture identifiers, and explicit
control fields. Na–Ag–Bi–S appears in one task only.

Metrics are computed from structured hypotheses, evidence, and typed runtime
events. Proposal counts and traceability do not imply scientific validation.
Missing properties remain unknown, and generated or prior evidence cannot
produce scientific success. The validity rate is unavailable until an
independent calculation or experiment is present.

Each task names a typed, repository-owned evidence fixture. Fixtures contain
deterministic `ScientificEvidence` records and open questions, are copied into
each fresh repeat as prior evidence, and are hashed into the configuration
snapshot. They never receive runtime evidence authority. Empty literature,
known-material prior, Na–Ag–Bi–S endmember prior, missing competitors, and
forged-fidelity adversarial prior are represented explicitly.

The only supported ablation is a declared workflow arm. Provider, model, task
set, budget, and evidence snapshot must match. Real model, remote, and HPC
experiments are explicit follow-up work and are never started by this pilot.
`controlled_fields` always includes the mandatory core entries provider, model,
task, budget, and evidence snapshot; these values must remain equal. Additional
workflow-related entries such as system prompt, skill index, context, or tool
surface explicitly authorize that treatment difference to vary.

## P0 validation record

The offline acceptance command is:

```text
PYTHONPATH=src /home/shiqiany/miniconda3/bin/pytest -q tests/test_discovery_experiments.py
```

The environment used while implementing this task did not provide `pymatgen`,
so the structured hypothesis fixture tests were skipped by the capability
guard. The ablation tests passed. The repository's prescribed `uv run` command
could not acquire its cache lock in the restricted environment. Full suite,
full mypy, and real-model ablation were not run here; their results must be
recorded before release.

The local Python 3.12 project environment completed the offline pilot:
12 tasks × 3 repeats = 36 runs, 36 completed, 0 failed, and 36 unevaluated
expectation checks. Raw aggregate metrics were proposal 0/36, unique
composition 0/36, basis 0/36, traceable basis 0/36, unknown property 0/36,
unsupported validation 0/36, 108 tool calls, input/output tokens unavailable,
and scientific validity unavailable. The fake provider produced completion
traces for every task; it did not perform real scientific validation.
