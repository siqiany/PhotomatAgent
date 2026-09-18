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

## P1 structure and validation boundary

Structure derivations are projected as first-class structure candidates. Their
candidate ID is derived from the canonical structure hash, not from a CIF
filename or path. Records that point to the same geometry merge into one
candidate while retaining derivation IDs, hypotheses, origins, and source
paths. Distinct geometries with the same normalized composition remain separate
candidates and can be checked independently. A parent structure is assigned
only when the runtime matches the operation input SHA-256 to a previously
registered structure artifact; an external mother structure may be linked to a
composition hypothesis without being represented as a structure parent.

CHGNet evidence is scoped from the actual workspace file and includes both the
input SHA-256 and canonical `structure_hash`/`candidate_id`. CHGNet energies and
forces are ML-potential observations for same-composition screening or
pre-relaxation. They cannot be relabeled as `E_hull`, formation energy,
stability, or detector performance. If relaxation changes geometry, the output
gets a new structure identity and downstream evidence must use that identity.
Downstream CHGNet and VASP evidence projects to structure candidates even when
its property is an energy or force. Candidate lineage uses the attested input
structure as parent for changed geometries; unchanged results omit a
self-parent. `representation.cif_hash` is the actual published file-byte
SHA-256, while `structure_hash` remains the canonical geometry identity.

VASP continues through the existing preparation, input validation, resource and
approval gates, and `SubmitOnceSession`. Slurm `COMPLETED` and command success
are scheduler/execution observations; scientific evidence requires collected,
validated artifacts. The pilot uses only fake/local calculation backends: its
fixtures are algorithmic boundary tests, x=0.25 is a stoichiometric test
point, and no real DFT, experiment, or material discovery is claimed. A whole
composition family may be rejected if its parent-structure or evidence scope
cannot be established.

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
