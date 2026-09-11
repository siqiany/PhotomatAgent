# Final whole-branch repair report

Base reviewed: `be07edf` (`codex/chgnet-mattergen`). No real MatterGen
generation, HPC submission, or repository-wide test suite was run.

## TDD RED evidence

- The new 33-atom CHGNet force test initially reported `0.0` instead of the
  final atom's `123.0` force. The missing/malformed/NaN/Inf force parameterized
  test reported **7 failed, 1 passed, 8 deselected**.
- The cell-relaxation counterexample initially reported `converged=True` from
  zero atomic forces despite no joint cell-filter evidence.
- The MatterGen funnel test initially raised `KeyError: 'path'` because the
  generated candidate exposed only the absolute `structure_path`.
- The candidate-path containment test initially did not raise for
  `../outside.cif`.
- The unrounded-force test initially observed `0.05` for a raw force of
  `0.050000004`.

## Implemented repair

- CHGNet force summaries now require a complete finite `(n_atoms, 3)` matrix,
  validate dimensions against the structure, iterate over every atom without
  the observation bound, and return typed `invalid_prediction` errors for
  missing, malformed, NaN, or Inf forces. Max-force convergence uses the
  unrounded scalar.
- Fixed-cell relaxation uses the complete final force matrix. Cell relaxation
  reports `converged=None` and an explicit `cell convergence unconfirmed`
  limitation unless the optimizer/ASE result supplies an explicit joint
  convergence marker; explicit fake optimizer markers are accepted.
- CHGNet relaxation evidence now carries relative input/output paths,
  `fmax_eV_A`, `steps`, `relax_cell`, convergence, and structure provenance.
- MatterGen manifests and normalized candidates retain a workspace-relative
  `path` alongside the absolute `structure_path`; candidate relative paths are
  checked for containment and consistency, and the funnel test calls
  `chgnet.screen` directly with the relative path.
- Removed the unsupported `category` and `tags` keys from the new
  `ml-potential-screening` skill frontmatter only.
- Removed the extra blank EOF line from the CHGNet/MatterGen design document.
- Bound the typed CHGNet loader dynamically so the existing runtime compatibility
  fallback remains accepted by mypy with CHGNet 0.4.2.

## Focused verification

Commands were run from the shared worktree with the repository Python 3.12
environment:

```text
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q tests/test_chgnet.py -k 'not pack_tools_are_deferred_and_registered'
16 passed, 1 deselected in 0.78s
```

The deselected registry test remains an environment-only pre-existing failure:
the full test imports the optional chemistry pack and this venv has no
`rdkit`.

```text
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m pytest -q tests/test_generation.py tests/test_skills.py
56 passed, 1 skipped, 12 warnings in 8.11s
```

The 12 warnings are the existing pymatgen CIF parsing warnings; the single skip
is the optional real MatterGen CLI contract test when no executable is
configured.

```text
PYTHONPATH=src /home/shiqiany/AIagent/PhomatAgent/.venv/bin/python -m mypy src/photomatagent/scientific/capabilities/chgnet/__init__.py src/photomatagent/scientific/capabilities/generation/mattergen.py src/photomatagent/scientific/capabilities/generation/mattergen_runner.py src/photomatagent/scientific/capabilities/generation/tools.py
Success: no issues found in 4 source files

/home/shiqiany/AIagent/PhomatAgent/.venv/bin/python /mnt/c/Users/牧之原/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/ml-potential-screening
Skill is valid!

git diff --check a6eaa70
exit=0

git diff --check
exit=0
```
