# Convergence diagnostics and deterministic recovery

## Verdict inputs (never Slurm state)

For a relax stage the convergence verdict comes from the OUTCAR + the
submitted INCAR only:

| Input | Source | Meaning |
|---|---|---|
| `reached required accuracy - stopping structural energy minimisation` | OUTCAR | VASP's formal ionic stop marker |
| TOTAL-FORCE last block | OUTCAR | per-atom forces; max row norm = max force |
| `EDIFFG` | submitted INCAR (mirrored) | force threshold `max|F| <= \|EDIFFG\|` |
| NSW / ionic step count | INCAR + OSZICAR / force blocks | exhaustion detection |
| `FORCES: max atom, RMS` | OUTCAR | per-step max-force history (plateau/excursion) |
| error tokens (`out of memory`, `BRMIX`, `ZHEGV`, MPI abort, segmentation, fatal) | OUTCAR | hard failures: never VALIDATED |

Adjacent ionic-step total-energy differences are NEVER a convergence
criterion. Slurm COMPLETED → COLLECTED → VALIDATED is the only path to
scientific evidence.

## Failure classification (closed set)

| Failure class | Detected when | Can auto-retry? | Restart artifact | Parameters changed by typed policy | Stop when |
|---|---|---|---|---|---|
| NSW_EXHAUSTED | steps >= NSW and not force-converged | yes (bounded) | CONTCAR (never initial POSCAR) | none; optional EDIFFG relaxation recorded as `practical_convergence` (old/new values + reason) | no CONTCAR, or attempts cap reached |
| WALLTIME | scheduler TIMEOUT | yes (bounded) | last complete CONTCAR | none | no CONTCAR, or attempts cap |
| FORCE_PLATEAU | last 3 max forces within 10% of each other, still > EDIFFG | yes (bounded) | CONTCAR | POTIM × 0.5, or IBRION → 1 (policy-gated) | no CONTCAR, or no allowed change |
| LINE_SEARCH_EXCURSION | last max force > 2× earlier max, no marker | only from a historical best | XDATCAR / lowest-force snapshot | none | no historical best: STOP (never continue from the worsened latest geometry) |
| OOM | scheduler OUT_OF_MEMORY or `out of memory` token | NO | — | resources must change (tasks/memory/LREAL) | always stop the identical attempt |
| SCF_NOT_CONVERGED | electronic SCF unconverged | NO | — | inspect mixing/SIGMA/NELM | always stop |
| AMBIGUOUS_SUBMISSION | sbatch outcome unknown | NO (reconcile first) | — | — | reconcile, then decide |
| STATUS_QUERY_FAILED | status refresh failed | NO — STATUS_ONLY | — | — | refresh/reconcile only; never resubmit |
| STATUS_UNKNOWN | everything else (crash, unreadable) | NO | — | — | manual inspection of OUTCAR/stderr |

## Automatic-attempt contract

- Every recovery attempt gets a NEW `attempt_id` and a NEW unique remote
  directory (submit-once + unique-dir invariants are preserved).
- Auto attempts are bounded by `RecoveryPolicy.max_auto_attempts`; after
  that the stage stops and reports the reason.
- Recovery parameters are produced ONLY by the typed policy (POTIM/IBRION/
  EDIFFG); the model never free-edits INCAR.
- Relaxing EDIFFG requires `practical_convergence` recorded with the OLD
  value, the NEW value, the reason and an explicit statement that the
  original threshold was NOT met; the workflow never presents it as the
  original accuracy.
- Every decision, restart source and parameter change is written into
  `recovery_provenance.json` plus the `recovery_attempts` ledger of
  `task_state.json`.

## Provenance fields per attempt

`attempt_id`, `failure` class, `restart_from` (CONTCAR/XDATCAR_BEST),
`parameter_changes`, `reason`, `request_id`, `job_id`, `remote_directory`,
`validated`, `practical_convergence` and the practical-convergence note.

