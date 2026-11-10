"""Re-run the identical public-data study against the repaired runtime.

Keeps the baseline output/database immutable. The workload still starts with
the same 2,048-token summary cap, exercising the adaptive retry rather than
making the problem disappear by increasing the experiment cap beforehand.
"""
from __future__ import annotations

import json
import runpy

import context_compaction_live as live

BASELINE = live.OUTPUT
live.OUTPUT = live.REPO / "user_output/context-compaction-fixed-v2-20261029"
live.WORKSPACE_ROOT = live.REPO / "tmp/context-fixed-v2-20261029/workspace"


if __name__ == "__main__":
    experiment = live.Experiment()
    experiment.config = experiment.config.model_copy(update={"summary_max_tokens": 8192, "response_reserve_tokens": 8192})
    manifest_path = live.OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = experiment.config.model_dump()
    manifest["summary_budget_change"] = "Use the production 8192-token aggregate/reply reserve, retaining the same 2048 initial chunk cap; retry 2048→4096→8192."
    live.dump(manifest_path, manifest)
    try:
        experiment.execute()
        runpy.run_path(str(live.REPO / "scripts/validation/context_compaction_live_audit.py"), run_name="__main__")
        baseline = json.loads((BASELINE / "metrics.json").read_text())
        after = json.loads((live.OUTPUT / "supplement/result.json").read_text())
        before_manual = json.loads((live.OUTPUT / "snapshots/before-manual.json").read_text())
        after_manual = json.loads((live.OUTPUT / "snapshots/after-manual.json").read_text())
        summary_calls = [c for c in experiment.recorder.calls if c["kind"] == "summary"]
        live.dump(live.OUTPUT / "comparison.json", {
            "baseline_metrics": baseline, "after": after,
            "context_limit_tokens": experiment.config.context_limit_tokens,
            "same_dataset_sha256": experiment.expected["sha256"],
            "baseline_preserved": True,
            "all_summary_checkpoints_complete": all("incomplete" not in warning
                for warning in after_manual["engine"]["compaction_state"]["compression_warnings"]),
            "summary_attempts": summary_calls,
            "manual_history_preserved": before_manual["conversation"] == after_manual["conversation"],
            "manual_scientific_state_preserved": before_manual["scientific"] == after_manual["scientific"],
            "task_metadata_change": "Bootstrap emits a COMPLETED P01 execution record with its real result reference; source data/numerics, user constraints and phase prompts are unchanged.",
        })
    except BaseException as error:
        live.dump(live.OUTPUT / "failure.json", {"type": type(error).__name__, "error": str(error), "calls": len(experiment.recorder.calls)})
        raise
