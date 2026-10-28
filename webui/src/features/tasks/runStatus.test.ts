import { describe, expect, it } from "vitest";

import type { TaskDetail } from "../../api";
import { canApplyStatusRefresh, persistedRunStatus, replayedRunStartedStatus } from "./runStatus";

const detail = {
  task: { task_id: "task_1", session_id: "session_1", title: "目标", goal: "目标", attachment_ids: [], created_at: "now" },
  runs: [
    { run_id: "run_partial", task_id: "task_1", request: { text: "请求", strategy: "standard", budget: { max_model_calls: 2, max_tokens: 200, wall_seconds: 20 } }, status: "PARTIAL", created_at: "now" },
  ],
  links: [],
  artifacts: [],
} satisfies TaskDetail;

describe("persisted run status", () => {
  it("preserves PARTIAL instead of inferring success from run_completed", () => {
    expect(persistedRunStatus(detail, "run_partial")).toBe("PARTIAL");
  });

  it("does not invent a status when the run is absent", () => {
    expect(persistedRunStatus(detail, "missing")).toBeNull();
  });

  it("ignores a GET status when a newer run event arrived during the request", () => {
    expect(canApplyStatusRefresh(4, 5)).toBe(false);
    expect(canApplyStatusRefresh(5, 5)).toBe(true);
  });

  it("does not regress a persisted terminal status while replaying run_started", () => {
    expect(replayedRunStartedStatus("PARTIAL")).toBe("PARTIAL");
    expect(replayedRunStartedStatus("SUCCEEDED")).toBe("SUCCEEDED");
    expect(replayedRunStartedStatus("QUEUED")).toBe("RUNNING");
  });
});
