import { expect, it } from "vitest";
import { budgetFailure, pendingApproval } from "./runDiagnostics";

it("reconstructs only unresolved approval for the current run", () => {
  const events = [
    { seq: 1, run_id: "a", type: "approval_pending", payload: { request_id: "old", tool_name: "vasp.submit" } },
    { seq: 2, run_id: "a", type: "approval_resolved", payload: { request_id: "old" } },
    { seq: 3, run_id: "b", type: "approval_pending", payload: { request_id: "other" } },
    { seq: 4, run_id: "a", type: "approval_pending", payload: { request_id: "current", tool_name: "vasp.prepare" } },
  ];
  expect(pendingApproval(events, "a")?.requestId).toBe("current");
  expect(pendingApproval(events, "b")?.requestId).toBe("other");
  expect(pendingApproval(events, null)).toBeNull();
});

it("reads conservative token preflight diagnostics from a failed run", () => {
  const events = [{ seq: 1, run_id: "a", type: "loop_failed", payload: {
    error: "RuntimeError: TOKEN_BUDGET_EXHAUSTED: total_limit=18000, consumed=14400, remaining=3600, input_upper_bound=4200",
  } }];
  expect(budgetFailure(events, "a")).toEqual({ totalLimit: 18000, consumed: 14400, remaining: 3600, inputUpperBound: 4200 });
  expect(budgetFailure(events, "b")).toBeNull();
});
