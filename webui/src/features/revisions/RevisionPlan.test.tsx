import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RevisionPlan as Plan } from "../../api";
import { RevisionPlan } from "./RevisionPlan";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let mounted: Array<{ root: Root; host: HTMLElement }> = [];
afterEach(async () => {
  await act(async () => { for (const item of mounted) item.root.unmount(); });
  for (const item of mounted) item.host.remove();
  mounted = [];
});

async function render(runStatus?: string) {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  mounted.push({ root, host });
  const plan = {
    plan_id: "plan_1", task_id: "task_1", feedback_id: "feedback_1", source_run_id: "run_1",
    source_digest: "digest", source_run_status: "FAILED", source_artifacts: [], feedback_text: "fix this",
    status: "READY", issues: [], warnings: [], attempt: 1, created_at: "now", run_id: null,
    run_attempts: runStatus ? [{ attempt: 1, run_id: "run_attempt", status: runStatus }] : [],
    compiler_cost: { model_calls: 2, input_tokens: 100, output_tokens: 20, total_tokens: 120, estimated: false, settled: true, budget: { max_model_calls: 5, max_tokens: 500, wall_seconds: 60 } },
  } as unknown as Plan;
  await act(async () => { root.render(<RevisionPlan plan={plan} busy={false} runDisabled={false} onRun={vi.fn()} onClarify={vi.fn()} onRetry={vi.fn()} />); });
  return host;
}

describe("RevisionPlan", () => {
  it("shows compiler usage and permits an initial run", async () => {
    const host = await render();
    expect(host.textContent).toContain("2 次模型调用");
    expect(host.textContent).toContain("120 tokens");
    expect(host.querySelector("button")?.textContent).toContain("开始补充工作");
  });

  it.each(["FAILED", "CANCELLED", "INTERRUPTED"])("offers retry after %s", async (status) => {
    const host = await render(status);
    expect(host.querySelector("button")?.textContent).toContain("重试本次修订");
  });

  it.each(["SUCCEEDED", "PARTIAL", "RUNNING", "QUEUED", "CANCEL_REQUESTED"])("does not offer retry after %s", async (status) => {
    const host = await render(status);
    expect(host.querySelector("button")).toBeNull();
  });
});
