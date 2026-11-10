import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";
import type { RunRecord, WorkbenchEvent } from "../../api";
import { Conversation } from "./Conversation";
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("Conversation long requests", () => {
  it("keeps long revision instructions folded while preserving their full content", async () => {
    const text = "Continue the existing research task in a new round.\n" + "完整修订依据".repeat(500);
    const host = document.createElement("div"); const root = createRoot(host);
    const run = { run_id: "run_one", status: "PARTIAL", request: { text } } as RunRecord;
    await act(async () => root.render(<Conversation runs={[run]} events={[]} activeRunId={null} activeStatus="PARTIAL" onInspect={() => undefined} />));
    const details = host.querySelector(".conversation-request-details") as HTMLDetailsElement;
    expect(details).not.toBeNull(); expect(details.open).toBe(false);
    expect(details.querySelector("p")?.textContent).toBe(text);
    expect(host.querySelector(".conversation-request-preview")?.textContent?.length).toBeLessThan(400);
    await act(async () => root.unmount());
  });

  it("shows a host QD summary only for the completed run that owns the event", async () => {
    const host = document.createElement("div"); const root = createRoot(host);
    const run = { run_id: "run_one", status: "COMPLETED", request: { text: "Search" } } as RunRecord;
    const events = [
      { seq: 1, type: "qd_completed", run_id: "other_run", payload: { status: "SOLVED", best_trial_id: "wrong" } },
      { seq: 2, type: "qd_completed", run_id: "run_one", payload: { status: "SOLVED", best_trial_id: "trial_1", cost: { model_calls: 3, tokens: 8562 } } },
    ] as WorkbenchEvent[];
    await act(async () => root.render(<Conversation runs={[run]} events={events} activeRunId={null} activeStatus="COMPLETED" onInspect={() => undefined} />));
    expect(host.querySelector(".qd-completion-summary")?.textContent).toContain("trial_1");
    expect(host.querySelector(".qd-completion-summary")?.textContent).not.toContain("wrong");
    expect(host.textContent).not.toContain("本轮没有完成最终回复");
    await act(async () => root.unmount());
  });
});
