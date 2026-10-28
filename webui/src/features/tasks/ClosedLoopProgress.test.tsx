import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it } from "vitest";

import type { WorkbenchEvent } from "../../api";
import { ClosedLoopProgress } from "./ClosedLoopProgress";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("shows separate delivery and scientific outcomes plus the explicit stop reason", async () => {
  const events: WorkbenchEvent[] = [
    { seq: 1, run_id: "run_1", type: "closed_loop_round_started", payload: { round_index: 1, phase: "MAKER" } },
    { seq: 2, run_id: "run_1", type: "closed_loop_checked", payload: {
      round_index: 1, delivery_status: "PASS", scientific_status: "UNKNOWN", issues: [], artifact_ids: ["artifact_1"],
    } },
    { seq: 3, run_id: "run_1", type: "closed_loop_completed", payload: {
      status: "NEEDS_EVIDENCE", rounds: 1, delivery_status: "PASS", scientific_status: "UNKNOWN",
    } },
  ];
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  try {
    await act(async () => { root.render(<ClosedLoopProgress events={events} />); });
    expect(host.textContent).toContain("交付：通过");
    expect(host.textContent).toContain("科研证据：未知");
    expect(host.textContent).toContain("停止原因：仍需证据");
  } finally {
    await act(async () => { root.unmount(); });
    host.remove();
  }
});

it("keeps scientific evidence unknown when a delivery-complete run lacks science evidence", async () => {
  const events: WorkbenchEvent[] = [{
    seq: 1, run_id: "run_1", type: "closed_loop_completed", payload: {
      status: "WALL_TIME_EXHAUSTED", rounds: 1, delivery_status: "PASS", scientific_status: "UNKNOWN",
    },
  }];
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  try {
    await act(async () => { root.render(<ClosedLoopProgress events={events} />); });
    expect(host.textContent).toContain("达到运行时间上限");
    expect(host.textContent).toContain("科研证据 未知");
    expect(host.textContent).not.toContain("科研证据通过");
  } finally {
    await act(async () => { root.unmount(); });
    host.remove();
  }
});
