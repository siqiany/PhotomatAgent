import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it } from "vitest";

import type { WorkbenchEvent } from "../../api";
import { ActivityTimeline } from "./ActivityTimeline";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function event(seq: number): WorkbenchEvent {
  return { seq, run_id: "run_1", type: "tool_completed", payload: { source: seq } };
}

it("keeps an older event in view while new stream events arrive", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const render = async (events: WorkbenchEvent[]) => {
    await act(async () => {
      root.render(<ActivityTimeline events={events} status="运行中" label={(type) => type} toolResultText={() => "done"} />);
    });
  };
  const click = async (text: string) => {
    const button = [...host.querySelectorAll("button")].find((item) => item.textContent?.includes(text));
    if (!button) throw new Error(`Missing button: ${text}`);
    await act(async () => { button.click(); });
  };

  try {
    await render(Array.from({ length: 60 }, (_, index) => event(index + 1)));
    await click("查看更早的事件");
    expect(host.textContent).toContain("事件 11");

    await render(Array.from({ length: 90 }, (_, index) => event(index + 1)));
    expect(host.textContent).toContain("事件 11");
    expect(host.textContent).toContain("查看最新事件");

    await click("查看最新事件");
    expect(host.textContent).not.toContain("事件 11");
    expect(host.textContent).toContain("事件 90");
  } finally {
    await act(async () => { root.unmount(); });
    host.remove();
  }
});
