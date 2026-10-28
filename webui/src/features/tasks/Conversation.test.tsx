import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";
import type { RunRecord } from "../../api";
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
});
