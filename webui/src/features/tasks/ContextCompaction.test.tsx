import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import * as apiModule from "../../api";
import { api, type TaskDetail, type WorkbenchEvent } from "../../api";
import { TaskWorkspace } from "./TaskWorkspace";
import { Conversation } from "./Conversation";
import { contextCompactionSummary } from "./conversation";

vi.mock("../skills/SkillPanel", () => ({ SkillPanel: () => null }));
vi.mock("../skills/CapabilityLearningPanel", () => ({ CapabilityLearningPanel: () => null }));
vi.mock("../skills/ProcedurePanel", () => ({ ProcedurePanel: () => null }));
vi.mock("../evaluations/EvaluationPanel", () => ({ EvaluationPanel: () => null }));
vi.mock("./InformationPanel", () => ({ TaskManagement: () => null }));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const detail = {
  task: { task_id: "task_context", session_id: "session_context", title: "Research", goal: "Keep original goal", attachment_ids: [], created_at: "now" },
  runs: [{ run_id: "run_research", task_id: "task_context", status: "SUCCEEDED", created_at: "now", request: { text: "Research", strategy: "standard" } }],
  links: [], artifacts: [],
} as TaskDetail;
let mounted: { host: HTMLDivElement; root: Root } | undefined;
afterEach(async () => {
  if (mounted) { await act(async () => mounted!.root.unmount()); mounted.host.remove(); }
  mounted = undefined;
  vi.restoreAllMocks();
  window.sessionStorage.clear();
});

async function mountWorkspace(value = detail) {
  vi.spyOn(api, "task").mockResolvedValue(value);
  vi.spyOn(api, "capabilities").mockResolvedValue({ items: [] });
  vi.spyOn(api, "revisionPlans").mockResolvedValue({ items: [] });
  vi.spyOn(apiModule, "subscribeToTask").mockReturnValue({ close: vi.fn() } as unknown as EventSource);
  const host = document.createElement("div"); document.body.append(host);
  const root = createRoot(host); mounted = { host, root };
  await act(async () => { root.render(<TaskWorkspace taskId={value.task.task_id} mobilePanel="workspace" onTaskChanged={() => undefined} />); });
  return host;
}

it("queues bounded compaction without sending or clearing a research draft", async () => {
  const compact = vi.spyOn(api, "compactContext").mockResolvedValue({ task_id: detail.task.task_id, run_id: "run_compact" });
  const start = vi.spyOn(api, "startRun");
  const host = await mountWorkspace();
  const draft = host.querySelector("[aria-label='本轮需求']") as HTMLTextAreaElement;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(draft, "Keep my unsent question");
    draft.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const button = [...host.querySelectorAll("button")].find((item) => item.textContent === "压缩上下文")!;
  await act(async () => { button.click(); });
  expect(compact).toHaveBeenCalledWith("task_context", { max_model_calls: 20, max_tokens: 100_000, wall_seconds: 1800 }, expect.any(String));
  expect(start).not.toHaveBeenCalled();
  expect(draft.value).toBe("Keep my unsent question");
  expect(host.textContent).toContain("完整历史保留");
});

it("disables manual compaction while a research run is active", async () => {
  const host = await mountWorkspace({ ...detail, runs: [{ ...detail.runs[0], status: "RUNNING" }] });
  const button = [...host.querySelectorAll("button")].find((item) => item.textContent === "压缩上下文")!;
  expect(button.disabled).toBe(true);
});

it("keeps a stable idempotency key when retrying a lost compaction response", async () => {
  const compact = vi.spyOn(api, "compactContext").mockRejectedValueOnce(new Error("Response lost")).mockResolvedValue({ task_id: "task_context", run_id: "run_compact" });
  const host = await mountWorkspace();
  const button = [...host.querySelectorAll("button")].find((item) => item.textContent === "压缩上下文")!;
  await act(async () => { button.click(); });
  await act(async () => { button.click(); });
  expect(compact.mock.calls).toHaveLength(2);
  expect(compact.mock.calls[0][2]).toBe(compact.mock.calls[1][2]);
});

it("renders maintenance outcome without claiming a new scientific answer", async () => {
  const host = document.createElement("div"); document.body.append(host);
  const root = createRoot(host); mounted = { host, root };
  const run = { ...detail.runs[0], request: { ...detail.runs[0].request, host_operation: { operation: "context_compaction" } } };
  const events = [{ seq: 1, run_id: run.run_id, type: "context_compaction_completed", payload: { tokens_before: 24000, tokens_after: 6000 } }] as WorkbenchEvent[];
  await act(async () => root.render(<Conversation runs={[run]} events={events} activeRunId={null} activeStatus="SUCCEEDED" onInspect={() => undefined} />));
  expect(host.textContent).toContain("24,000 → 6,000");
  expect(host.textContent).not.toContain("本轮没有完成最终回复");
  expect(contextCompactionSummary(events, "other_run")).toBeNull();
});
