import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as apiModule from "../../api";
import { api, type RevisionPlan, type RunRecord, type TaskDetail, type WorkbenchEvent } from "../../api";
import { TaskWorkspace } from "./TaskWorkspace";

vi.mock("../skills/SkillPanel", () => ({ SkillPanel: () => null }));
vi.mock("../evaluations/EvaluationPanel", () => ({ EvaluationPanel: () => null }));
vi.mock("../revisions/FeedbackForm", () => ({ FeedbackForm: () => null }));
vi.mock("../revisions/RevisionPlan", () => ({
  RevisionPlan: ({ plan }: { plan: { plan_id: string; feedback_text: string } }) => <output data-testid="revision-plan-id">{plan.plan_id}|{plan.feedback_text}</output>,
}));
vi.mock("../revisions/RevisionComparison", () => ({ RevisionComparison: () => null }));
vi.mock("../revisions/QDRevisionPanel", () => ({ QDRevisionPanel: () => null }));
vi.mock("./ActivityTimeline", () => ({ ActivityTimeline: () => null }));
vi.mock("./Conversation", () => ({ Conversation: () => null }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const detail = {
  task: {
    task_id: `task_${"a".repeat(32)}`, session_id: "session_1", title: "Test task",
    goal: "Check that the task workspace mounts", attachment_ids: [], created_at: "now",
  },
  runs: [], links: [], artifacts: [],
} as TaskDetail;
let mounted: Array<{ root: Root; host: HTMLElement }> = [];
afterEach(async () => {
  await act(async () => { for (const item of mounted) item.root.unmount(); });
  for (const item of mounted) item.host.remove();
  mounted = [];
  vi.restoreAllMocks();
  window.sessionStorage.clear();
});

describe("TaskWorkspace mount", () => {
  it("loads a task without recursive revision-plan state updates", async () => {
    vi.spyOn(api, "task").mockResolvedValue(detail);
    vi.spyOn(api, "capabilities").mockResolvedValue({ items: [] });
    vi.spyOn(api, "procedures").mockResolvedValue({ items: [] });
    vi.spyOn(api, "revisionPlans").mockResolvedValue({ items: [] });
    vi.spyOn(apiModule, "subscribeToTask").mockReturnValue({ close: vi.fn() } as unknown as EventSource);
    const logged = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push({ root, host });

    await act(async () => {
      root.render(<TaskWorkspace taskId={detail.task.task_id} mobilePanel="workspace" onTaskChanged={vi.fn()} />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(host.textContent).toContain("Check that the task workspace mounts");
    expect(host.querySelector("[aria-label='科研流程经验']")).not.toBeNull();
    expect(logged).not.toHaveBeenCalled();
  });

  it("submits bounded evolution defaults and the user's explicit delivery filenames", async () => {
    vi.spyOn(api, "task").mockResolvedValue(detail);
    vi.spyOn(api, "capabilities").mockResolvedValue({ items: [] });
    vi.spyOn(api, "procedures").mockResolvedValue({ items: [] });
    vi.spyOn(api, "revisionPlans").mockResolvedValue({ items: [] });
    vi.spyOn(api, "startRun").mockResolvedValue({ task_id: detail.task.task_id, run_id: "run_new" });
    vi.spyOn(apiModule, "subscribeToTask").mockReturnValue({ close: vi.fn() } as unknown as EventSource);
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push({ root, host });

    await act(async () => {
      root.render(<TaskWorkspace taskId={detail.task.task_id} mobilePanel="workspace" onTaskChanged={vi.fn()} />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    const budgetDisclosure = host.querySelector(".budget-controls") as HTMLDetailsElement;
    await act(async () => { budgetDisclosure.open = true; budgetDisclosure.dispatchEvent(new Event("toggle")); });
    expect((host.querySelector("[aria-label='启用自动修订']") as HTMLInputElement).checked).toBe(true);
    expect((host.querySelector("[aria-label='最多修订轮数']") as HTMLInputElement).value).toBe("3");

    const setInput = async (selector: string, value: string) => {
      const input = host.querySelector(selector) as HTMLInputElement;
      await act(async () => {
        const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), "value")?.set;
        setter?.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      });
    };
    await setInput("[aria-label='最多修订轮数']", "5");
    await setInput("[aria-label='必须交付的文件名']", "report.md, data.csv, report.md");
    await setInput("[aria-label='本轮需求']", "Write a report");
    const checkbox = host.querySelector("[aria-label='启用自动修订']") as HTMLInputElement;
    await act(async () => {
      checkbox.click();
      host.querySelector("form.composer")?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(api.startRun).toHaveBeenCalledWith(detail.task.task_id, expect.objectContaining({
      evolution: {
        enabled: false, max_rounds: 5, patience: 2,
        required_artifacts: ["report.md", "data.csv"], checks: [], target: null,
      },
    }));
  });

  it("ignores a pending revision-plan refresh after switching tasks", async () => {
    const taskOne = detail.task.task_id;
    const taskTwo = `task_${"b".repeat(32)}`;
    const runOne = `run_${"c".repeat(32)}`;
    const runTwo = `run_${"d".repeat(32)}`;
    const plan = (planId: string, taskId: string, runId: string): RevisionPlan => ({
      plan_id: planId, task_id: taskId, feedback_id: `feedback_${planId}`, source_run_id: runId,
      source_digest: "digest", source_run_status: "SUCCEEDED", source_artifacts: [],
      feedback_text: "update", status: "READY", issues: [], warnings: [], attempt: 1,
      created_at: "now", run_id: runId,
      run_attempts: [{ attempt: 1, run_id: runId, status: "SUCCEEDED" }], compiler_cost: {},
    });
    const oldPlan = plan("plan_old", taskOne, runOne);
    const newPlan = plan("plan_new", taskTwo, runTwo);
    const taskOneDetail = {
      ...detail,
      runs: [{ run_id: runOne, task_id: taskOne, status: "SUCCEEDED", created_at: "now", request: { text: "done", strategy: "standard" } } as RunRecord],
    };
    const taskTwoDetail = {
      ...detail,
      task: { ...detail.task, task_id: taskTwo, title: "Second task", goal: "Second task remains selected" },
      runs: [],
    };
    window.sessionStorage.setItem(`photomatagent:revision-context:${taskOne}`, JSON.stringify({ planId: oldPlan.plan_id, sourceRunId: runOne, revisionRunId: runOne }));
    window.sessionStorage.setItem(`photomatagent:revision-context:${taskTwo}`, JSON.stringify({ planId: newPlan.plan_id, sourceRunId: runTwo, revisionRunId: runTwo }));
    vi.spyOn(api, "task").mockImplementation(async (requestedTaskId) => requestedTaskId === taskOne ? taskOneDetail : taskTwoDetail);
    vi.spyOn(api, "capabilities").mockResolvedValue({ items: [] });
    vi.spyOn(api, "procedures").mockResolvedValue({ items: [] });
    vi.spyOn(api, "revisionPlans").mockResolvedValue({ items: [] });
    let onEvent: ((event: WorkbenchEvent) => void) | undefined;
    vi.spyOn(apiModule, "subscribeToTask").mockImplementation((_id, callback) => {
      onEvent = callback;
      return { close: vi.fn() } as unknown as EventSource;
    });
    let resolveOldRefresh: ((value: RevisionPlan) => void) | undefined;
    let oldPlanFetches = 0;
    vi.spyOn(api, "revisionPlan").mockImplementation(async (_requestedTaskId, planId) => {
      if (planId === oldPlan.plan_id && ++oldPlanFetches === 2) {
        return new Promise<RevisionPlan>((resolve) => { resolveOldRefresh = resolve; });
      }
      return planId === oldPlan.plan_id ? oldPlan : newPlan;
    });
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push({ root, host });

    await act(async () => {
      root.render(<TaskWorkspace taskId={taskOne} mobilePanel="workspace" onTaskChanged={vi.fn()} />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(host.querySelector("[data-testid='revision-plan-id']")?.textContent).toBe(`${oldPlan.plan_id}|${oldPlan.feedback_text}`);
    await act(async () => {
      onEvent?.({ seq: 1, run_id: runOne, type: "run_completed", payload: {} });
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(resolveOldRefresh).toBeDefined();

    await act(async () => {
      root.render(<TaskWorkspace taskId={taskTwo} mobilePanel="workspace" onTaskChanged={vi.fn()} />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(host.querySelector("[data-testid='revision-plan-id']")?.textContent).toBe(`${newPlan.plan_id}|${newPlan.feedback_text}`);
    await act(async () => {
      resolveOldRefresh?.(oldPlan);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(host.querySelector("[data-testid='revision-plan-id']")?.textContent).toBe(`${newPlan.plan_id}|${newPlan.feedback_text}`);
  });

  it("recovers the latest persisted plan without a tab session and lets the user switch history", async () => {
    const taskId = detail.task.task_id;
    const runOne = `run_${"e".repeat(32)}`;
    const runTwo = `run_${"f".repeat(32)}`;
    const makePlan = (planId: string, sourceRunId: string, feedback: string): RevisionPlan => ({
      plan_id: planId, task_id: taskId, feedback_id: `feedback_${planId}`, source_run_id: sourceRunId,
      source_digest: "digest", source_run_status: "SUCCEEDED", source_artifacts: [],
      feedback_text: feedback, status: "READY", issues: [], warnings: [], attempt: 1,
      created_at: "2026-10-08T10:00:00Z", run_id: null, run_attempts: [], compiler_cost: {},
    });
    const older = makePlan("plan_older", runOne, "Keep the original expert feedback");
    const latest = makePlan("plan_latest", runTwo, "Latest expert feedback remains visible");
    vi.spyOn(api, "task").mockResolvedValue(detail);
    vi.spyOn(api, "capabilities").mockResolvedValue({ items: [] });
    vi.spyOn(api, "procedures").mockResolvedValue({ items: [] });
    vi.spyOn(api, "revisionPlans").mockResolvedValue({ items: [older, latest] });
    const readPlan = vi.spyOn(api, "revisionPlan").mockImplementation(async (_id, planId) => planId === older.plan_id ? older : latest);
    vi.spyOn(apiModule, "subscribeToTask").mockReturnValue({ close: vi.fn() } as unknown as EventSource);
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    mounted.push({ root, host });

    await act(async () => {
      root.render(<TaskWorkspace taskId={taskId} mobilePanel="workspace" onTaskChanged={vi.fn()} />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(host.querySelector("[data-testid='revision-plan-id']")?.textContent)
      .toBe(`${latest.plan_id}|${latest.feedback_text}`);
    const history = host.querySelector("[aria-label='历史修订计划']") as HTMLSelectElement | null;
    expect(history).not.toBeNull();
    expect([...history!.options].map((option) => option.textContent).join(" ")).toContain(older.feedback_text);
    expect([...history!.options].map((option) => option.textContent).join(" ")).toContain(latest.feedback_text);

    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(history!), "value")?.set;
      setter?.call(history, older.plan_id);
      history!.dispatchEvent(new Event("change", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(readPlan).toHaveBeenCalledWith(taskId, older.plan_id);
    expect(host.querySelector("[data-testid='revision-plan-id']")?.textContent)
      .toBe(`${older.plan_id}|${older.feedback_text}`);
    expect(JSON.parse(window.sessionStorage.getItem(`photomatagent:revision-context:${taskId}`) ?? "{}"))
      .toMatchObject({ planId: older.plan_id, sourceRunId: older.source_run_id });
  });
});
