import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api, type Artifact, type Procedure, type RunRecord } from "../../api";
import { ProcedurePanel } from "./ProcedurePanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const taskId = `task_${"a".repeat(32)}`;
const runId = `run_${"b".repeat(32)}`;
const source: Artifact = {
  artifact_id: "artifact_report", task_id: taskId, run_id: runId, filename: "procedure.md",
  media_type: "text/markdown", size_bytes: 40, sha256: "digest",
};
let mounted: Array<{ root: Root; host: HTMLElement }> = [];
afterEach(async () => {
  await act(async () => { for (const item of mounted) item.root.unmount(); });
  for (const item of mounted) item.host.remove();
  mounted = [];
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function render(artifacts: Artifact[] = [source], runs: RunRecord[] = []) {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  mounted.push({ root, host });
  await act(async () => { root.render(<ProcedurePanel taskId={taskId} artifacts={artifacts} runs={runs} />); });
  await act(async () => { await Promise.resolve(); });
  return host;
}

async function rerender(host: HTMLElement, artifacts: Artifact[]) {
  const entry = mounted.find((item) => item.host === host);
  if (!entry) throw new Error("Missing mounted procedure panel");
  await act(async () => { entry.root.render(<ProcedurePanel taskId={taskId} artifacts={artifacts} />); });
}

async function setValue(host: HTMLElement, label: string, value: string) {
  const input = host.querySelector(`[aria-label="${label}"]`) as HTMLInputElement | HTMLTextAreaElement | null;
  if (!input) throw new Error(`Missing control ${label}`);
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function click(host: HTMLElement, label: string) {
  const button = [...host.querySelectorAll("button")].find((item) => item.textContent?.includes(label));
  if (!button) throw new Error(`Missing button ${label}`);
  await act(async () => { button.click(); await Promise.resolve(); });
}

describe("ProcedurePanel", () => {
  it("creates guidance from a task-owned report, then records review before approval", async () => {
    vi.spyOn(api, "procedures").mockResolvedValue({ items: [] });
    const created: Procedure = {
      skill_id: "procedure_1", name: "ftir-baseline", index_name: "procedure-ftir-baseline-a1b2c3d4",
      description: "FTIR baseline handling", scope: "FTIR spectra", task_id: taskId, run_id: runId,
      status: "CANDIDATE", validation_status: "UNKNOWN", instructions: "1. inspect baseline",
    };
    vi.spyOn(api, "createProcedure").mockResolvedValue(created);
    vi.spyOn(api, "procedure").mockResolvedValue(created);
    const approve = vi.spyOn(api, "approveProcedure").mockResolvedValue({ ...created, status: "APPROVED_PROCEDURE" });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("1. inspect baseline\n2. subtract blank", { status: 200 })));
    const host = await render();
    expect(host.textContent).toContain("科研流程经验");

    await setValue(host, "经验来源产物", source.artifact_id);
    await act(async () => { await Promise.resolve(); });
    await setValue(host, "流程说明", "Reviewed sequence: inspect, then subtract blank.");
    await rerender(host, [{ ...source }]);
    expect((host.querySelector('[aria-label="流程说明"]') as HTMLTextAreaElement).value)
      .toBe("Reviewed sequence: inspect, then subtract blank.");
    await setValue(host, "名称 slug", "ftir-baseline");
    await setValue(host, "经验描述", "FTIR baseline handling");
    await setValue(host, "适用范围", "FTIR spectra");
    await click(host, "保存为候选");
    expect(api.createProcedure).toHaveBeenCalledWith(expect.objectContaining({
      name: "ftir-baseline", task_id: taskId, run_id: runId, artifact_id: source.artifact_id,
      instructions: "Reviewed sequence: inspect, then subtract blank.",
    }));
    expect(host.textContent).toContain("UNKNOWN");
    expect(host.textContent).toContain(created.index_name);
    expect(host.textContent).toContain("批准仅允许后续任务检索");

    await setValue(host, "审阅说明", "步骤适用于这类基线校正，使用时仍需检查空白谱。 ");
    await click(host, "批准并供后续任务检索");
    expect(approve).toHaveBeenCalledWith(created.skill_id, "步骤适用于这类基线校正，使用时仍需检查空白谱。");
    expect(host.textContent).toContain("已批准 · 可检索");
    expect(host.textContent).toContain("UNKNOWN");

    vi.spyOn(api, "disableProcedure").mockResolvedValue({ ...created, status: "DISABLED" });
    await click(host, "停用流程经验");
    expect(host.textContent).toContain("已停用");
  });

  it("does not offer uploads or other task files as procedure evidence", async () => {
    vi.spyOn(api, "procedures").mockResolvedValue({ items: [] });
    const upload = { ...source, artifact_id: "upload", run_id: null };
    const otherRun = { ...source, artifact_id: "other", run_id: `run_${"c".repeat(32)}` };
    const terminalRun = { run_id: runId, task_id: taskId, status: "SUCCEEDED", created_at: "now", request: { text: "done", strategy: "standard" } } as RunRecord;
    const host = await render([upload, otherRun], [terminalRun]);
    expect(host.textContent).toContain("本任务没有可用的运行产物");
    expect(host.textContent).not.toContain("保存为候选");
  });
});
