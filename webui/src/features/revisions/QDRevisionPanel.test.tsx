import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api, type SkillExecutionSource } from "../../api";
import { QDRevisionPanel } from "./QDRevisionPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const taskId = "task_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const failedSource: SkillExecutionSource = {
  execution_id: `exec_${"a".repeat(32)}`,
  task_id: taskId,
  kind: "ordinary_run",
  run_id: "run_failed_baseline",
  script_path: "tmp/task/calc.py",
  argv: ["--scale", "1.0"],
  input_keys: ["data"],
  artifacts: ["result.json"],
  state: "COMPLETED",
};
let mounted: Array<{ root: Root; host: HTMLElement }> = [];

afterEach(async () => {
  await act(async () => { for (const item of mounted) item.root.unmount(); });
  for (const item of mounted) item.host.remove();
  mounted = [];
  vi.restoreAllMocks();
});

async function render() {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  mounted.push({ root, host });
  await act(async () => { root.render(<QDRevisionPanel taskId={taskId} disabled={false} busy={false} events={[]} onRun={vi.fn()} />); });
  await act(async () => { await Promise.resolve(); });
  return host;
}

describe("QDRevisionPanel", () => {
  it("keeps a completed baseline eligible even when its selected checker may fail", async () => {
    vi.spyOn(api, "skillSources").mockResolvedValue({ items: [failedSource] });
    const host = await render();

    expect(host.textContent).toContain("calc.py");
    expect(host.textContent).toContain("--scale 1.0");
    expect(host.textContent).not.toContain("QD 暂不可用");
    expect(host.textContent).toContain("JSON 数值范围");
    expect(host.textContent).toContain("JSON 数值求和");
  });

  it("leaves QD unavailable when this task has no representable source", async () => {
    vi.spyOn(api, "skillSources").mockResolvedValue({ items: [] });
    const host = await render();

    expect(host.textContent).toContain("本任务没有带有基线参数的真实脚本执行来源");
    expect(host.querySelector("button" )).toBeNull();
  });

  it("does not offer a process-failed script as a QD source", async () => {
    vi.spyOn(api, "skillSources").mockResolvedValue({ items: [{ ...failedSource, state: "FAILED" }] });
    const host = await render();

    expect(host.textContent).toContain("本任务没有带有基线参数的真实脚本执行来源");
    expect(host.textContent).not.toContain("run_failed_baseline");
  });
});
