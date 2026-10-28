import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api, type Artifact, type LocalSkill, type SkillExecutionSource } from "../../api";
import { SkillPanel } from "./SkillPanel";

const taskA = "task_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const taskB = "task_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const executionId = `exec_${"a".repeat(32)}`;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function source(taskId = taskA): SkillExecutionSource {
  return {
    execution_id: executionId,
    task_id: taskId,
    run_id: "run_1",
    script_path: `tmp/workbench/${taskId}/run_1/sum.py`,
    input_keys: ["data"],
    artifacts: ["result.json"],
    state: "COMPLETED",
  };
}

function skill(overrides: Partial<LocalSkill> = {}): LocalSkill {
  return {
    skill_id: `skill_${"b".repeat(32)}`,
    name: "array-sum",
    description: "对数值数组求和",
    scope: "数值教学示例",
    source: { kind: "ordinary_run", task_id: taskA, run_id: "run_1", execution_id: executionId },
    request: { input_refs: { data: "tmp/source.json" } },
    status: "CANDIDATE",
    source_evidence: { status: "PASS", checker_id: "json_numeric_sum" },
    uses: [],
    ...overrides,
  };
}

let mounted: Array<{ root: Root; host: HTMLElement }> = [];

afterEach(async () => {
  await act(async () => {
    for (const item of mounted) item.root.unmount();
  });
  for (const item of mounted) item.host.remove();
  mounted = [];
  vi.restoreAllMocks();
  vi.useRealTimers();
});

async function render(taskId: string, artifacts: Artifact[] = [], onRunQueued = vi.fn(), disabled = false) {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  mounted.push({ root, host });
  await act(async () => {
    root.render(<SkillPanel taskId={taskId} artifacts={artifacts} disabled={disabled} onRunQueued={onRunQueued} />);
  });
  return { host, onRunQueued };
}

async function click(element: Element | null) {
  if (!element) throw new Error("Expected button was not found.");
  await act(async () => { element.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
}

async function change(element: HTMLInputElement | HTMLSelectElement | null, value: string) {
  if (!element) throw new Error("Expected form control was not found.");
  const prototype = element instanceof HTMLInputElement ? HTMLInputElement.prototype : HTMLSelectElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value")?.set?.call(element, value);
  await act(async () => { element.dispatchEvent(new Event("input", { bubbles: true })); });
  await act(async () => { element.dispatchEvent(new Event("change", { bubbles: true })); });
}

describe("SkillPanel", () => {
  it("keeps missing source evidence UNKNOWN and prevents replay", async () => {
    vi.spyOn(api, "skills").mockResolvedValue({ items: [skill({ source_evidence: { status: "UNKNOWN", gap: "CHECKER_NOT_SELECTED" } })] });
    vi.spyOn(api, "skillSources").mockResolvedValue({ items: [] });
    const { host } = await render(taskB);

    await click(host.querySelector("[aria-expanded='false']"));

    expect(host.textContent).toContain("检查未知");
    expect(host.querySelector(".skill-checker")?.getAttribute("aria-label")).toBe("UNKNOWN");
    expect(host.querySelector("button[aria-expanded='true']")).not.toBeNull();
    expect(host.textContent).not.toContain("回放到当前任务");
  });

  it("binds a different task attachment and queues replay with the fixed scope", async () => {
    const candidate = skill();
    vi.spyOn(api, "skills").mockResolvedValue({ items: [candidate] });
    vi.spyOn(api, "skillSources").mockResolvedValue({ items: [source()] });
    const replay = vi.spyOn(api, "replaySkill").mockResolvedValue({ task_id: taskB, run_id: "run_replay" });
    const attachment = { artifact_id: "artifact_input", task_id: taskB, run_id: null, filename: "new-input.json", media_type: "application/json", size_bytes: 32, sha256: "digest" };
    const { host, onRunQueued } = await render(taskB, [attachment]);

    await click(host.querySelector("[aria-expanded='false']"));
    await change(host.querySelector<HTMLSelectElement>("[aria-label='技能输入 data']"), attachment.artifact_id);
    await click([...host.querySelectorAll("button")].find((button) => button.textContent?.includes("回放到当前任务")) ?? null);

    expect(replay).toHaveBeenCalledWith(candidate.skill_id, {
      task_id: taskB,
      scope: "数值教学示例",
      input_refs: { data: "artifact:artifact_input" },
    });
    expect(onRunQueued).toHaveBeenCalledWith({ task_id: taskB, run_id: "run_replay" });
  });

  it("freezes the chosen numeric bounds in a candidate checker", async () => {
    vi.spyOn(api, "skills").mockResolvedValue({ items: [] });
    vi.spyOn(api, "skillSources").mockResolvedValue({ items: [{ ...source(taskB), kind: "qd_trial", run_id: undefined, trial_id: "trial_1" }] });
    const create = vi.spyOn(api, "createSkill").mockResolvedValue(skill({ source: { kind: "qd_trial", task_id: taskB, trial_id: "trial_1", execution_id: executionId } }));
    const { host } = await render(taskB);

    await click(host.querySelector(".skill-create-details summary"));
    await change(host.querySelector<HTMLSelectElement>("[aria-label='独立规则检查']"), "json_numeric_bounds");
    await change(host.querySelector<HTMLInputElement>("[aria-label='技能名称']"), "bounded-value");
    await change(host.querySelector<HTMLInputElement>("[aria-label='方法说明']"), "提取并检查一个数值");
    await change(host.querySelector<HTMLInputElement>("[aria-label='适用范围']"), "教学数据");
    await change(host.querySelector<HTMLInputElement>("[aria-label='JSON 字段']"), "score");
    await change(host.querySelector<HTMLInputElement>("[aria-label='允许最小值']"), "0.2");
    await change(host.querySelector<HTMLInputElement>("[aria-label='允许最大值']"), "0.8");
    await change(host.querySelector<HTMLInputElement>("[aria-label='检查单位']"), "eV");
    const form = host.querySelector(".skill-create-form");
    if (!form) throw new Error("Candidate form was not rendered.");
    await act(async () => { form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); });

    expect(create).toHaveBeenCalledWith(expect.objectContaining({
      source: { kind: "qd_trial", task_id: taskB, trial_id: "trial_1", execution_id: executionId },
      checker: { checker_id: "json_numeric_bounds", output_path: "result.json", field: "score", minimum: 0.2, maximum: 0.8, units: "eV" },
    }));
    expect(host.textContent).toContain("范围检查不验证科学假设");
  });

  it("refreshes persisted skill-use status while the associated run is active", async () => {
    vi.useFakeTimers();
    const candidate = skill();
    vi.spyOn(api, "skills")
      .mockResolvedValueOnce({ items: [candidate] })
      .mockResolvedValue({ items: [skill({
        status: "TRIALABLE",
        uses: [{ use_id: "use_1", task_id: taskB, code: "OK", checker: { status: "PASS" }, artifacts: ["user_output/task_b/result.json"] }],
      })] });
    vi.spyOn(api, "skillSources").mockResolvedValue({ items: [] });
    const { host } = await render(taskB, [], vi.fn(), true);

    await act(async () => { await vi.advanceTimersByTimeAsync(2500); });

    expect(host.textContent).toContain("本地可试用");
    await click(host.querySelector("[aria-expanded='false']"));
    expect(host.textContent).toContain("user_output/task_b/result.json");
  });
});
