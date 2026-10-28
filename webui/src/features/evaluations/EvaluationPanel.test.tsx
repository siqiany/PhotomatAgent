import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api, type Artifact, type EvaluationCapabilities, type EvaluationStatus } from "../../api";
import { EvaluationPanel } from "./EvaluationPanel";

const taskId = "task_evaluation_test";
const artifacts: Artifact[] = ["input_a", "reference_a", "input_b", "reference_b"].map((artifact_id, index) => ({
  artifact_id,
  task_id: taskId,
  run_id: null,
  filename: `${artifact_id}.json`,
  media_type: "application/json",
  size_bytes: index + 20,
  sha256: "a".repeat(64),
}));

const capabilities: EvaluationCapabilities = {
  conditions: [
    { id: "STATIC", available: true, reason: null },
    { id: "FEEDBACK_CHAIN", available: true, reason: null },
    { id: "RANDOM_CANDIDATE", available: true, reason: null },
    { id: "DIRECTED_MUTATION", available: true, reason: null },
    { id: "FULL_QD_SKILLS", available: false, reason: "冻结依赖尚未接入。" },
  ],
  checker_ids: ["json_exact_v1", "json_numeric_tolerance_v1"],
  frozen_skill_sources: ["published_qd"],
  frozen_skill_versions: [],
};

const completed: EvaluationStatus = {
  evaluation_id: "eval_persisted_1234567890",
  task_id: taskId,
  run_id: "run_eval_1",
  status: "SUCCEEDED",
  manifest_sha256: "b".repeat(64),
  created_at: "2026-10-04T00:00:00Z",
  updated_at: "2026-10-04T00:01:00Z",
  error_code: null,
  report: {
    evaluation_id: "eval_persisted_1234567890",
    manifest_sha256: "b".repeat(64),
    results: [{
      case_id: "case_1",
      task_group_id: "heldout_a",
      condition: "STATIC",
      verdict: "PASS",
      cost: { model_calls: 1, tokens: 17 },
      provenance: "runtime",
      diagnostics: [],
      checker: {
        checker_id: "json_numeric_tolerance_v1",
        checker_version: "1",
        verdict: "PASS",
        reason_codes: ["NUMERIC_TOLERANCE_WITHIN_LIMIT"],
        evidence_refs: ["sha256:output", "sha256:reference"],
        checked: [{
          name: "metric",
          observed: 1.005,
          reference: 1,
          units: "nm",
          absolute_error: 0.005,
          allowed_error: 0.01,
          verdict: "PASS",
        }],
      },
    }],
    condition_success: { STATIC: 1 },
    condition_metrics: { STATIC: { cases: 1, unknown_rate: 0, budget_exhaustion_rate: 0 } },
    condition_cost: { STATIC: { model_calls: 1, tokens: 17 } },
    total_cost: { model_calls: 1, tokens: 17 },
    evidence_class: "runtime_observation",
  },
};

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let mounted: Array<{ root: Root; host: HTMLElement }> = [];

afterEach(async () => {
  await act(async () => { for (const item of mounted) item.root.unmount(); });
  for (const item of mounted) item.host.remove();
  mounted = [];
  vi.restoreAllMocks();
  vi.useRealTimers();
});

async function render(records: EvaluationStatus[] = [], available = capabilities) {
  vi.spyOn(api, "evaluationCapabilities").mockResolvedValue(available);
  vi.spyOn(api, "evaluationsForTask").mockResolvedValue(records);
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  mounted.push({ root, host });
  await act(async () => { root.render(<EvaluationPanel taskId={taskId} artifacts={artifacts} />); });
  return host;
}

async function change(element: HTMLInputElement | HTMLSelectElement | null, value: string) {
  if (!element) throw new Error("Expected evaluation form control was not found.");
  const prototype = element instanceof HTMLInputElement ? HTMLInputElement.prototype : HTMLSelectElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value")?.set?.call(element, value);
  await act(async () => { element.dispatchEvent(new Event("input", { bubbles: true })); });
  await act(async () => { element.dispatchEvent(new Event("change", { bubbles: true })); });
}

async function click(element: Element | null) {
  if (!element) throw new Error("Expected evaluation button was not found.");
  await act(async () => { element.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
}

describe("EvaluationPanel", () => {
  it("freezes the explicitly selected published skill when the backend is ready", async () => {
    const version = { source_type: "published_qd" as const, skill_id: "diagnostic", bundle_sha256: "c".repeat(64) };
    const host = await render([], { ...capabilities,
      conditions: capabilities.conditions.map((item) => item.id === "FULL_QD_SKILLS" ? { ...item, available: true, reason: null } : item),
      frozen_skill_versions: [version],
    });
    const start = vi.spyOn(api, "startEvaluation").mockResolvedValue({ evaluation_id: "eval_full", task_id: taskId, run_id: "run_full", status: "QUEUED" });
    for (const [label, value] of [["案例 1 输入 JSON", "input_a"], ["案例 1 参考 JSON", "reference_a"], ["案例 2 输入 JSON", "input_b"], ["案例 2 参考 JSON", "reference_b"]]) {
      await change(host.querySelector<HTMLSelectElement>(`[aria-label='${label}']`), value);
    }
    const full = [...host.querySelectorAll<HTMLInputElement>(".evaluation-conditions input")].find((item) => item.parentElement?.textContent?.includes("完整 QD 技能搜索"));
    await click(full ?? null);
    await click(host.querySelector(".evaluation-start"));
    expect(start).not.toHaveBeenCalled();
    expect(host.textContent).toContain("请选择至少一个冻结技能版本");
    await click(host.querySelector("[aria-label='冻结技能 diagnostic']"));
    await click(host.querySelector(".evaluation-start"));
    expect(start).toHaveBeenCalledWith(expect.objectContaining({ conditions: ["STATIC", "FULL_QD_SKILLS"], frozen_qd_skills: [version] }));
    expect(host.textContent).not.toContain("当前完整技能条件仍不可用");
  });
  it("submits selected task artifacts and a frozen checker contract to the backend", async () => {
    const host = await render();
    const start = vi.spyOn(api, "startEvaluation").mockResolvedValue({
      evaluation_id: "eval_new", task_id: taskId, run_id: "run_eval", status: "QUEUED",
    });
    vi.spyOn(api, "evaluationsForTask").mockResolvedValueOnce([]).mockResolvedValue([completed]);

    await change(host.querySelector<HTMLSelectElement>("[aria-label='案例 1 输入 JSON']"), "input_a");
    await change(host.querySelector<HTMLSelectElement>("[aria-label='案例 1 参考 JSON']"), "reference_a");
    await change(host.querySelector<HTMLSelectElement>("[aria-label='案例 2 输入 JSON']"), "input_b");
    await change(host.querySelector<HTMLSelectElement>("[aria-label='案例 2 参考 JSON']"), "reference_b");
    await click(host.querySelector(".evaluation-start"));

    expect(start).toHaveBeenCalledWith(expect.objectContaining({
      task_id: taskId,
      conditions: ["STATIC"],
      cases: [
        expect.objectContaining({ input_artifact_id: "input_a", reference_artifact_id: "reference_a", task_group_id: "heldout_a" }),
        expect.objectContaining({ input_artifact_id: "input_b", reference_artifact_id: "reference_b", task_group_id: "heldout_b" }),
      ],
      checker: expect.objectContaining({
        checker_id: "json_numeric_tolerance_v1",
        numeric_fields: { metric: expect.objectContaining({ value_path: "/peak/value", unit_path: "/peak/unit", units: "nm", absolute_tolerance: 0.01 }) },
      }),
      frozen_qd_skills: [],
    }));
    expect(host.textContent).toContain("检查规则与预算");
  });

  it("shows the persisted outcome and leaves unsupported frozen-skill comparison disabled", async () => {
    const host = await render([completed]);

    expect(host.textContent).toContain("运行观测：结论仅针对本次冻结 JSON 检查合同");
    expect(host.textContent).toContain("通过率 100%");
    expect(host.textContent).toContain("误差 0.005 / 容许 0.01");
    const fullSkills = [...host.querySelectorAll<HTMLInputElement>(".evaluation-conditions input")]
      .find((element) => element.parentElement?.textContent?.includes("完整 QD 技能搜索"));
    expect(fullSkills?.disabled).toBe(true);
    expect(host.textContent).toContain("冻结依赖尚未接入。");
  });
});
