import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./api";

describe("workbench API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends a scoped one-time approval decision", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ request_id: "approval-1", decision: "denied" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await api.resolveApproval("task-1", "run-1", "approval-1", false);
    expect(fetchMock).toHaveBeenCalledWith("/api/tasks/task-1/runs/run-1/approvals/approval-1", expect.objectContaining({
      method: "POST", body: JSON.stringify({ approved: false }),
    }));
  });

  it("sends a typed research request with an idempotency key", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ task_id: "task_1", run_id: "run_1" }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("crypto", { ...crypto, randomUUID: () => "request-1" });

    const result = await api.startRun("task_1", { text: "比较吸收边", strategy: "standard" });

    expect(result).toEqual({ task_id: "task_1", run_id: "run_1" });
    expect(fetchMock).toHaveBeenCalledWith("/api/tasks/task_1/runs", expect.objectContaining({
      method: "POST",
      headers: expect.objectContaining({ "Idempotency-Key": "request-1" }),
      body: JSON.stringify({ text: "比较吸收边", strategy: "standard" }),
    }));
  });

  it("shows the backend next action when the request is rejected", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        error: {
          code: "STRATEGY_UNAVAILABLE",
          message: "当前阶段尚未接入 QD 策略。",
          next_action: "切换为普通科研运行。",
        },
      }), { status: 409, headers: { "Content-Type": "application/json" } }),
    ));

    await expect(api.startRun("task_1", { text: "继续", strategy: "qd" })).rejects.toThrow(
      "当前阶段尚未接入 QD 策略。 切换为普通科研运行。",
    );
  });

  it("lists skill candidates and real execution sources", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ items: [{ skill_id: "skill_1", status: "CANDIDATE", uses: [] }] }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ items: [{ execution_id: "exec_1", task_id: "task_1", run_id: "run_1", input_keys: ["data"], artifacts: ["result.json"] }] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.skills()).resolves.toMatchObject({ items: [{ skill_id: "skill_1", status: "CANDIDATE" }] });
    await expect(api.skillSources()).resolves.toMatchObject({ items: [{ execution_id: "exec_1", input_keys: ["data"] }] });
    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/skills", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/skills/sources", undefined);
  });

  it("starts a QD revision with only the explicitly frozen source, checker, and token choices", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ task_id: "task_1", run_id: "run_qd" }), { status: 202 },
    ));
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("crypto", { ...crypto, randomUUID: () => "revision-qd-key" });
    const request = {
      strategy: "qd" as const,
      budget: { max_model_calls: 5, max_tokens: 2000, wall_seconds: 60 },
      qd: {
        execution_id: "exec_abc",
        checker: { checker_id: "json_numeric_bounds" as const, output_path: "result.json", field: "value", minimum: 2, maximum: 4, units: "eV" },
        search_mode: "directed" as const,
        argv_choices: [["--scale", "1.0"], ["--scale", "2.0"]],
        max_trials: 6,
        seed: 7,
      },
    };

    await api.executeRevision("task_1", "plan_1", request);

    expect(fetchMock).toHaveBeenCalledWith("/api/tasks/task_1/revisions/plan_1/runs", expect.objectContaining({
      method: "POST",
      headers: expect.objectContaining({ "Idempotency-Key": "revision-qd-key" }),
      body: JSON.stringify(request),
    }));
  });

  it("sends immutable checker contracts for skill creation and replay", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ skill_id: "skill_1", status: "CANDIDATE" }), { status: 201 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ task_id: "task_2", run_id: "run_2" }), { status: 202 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ skill_id: "skill_1", status: "DISABLED" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("crypto", { ...crypto, randomUUID: () => "skill-replay-key" });
    const create = {
      name: "sum-array",
      description: "对数组求和",
      scope: "数值教学示例",
      source: { kind: "ordinary_run" as const, task_id: "task_1", run_id: "run_1", execution_id: "exec_1" },
      checker: { checker_id: "json_numeric_sum" as const, input_key: "data", output_path: "result.json", field: "sum", tolerance: 1e-8 },
    };

    await api.createSkill(create);
    await api.replaySkill("skill_1", { task_id: "task_2", scope: "数值教学示例", input_refs: { data: "artifact:artifact_1" } });
    await api.disableSkill("skill_1");

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/skills", expect.objectContaining({ method: "POST", body: JSON.stringify(create) }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/skills/skill_1/replay", expect.objectContaining({
      method: "POST",
      headers: expect.objectContaining({ "Content-Type": "application/json", "Idempotency-Key": "skill-replay-key" }),
      body: JSON.stringify({ task_id: "task_2", scope: "数值教学示例", input_refs: { data: "artifact:artifact_1" } }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, "/api/skills/skill_1/disable", expect.objectContaining({ method: "POST" }));
  });
});
