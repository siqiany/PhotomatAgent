import { expect, test, type Page } from "@playwright/test";

// Engineering acceptance only: deterministic model and dependency plan; the
// source and replay scripts execute as real local Python subprocesses. The
// numeric rules below check software behavior, not a materials-science claim.
type Task = { task: { task_id: string; title: string }; runs: Array<{ run_id: string; status: string; request: { text: string } }>; artifacts: Array<{ artifact_id: string; filename: string; run_id: string | null; sha256: string }> };
type Event = { run_id: string; type: string; payload: Record<string, unknown> };

async function createTask(page: Page, title: string, goal: string): Promise<string> {
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill(title);
  await page.getByLabel("研究目标").fill(goal);
  await page.getByRole("button", { name: "创建任务" }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByRole("heading", { name: title })).toBeVisible();
  const listing = await (await page.request.get("/api/tasks")).json() as { items: Array<{ task_id: string; title: string }> };
  const taskId = listing.items.find((item) => item.title === title)?.task_id;
  expect(taskId).toMatch(/^task_[0-9a-f]{32}$/);
  return taskId!;
}

async function detail(page: Page, taskId: string): Promise<Task> {
  const response = await page.request.get(`/api/tasks/${taskId}`);
  expect(response.ok(), await response.text()).toBeTruthy();
  return response.json() as Promise<Task>;
}

async function events(page: Page, taskId: string): Promise<Event[]> {
  const response = await page.request.get(`/api/__e2e__/tasks/${taskId}/events`);
  expect(response.ok(), await response.text()).toBeTruthy();
  return (await response.json() as { items: Event[] }).items;
}

async function upload(page: Page, taskId: string, name: string, content: string, mimeType = "application/json") {
  await page.getByLabel("添加任务附件").setInputFiles({ name, mimeType, buffer: Buffer.from(content) });
  await expect(page.getByRole("region", { name: "任务资料" })).toContainText(name);
  const item = (await detail(page, taskId)).artifacts.find((artifact) => artifact.filename === name);
  expect(item?.artifact_id).toBeTruthy();
  return item!;
}

test("engineering research journey persists feedback, real script execution, cross-task skill reuse, and frozen checks", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  const taskA = await createTask(page, `工程研究源任务 ${Date.now()}`, "以教学数值输入检查工作流，所有物理解释均待验证。");
  const sourceInput = await upload(page, taskA, "源数值.json", "[2,3]");
  await page.getByRole("region", { name: "任务资料" }).getByRole("button", { name: "预览前 32 KiB" }).click();
  await expect(page.getByRole("region", { name: "任务资料" })).toContainText("[2,3]");
  const image = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==", "base64");
  await page.getByLabel("添加任务附件").setInputFiles({ name: "教学像素.png", mimeType: "image/png", buffer: image });
  await expect(page.getByRole("img", { name: "资料预览：教学像素.png" })).toBeVisible();
  const downloaded = await page.request.get(`/api/artifacts/${sourceInput.artifact_id}/content`);
  expect(downloaded.ok()).toBeTruthy();
  expect(await downloaded.text()).toBe("[2,3]");

  const prepared = await page.request.post(`/api/__e2e__/tasks/${taskA}/script-source/${sourceInput.artifact_id}`);
  expect(prepared.ok(), await prepared.text()).toBeTruthy();
  expect((await prepared.json() as { kind: string }).kind).toBe("engineering-only");
  await page.getByLabel("本轮需求").fill("执行教学求和脚本，并标注材料结论仍待验证。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  await expect(page.locator(".run-status")).toHaveText("已完成", { timeout: 20_000 });
  await expect(page.getByRole("region", { name: "本轮报告" })).toContainText("不代表材料科学结论");
  const sourceRun = (await detail(page, taskA)).runs.at(-1)!;
  expect(sourceRun.request.text).toContain("执行教学求和脚本");
  const sourceEvents = await events(page, taskA);
  expect(sourceEvents.some((event) => event.run_id === sourceRun.run_id && event.type === "tool_completed" && JSON.stringify(event.payload).includes("script_run"))).toBeTruthy();
  expect(sourceEvents.some((event) => event.run_id === sourceRun.run_id && event.type === "run_completed" && "cost" in event.payload)).toBeTruthy();
  const sourceResponse = await page.request.get("/api/skills/sources");
  const source = (await sourceResponse.json() as { items: Array<{ task_id: string; run_id: string; execution_id: string; artifacts: string[] }> }).items.find((item) => item.run_id === sourceRun.run_id);
  expect(source?.task_id).toBe(taskA);
  expect(source?.artifacts).toContain("result.json");

  const registered = await page.request.post(`/api/__e2e__/tasks/${taskA}/runs/${sourceRun.run_id}/artifact`);
  expect(registered.ok(), await registered.text()).toBeTruthy();
  const sourceArtifactId = (await registered.json() as { artifact_id: string }).artifact_id;
  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByTestId("active-run-id")).toHaveText(sourceRun.run_id);
  await expect(page.getByRole("region", { name: "本轮产物" })).toContainText("SAM 演示指标.md");
  await page.getByRole("button", { name: "继续改进" }).click();
  await expect(page.getByLabel("基于哪一轮结果改进")).toHaveValue(sourceRun.run_id);
  await page.getByRole("checkbox", { name: "SAM 演示指标.md" }).check();
  await page.getByLabel("对这轮结果的意见").fill("请确认目标边界，并补充可核验的一手来源。");
  const feedbackResponse = page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname === `/api/tasks/${taskA}/feedback`);
  await page.getByRole("button", { name: "分析意见并生成计划" }).click();
  const sourcePlan = await (await feedbackResponse).json() as { plan_id: string; task_id: string; source_run_id: string; source_digest: string; source_artifacts: Array<{ artifact_id: string }> };
  expect(sourcePlan.task_id).toBe(taskA);
  expect(sourcePlan.source_run_id).toBe(sourceRun.run_id);
  expect(sourcePlan.source_digest).toMatch(/^[0-9a-f]{64}$/);
  expect(sourcePlan.source_artifacts.map((artifact) => artifact.artifact_id)).toEqual([sourceArtifactId]);
  await expect(page.getByRole("region", { name: "修订计划" })).toContainText("需要补充目标说明");
  await page.getByLabel("补充目标说明").fill("仅演示数值程序检查，不提出已验证的材料结论。");
  await page.getByRole("button", { name: "用补充说明重新生成计划" }).click();
  await expect(page.getByRole("region", { name: "修订计划" })).toContainText("UNKNOWN");
  const durablePlan = await (await page.request.get(`/api/tasks/${taskA}/revisions/${sourcePlan.plan_id}`)).json() as { source_run_id: string; issues: unknown[] };
  expect(durablePlan.source_run_id).toBe(sourceRun.run_id);
  expect(durablePlan.issues.length).toBeGreaterThan(0);
  await page.getByRole("button", { name: "开始补充工作" }).click();
  await expect(page.getByRole("region", { name: "新旧结果对照" })).toContainText("待核查（UNKNOWN）", { timeout: 20_000 });
  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByRole("region", { name: "新旧结果对照" })).toContainText("待核查（UNKNOWN）");
  await expect(page.getByRole("checkbox", { name: "SAM 演示指标.md" })).toBeChecked();

  const skillPanel = page.getByRole("region", { name: "技能复用" });
  await skillPanel.getByText("从本任务执行记录创建候选").click();
  await expect(page.getByLabel("来源执行")).toHaveValue(source!.execution_id);
  await page.getByLabel("技能名称").fill("numeric-bounds");
  await page.getByLabel("方法说明").fill("教学数组求和，数值上界为十。");
  await page.getByLabel("适用范围").fill("教学数值数组");
  await page.getByLabel("独立规则检查").selectOption("json_numeric_bounds");
  await page.getByLabel("JSON 字段").fill("sum");
  await page.getByLabel("允许最小值").fill("0");
  await page.getByLabel("允许最大值").fill("10");
  await page.getByRole("button", { name: "保存候选" }).click();
  await expect(skillPanel).toContainText("源检查");
  let skills = await (await page.request.get("/api/skills")).json() as { items: Array<{ skill_id: string; source: { task_id: string; run_id: string }; source_evidence: { status: string }; status: string; uses: Array<{ checker: { status: string }; input_sha256: Record<string, string>; task_id: string }> }> };
  const skillId = skills.items.find((item) => item.source?.run_id === sourceRun.run_id)?.skill_id;
  expect(skillId).toMatch(/^skill_[0-9a-f]{32}$/);
  expect(skills.items.find((item) => item.skill_id === skillId)?.source_evidence.status).toBe("PASS");
  await skillPanel.getByRole("button", { name: "回放到当前任务" }).click();
  await expect.poll(async () => (await (await page.request.get("/api/skills")).json() as typeof skills).items.find((item) => item.skill_id === skillId)?.status).toBe("TRIALABLE");
  skills = await (await page.request.get("/api/skills")).json();
  const sourceDigest = skills.items.find((item) => item.skill_id === skillId)!.uses.at(-1)!.input_sha256.data;

  const taskB = await createTask(page, `工程研究复用任务 ${Date.now()}`, "用不同教学输入复用相同程序；物理结论未知。");
  const failingInput = await upload(page, taskB, "超界数值.json", "[4,8]");
  const passingInput = await upload(page, taskB, "新数值.json", "[4,5]");
  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  const targetSkill = page.getByRole("region", { name: "技能复用" });
  await targetSkill.getByRole("button", { name: /numeric-bounds/ }).click();
  await page.getByLabel("技能输入 data").selectOption(failingInput.artifact_id);
  await targetSkill.getByRole("button", { name: "回放到当前任务" }).click();
  await expect.poll(async () => (await (await page.request.get("/api/skills")).json() as typeof skills).items.find((item) => item.skill_id === skillId)?.uses.at(-1)?.checker.status).toBe("FAIL");
  await expect(page.locator(".run-status")).toHaveText(/部分完成|已完成/);
  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await targetSkill.getByRole("button", { name: /numeric-bounds/ }).click();
  await page.getByLabel("技能输入 data").selectOption(passingInput.artifact_id);
  await targetSkill.getByRole("button", { name: "回放到当前任务" }).click();
  await expect.poll(async () => (await (await page.request.get("/api/skills")).json() as typeof skills).items.find((item) => item.skill_id === skillId)?.status).toBe("REUSED");
  const finalSkill = (await (await page.request.get("/api/skills")).json() as typeof skills).items.find((item) => item.skill_id === skillId)!;
  expect(finalSkill.uses.at(-1)?.task_id).toBe(taskB);
  expect(finalSkill.uses.at(-1)?.checker.status).toBe("PASS");
  expect(finalSkill.uses.at(-1)?.input_sha256.data).not.toBe(sourceDigest);
  const target = await detail(page, taskB);
  const resultArtifact = target.artifacts.find((artifact) => artifact.run_id === target.runs.at(-1)?.run_id && artifact.filename === "result.json");
  expect(resultArtifact).toBeTruthy();
  expect(await (await page.request.get(`/api/artifacts/${resultArtifact!.artifact_id}/content`)).json()).toEqual({ sum: 9 });
  expect((await events(page, taskB)).some((event) => event.type === "skill_use" && (event.payload.checker as { status?: string })?.status === "PASS")).toBeTruthy();

  for (const [name, content] of [
    ["input_a.json", '{"case":"a"}'], ["reference_a.json", '{"required":"present","peak":{"value":1,"unit":"nm"}}'],
    ["input_b.json", '{"case":"b"}'], ["reference_b.json", '{"required":"present","peak":{"value":1,"unit":"nm"}}'],
  ]) await upload(page, taskB, name, content);
  await page.getByLabel("案例 1 输入 JSON").selectOption({ label: "input_a.json" });
  await page.getByLabel("案例 1 参考 JSON").selectOption({ label: "reference_a.json" });
  await page.getByLabel("案例 2 输入 JSON").selectOption({ label: "input_b.json" });
  await page.getByLabel("案例 2 参考 JSON").selectOption({ label: "reference_b.json" });
  const evaluation = page.getByRole("region", { name: "冻结对照评测" });
  await evaluation.getByRole("button", { name: "开始冻结对照评测" }).click();
  await expect(evaluation).toContainText("评测已完成", { timeout: 20_000 });
  await expect(evaluation).toContainText("演示模拟结果：仅为工程测试证据，不是科研验证。");
  const evaluations = await (await page.request.get(`/api/tasks/${taskB}/evaluations`)).json() as Array<{ status: string; run_id: string; report?: { evidence_class: string; results: Array<{ verdict: string }>; total_cost: { model_calls: number; tokens: number } } }>;
  expect(evaluations[0]?.status).toBe("SUCCEEDED");
  expect(evaluations[0]?.report?.evidence_class).toBe("fake_engineering_only");
  expect(evaluations[0]?.report?.results.map((result) => result.verdict)).toEqual(["PASS", "PASS"]);
  expect(evaluations[0]?.report?.total_cost.model_calls).toBeGreaterThan(0);
  expect((await events(page, taskB)).some((event) => event.run_id === evaluations[0]?.run_id && event.type === "evaluation_completed" && "cost" in event.payload)).toBeTruthy();
  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByRole("region", { name: "冻结对照评测" })).toContainText("评测已完成");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("tab", { name: "详情" }).click();
  await expect(page.getByRole("region", { name: "本轮产物" })).toBeVisible();
  await page.getByRole("tab", { name: "工作区" }).click();
  await expect(page.getByLabel("本轮需求")).toBeVisible();
});

test("a cancelled engineering run keeps its cost record and can be followed by another run", async ({ page }) => {
  await page.goto("/");
  const taskId = await createTask(page, `工程取消任务 ${Date.now()}`, "检查取消后的任务恢复。");
  await page.getByLabel("本轮需求").fill("开始一轮长时间分析。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  await expect(page.locator(".run-status")).toHaveText("正在分析");
  const cancelledRun = await page.getByTestId("active-run-id").textContent();
  await page.getByRole("button", { name: "请求取消" }).click();
  await expect(page.locator(".run-status")).toHaveText("已取消", { timeout: 10_000 });
  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  expect((await detail(page, taskId)).runs.find((run) => run.run_id === cancelledRun)?.status).toBe("CANCELLED");
  expect((await events(page, taskId)).some((event) => event.run_id === cancelledRun && event.type === "run_cancelled" && "cost" in event.payload)).toBeTruthy();
  await page.getByLabel("本轮需求").fill("取消后重试：仅输出待验证的演示说明。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  await expect(page.locator(".run-status")).toHaveText("已完成", { timeout: 10_000 });
  expect((await detail(page, taskId)).runs).toHaveLength(2);
});
