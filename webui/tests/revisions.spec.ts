import { expect, test } from "@playwright/test";

test("a reviewer can clarify goals, inspect UNKNOWN checks, and continue in the same task", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "导入历史任务" }).click();
  await page.getByRole("button", { name: /e2e-archived-session/ }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  const source = page.getByRole("region", { name: "导入历史来源" });
  await expect(source).toContainText("只读来源");
  await expect(source).toContainText("SAM");
  await page.getByLabel("本轮需求").fill("先整理一份候选材料简报，并注明哪些内容尚未验证。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  const report = page.getByRole("region", { name: "本轮报告" });
  await expect(report).toContainText("初稿演示：吸收边和载流子输运仍需实验验证。");
  await expect(page.locator(".run-status")).toHaveText("已完成");

  await page.setViewportSize({ width: 390, height: 844 });
  const taskTitle = await page.locator(".breadcrumbs h1").innerText();
  const tasks = await (await page.request.get("/api/tasks")).json() as { items: Array<{ task_id: string; title: string }> };
  const taskId = tasks.items.find((task) => task.title === taskTitle)?.task_id;
  const sourceRunId = await page.getByTestId("active-run-id").textContent();
  expect(taskId).toBeTruthy();
  expect(sourceRunId).toMatch(/^run_/);
  const artifactResponse = await page.request.post(`/api/__e2e__/tasks/${taskId}/runs/${sourceRunId}/artifact`);
  expect(artifactResponse.ok(), await artifactResponse.text()).toBeTruthy();
  await page.reload();
  await page.getByRole("tab", { name: "详情" }).click();
  await expect(page.getByRole("region", { name: "本轮产物" })).toContainText("SAM 演示指标.md");
  await page.getByRole("tab", { name: "工作区" }).click();
  await page.getByRole("button", { name: "继续改进" }).click();
  await expect(page.getByLabel("基于哪一轮结果改进")).toHaveValue(/run_/);
  const sourceArtifact = page.getByRole("checkbox", { name: "SAM 演示指标.md" });
  await sourceArtifact.check();
  await page.getByLabel("对这轮结果的意见").fill("请确认目标边界，并补充可核验的一手来源。");
  await page.getByRole("button", { name: "分析意见并生成计划" }).click();
  await expect(sourceArtifact).toBeChecked();
  await expect(page.getByRole("region", { name: "修订计划" })).toContainText("需要补充目标说明");
  await page.getByLabel("补充目标说明").fill("先整理候选，不要求结论已由实验验证。");
  await page.getByRole("button", { name: "用补充说明重新生成计划" }).click();

  const plan = page.getByRole("region", { name: "修订计划" });
  await expect(plan).toContainText("UNKNOWN");
  await expect(plan).toContainText("证据不足");
  await expect(plan).toContainText("补充两篇可核验原始研究");
  await expect(plan).not.toContainText(/plan_[a-f0-9]+/);
  await page.getByRole("button", { name: "开始补充工作" }).click();

  const comparison = page.getByRole("region", { name: "新旧结果对照" });
  await expect(comparison).toContainText("修订前");
  await expect(comparison).toContainText("修订后");
  await expect(comparison).toContainText("初稿演示：吸收边和载流子输运仍需实验验证。");
  await expect(comparison).toContainText("修订演示：新增来源仍待核对");
  await expect(comparison).toContainText("演示结果（fake），不代表研究结果");
  await expect(comparison).toContainText("待核查（UNKNOWN）");
  await comparison.getByLabel("回应状态：关键候选仍缺可核验的一手来源").selectOption("EVIDENCE_ADDED");
  await expect(comparison).toContainText("人工判断");
  await expect(page.evaluate(() => Object.entries(localStorage).some(([key, value]) =>
    key.startsWith("photomatagent:revision-review:") && value.includes("EVIDENCE_ADDED"),
  ))).toBeTruthy();
  let releasePlanFetch: (() => void) | undefined;
  await page.route("**/api/tasks/*/revisions/*", async (route) => {
    if (route.request().method() === "GET") {
      await new Promise<void>((resolve) => { releasePlanFetch = resolve; });
    }
    await route.continue();
  });
  await page.reload();
  await expect(page.getByLabel("对这轮结果的意见")).toBeDisabled();
  releasePlanFetch?.();
  const restoredComparison = page.getByRole("region", { name: "新旧结果对照" });
  await expect(restoredComparison).toBeVisible();
  await expect(restoredComparison.getByLabel("回应状态：关键候选仍缺可核验的一手来源")).toHaveValue("EVIDENCE_ADDED");
  await expect(restoredComparison.locator(".revision-plan-state")).toHaveText("已完成");
  const boundSourceRunId = await page.getByLabel("基于哪一轮结果改进").inputValue();
  await expect(page.getByRole("checkbox", { name: "SAM 演示指标.md" })).toBeChecked();
  await page.unroute("**/api/tasks/*/revisions/*");

  await page.getByLabel("本轮需求").fill("开始一轮长时间分析。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  await expect(page.locator(".run-status")).toHaveText("正在分析");
  await expect(restoredComparison.locator(".revision-plan-state")).toHaveText("已完成");
  await expect(restoredComparison.getByLabel("回应状态：关键候选仍缺可核验的一手来源")).toBeEnabled();
  await expect(page.locator(".run-status")).toHaveText("已完成", { timeout: 5_000 });

  let failRestoreOnce = true;
  await page.route("**/api/tasks/*/revisions/*", async (route) => {
    if (route.request().method() === "GET" && failRestoreOnce) {
      failRestoreOnce = false;
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { message: "临时恢复故障" } }) });
      return;
    }
    await route.continue();
  });
  await page.reload();
  await expect(page.getByRole("alert")).toContainText("重试会保留恢复信息");
  await expect(page.getByLabel("对这轮结果的意见")).toBeDisabled();
  await expect(page.getByRole("button", { name: "重试恢复" })).toBeVisible();
  await expect(page.evaluate(() => Object.keys(sessionStorage).some((key) => key.startsWith("photomatagent:revision-context:")))).toBeTruthy();
  await page.getByRole("button", { name: "重试恢复" }).click();
  await expect(page.getByRole("region", { name: "新旧结果对照" })).toBeVisible();
  await expect(page.getByLabel("基于哪一轮结果改进")).toHaveValue(boundSourceRunId);
  await expect(page.getByLabel("回应状态：关键候选仍缺可核验的一手来源")).toHaveValue("EVIDENCE_ADDED");
});

test("feedback and materials survive a failed revision run", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("反馈失败保留");
  await page.getByLabel("研究目标").fill("验证失败时不丢失反馈或既有资料。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  const taskList = await (await page.request.get("/api/tasks")).json() as {
    items: Array<{ task_id: string; title: string }>;
  };
  const taskId = taskList.items.find((task) => task.title === "反馈失败保留")?.task_id;
  expect(taskId).toBeTruthy();
  const uploaded = await page.request.post(`/api/tasks/${taskId}/attachments`, {
    multipart: { file: { name: "已有资料.txt", mimeType: "text/plain", buffer: Buffer.from("既有研究笔记") } },
  });
  expect(uploaded.ok()).toBeTruthy();
  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByRole("heading", { name: "反馈失败保留" })).toBeVisible();
  await expect(page.getByRole("region", { name: "任务资料" })).toContainText("已有资料.txt");
  await page.getByLabel("本轮需求").fill("保存一份可供复核的报告。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  await expect(page.locator(".run-status")).toHaveText("已完成");

  await page.getByRole("button", { name: "继续改进" }).click();
  const feedback = "保留已有结论，并补充来源；故意触发运行失败。";
  await page.getByLabel("对这轮结果的意见").fill(feedback);
  await page.getByRole("button", { name: "分析意见并生成计划" }).click();
  const plan = page.getByRole("region", { name: "修订计划" });
  await expect(plan).toContainText("需要补充目标说明");
  await page.getByLabel("补充目标说明").fill("保留当前任务范围，并记录失败结果。");
  await page.getByRole("button", { name: "用补充说明重新生成计划" }).click();
  await expect(plan).toContainText("UNKNOWN");
  await page.getByRole("button", { name: "开始补充工作" }).click();
  await expect(page.locator(".run-status")).toHaveText("运行失败");
  await expect(page.getByLabel("对这轮结果的意见")).toHaveValue(/故意触发运行失败/);
  await expect(plan).toBeVisible();
  await expect(page.getByRole("region", { name: "任务资料" })).toContainText("已有资料.txt");
});

test("a revision cannot overlap another write run when a ready plan exists", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("任务写入互斥");
  await page.getByLabel("研究目标").fill("同一任务同一时刻只允许一轮写入。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByRole("heading", { name: "任务写入互斥" })).toBeVisible();

  await page.getByLabel("本轮需求").fill("先生成用于后续反馈的基准结果。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  await expect(page.locator(".run-status")).toHaveText("已完成");
  await page.getByRole("button", { name: "继续改进" }).click();
  await page.getByLabel("对这轮结果的意见").fill("补充可核验来源，并保留不确定性。");
  let preparedPlan: { plan_id: string } | undefined;
  page.on("response", async (response) => {
    if (response.request().method() === "POST" && new URL(response.url()).pathname.endsWith("/feedback") && response.status() === 201) {
      preparedPlan = await response.json() as { plan_id: string };
    }
  });
  await page.getByRole("button", { name: "分析意见并生成计划" }).click();
  await expect(page.getByRole("region", { name: "修订计划" })).toContainText("需要补充目标说明");
  await page.getByLabel("补充目标说明").fill("只整理来源，不声称已实验验证。");
  await page.getByRole("button", { name: "用补充说明重新生成计划" }).click();
  await expect(page.getByRole("button", { name: "开始补充工作" })).toBeEnabled();
  await expect.poll(() => preparedPlan?.plan_id).toBeTruthy();

  await page.getByLabel("本轮需求").fill("开始一轮长时间分析。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  await expect(page.locator(".run-status")).toHaveText("正在分析");
  await expect(page.locator(".submit-button")).toBeDisabled();
  await expect(page.getByRole("button", { name: "开始补充工作" })).toBeDisabled();
  const tasks = await (await page.request.get("/api/tasks")).json() as { items: Array<{ task_id: string; title: string }> };
  const taskId = tasks.items.find((task) => task.title === "任务写入互斥")?.task_id;
  expect(taskId).toBeTruthy();
  const rejectedRevision = await page.request.post(`/api/tasks/${taskId}/revisions/${preparedPlan!.plan_id}/runs`, {
    headers: { "Idempotency-Key": "revision-overlap-e2e" },
    data: { budget: { max_model_calls: 20, max_tokens: 10_000, wall_seconds: 300 } },
  });
  expect(rejectedRevision.status()).toBe(409);
  await expect(page.locator(".run-status")).toHaveText("已完成", { timeout: 5_000 });
});
