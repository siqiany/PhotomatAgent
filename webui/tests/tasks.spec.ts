import { expect, test } from "@playwright/test";

test("a research task runs through the backend and remains visible after refresh", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("中波红外材料初筛");
  await page.getByLabel("研究目标").fill("整理可用于中波红外探测的候选材料与证据缺口。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();

  await expect(page.getByRole("heading", { name: "中波红外材料初筛" })).toBeVisible();
  await page.getByText("本轮预算").click();
  await page.getByLabel("模型调用上限", { exact: true }).fill("7");
  await page.getByLabel("Token 上限", { exact: true }).fill("24000");
  await page.getByLabel("运行时间上限（秒）").fill("900");
  await page.getByLabel("添加任务附件").setInputFiles({
    name: "候选材料.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("材料,带隙_eV\n候选 A,0.32\n"),
  });
  const materials = page.getByRole("region", { name: "任务资料" });
  await expect(materials).toContainText("候选材料.csv");
  await materials.getByRole("button", { name: "预览前 32 KiB" }).click();
  await expect(materials).toContainText("候选 A,0.32");

  let submittedBudget: unknown;
  page.on("request", (request) => {
    if (request.url().includes("/runs") && request.method() === "POST") {
      submittedBudget = request.postDataJSON().budget;
    }
  });
  await page.getByLabel("本轮需求").fill("先比较候选材料的带隙范围，并标注未验证的数据。");
  await page.getByRole("button", { name: "提交给智能体" }).click();
  await expect.poll(() => submittedBudget).toEqual({ max_model_calls: 7, max_tokens: 24000, wall_seconds: 900 });
  const report = page.getByRole("region", { name: "本轮报告" });
  await expect(report).toContainText("初稿演示：吸收边和载流子输运仍需实验验证。");
  await expect(page.locator(".run-status")).toHaveText("已完成");
  const runId = await page.getByTestId("active-run-id").textContent();
  expect(runId).toMatch(/^run_/);

  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByRole("heading", { name: "中波红外材料初筛" })).toBeVisible();
  await expect(page.getByTestId("active-run-id")).toHaveText(runId ?? "");
  await expect(page.getByLabel("模型调用上限", {exact: true})).toHaveValue("7");
  await expect(page.getByLabel("Token 上限", {exact: true})).toHaveValue("24000");
  await expect(page.getByRole("region", { name: "本轮报告" })).toContainText(
    "初稿演示：吸收边和载流子输运仍需实验验证。",
  );
});

test("an unavailable optional capabilities endpoint does not block the task workspace", async ({ page }) => {
  await page.route("**/api/capabilities", (route) => route.abort());
  await page.goto("/");
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("能力接口离线");
  await page.getByLabel("研究目标").fill("可选能力状态服务离线时，仍能进入任务工作区。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByRole("heading", { name: "能力接口离线" })).toBeVisible();
  await expect(page.getByLabel("本轮需求")).toBeVisible();
  await expect(page.getByLabel("功能状态")).toContainText("状态暂不可读");
});

test("task list and task detail failures can be retried in place", async ({ page }) => {
  let taskListAttempts = 0;
  await page.route("**/api/tasks", async (route) => {
    if (route.request().method() === "GET" && taskListAttempts++ === 0) {
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { message: "暂时不可用" } }) });
      return;
    }
    await route.continue();
  });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("暂时不可用");
  await page.getByRole("button", { name: "重试" }).click();
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("重试入口");
  await page.getByLabel("研究目标").fill("页面可以从临时加载错误中恢复。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();

  let detailAttempts = 0;
  await page.route("**/api/tasks/*", async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === "GET" && !url.pathname.endsWith("/events") && detailAttempts++ === 0) {
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { message: "详情暂时不可用" } }) });
      return;
    }
    await route.continue();
  });
  await page.reload();
  await expect(page.getByRole("button", { name: "重试打开" })).toBeVisible();
  await page.getByRole("button", { name: "重试打开" }).click();
  await expect(page.getByLabel("本轮需求")).toBeVisible();
});

test("imported history displays its read-only conversation and source digest", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "导入历史任务" }).click();
  await page.getByRole("button", { name: /e2e-archived-session/ }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  const source = page.getByRole("region", { name: "导入历史来源" });
  await expect(source).toContainText("e2e-archived-session");
  await source.getByText("查看历史对话（只读）").click();
  await expect(source).toContainText("请复核这组 SAM 分割结果。");
  await expect(source).toContainText("历史记录：边界误差需要检查。");
});

test("a failed history import stays retryable and shows its progress", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "导入历史任务" }).click();
  const historyItem = page.getByRole("button", { name: /e2e-archived-session/ });
  await expect(historyItem).toBeVisible();
  let releaseFirst: (() => void) | undefined;
  let attempts = 0;
  await page.route("**/api/history/import", async (route) => {
    attempts += 1;
    if (attempts === 1) {
      await new Promise<void>((resolve) => { releaseFirst = resolve; });
      await route.fulfill({ status: 422, contentType: "application/json", body: JSON.stringify({ error: { message: "临时导入错误" } }) });
    } else {
      await route.continue();
    }
  });

  await historyItem.click();
  await expect(historyItem).toContainText("正在导入…");
  await expect(historyItem).toBeDisabled();
  releaseFirst?.();
  await expect(page.getByRole("alert")).toContainText("临时导入错误");
  await expect(historyItem).toBeEnabled();
  await historyItem.click();
  await expect(page.getByRole("region", { name: "导入历史来源" })).toBeVisible();
});

test("the narrow layout can switch between task, workspace, and result panels", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("tab", { name: "任务" }).click();
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("窄屏工作区检查");
  await page.getByLabel("研究目标").fill("验证窄屏下仍可访问研究目标和结果面板。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();

  await expect(page.getByRole("heading", { name: "窄屏工作区检查" })).toBeVisible();
  await page.getByRole("tab", { name: "详情" }).click();
  await expect(page.getByRole("region", { name: "本轮报告" })).toBeVisible();
  await page.getByRole("tab", { name: "工作区" }).click();
  await expect(page.getByLabel("本轮需求")).toBeVisible();
});

test("a long event stream keeps the composer visible and lets readers inspect a bounded timeline", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("事件密集型任务");
  await page.getByLabel("研究目标").fill("检查大量工具流事件时仍能输入问题。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await expect(page.getByRole("heading", { name: "事件密集型任务" })).toBeVisible();

  const runId = "run_stream_fixture";
  const entries = [
    { type: "run_started", payload: {} },
    ...Array.from({ length: 1_200 }, (_, index) => ({ type: "tool_call_arguments_delta", payload: { delta: `part-${index}` } })),
    { type: "tool_completed", payload: { tool_name: "read_file", result: "checked source" } },
  ];
  const stream = entries.map((entry, index) => {
    const seq = index + 1;
    return `id: ${seq}\nevent: ${entry.type}\ndata: ${JSON.stringify({ run_id: runId, ...entry })}\n\n`;
  }).join("");
  await page.route("**/api/tasks/*/events?after=0", (route) => route.fulfill({
    status: 200,
    contentType: "text/event-stream",
    body: stream,
  }));
  await page.reload();

  const process = page.locator(".activity-panel");
  await expect(process.locator(":scope > summary")).toContainText("1202");
  await expect(process).not.toHaveAttribute("open", "");
  await expect(page.getByLabel("本轮需求")).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBeLessThanOrEqual(900);

  await process.locator(":scope > summary").click();
  await expect(process.locator(".activity-item")).toHaveCount(2);
  await process.getByText("科研工具返回").click();
  await expect(process).toContainText('"checked source"');
  await process.getByRole("button", { name: /查看全部事件/ }).click();
  await expect(process.locator(".activity-item")).toHaveCount(25);
  await process.getByRole("button", { name: "只看关键事件" }).click();
  await expect(process.locator(".activity-item")).toHaveCount(2);
  await expect(page.getByLabel("本轮需求")).toBeInViewport();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByLabel("本轮需求")).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBeLessThanOrEqual(844);
});

test("conversation keeps multiple turns, opens details on demand, and exposes a no-progress stop", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("连续科研对话");
  await page.getByLabel("研究目标").fill("检查连续对话与停止诊断。这里使用工程模拟模型。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await expect(page.getByRole("heading", {name: "连续科研对话"})).toBeVisible();
  await expect(page.locator(".skill-inspector")).not.toBeVisible();
  const composer = page.getByLabel("本轮需求");
  await composer.fill("比较材料候选，标出证据缺口。");
  await composer.press("Enter");
  await expect(page.locator(".run-status")).toHaveText("已完成");
  await expect(page.locator(".conversation-turn")).toHaveCount(1);
  await composer.fill("继续核对来源");
  await composer.press("Shift+Enter");
  await expect(composer).toHaveValue("继续核对来源\n");
  await composer.press("Enter");
  await expect(page.locator(".run-status")).toHaveText("已完成");
  await expect(page.locator(".conversation-turn")).toHaveCount(2);
  await page.reload();
  await expect(page.locator(".conversation-turn")).toHaveCount(2);
  await expect(page.locator(".conversation-turn").first()).toContainText("比较材料候选");
  await expect(page.locator(".conversation-turn").last()).toContainText("继续核对来源");
  await expect(page.locator(".conversation-turn .markdown-body")).toHaveCount(2);
  const latestBeforeInspection = await page.getByTestId("active-run-id").textContent();
  await page.locator(".conversation-turn").first().getByRole("button", {name: "查看本轮详情"}).click();
  await expect(page.locator(".skill-inspector .activity-panel")).toBeVisible();
  await expect(page.getByTestId("active-run-id")).toHaveText(latestBeforeInspection ?? "");
  await page.locator(".header-actions").getByRole("button", {name: "收起详情", exact: true}).click();
  await expect(composer).toBeInViewport();
  await page.screenshot({ path: "../tmp/workbench-conversation-desktop.png" });

  const runId = await page.getByTestId("active-run-id").textContent();
  const entries = [
    {type: "model_stream_started", payload: {}},
    {type: "text_delta", payload: {text: "## 临时分析\n\n尚未验证，准备登记假设。"}},
    {type: "model_response_completed", payload: {tool_call_count: 1, finish_reason: "tool_calls"}},
    {type: "tool_failed", payload: {tool_name: "generation.register_hypothesis", error: "unknown evidence id(s): 2008.13386v1"}},
    {type: "loop_completed", payload: {reason: "no_progress"}},
  ];
  await page.route("**/api/tasks/*/events?after=0", (route) => route.fulfill({status: 200, contentType: "text/event-stream", body: entries.map((entry, i) => `id: ${i+1}\nevent: ${entry.type}\ndata: ${JSON.stringify({run_id: runId, ...entry})}\n\n`).join("")}));
  await page.reload();
  await expect(page.locator(".stop-explanation")).toContainText("generation.register_hypothesis");
  await expect(page.locator(".stop-explanation")).toContainText("unknown evidence id");
  await page.locator(".research-progress > summary").click();
  await expect(page.getByRole("heading", {name: "临时分析"})).toBeVisible();
  await page.setViewportSize({width: 390, height: 844});
  await page.locator(".conversation-turn").last().getByRole("button", {name: "查看本轮详情"}).click();
  await expect(page.getByRole("region", {name: "本轮报告"})).toBeVisible();
  await page.screenshot({ path: "../tmp/workbench-conversation-mobile.png" });
});
