import { expect, test } from "@playwright/test";

test("a frozen evaluation runs on the backend and restores its checked report after refresh", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "新建任务" }).click();
  await page.getByLabel("任务名称").fill("冻结评测演示");
  await page.getByLabel("研究目标").fill("比较两个留出组中的数值 JSON 结果。此教学数据不构成科学结论。");
  await page.getByRole("button", { name: "创建任务" }).click();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  await expect(page.getByRole("heading", { name: "冻结评测演示" })).toBeVisible();

  const uploads = [
    ["input_a.json", '{"case":"a"}'],
    ["reference_a.json", '{"required":"present","peak":{"value":1.0,"unit":"nm"}}'],
    ["input_b.json", '{"case":"b"}'],
    ["reference_b.json", '{"required":"present","peak":{"value":1.0,"unit":"nm"}}'],
  ] as const;
  for (const [name, content] of uploads) {
    await page.getByLabel("添加任务附件").setInputFiles({
      name,
      mimeType: "application/json",
      buffer: Buffer.from(content),
    });
  }

  await page.getByLabel("案例 1 输入 JSON").selectOption({ label: "input_a.json" });
  await page.getByLabel("案例 1 参考 JSON").selectOption({ label: "reference_a.json" });
  await page.getByLabel("案例 2 输入 JSON").selectOption({ label: "input_b.json" });
  await page.getByLabel("案例 2 参考 JSON").selectOption({ label: "reference_b.json" });
  const panel = page.getByRole("region", { name: "冻结对照评测" });
  await panel.getByRole("button", { name: "开始冻结对照评测" }).click();

  await expect(panel).toContainText("评测已完成", { timeout: 20_000 });
  await expect(panel).toContainText("演示模拟结果：仅为工程测试证据，不是科研验证。");
  await expect(panel).toContainText("通过率 100%");
  await expect(panel).toContainText("误差 0 / 容许 0.01");

  await page.reload();
  await page.getByRole("button", { name: "产物与自进化", exact: true }).click();
  const restored = page.getByRole("region", { name: "冻结对照评测" });
  await expect(restored).toContainText("评测已完成", { timeout: 10_000 });
  await expect(restored).toContainText("演示模拟结果：仅为工程测试证据，不是科研验证。");
  await expect(restored).toContainText("通过率 100%");
});
