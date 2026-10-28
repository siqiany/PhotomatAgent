import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  reporter: "list",
  use: {
    baseURL: `http://127.0.0.1:${process.env.WORKBENCH_E2E_PORT ?? "4173"}`,
    trace: "retain-on-failure",
    channel: "chromium",
    ...devices["Desktop Chrome"],
  },
  webServer: [
    {
      command: "uv run python webui/tests/fixture_server.py",
      env: { WORKBENCH_API_E2E_PORT: process.env.WORKBENCH_API_E2E_PORT ?? "18765" },
      cwd: "..",
      url: `http://127.0.0.1:${process.env.WORKBENCH_API_E2E_PORT ?? "18765"}/api/tasks`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      env: { WORKBENCH_API_E2E_PORT: process.env.WORKBENCH_API_E2E_PORT ?? "18765" },
      command: `npm run dev -- --port ${process.env.WORKBENCH_E2E_PORT ?? "4173"} --strictPort`,
      url: `http://127.0.0.1:${process.env.WORKBENCH_E2E_PORT ?? "4173"}`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
  ],
});
