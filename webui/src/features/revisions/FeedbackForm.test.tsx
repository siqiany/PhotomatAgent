import { act } from "react";
import type { ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Artifact, RunRecord } from "../../api";
import { FeedbackForm } from "./FeedbackForm";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let mounted: Array<{ root: Root; host: HTMLElement }> = [];
afterEach(async () => {
  await act(async () => { for (const item of mounted) item.root.unmount(); });
  for (const item of mounted) item.host.remove();
  mounted = [];
});

function run(runId: string, status: string): RunRecord {
  return { run_id: runId, task_id: "task_1", status, created_at: "now", request: { text: runId, strategy: "standard" } } as RunRecord;
}

async function render(runs: RunRecord[], artifacts: Artifact[] = [], overrides: Partial<ComponentProps<typeof FeedbackForm>> = {}) {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  mounted.push({ root, host });
  await act(async () => { root.render(<FeedbackForm runs={runs} artifacts={artifacts} disabled={false} submitting={false} feedback="" preferredSourceRunId={null} preferredArtifactIds={[]} onFeedbackChange={vi.fn()} onInvalidatePlan={vi.fn()} onPrepare={vi.fn()} {...overrides} />); });
  return host;
}

describe("FeedbackForm", () => {
  it("allows feedback to use artifacts from any terminal run status", async () => {
    const host = await render([run("run_failed", "FAILED"), run("run_cancelled", "CANCELLED"), run("run_interrupted", "INTERRUPTED")]);
    const options = [...host.querySelectorAll("#revision-source-run option")];
    expect(options.map((option) => option.getAttribute("value"))).toEqual(["run_failed", "run_cancelled", "run_interrupted"]);
    expect(host.textContent).toContain("失败");
    expect(host.textContent).toContain("已取消");
    expect(host.textContent).toContain("已中断");
  });

  it("preselects only the latest snapshot per filename from the selected run and preserves deselection", async () => {
    const selectedRun = "run_latest";
    const otherRun = "run_other";
    const artifact = (artifact_id: string, filename: string, run_id: string): Artifact => ({
      artifact_id, filename, run_id, task_id: "task_1", media_type: "text/markdown", size_bytes: 10, sha256: artifact_id,
    });
    const prepare = vi.fn();
    const props = {
      feedback: "keep this",
      onPrepare: prepare,
    };
    const host = await render(
      [run(otherRun, "FAILED"), run(selectedRun, "SUCCEEDED")],
      [artifact("old_report", "report.md", selectedRun), artifact("new_report", "report.md", selectedRun),
        artifact("notes", "notes.md", selectedRun), artifact("other_run", "other.md", otherRun)],
      props,
    );
    const checks = [...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')];
    expect(checks.map((input) => [input.parentElement?.textContent, input.checked])).toEqual([
      ["report.md", true], ["notes.md", true],
    ]);
    expect(host.textContent).toContain("已选 2 项");

    await act(async () => {
      checks.find((input) => (input.parentElement?.textContent ?? "").includes("report.md"))?.click();
    });
    // Parent rerender from artifact polling must not restore a user's explicit deselection.
    const entry = mounted.find((item) => item.host === host)!;
    await act(async () => {
      entry.root.render(<FeedbackForm runs={[run(selectedRun, "SUCCEEDED"), run(otherRun, "FAILED")]} artifacts={[
        artifact("old_report", "report.md", selectedRun), artifact("new_report", "report.md", selectedRun),
        artifact("notes", "notes.md", selectedRun), artifact("other_run", "other.md", otherRun),
      ]} disabled={false} submitting={false} preferredSourceRunId={null} preferredArtifactIds={[]} onFeedbackChange={vi.fn()} onInvalidatePlan={vi.fn()} onPrepare={prepare} feedback="keep this" />);
    });
    expect(host.textContent).toContain("已选 1 项");
    const form = host.querySelector("form");
    await act(async () => { form?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); });
    expect(prepare).toHaveBeenCalledWith(selectedRun, "keep this", ["notes"]);
  });

  it("shows an explicit empty-source warning when the selected run has no artifacts", async () => {
    const host = await render([run("run_latest", "SUCCEEDED")]);
    expect(host.textContent).toContain("本轮没有可关联的产物");
  });

  it("defaults preferred runs to latest artifacts unless a saved plan explicitly selected none", async () => {
    const selectedRun = "run_preferred";
    const artifact = (artifact_id: string, filename: string): Artifact => ({
      artifact_id, filename, run_id: selectedRun, task_id: "task_1", media_type: "text/markdown", size_bytes: 10, sha256: artifact_id,
    });
    const runs = [run("run_older", "FAILED"), run(selectedRun, "SUCCEEDED")];
    const artifacts = [artifact("first", "first.md"), artifact("second", "second.md")];
    const host = await render(runs, artifacts, { preferredSourceRunId: selectedRun });
    expect([...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].every((item) => item.checked)).toBe(true);

    const entry = mounted.find((item) => item.host === host)!;
    await act(async () => {
      entry.root.render(<FeedbackForm runs={runs} artifacts={artifacts} disabled={false} submitting={false}
        feedback="" preferredSourceRunId={selectedRun} preferredArtifactIds={[]} preservePreferredArtifactSelection
        onFeedbackChange={vi.fn()} onInvalidatePlan={vi.fn()} onPrepare={vi.fn()} />);
    });
    expect([...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].every((item) => !item.checked)).toBe(true);
  });

  it("initializes when artifacts arrive later and preserves choices while tracking the latest same-name snapshot", async () => {
    const selectedRun = "run_latest";
    const artifact = (artifact_id: string, filename: string): Artifact => ({
      artifact_id, filename, run_id: selectedRun, task_id: "task_1", media_type: "text/markdown", size_bytes: 10, sha256: artifact_id,
    });
    const runs = [run(selectedRun, "SUCCEEDED")];
    const entryHost = await render(runs, []);
    const entry = mounted.find((item) => item.host === entryHost)!;
    const renderArtifacts = async (items: Artifact[]) => act(async () => {
      entry.root.render(<FeedbackForm runs={runs} artifacts={items} disabled={false} submitting={false} feedback=""
        preferredSourceRunId={null} preferredArtifactIds={[]} onFeedbackChange={vi.fn()} onInvalidatePlan={vi.fn()} onPrepare={vi.fn()} />);
    });

    await renderArtifacts([artifact("report_v1", "report.md"), artifact("notes", "notes.md")]);
    const checks = [...entryHost.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')];
    expect(checks.every((item) => item.checked)).toBe(true);
    await act(async () => { checks.find((item) => item.parentElement?.textContent === "report.md")?.click(); });

    await renderArtifacts([artifact("report_v1", "report.md"), artifact("report_v2", "report.md"), artifact("notes", "notes.md"), artifact("extra", "extra.md")]);
    const latestChecks = [...entryHost.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')];
    expect(latestChecks.map((item) => [item.parentElement?.textContent, item.checked])).toEqual([
      ["report.md", false], ["notes.md", true], ["extra.md", true],
    ]);
  });
});
