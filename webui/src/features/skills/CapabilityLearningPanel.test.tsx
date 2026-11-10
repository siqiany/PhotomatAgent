import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "../../api";
import { CapabilityLearningPanel } from "./CapabilityLearningPanel";
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let cleanup: (() => Promise<void>) | undefined;
afterEach(async () => { await cleanup?.(); vi.restoreAllMocks(); });
async function mount(taskId: string) {
  const host=document.createElement("div");document.body.append(host);const root=createRoot(host);
  cleanup=async () => { await act(async () => root.unmount());host.remove(); };
  await act(async () => root.render(<CapabilityLearningPanel taskId={taskId} sourceRunId="run_source" />));
  return {host,root};
}
it("queues bounded learning and shows independently qualified procedure",async () => {
  vi.spyOn(api,"capabilityLearning").mockResolvedValue({items:[{job_task_id:"job",run_id:"run_job",status:"SUCCEEDED",chain_sha256:"abc",candidates:[{skill_id:"method",validation_status:"PASS",procedure_id:"procedure"}]}]});
  vi.spyOn(api,"capabilityBenchmarks").mockResolvedValue({items:[]});
  const start=vi.spyOn(api,"startCapabilityLearning").mockResolvedValue({task_id:"job",run_id:"run_job"});
  const {host}=await mount("task_a");
  expect(host.textContent).toContain("自动启用");expect(host.textContent).toContain("科研有效性：未知");
  const button=[...host.querySelectorAll("button")].find(b => b.textContent==="重新验证并搜索能力版本")!;
  await act(async () => button.click());
  expect(start).toHaveBeenCalledWith("task_a",{source_run_id:"run_source",max_candidates:3,max_executions:24,search_mode:"directed"});
});
it("clears old benchmark and review details on task changes",async () => {
  vi.spyOn(api,"capabilityLearning").mockResolvedValue({items:[]});
  vi.spyOn(api,"capabilityBenchmarks").mockImplementation(async taskId => ({items:taskId==="task_a"?[{job_task_id:"old_job",status:"SUCCEEDED",report:{marker:"old private report"}}]:[]}));
  const {host,root}=await mount("task_a");expect(host.textContent).toContain("old private report");
  await act(async () => root.render(<CapabilityLearningPanel taskId="task_b" sourceRunId="run_b" />));
  expect(host.textContent).not.toContain("old private report");
});
