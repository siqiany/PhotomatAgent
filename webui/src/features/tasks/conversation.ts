import type { WorkbenchEvent } from "../../api";

export interface Reply { text: string; final: boolean }

// Model replies containing tool calls are progress, never a final answer.
export function repliesForRun(events: WorkbenchEvent[], runId: string): Reply[] {
  const replies: Reply[] = [];
  let current: Reply | undefined;
  for (const event of events) {
    if (event.run_id !== runId) continue;
    if (event.type === "closed_loop_round_started") {
      for (const reply of replies) reply.final = false;
    }
    if (event.type === "model_stream_started") current = undefined;
    if (event.type === "text_delta" && typeof event.payload.text === "string") {
      if (!current) { current = { text: "", final: false }; replies.push(current); }
      current.text += event.payload.text;
    }
    if (event.type === "model_response_completed" && current) {
      current.final = event.payload.tool_call_count === 0 && event.payload.finish_reason === "stop";
    }
  }
  const completion = [...events].reverse().find((event) => event.run_id === runId && event.type === "loop_completed");
  const checked = [...events].reverse().find((event) => event.run_id === runId && event.type === "closed_loop_completed");
  if ((completion?.payload.reason !== "final_response" && completion?.payload.reason !== "completed") ||
      (checked && checked.payload.delivery_status !== "PASS")) {
    for (const reply of replies) reply.final = false;
  }
  return replies;
}

export function stopExplanation(events: WorkbenchEvent[], runId: string): string | null {
  const last = [...events].reverse().find((e) => e.run_id === runId && e.type === "loop_completed");
  if (last?.payload.reason !== "no_progress") return null;
  const failure = [...events].reverse().find((e) => e.run_id === runId && e.type === "tool_failed");
  const tool = failure?.payload.underlying_tool ?? failure?.payload.tool_name;
  const error = failure?.payload.error;
  return `连续多轮没有取得新进展，本轮已停止，已有回复与产物已保留。${failure ? `最后失败的工具：${String(tool ?? "未知")}。${typeof error === "string" ? `原因：${error}` : ""}` : ""}`;
}
