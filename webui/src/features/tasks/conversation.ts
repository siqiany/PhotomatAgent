import type { WorkbenchEvent } from "../../api";

export interface Reply { text: string; final: boolean }

export function contextCompactionSummary(events: WorkbenchEvent[], runId: string): string | null {
  const last = [...events].reverse().find((event) => event.run_id === runId && [
    "context_compaction_completed", "context_compaction_skipped", "context_compaction_failed",
  ].includes(event.type));
  if (!last) return null;
  if (last.type === "context_compaction_completed") {
    const before = last.payload.tokens_before;
    const after = last.payload.tokens_after;
    return `工作上下文已压缩${typeof before === "number" && typeof after === "number" ? `：估计 ${before.toLocaleString()} → ${after.toLocaleString()} tokens` : ""}。完整对话、科研证据与权限边界保留。`;
  }
  if (last.type === "context_compaction_failed") return "上下文压缩未完成，原工作上下文已保留。可在运行详情查看原因和已消耗预算。";
  const reasons: Record<string, string> = {
    no_eligible_history: "暂无可压缩的旧历史，最近交互继续保留。",
    no_reduction: "摘要没有缩小上下文，继续使用原工作上下文。",
    no_summarizer: "当前运行时未配置摘要器。",
    inflight_tool_transaction: "工具调用尚未结束，请在安全结束后重试。",
    retry_cooldown: "自动压缩正在冷却，可手动重试。",
  };
  return reasons[String(last.payload.reason)] ?? `本次压缩已跳过：${String(last.payload.reason ?? "未知原因")}。`;
}

export function qdCompletionSummary(events: WorkbenchEvent[], runId: string): string | null {
  const completion = [...events].reverse().find((event) => event.run_id === runId && event.type === "qd_completed");
  if (!completion) return null;
  const payload = completion.payload;
  const status = typeof payload.status === "string" ? payload.status : null;
  if (!status) return null;
  const labels: Record<string, string> = {
    SOLVED: "已找到满足搜索目标的候选",
    PARTIAL: "部分完成",
    FAILED: "失败",
    CANCELLED: "已取消",
    EXHAUSTED: "搜索空间已耗尽",
  };
  const lines = [`QD 候选搜索状态（主机记录）：${labels[status] ?? status}。`];
  if (typeof payload.best_trial_id === "string") lines.push(`最佳试验：${payload.best_trial_id}。`);
  const cost = payload.cost;
  if (cost && typeof cost === "object") {
    const value = cost as Record<string, unknown>;
    const calls = typeof value.model_calls === "number" ? `${value.model_calls} 次模型调用` : null;
    const tokenCount = typeof value.tokens === "number" ? value.tokens : value.model_tokens;
    const tokens = typeof tokenCount === "number" ? `${tokenCount} tokens` : null;
    if (calls || tokens) lines.push(`搜索成本：${[calls, tokens].filter(Boolean).join(" · ")}。`);
  }
  const checked = [...events].reverse().find((event) => event.run_id === runId && event.type === "closed_loop_completed");
  if (checked && typeof checked.payload.scientific_status === "string") {
    lines.push(`科研证据状态：${checked.payload.scientific_status}。`);
  }
  if (Array.isArray(payload.unresolved_issues) && payload.unresolved_issues.length) {
    lines.push("方法或格式检查仍有未解决问题；此状态不代表材料性能已通过验证。");
  } else {
    lines.push("此搜索状态不代表材料性能已通过验证。");
  }
  return lines.join(" ");
}

// Model replies containing tool calls are progress, never a final answer.
export function repliesForRun(events: WorkbenchEvent[], runId: string): Reply[] {
  const replies: Reply[] = [];
  let current: Reply | undefined;
  let researchDelivery: Reply | undefined;
  for (const event of events) {
    if (event.run_id !== runId) continue;
    if (event.type === "closed_loop_round_started") {
      for (const reply of replies) reply.final = false;
    }
    if (event.type === "model_stream_started") current = undefined;
    if (event.type === "text_delta" && typeof event.payload.text === "string") {
      if (typeof event.payload.research_delivery === "string") {
        researchDelivery = { text: event.payload.text, final: false };
        replies.push(researchDelivery);
        current = undefined;
        continue;
      }
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
  if (researchDelivery && (completion?.payload.reason === "research_delivery" || completion?.payload.reason === "research_partial")) {
    researchDelivery.final = true;
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
