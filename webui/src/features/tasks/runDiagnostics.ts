import type { WorkbenchEvent } from "../../api";

export interface PendingApproval {
  requestId: string;
  toolName: string;
  argumentsPreview: string;
  reason: string;
}

export function pendingApproval(events: WorkbenchEvent[], runId: string | null): PendingApproval | null {
  if (!runId) return null;
  const resolved = new Set(events.filter((event) => event.run_id === runId && event.type === "approval_resolved")
    .map((event) => event.payload.request_id));
  const pending = [...events].reverse().find((event) => event.run_id === runId && event.type === "approval_pending"
    && typeof event.payload.request_id === "string" && !resolved.has(event.payload.request_id));
  if (!pending) return null;
  return {
    requestId: pending.payload.request_id as string,
    toolName: typeof pending.payload.tool_name === "string" ? pending.payload.tool_name : "科研工具",
    argumentsPreview: typeof pending.payload.arguments_preview === "string" ? pending.payload.arguments_preview : "{}",
    reason: typeof pending.payload.reason === "string" ? pending.payload.reason : "需要确认",
  };
}

export interface BudgetFailure {
  totalLimit: number;
  consumed: number;
  remaining: number;
  inputUpperBound: number;
}

export function budgetFailure(events: WorkbenchEvent[], runId: string | null): BudgetFailure | null {
  const failure = [...events].reverse().find((event) => event.run_id === runId && event.type === "loop_failed"
    && typeof event.payload.error === "string" && event.payload.error.includes("TOKEN_BUDGET_EXHAUSTED"));
  if (!failure) return null;
  const error = failure.payload.error as string;
  const read = (key: string) => Number(new RegExp(`${key}=(-?\\d+)`).exec(error)?.[1]);
  const values = [read("total_limit"), read("consumed"), read("remaining"), read("input_upper_bound")];
  if (values.some((value) => !Number.isFinite(value))) return null;
  return { totalLimit: values[0], consumed: values[1], remaining: values[2], inputUpperBound: values[3] };
}
