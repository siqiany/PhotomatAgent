import type { RunRecord } from "../api";

const labels: Record<string, string> = {
  QUEUED: "排队中",
  RUNNING: "正在分析",
  WAITING_INPUT: "等待补充",
  CANCEL_REQUESTED: "正在取消",
  CANCELLED: "已取消",
  SUCCEEDED: "运行结束",
  PARTIAL: "部分完成",
  FAILED: "运行失败",
  INTERRUPTED: "运行中断",
};

export function statusLabel(status: string | undefined): string {
  return status ? labels[status] ?? status : "草稿";
}

export function isRunActive(status: string | undefined): boolean {
  return status !== undefined && ["QUEUED", "RUNNING", "WAITING_INPUT", "CANCEL_REQUESTED"].includes(status);
}

export function RunStatus({ run }: { run: RunRecord | null }) {
  const status = run?.status;
  return (
    <span className={`run-status status-${status?.toLowerCase() ?? "draft"}`} role="status">
      <span className="status-dot" aria-hidden="true" />
      {statusLabel(status)}
    </span>
  );
}
