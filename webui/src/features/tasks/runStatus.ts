import type { TaskDetail } from "../../api";

/** Resolve terminal display state from the persisted run record, never the event name. */
export function persistedRunStatus(detail: TaskDetail, runId: string): string | null {
  return detail.runs.find((run) => run.run_id === runId)?.status ?? null;
}

export function canApplyStatusRefresh(revisionAtRequest: number, currentRevision: number): boolean {
  return revisionAtRequest === currentRevision;
}

export function replayedRunStartedStatus(persistedStatus: string | undefined): string {
  const activeStatuses = new Set(["QUEUED", "RUNNING", "WAITING_INPUT", "CANCEL_REQUESTED"]);
  return persistedStatus && !activeStatuses.has(persistedStatus) ? persistedStatus : "RUNNING";
}
