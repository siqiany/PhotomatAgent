import type { components } from "./generated/api-schema";

export type TaskCreate = components["schemas"]["TaskCreate"];
export type RunRequest = components["schemas"]["RunRequest-Input"];
export type ResearchTask = components["schemas"]["ResearchTask"];
type ApiRunRecord = components["schemas"]["RunRecord"];
export type RunRecord = Omit<ApiRunRecord, "request"> & {
  request: Pick<RunRequest, "text" | "strategy"> & {
    budget?: components["schemas"]["BudgetSpec"];
    evolution?: components["schemas"]["EvolutionOptions"];
    host_operation?: Record<string, unknown> | null;
  };
};
type ApiTaskSummary = components["schemas"]["TaskSummaryResponse"];
export type TaskSummary = Omit<ApiTaskSummary, "latest_run"> & {
  latest_run?: RunRecord | null;
};
type ApiTaskDetail = components["schemas"]["TaskDetailResponse"];
export type TaskDetail = Omit<ApiTaskDetail, "runs"> & { runs: RunRecord[] };
export type Artifact = components["schemas"]["ArtifactMetadataResponse"];
export type Capability = components["schemas"]["CapabilityResponse"];
export type HistorySession = components["schemas"]["HistorySessionResponse"];
export type RunRef = components["schemas"]["RunRefResponse"];
export type HistoricalTask = components["schemas"]["HistoricalTaskResponse"];
export type RevisionPlan = components["schemas"]["RevisionPlanResponse"];
export type RevisionQDChecker = SkillCheckerSpec & { units: string };
export type RevisionQDRequest = {
  execution_id: string;
  checker: RevisionQDChecker;
  search_mode: "directed" | "random";
  argv_choices: string[][];
  max_trials: number;
  seed: number;
};
export type RevisionRunRequest = {
  budget?: components["schemas"]["BudgetSpec"];
  strategy: "standard" | "qd";
  qd?: RevisionQDRequest;
};

export type SkillStatus = "CANDIDATE" | "TRIALABLE" | "REUSED" | "DISABLED" | "REVALIDATION_REQUIRED";
export type SkillCheckerSpec = {
  checker_id: "json_numeric_sum" | "json_numeric_bounds";
  input_key?: string;
  output_path: string;
  field: string;
  tolerance?: number;
  minimum?: number;
  maximum?: number;
  units?: string;
};
export type SkillSourceRef =
  | { kind: "ordinary_run"; task_id: string; run_id: string; execution_id: string }
  | { kind: "qd_trial"; task_id: string; trial_id: string; execution_id: string };
export type SkillCreate = {
  name: string;
  description: string;
  scope: string;
  source: SkillSourceRef;
  checker: SkillCheckerSpec | null;
};
export type SkillReplay = {
  task_id: string;
  scope: string;
  input_refs?: Record<string, string>;
  argv?: string[];
};
export interface SkillCheckerResult {
  status: "PASS" | "FAIL" | "UNKNOWN" | string;
  gap?: string;
  checker_id?: string;
  method?: string;
  units?: string;
  limitations?: string[];
}
export interface SkillUse {
  use_id?: string;
  task_id?: string;
  code: string;
  checker: SkillCheckerResult;
  artifacts: string[];
}
export interface LocalSkill {
  skill_id: string;
  name?: string;
  description?: string;
  scope?: string;
  source?: SkillSourceRef;
  request?: { input_refs?: Record<string, string> };
  status: SkillStatus;
  source_evidence?: SkillCheckerResult;
  uses: SkillUse[];
}
export interface SkillExecutionSource {
  execution_id: string;
  task_id: string;
  kind?: "ordinary_run" | "qd_trial";
  run_id?: string;
  trial_id?: string;
  script_path: string;
  argv?: string[];
  input_keys: string[];
  artifacts: string[];
  state?: string;
}

export interface Procedure {
  skill_id: string;
  name: string;
  index_name: string;
  description: string;
  scope: string;
  task_id: string;
  run_id: string;
  status: "CANDIDATE" | "APPROVED_PROCEDURE" | "DISABLED";
  validation_status: "UNKNOWN";
  instructions?: string;
  retrieval_count?: number;
  review_note?: string | null;
}
export interface ProcedureCreate {
  name: string;
  description: string;
  scope: string;
  instructions: string;
  task_id: string;
  run_id: string;
  artifact_id?: string;
  plan_id?: string;
}

export interface WorkbenchEvent {
  seq: number;
  run_id: string;
  type: string;
  payload: Record<string, unknown>;
}

export type EvaluationCondition =
  | "STATIC"
  | "FEEDBACK_CHAIN"
  | "RANDOM_CANDIDATE"
  | "DIRECTED_MUTATION"
  | "FULL_QD_SKILLS";
export interface EvaluationConditionCapability {
  id: EvaluationCondition;
  available: boolean;
  reason: string | null;
}
export interface FrozenEvaluationSkillVersion {
  source_type: "published_qd";
  skill_id: string;
  bundle_sha256: string;
}
export interface EvaluationCapabilities {
  conditions: EvaluationConditionCapability[];
  checker_ids: string[];
  frozen_skill_sources: string[];
  frozen_skill_versions: FrozenEvaluationSkillVersion[];
}
export interface EvaluationNumericField {
  value_path: string;
  unit_path: string;
  units: string;
  absolute_tolerance: number;
  relative_tolerance: number;
}
export interface EvaluationCheckerSpec {
  checker_id: "json_exact_v1" | "json_numeric_tolerance_v1";
  version: "1";
  required_fields: string[];
  numeric_fields: Record<string, EvaluationNumericField>;
}
export interface EvaluationCaseSelection {
  case_id: string;
  task_group_id: string;
  signal_family: string;
  source_id: string;
  input_artifact_id: string;
  reference_artifact_id: string;
}
export interface EvaluationStartRequest {
  task_id: string;
  cases: EvaluationCaseSelection[];
  conditions: EvaluationCondition[];
  checker: EvaluationCheckerSpec;
  budget: {
    max_trials: number;
    max_model_calls: number;
    max_model_tokens: number;
    max_wall_seconds: number;
    max_proposals: number;
  };
  seed: number;
  development_groups: string[];
  frozen_qd_skills: FrozenEvaluationSkillVersion[];
}
export interface EvaluationReport {
  evaluation_id: string;
  manifest_sha256: string;
  results: Array<{
    case_id: string;
    task_group_id: string;
    condition: EvaluationCondition;
    verdict: "PASS" | "FAIL" | "UNKNOWN";
    output_sha256?: string | null;
    cost: Record<string, number>;
    provenance: "fake" | "runtime";
    diagnostics: string[];
    checker?: {
      checker_id: string;
      checker_version: string;
      verdict: "PASS" | "FAIL" | "UNKNOWN";
      reason_codes: string[];
      evidence_refs: string[];
      checked: Array<{
        name: string;
        observed: number;
        reference: number;
        units: string;
        absolute_error: number;
        allowed_error: number;
        verdict: "PASS" | "FAIL";
      }>;
    } | null;
  }>;
  condition_success: Record<string, number | null>;
  condition_metrics: Record<string, { cases: number; unknown_rate: number; budget_exhaustion_rate: number }>;
  condition_cost: Record<string, Record<string, number>>;
  total_cost: Record<string, number>;
  evidence_class: "fake_engineering_only" | "runtime_observation";
}
export interface EvaluationStatus {
  evaluation_id: string;
  task_id: string;
  run_id: string;
  status: string;
  manifest_sha256: string;
  created_at: string;
  updated_at: string;
  report: EvaluationReport | null;
  error_code: string | null;
}

interface ErrorBody {
  error?: {
    code?: string;
    message?: string;
    next_action?: string | null;
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const body = (await response.json().catch(() => ({}))) as T & ErrorBody;
  if (!response.ok) {
    const detail = body.error;
    const message = detail?.next_action
      ? `${detail.message ?? "请求失败。"} ${detail.next_action}`
      : detail?.message ?? `请求失败（${response.status}）。`;
    throw new Error(message);
  }
  return body;
}

export const api = {
  tasks: () => request<{ items: TaskSummary[] }>("/api/tasks"),
  task: (taskId: string) => request<TaskDetail>(`/api/tasks/${encodeURIComponent(taskId)}`),
  historicalTask: (taskId: string) => request<HistoricalTask>(`/api/tasks/${encodeURIComponent(taskId)}/history`),
  createTask: (body: TaskCreate) =>
    request<{ task_id: string }>("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  startRun: (taskId: string, body: RunRequest) =>
    request<RunRef>(`/api/tasks/${encodeURIComponent(taskId)}/runs`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": crypto.randomUUID(),
      },
      body: JSON.stringify(body),
    }),
  cancelRun: (runId: string) =>
    request<{ run_id: string; status: string }>(`/api/runs/${encodeURIComponent(runId)}/cancel`, {
      method: "POST",
    }),
  resolveApproval: (taskId: string, runId: string, requestId: string, approved: boolean) =>
    request<{ request_id: string; decision: "approved" | "denied" }>(
      `/api/tasks/${encodeURIComponent(taskId)}/runs/${encodeURIComponent(runId)}/approvals/${encodeURIComponent(requestId)}`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approved }) },
    ),
  resumeRun: (runId: string) =>
    request<RunRef>(`/api/runs/${encodeURIComponent(runId)}/resume`, {
      method: "POST",
      headers: { "Idempotency-Key": crypto.randomUUID() },
    }),
  prepareFeedback: (taskId: string, body: components["schemas"]["FeedbackRequest"]) =>
    request<RevisionPlan>(`/api/tasks/${encodeURIComponent(taskId)}/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": crypto.randomUUID(),
      },
      body: JSON.stringify(body),
    }),
  revisionPlan: (taskId: string, planId: string) =>
    request<RevisionPlan>(
      `/api/tasks/${encodeURIComponent(taskId)}/revisions/${encodeURIComponent(planId)}`,
    ),
  revisionPlans: (taskId: string) => request<{ items: RevisionPlan[] }>(`/api/tasks/${encodeURIComponent(taskId)}/revisions`),
  executeRevision: (taskId: string, planId: string, body: RevisionRunRequest) =>
    request<RunRef>(
      `/api/tasks/${encodeURIComponent(taskId)}/revisions/${encodeURIComponent(planId)}/runs`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": crypto.randomUUID(),
        },
        body: JSON.stringify(body),
      },
    ),
  upload: (taskId: string, file: File) => {
    const body = new FormData();
    body.set("file", file);
    return request<Artifact & { attachment_id: string }>(
      `/api/tasks/${encodeURIComponent(taskId)}/attachments`,
      { method: "POST", body },
    );
  },
  capabilities: () => request<{ items: Capability[] }>("/api/capabilities"),
  evaluationCapabilities: () => request<EvaluationCapabilities>("/api/evaluations/capabilities"),
  evaluationsForTask: (taskId: string) => request<EvaluationStatus[]>(`/api/tasks/${encodeURIComponent(taskId)}/evaluations`),
  evaluation: (evaluationId: string) => request<EvaluationStatus>(`/api/evaluations/${encodeURIComponent(evaluationId)}`),
  startEvaluation: (body: EvaluationStartRequest) => request<Pick<EvaluationStatus, "evaluation_id" | "task_id" | "run_id" | "status">>("/api/evaluations", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Idempotency-Key": crypto.randomUUID() },
    body: JSON.stringify(body),
  }),
  skills: () => request<{ items: LocalSkill[] }>("/api/skills"),
  procedures: () => request<{ items: Procedure[] }>("/api/procedures"),
  procedure: (skillId: string) => request<Procedure>(`/api/procedures/${encodeURIComponent(skillId)}`),
  createProcedure: (body: ProcedureCreate) => request<Procedure>("/api/procedures", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }),
  approveProcedure: (skillId: string, reviewNote: string) => request<Procedure>(`/api/procedures/${encodeURIComponent(skillId)}/approve`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ review_note: reviewNote }),
  }),
  disableProcedure: (skillId: string) => request<Procedure>(`/api/procedures/${encodeURIComponent(skillId)}/disable`, { method: "POST" }),
  skillSources: () => request<{ items: SkillExecutionSource[] }>("/api/skills/sources"),
  createSkill: (body: SkillCreate) =>
    request<LocalSkill>("/api/skills", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  replaySkill: (skillId: string, body: SkillReplay) =>
    request<RunRef>(`/api/skills/${encodeURIComponent(skillId)}/replay`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": crypto.randomUUID() },
      body: JSON.stringify(body),
    }),
  disableSkill: (skillId: string) =>
    request<{ skill_id: string; status: "DISABLED" }>(
      `/api/skills/${encodeURIComponent(skillId)}/disable`,
      { method: "POST" },
    ),
  history: () => request<{ items: HistorySession[] }>("/api/history/sessions"),
  importHistory: (sessionId: string) =>
    request<{ task_id: string }>("/api/history/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }),
};

const eventNames = [
  "run_started", "run_completed", "run_failed", "run_cancelled", "run_interrupted",
  "evaluation_completed",
  "evaluation_experience_recorded", "closed_loop_round_started", "closed_loop_checked",
  "closed_loop_action_selected", "closed_loop_completed", "evolution_guidance_retrieved",
  "evolution_guidance_scope_miss", "evolution_guidance_out_of_range", "evolution_guidance_outcome",
  "evolution_observation", "evolution_candidate_captured", "evolution_skill_executed",
  "qd_compiled", "qd_completed", "qd_trial", "qd_cost",
  "cancel_requested", "recovery_required", "approval_pending", "approval_resolved", "loop_started", "loop_iteration_started",
  "model_request_started", "model_response_completed", "model_stream_started", "text_delta",
  "tool_call_started", "tool_call_arguments_delta", "tool_call_completed", "tool_requested",
  "tool_started", "tool_completed", "tool_failed", "tool_approval_required", "tool_permission_denied",
  "sensitive_access_blocked", "provider_failed", "budget_updated", "loop_completed", "loop_failed",
  "scientific_state_updated", "hypothesis_registered", "hypothesis_registration_rejected", "structure_derived",
  "scientific_trace_meta", "scientific_loop_started", "candidate_proposed", "candidate_evaluated",
  "candidate_judged", "scientific_feedback_generated", "scientific_loop_decision_made",
  "scientific_loop_completed", "scientific_loop_stalled", "evolution_task_created",
  "evolution_episode_started", "evolution_episode_completed", "evolution_episode_failed",
  "expert_feedback_recorded", "expert_feedback_compiled", "revision_plan_confirmed",
  "evolution_iteration_started", "evolution_comparison_completed", "experience_state_changed",
  "evolution_task_accepted", "evolution_task_stopped", "evolution_task_reopened",
  "qd_trial_started", "qd_trial_collected", "qd_trial_failed", "qd_genome_mismatch",
  "skill_use",
  "context_prune_started", "context_prune_completed", "context_compaction_started",
  "context_compaction_completed", "context_compaction_failed", "context_compaction_skipped",
];

export function subscribeToTask(
  taskId: string,
  onEvent: (event: WorkbenchEvent) => void,
  onError: () => void,
): EventSource {
  const source = new EventSource(`/api/tasks/${encodeURIComponent(taskId)}/events?after=0`);
  for (const type of eventNames) {
    source.addEventListener(type, (rawEvent) => {
      const event = rawEvent as MessageEvent<string>;
      try {
        const parsed = JSON.parse(event.data) as Record<string, unknown>;
        const seq = Number(event.lastEventId);
        if (
          !Number.isSafeInteger(seq) ||
          seq < 1 ||
          typeof parsed.run_id !== "string" ||
          typeof parsed.type !== "string" ||
          parsed.type !== type ||
          !parsed.payload ||
          typeof parsed.payload !== "object"
        ) return;
        onEvent({
          seq,
          run_id: parsed.run_id,
          type,
          payload: parsed.payload as Record<string, unknown>,
        });
      } catch {
        onError();
      }
    });
  }
  source.onerror = onError;
  return source;
}
