import { useState } from "react";

import type { RevisionPlan as Plan } from "../../api";

interface Props {
  plan: Plan;
  busy: boolean;
  runDisabled: boolean;
  onRun: () => void;
  onClarify: (text: string) => void;
  onRetry: () => void;
}

type PlanRunAttempt = { attempt: number; run_id: string; status: string };
type PlanWithRunDetails = Plan & {
  run_attempts?: PlanRunAttempt[];
  compiler_cost?: Record<string, unknown>;
};

export function RevisionPlan({ plan, busy, runDisabled, onRun, onClarify, onRetry }: Props) {
  const [clarification, setClarification] = useState("");
  const needsGoalClarification = plan.issues?.some(
    (issue) => issue.category === "TASK_DEFINITION" && issue.status === "QUERY",
  ) ?? false;
  const unavailable = plan.status === "UNAVAILABLE";
  const planDetails = plan as PlanWithRunDetails;
  const attempts = planDetails.run_attempts ?? [];
  const latestRunStatus = attempts.at(-1)?.status;
  const retryableRun = ["FAILED", "CANCELLED", "INTERRUPTED"].includes(latestRunStatus ?? "");
  const hasRun = attempts.length > 0;
  const activeRun = ["QUEUED", "RUNNING", "WAITING_INPUT", "CANCEL_REQUESTED"].includes(latestRunStatus ?? "");

  return (
    <section className="revision-plan" aria-label="修订计划" aria-live="polite">
      <div className="revision-section-heading">
        <div>
          <span className="eyebrow">REVIEW RESPONSE</span>
          <h3>修订计划</h3>
        </div>
        <span className={`revision-plan-state state-${plan.status.toLowerCase()}`}>
          {plan.status === "READY" ? "待开始" : plan.status === "PREPARING" ? "正在分析" : "暂不可用"}
        </span>
      </div>
      <blockquote>{plan.feedback_text}</blockquote>
      {plan.error ? <p className="inline-error" role="alert">{plan.error}</p> : null}
      {plan.warnings?.map((warning, index) => <p className="revision-warning" key={`${index}:${warning}`}>{warning}</p>)}
      {planDetails.compiler_cost ? <CompilerCost cost={planDetails.compiler_cost} /> : null}
      {attempts.length ? (
        <div className="revision-run-attempts" aria-label="修订运行记录">
          <h4>修订运行记录</h4>
          <ol>{attempts.map((attempt) => <li key={attempt.run_id}>第 {attempt.attempt} 次 · {runStatusLabel(attempt.status)} · <code>{attempt.run_id}</code></li>)}</ol>
        </div>
      ) : null}

      <ol className="revision-issue-list">
        {plan.issues?.map((issue) => {
          const checkMethod = issue.check_method ?? issue.acceptance_test;
          const unknownCheck = !checkMethod || /\bUNKNOWN\b|机器状态保持 UNKNOWN/.test(checkMethod);
          return (
            <li key={issue.issue_id}>
            <div className="revision-issue-heading">
              <strong>{issue.problem}</strong>
              <span>{severityLabel(issue.severity)}</span>
            </div>
            {issue.original_feedback_span ? <p className="revision-quote">对应意见：{issue.original_feedback_span}</p> : null}
            {issue.actions?.length ? (
              <div><span className="revision-label">拟采取行动</span><ul>{issue.actions.map((action) => <li key={action}>{action}</li>)}</ul></div>
            ) : null}
            {issue.expected_artifacts?.length ? <p><span className="revision-label">预期产物</span>{issue.expected_artifacts.join("、")}</p> : null}
            {unknownCheck ? (
              <p className="revision-unknown"><strong>UNKNOWN · 证据不足</strong>：{checkMethod ?? "当前没有明确的检查方式"}；本计划不代表相关主张已验证。</p>
            ) : <p><span className="revision-label">检查方式</span>{checkMethod}</p>}
            {issue.cannot_process_reason ? <p className="revision-warning">暂不能处理：{issue.cannot_process_reason}</p> : null}
            {issue.preserve?.length ? <p><span className="revision-label">需要保留</span>{issue.preserve.join("；")}</p> : null}
            {issue.original_evidence?.length ? (
              <details className="revision-evidence">
                <summary>查看此项关联的原始证据引用（{issue.original_evidence.length}）</summary>
                <ul>{issue.original_evidence.map((reference) => <li key={reference}><code>{reference}</code></li>)}</ul>
              </details>
            ) : null}
            </li>
          );
        })}
      </ol>

      {needsGoalClarification ? (
        <div className="revision-clarification">
          <label htmlFor="revision-clarification">需要补充目标说明</label>
          <textarea
            id="revision-clarification"
            aria-label="补充目标说明"
            rows={3}
            value={clarification}
            onChange={(event) => setClarification(event.target.value)}
            placeholder="补充范围、约束或期望结果；原任务目标不会被静默改写。"
            disabled={busy}
          />
          <button
            className="button button-secondary"
            disabled={busy || !clarification.trim()}
            onClick={() => onClarify(clarification.trim())}
          >用补充说明重新生成计划</button>
        </div>
      ) : null}

      <div className="revision-plan-actions">
        {unavailable ? <button className="button button-secondary" disabled={busy} onClick={onRetry}>重试生成计划</button> : null}
        {plan.status === "READY" && (!hasRun || retryableRun) ? (
          <button className="button button-primary" disabled={busy || runDisabled || needsGoalClarification} onClick={onRun}>
            {busy ? "正在启动…" : retryableRun ? "重试本次修订" : "开始补充工作"}
          </button>
        ) : null}
        {hasRun && !retryableRun ? <p className="revision-muted" role="status">{activeRun ? "修订运行正在进行；完成或终止后才能重试。" : `最近一次修订${runStatusLabel(latestRunStatus ?? "")}；不会重复启动。`}</p> : null}
      </div>
    </section>
  );
}

function severityLabel(severity: string): string {
  return ({ CRITICAL: "紧急", HIGH: "重要", MEDIUM: "需要处理", LOW: "一般" } as Record<string, string>)[severity] ?? "待复核";
}

function runStatusLabel(status: string): string {
  return ({
    SUCCEEDED: "已完成",
    PARTIAL: "部分完成",
    FAILED: "失败",
    CANCELLED: "已取消",
    INTERRUPTED: "已中断",
    QUEUED: "排队中",
    RUNNING: "运行中",
    WAITING_INPUT: "等待输入",
    CANCEL_REQUESTED: "正在取消",
  } as Record<string, string>)[status] ?? status;
}

function CompilerCost({ cost }: { cost: Record<string, unknown> }) {
  const budget = typeof cost.budget === "object" && cost.budget !== null
    ? cost.budget as Record<string, unknown>
    : {};
  const value = (key: string) => typeof cost[key] === "number" ? cost[key] as number : "—";
  return (
    <p className="revision-muted" aria-label="计划生成成本">
      计划生成成本：{value("model_calls")} 次模型调用 · {value("total_tokens")} tokens
      （输入 {value("input_tokens")} / 输出 {value("output_tokens")}）
      · 预算 {displayBudgetValue(budget.max_model_calls)} 次调用、{displayBudgetValue(budget.max_tokens)} tokens、{displayBudgetValue(budget.wall_seconds)} 秒
      {cost.estimated === true ? " · 估算值" : ""}{cost.settled === false ? " · 尚未结算" : ""}
    </p>
  );
}

function displayBudgetValue(value: unknown): string {
  return typeof value === "number" || typeof value === "string" ? String(value) : "—";
}
