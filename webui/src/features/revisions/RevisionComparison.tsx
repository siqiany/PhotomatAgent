import { useState } from "react";

import type { RevisionPlan } from "../../api";

type ResponseKind = "PENDING" | "EVIDENCE_ADDED" | "TEXT_ONLY" | "UNRESOLVED";
type Responses = Record<string, ResponseKind>;

interface Props {
  plan: RevisionPlan;
  taskId: string;
  revisionRunId: string;
  before: string;
  after: string;
  sourceProvider: string | null;
  revisionProvider: string | null;
  revisionStatus: string;
  revisionActive: boolean;
}

export function RevisionComparison({
  plan,
  taskId,
  revisionRunId,
  before,
  after,
  sourceProvider,
  revisionProvider,
  revisionStatus,
  revisionActive,
}: Props) {
  const storageKey = `photomatagent:revision-review:${taskId}:${plan.plan_id}:${revisionRunId}`;
  const [responses, setResponses] = useState<Responses>(() => readResponses(storageKey));
  const issues = plan.issues ?? [];
  const canAnnotate = !revisionActive && Boolean(after.trim());

  const updateResponse = (issueId: string, value: ResponseKind) => {
    setResponses((current) => {
      const updated = { ...current, [issueId]: value };
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(updated));
      } catch {
        // The comparison remains usable if browser storage is unavailable.
      }
      return updated;
    });
  };

  return (
    <section className="revision-comparison" aria-label="新旧结果对照">
      <div className="revision-section-heading">
        <div>
          <span className="eyebrow">REVISION REVIEW</span>
          <h3>新旧结果对照</h3>
        </div>
        <span className="revision-plan-state">{revisionStatus}</span>
      </div>
      <div className="revision-comparison-columns">
        <article>
          <h4>修订前</h4>
          <p className="revision-provenance">{provenanceLabel(sourceProvider)}</p>
          <p>{before || "原始结果暂不可读。"}</p>
        </article>
        <article>
          <h4>修订后</h4>
          <p className="revision-provenance">{provenanceLabel(revisionProvider)}</p>
          <p>{after || (revisionActive ? "等待新一轮生成结果…" : "本轮没有可显示的文本结果。")}</p>
        </article>
      </div>
      <div className="revision-response-list">
        <h4>逐项人工记录</h4>
        <p className="revision-manual-note">
          下列内容是保存在此浏览器中的人工判断，不是自动评测结果，也不表示证据真实或主张已经通过。请按计划中的检查方式逐项核对。
        </p>
        {issues.map((issue) => (
          <label key={issue.issue_id}>
            <span>{issue.problem}</span>
            <select
              aria-label={responseLabel(issue.problem)}
              value={responses[issue.issue_id] ?? "PENDING"}
              disabled={!canAnnotate}
              onChange={(event) => updateResponse(issue.issue_id, event.target.value as ResponseKind)}
            >
              <option value="PENDING">待核查（UNKNOWN）</option>
              <option value="EVIDENCE_ADDED">人工判断：发现新增证据</option>
              <option value="TEXT_ONLY">人工判断：仅有文字修改</option>
              <option value="UNRESOLVED">人工判断：仍未解决</option>
            </select>
          </label>
        ))}
        {!canAnnotate ? <p className="revision-muted">生成结束且出现修订文本后，才能保存人工记录。</p> : null}
      </div>
    </section>
  );
}

function readResponses(key: string): Responses {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(key) ?? "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return Object.fromEntries(
      Object.entries(value).filter((entry): entry is [string, ResponseKind] =>
        ["PENDING", "EVIDENCE_ADDED", "TEXT_ONLY", "UNRESOLVED"].includes(entry[1] as string),
      ),
    );
  } catch {
    return {};
  }
}

function provenanceLabel(provider: string | null): string {
  if (provider === "fake") return "演示结果（fake），不代表研究结果";
  if (provider) return `模型生成（${provider}），尚未独立科学验证`;
  return "来源未知，尚未验证";
}

function responseLabel(problem: string): string {
  return `回应状态：${problem}`;
}
