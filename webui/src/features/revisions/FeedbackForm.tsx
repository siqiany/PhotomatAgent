import { useEffect, useMemo, useRef, useState } from "react";

import type { Artifact, RunRecord } from "../../api";

interface Props {
  runs: RunRecord[];
  artifacts: Artifact[];
  disabled: boolean;
  submitting: boolean;
  feedback: string;
  preferredSourceRunId: string | null;
  preferredArtifactIds: string[];
  preservePreferredArtifactSelection?: boolean;
  onFeedbackChange: (value: string) => void;
  onInvalidatePlan: () => void;
  onPrepare: (sourceRunId: string, feedback: string, artifactIds: string[]) => void;
}

export function FeedbackForm({
  runs,
  artifacts,
  disabled,
  submitting,
  feedback,
  preferredSourceRunId,
  preferredArtifactIds,
  preservePreferredArtifactSelection = false,
  onFeedbackChange,
  onInvalidatePlan,
  onPrepare,
}: Props) {
  const eligibleRuns = useMemo(
    () => runs.filter((run) => ["SUCCEEDED", "PARTIAL", "FAILED", "CANCELLED", "INTERRUPTED"].includes(run.status)),
    [runs],
  );
  const [sourceRunId, setSourceRunId] = useState(eligibleRuns.at(-1)?.run_id ?? "");
  const [selectedArtifactIds, setSelectedArtifactIds] = useState<string[]>([]);
  const modifiedArtifactSelections = useRef(new Map<string, Map<string, boolean>>());
  const appliedPreferredSelection = useRef("");
  const sourceArtifacts = artifacts.filter((artifact) => artifact.run_id === sourceRunId);
  // artifacts_for_task is ordered by created_at, then artifact_id; the last snapshot per filename is latest.
  const latestSourceArtifacts = [...new Map(sourceArtifacts.map((artifact) => [artifact.filename, artifact])).values()];
  const latestArtifactKey = latestSourceArtifacts.map((artifact) => artifact.artifact_id).join("\u0000");
  const preferredArtifactKey = preferredArtifactIds.join("\u0000");

  useEffect(() => {
    if (!eligibleRuns.some((run) => run.run_id === sourceRunId)) {
      setSourceRunId(eligibleRuns.at(-1)?.run_id ?? "");
    }
  }, [eligibleRuns, sourceRunId]);

  useEffect(() => {
    if (preferredSourceRunId && eligibleRuns.some((run) => run.run_id === preferredSourceRunId)) {
      setSourceRunId(preferredSourceRunId);
    }
  }, [eligibleRuns, preferredSourceRunId]);

  useEffect(() => {
    const explicitPlanSelection = preservePreferredArtifactSelection && preferredSourceRunId === sourceRunId;
    let modified = modifiedArtifactSelections.current.get(sourceRunId) ?? new Map<string, boolean>();
    if (explicitPlanSelection) {
      const selectionKey = `${sourceRunId}\u0000${preferredArtifactKey}`;
      if (appliedPreferredSelection.current !== selectionKey) {
        modified = new Map(latestSourceArtifacts.map((artifact) => [
          artifact.filename, preferredArtifactIds.includes(artifact.artifact_id),
        ]));
        modifiedArtifactSelections.current.set(sourceRunId, modified);
        appliedPreferredSelection.current = selectionKey;
      } else {
        for (const artifact of latestSourceArtifacts) {
          if (!modified.has(artifact.filename)) modified.set(artifact.filename, false);
        }
      }
    } else {
      appliedPreferredSelection.current = "";
    }
    setSelectedArtifactIds(latestSourceArtifacts.filter((artifact) => {
      if (modified?.has(artifact.filename)) return modified.get(artifact.filename) === true;
      return !explicitPlanSelection || preferredArtifactIds.includes(artifact.artifact_id);
    }).map((artifact) => artifact.artifact_id));
  }, [latestArtifactKey, preferredArtifactKey, preferredSourceRunId, preservePreferredArtifactSelection, sourceRunId]);

  const toggleArtifact = (artifactId: string) => {
    const artifact = latestSourceArtifacts.find((item) => item.artifact_id === artifactId);
    if (!artifact) return;
    const modified = modifiedArtifactSelections.current.get(sourceRunId) ?? new Map<string, boolean>();
    const select = !selectedArtifactIds.includes(artifactId);
    modified.set(artifact.filename, select);
    modifiedArtifactSelections.current.set(sourceRunId, modified);
    setSelectedArtifactIds((current) => latestSourceArtifacts.filter((item) => {
      if (modified.has(item.filename)) return modified.get(item.filename) === true;
      return current.includes(item.artifact_id);
    }).map((item) => item.artifact_id));
  };

  return (
    <form
      className="revision-feedback-form"
      aria-label="继续改进"
      onSubmit={(event) => {
        event.preventDefault();
        if (sourceRunId && feedback.trim()) onPrepare(sourceRunId, feedback.trim(), selectedArtifactIds);
      }}
    >
      <label htmlFor="revision-source-run">基于哪一轮结果改进</label>
      <select
        id="revision-source-run"
        aria-label="基于哪一轮结果改进"
        value={sourceRunId}
        onChange={(event) => {
          onInvalidatePlan();
          setSourceRunId(event.target.value);
        }}
        disabled={disabled || submitting || eligibleRuns.length === 0}
      >
        {eligibleRuns.map((run) => (
          <option key={run.run_id} value={run.run_id}>
            {run.request.text.slice(0, 72)} · {runStatusLabel(run.status)}
          </option>
        ))}
      </select>
      {eligibleRuns.length === 0 ? <p className="revision-muted">完成一轮研究后，才能对结果提交修改意见。</p> : null}

      {sourceArtifacts.length > 0 ? (
        <fieldset className="revision-artifact-picker" disabled={disabled || submitting}>
          <legend>关联本轮产物（可选，已选 {selectedArtifactIds.length} 项）</legend>
          {latestSourceArtifacts.map((artifact) => (
            <label key={artifact.artifact_id}>
              <input
                type="checkbox"
                checked={selectedArtifactIds.includes(artifact.artifact_id)}
                onChange={() => {
                  onInvalidatePlan();
                  toggleArtifact(artifact.artifact_id);
                }}
              />
              <span>{artifact.filename}</span>
            </label>
          ))}
        </fieldset>
      ) : <p className="revision-warning">本轮没有可关联的产物；可以直接提交修改意见。</p>}

      <label htmlFor="revision-feedback">对这轮结果的意见</label>
      <textarea
        id="revision-feedback"
        aria-label="对这轮结果的意见"
        value={feedback}
        onChange={(event) => {
          onInvalidatePlan();
          onFeedbackChange(event.target.value);
        }}
        placeholder="指出需要补充、纠正或保留的内容…"
        rows={4}
        disabled={disabled || submitting || eligibleRuns.length === 0}
      />
      <div className="revision-form-footer">
        <p>原运行和产物会保留；修订结果会作为同一任务中的新一轮运行。</p>
        <button className="button button-primary" disabled={disabled || submitting || !sourceRunId || !feedback.trim()}>
          {submitting ? "正在分析意见…" : "分析意见并生成计划"}
        </button>
      </div>
    </form>
  );
}

function runStatusLabel(status: string): string {
  return ({
    SUCCEEDED: "已完成",
    PARTIAL: "部分完成",
    FAILED: "失败",
    CANCELLED: "已取消",
    INTERRUPTED: "已中断",
  } as Record<string, string>)[status] ?? status;
}
