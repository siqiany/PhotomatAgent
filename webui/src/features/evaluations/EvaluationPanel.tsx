import { useCallback, useEffect, useMemo, useState } from "react";

import {
  api,
  type Artifact,
  type EvaluationCapabilities,
  type EvaluationCaseSelection,
  type EvaluationCheckerSpec,
  type EvaluationCondition,
  type FrozenEvaluationSkillVersion,
  type RunRef,
  type EvaluationStartRequest,
  type EvaluationStatus,
} from "../../api";

interface Props {
  taskId: string;
  artifacts: Artifact[];
  disabled?: boolean;
  onRunQueued?: (reference: RunRef) => void;
}

interface CaseDraft {
  inputArtifactId: string;
  referenceArtifactId: string;
  groupId: string;
}

const CONDITION_LABELS: Record<EvaluationCondition, string> = {
  STATIC: "单次直接回答",
  FEEDBACK_CHAIN: "检查反馈后迭代",
  RANDOM_CANDIDATE: "随机候选",
  DIRECTED_MUTATION: "定向变异",
  FULL_QD_SKILLS: "完整 QD 技能搜索",
};

const INITIAL_CASES: CaseDraft[] = [
  { inputArtifactId: "", referenceArtifactId: "", groupId: "heldout_a" },
  { inputArtifactId: "", referenceArtifactId: "", groupId: "heldout_b" },
];

export function EvaluationPanel({ taskId, artifacts, disabled = false, onRunQueued }: Props) {
  const [capabilities, setCapabilities] = useState<EvaluationCapabilities | null>(null);
  const [evaluations, setEvaluations] = useState<EvaluationStatus[]>([]);
  const [cases, setCases] = useState<CaseDraft[]>(INITIAL_CASES);
  const [conditions, setConditions] = useState<EvaluationCondition[]>(["STATIC"]);
  const [selectedSkills, setSelectedSkills] = useState<FrozenEvaluationSkillVersion[]>([]);
  const [checkerId, setCheckerId] = useState<EvaluationCheckerSpec["checker_id"]>("json_numeric_tolerance_v1");
  const [requiredPath, setRequiredPath] = useState("/required");
  const [numericValuePath, setNumericValuePath] = useState("/peak/value");
  const [numericUnitPath, setNumericUnitPath] = useState("/peak/unit");
  const [units, setUnits] = useState("nm");
  const [absoluteTolerance, setAbsoluteTolerance] = useState("0.01");
  const [relativeTolerance, setRelativeTolerance] = useState("0");
  const [maxTrials, setMaxTrials] = useState(4);
  const [maxModelCalls, setMaxModelCalls] = useState(8);
  const [maxTokens, setMaxTokens] = useState(20_000);
  const [wallSeconds, setWallSeconds] = useState(300);
  const [seed, setSeed] = useState(31);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const jsonArtifacts = useMemo(
    () => artifacts.filter((artifact) => artifact.task_id === taskId && artifact.media_type === "application/json"),
    [artifacts, taskId],
  );

  const reload = useCallback(async () => {
    try {
      const [available, records] = await Promise.all([
        api.evaluationCapabilities(),
        api.evaluationsForTask(taskId),
      ]);
      setCapabilities(available);
      setEvaluations(records);
      setError(null);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => { void reload(); }, [reload]);

  const hasActive = evaluations.some((item) => item.status === "QUEUED" || item.status === "RUNNING");
  useEffect(() => {
    if (!hasActive) return;
    const timer = window.setInterval(() => { void reload(); }, 1200);
    return () => window.clearInterval(timer);
  }, [hasActive, reload]);

  const updateCase = (index: number, field: keyof CaseDraft, value: string) => {
    setCases((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, [field]: value } : item));
  };

  const toggleCondition = (condition: EvaluationCondition, checked: boolean) => {
    if (!checked) {
      setConditions((current) => current.filter((item) => item !== condition));
      return;
    }
    setConditions((current) => current.includes(condition) ? current : [...current, condition]);
  };

  const checker = (): EvaluationCheckerSpec => {
    if (checkerId === "json_exact_v1") {
      return { checker_id: checkerId, version: "1", required_fields: [], numeric_fields: {} };
    }
    const absolute = Number(absoluteTolerance);
    const relative = Number(relativeTolerance);
    if (!numericValuePath.startsWith("/") || !numericUnitPath.startsWith("/") || !units.trim()) {
      throw new Error("请输入以 / 开头的 JSON 路径和单位。");
    }
    if (!Number.isFinite(absolute) || absolute < 0 || !Number.isFinite(relative) || relative < 0) {
      throw new Error("容差必须是非负有限数值。");
    }
    const required = requiredPath.trim();
    if (required && !required.startsWith("/")) throw new Error("必需字段路径必须以 / 开头。");
    return {
      checker_id: checkerId,
      version: "1",
      required_fields: required ? [required] : [],
      numeric_fields: {
        metric: {
          value_path: numericValuePath.trim(),
          unit_path: numericUnitPath.trim(),
          units: units.trim(),
          absolute_tolerance: absolute,
          relative_tolerance: relative,
        },
      },
    };
  };

  const start = async () => {
    setError(null);
    setNotice(null);
    if (!conditions.length) {
      setError("至少选择一种可用评测条件。");
      return;
    }
    if (conditions.includes("FULL_QD_SKILLS") && !selectedSkills.length) {
      setError("请选择至少一个冻结技能版本。");
      return;
    }
    if (cases.some((item) => !item.inputArtifactId || !item.referenceArtifactId || !item.groupId.trim())) {
      setError("请为两个留出组分别选择 JSON 输入和参考结果，并填写组名。");
      return;
    }
    if (cases.some((item) => item.inputArtifactId === item.referenceArtifactId)) {
      setError("同一案例的输入和参考结果必须是不同文件。");
      return;
    }
    if (maxTrials < 1 || maxModelCalls < 1 || maxTokens < 1 || wallSeconds < 1) {
      setError("预算必须是正数。");
      return;
    }
    setBusy(true);
    try {
      const selectedCases: EvaluationCaseSelection[] = cases.map((item, index) => ({
        case_id: `case_${index + 1}`,
        task_group_id: item.groupId.trim().replaceAll(" ", "_"),
        signal_family: "user_selected_json",
        source_id: "workbench_artifacts",
        input_artifact_id: item.inputArtifactId,
        reference_artifact_id: item.referenceArtifactId,
      }));
      const body: EvaluationStartRequest = {
        task_id: taskId,
        cases: selectedCases,
        conditions,
        checker: checker(),
        budget: {
          max_trials: maxTrials,
          max_model_calls: maxModelCalls,
          max_model_tokens: maxTokens,
          max_wall_seconds: wallSeconds,
          max_proposals: maxTrials,
        },
        seed,
        development_groups: [],
        frozen_qd_skills: conditions.includes("FULL_QD_SKILLS") ? selectedSkills : [],
      };
      const started = await api.startEvaluation(body);
      onRunQueued?.(started);
      await reload();
      setNotice(`冻结评测已进入后端队列：${started.evaluation_id}。关闭页面不会停止运行。`);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (runId: string) => {
    setError(null);
    try {
      await api.cancelRun(runId);
      setNotice("已请求取消；已发生的模型调用和运行成本会保留。 ");
      await reload();
    } catch (reason) {
      setError(messageOf(reason));
    }
  };

  return (
    <section className="evaluation-panel" aria-label="冻结对照评测">
      <div className="evaluation-heading">
        <div>
          <span className="eyebrow">FROZEN COMPARISON</span>
          <h3>对照评测</h3>
        </div>
        {hasActive ? <span className="evaluation-live">评测运行中</span> : null}
      </div>
      <p className="evaluation-explainer">
        选择当前任务中的输入和参考 JSON，冻结检查规则后比较策略。参考文件只由后端检查器读取，不会进入模型工作区。
      </p>
      {loading ? <p className="activity-text">正在读取评测能力和历史记录…</p> : null}
      {capabilities ? (
        <div className="evaluation-conditions" aria-label="评测条件">
          {capabilities.conditions.map((condition) => (
            <label key={condition.id} className={condition.available ? "" : "evaluation-unavailable"}>
              <input
                type="checkbox"
                checked={conditions.includes(condition.id)}
                disabled={!condition.available || disabled || busy || hasActive}
                onChange={(event) => toggleCondition(condition.id, event.currentTarget.checked)}
              />
              <span>{CONDITION_LABELS[condition.id]}</span>
              {!condition.available ? <small>{condition.reason}</small> : null}
            </label>
          ))}
        </div>
      ) : null}

      {capabilities?.frozen_skill_versions.length ? (
        <fieldset disabled={disabled || busy || hasActive}>
          <legend>冻结技能版本</legend>
          <p className="evaluation-muted">选择本次执行的确切版本。每个版本会校验脚本、依赖环境与 bundle 摘要。</p>
          {capabilities.frozen_skill_versions.map((version) => (
            <label key={`${version.skill_id}:${version.bundle_sha256}`}>
              <input type="checkbox" aria-label={`冻结技能 ${version.skill_id}`}
                checked={selectedSkills.some((item) => item.skill_id === version.skill_id && item.bundle_sha256 === version.bundle_sha256)}
                onChange={(event) => { const checked = event.currentTarget.checked; setSelectedSkills((current) => checked
                  ? [...current, version]
                  : current.filter((item) => item.skill_id !== version.skill_id || item.bundle_sha256 !== version.bundle_sha256)); }} />
              {version.skill_id} · {version.bundle_sha256.slice(0, 12)} · published_qd
            </label>
          ))}
        </fieldset>
      ) : null}

      <div className="evaluation-case-list">
        {cases.map((item, index) => (
          <fieldset className="evaluation-case" key={index}>
            <legend>留出案例 {index + 1}</legend>
            <label>
              数据组名称
              <input value={item.groupId} onChange={(event) => updateCase(index, "groupId", event.currentTarget.value)} disabled={disabled || busy || hasActive} />
            </label>
            <label>
              模型输入 JSON
              <select aria-label={`案例 ${index + 1} 输入 JSON`} value={item.inputArtifactId} onChange={(event) => updateCase(index, "inputArtifactId", event.currentTarget.value)} disabled={disabled || busy || hasActive}>
                <option value="">选择当前任务中的 JSON</option>
                {jsonArtifacts.map((artifact) => <option key={artifact.artifact_id} value={artifact.artifact_id}>{artifact.filename}</option>)}
              </select>
            </label>
            <label>
              参考结果 JSON（只供后端检查）
              <select aria-label={`案例 ${index + 1} 参考 JSON`} value={item.referenceArtifactId} onChange={(event) => updateCase(index, "referenceArtifactId", event.currentTarget.value)} disabled={disabled || busy || hasActive}>
                <option value="">选择当前任务中的参考 JSON</option>
                {jsonArtifacts.map((artifact) => <option key={artifact.artifact_id} value={artifact.artifact_id}>{artifact.filename}</option>)}
              </select>
            </label>
          </fieldset>
        ))}
      </div>

      <details className="evaluation-options">
        <summary>检查规则与预算</summary>
        <label>
          独立检查器
          <select aria-label="评测检查器" value={checkerId} onChange={(event) => setCheckerId(event.currentTarget.value as EvaluationCheckerSpec["checker_id"])} disabled={disabled || busy || hasActive}>
            <option value="json_numeric_tolerance_v1">JSON 数值、单位与容差</option>
            <option value="json_exact_v1">JSON 精确对照（兼容旧规则）</option>
          </select>
        </label>
        {checkerId === "json_numeric_tolerance_v1" ? (
          <>
            <div className="evaluation-checker-fields">
              <label>必需字段路径<input aria-label="必需字段路径" value={requiredPath} onChange={(event) => setRequiredPath(event.currentTarget.value)} disabled={disabled || busy || hasActive} /></label>
              <label>数值 JSON 路径<input aria-label="数值 JSON 路径" value={numericValuePath} onChange={(event) => setNumericValuePath(event.currentTarget.value)} disabled={disabled || busy || hasActive} /></label>
              <label>单位 JSON 路径<input aria-label="单位 JSON 路径" value={numericUnitPath} onChange={(event) => setNumericUnitPath(event.currentTarget.value)} disabled={disabled || busy || hasActive} /></label>
              <label>单位<input aria-label="评测单位" value={units} onChange={(event) => setUnits(event.currentTarget.value)} disabled={disabled || busy || hasActive} /></label>
              <label>绝对容差<input aria-label="绝对容差" type="number" min="0" step="any" value={absoluteTolerance} onChange={(event) => setAbsoluteTolerance(event.currentTarget.value)} disabled={disabled || busy || hasActive} /></label>
              <label>相对容差<input aria-label="相对容差" type="number" min="0" step="any" value={relativeTolerance} onChange={(event) => setRelativeTolerance(event.currentTarget.value)} disabled={disabled || busy || hasActive} /></label>
            </div>
            <p className="evaluation-muted">数值字段需包含数值和单位，例如 <code>{'{"peak":{"value":1.0,"unit":"nm"}}'}</code>。缺少字段会标为 UNKNOWN。</p>
          </>
        ) : null}
        <div className="evaluation-budget-fields">
          <label>试验上限<input aria-label="评测试验上限" type="number" min="1" value={maxTrials} onChange={(event) => setMaxTrials(Math.max(1, Number(event.currentTarget.value) || 1))} disabled={disabled || busy || hasActive} /></label>
          <label>模型调用上限<input aria-label="评测模型调用上限" type="number" min="1" value={maxModelCalls} onChange={(event) => setMaxModelCalls(Math.max(1, Number(event.currentTarget.value) || 1))} disabled={disabled || busy || hasActive} /></label>
          <label>Token 上限<input aria-label="评测 Token 上限" type="number" min="1" value={maxTokens} onChange={(event) => setMaxTokens(Math.max(1, Number(event.currentTarget.value) || 1))} disabled={disabled || busy || hasActive} /></label>
          <label>运行秒数<input aria-label="评测运行秒数" type="number" min="1" value={wallSeconds} onChange={(event) => setWallSeconds(Math.max(1, Number(event.currentTarget.value) || 1))} disabled={disabled || busy || hasActive} /></label>
          <label>随机种子<input aria-label="评测随机种子" type="number" value={seed} onChange={(event) => setSeed(Number(event.currentTarget.value) || 0)} disabled={disabled || busy || hasActive} /></label>
        </div>
        <p>每种条件分别使用冻结预算。未被选入的开发组不会进入比较；评测产物不会被提取为技能。</p>
      </details>

      {jsonArtifacts.length < 4 ? <p className="evaluation-muted">至少准备两个留出组的输入 JSON 和参考 JSON；可在本任务中上传，也可使用任务生成的 JSON 产物。</p> : null}
      <button className="button button-primary evaluation-start" onClick={() => void start()} disabled={disabled || busy || hasActive || loading}>
        {busy ? "正在冻结评测配置…" : "开始冻结对照评测"}
      </button>
      {error ? <div className="inline-error" role="alert">{error}</div> : null}
      {notice ? <div className="inline-notice" role="status">{notice}</div> : null}

      {evaluations.length ? (
        <div className="evaluation-history" aria-label="已保存的评测记录">
          <h4>评测记录</h4>
          {evaluations.map((evaluation) => (
            <EvaluationResult key={evaluation.evaluation_id} evaluation={evaluation} onCancel={() => void cancel(evaluation.run_id)} />
          ))}
        </div>
      ) : null}

    </section>
  );
}

function EvaluationResult({ evaluation, onCancel }: { evaluation: EvaluationStatus; onCancel: () => void }) {
  const report = evaluation.report;
  const resultsToShow = report
    ? [...report.results.filter((item) => item.verdict !== "PASS"), ...report.results.filter((item) => item.verdict === "PASS")].slice(0, 16)
    : [];
  return (
    <details className="evaluation-record" open={Boolean(report)}>
      <summary>
        <span>{evaluation.status === "SUCCEEDED" ? "评测已完成" : evaluation.status === "RUNNING" ? "评测运行中" : evaluation.status === "QUEUED" ? "评测排队中" : evaluation.status === "INTERRUPTED" ? "进程中断" : evaluation.status}</span>
        <code>{evaluation.evaluation_id.slice(0, 20)}</code>
      </summary>
      {(evaluation.status === "RUNNING" || evaluation.status === "QUEUED") ? (
        <button className="text-button danger-text" onClick={onCancel}>请求取消评测</button>
      ) : null}
      {evaluation.error_code ? <p className="evaluation-muted">诊断：{evaluation.error_code}</p> : null}
      {report ? (
        <>
          <p className={report.evidence_class === "fake_engineering_only" ? "evaluation-evidence fake" : "evaluation-evidence"}>
            {report.evidence_class === "fake_engineering_only"
              ? "演示模拟结果：仅为工程测试证据，不是科研验证。"
              : "运行观测：结论仅针对本次冻结 JSON 检查合同，不单独证明科学假设。"}
          </p>
          <div className="evaluation-metrics">
            {Object.entries(report.condition_success).map(([condition, success]) => {
              const metrics = report.condition_metrics[condition];
              const cost = report.condition_cost[condition] ?? {};
              return (
                <div className="evaluation-metric" key={condition}>
                  <strong>{CONDITION_LABELS[condition as EvaluationCondition] ?? condition}</strong>
                  <span>通过率 {success === null ? "—" : `${(success * 100).toFixed(0)}%`}</span>
                  <span>未知率 {metrics ? `${(metrics.unknown_rate * 100).toFixed(0)}%` : "—"}</span>
                  <span>预算耗尽 {metrics ? `${(metrics.budget_exhaustion_rate * 100).toFixed(0)}%` : "—"}</span>
                  <span>{cost.model_calls ?? 0} 次调用 · {cost.tokens ?? 0} tokens</span>
                </div>
              );
            })}
          </div>
          <div className="evaluation-case-results" aria-label="案例检查结果">
            {resultsToShow.map((result, index) => (
              <article className={`evaluation-case-result verdict-${result.verdict.toLowerCase()}`} key={`${result.condition}:${result.case_id}:${index}`}>
                <strong>{result.case_id} · {CONDITION_LABELS[result.condition]}</strong>
                <span>独立检查：{result.verdict}</span>
                {result.checker?.checked.map((field) => (
                  <small key={field.name}>{field.name}: {field.observed} {field.units}（参考 {field.reference}，误差 {field.absolute_error} / 容许 {field.allowed_error}）</small>
                ))}
                {result.checker?.reason_codes.length ? <small>{result.checker.reason_codes.join(" · ")}</small> : null}
                {result.diagnostics.length ? <small>{result.diagnostics.join(" · ")}</small> : null}
              </article>
            ))}
          </div>
          {report.results.length > resultsToShow.length ? <p className="evaluation-muted">显示 {resultsToShow.length} 条样本；其余 {report.results.length - resultsToShow.length} 条仍保存在完整评测报告中。</p> : null}
          <p className="evaluation-muted">清单摘要：<code>{report.manifest_sha256}</code> · 总成本 {report.total_cost.model_calls ?? 0} 次模型调用、{report.total_cost.tokens ?? 0} tokens。</p>
        </>
      ) : null}
    </details>
  );
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : "评测请求失败，请重试。";
}
