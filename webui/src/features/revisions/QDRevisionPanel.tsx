import { useEffect, useMemo, useState } from "react";

import { api, type RevisionQDChecker, type RevisionQDRequest, type SkillExecutionSource, type WorkbenchEvent } from "../../api";

interface Props {
  taskId: string;
  disabled: boolean;
  busy: boolean;
  events: WorkbenchEvent[];
  runId?: string | null;
  onRun: (qd: RevisionQDRequest) => void;
}

type CheckerId = "" | "json_numeric_sum" | "json_numeric_bounds";

export function QDRevisionPanel({ taskId, disabled, busy, events, runId, onRun }: Props) {
  const [sources, setSources] = useState<SkillExecutionSource[]>([]);
  const [sourceId, setSourceId] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [checkerId, setCheckerId] = useState<CheckerId>("");
  const [searchMode, setSearchMode] = useState<"directed" | "random">("directed");
  const [outputPath, setOutputPath] = useState("");
  const [field, setField] = useState("");
  const [inputKey, setInputKey] = useState("");
  const [minimum, setMinimum] = useState("");
  const [maximum, setMaximum] = useState("");
  const [units, setUnits] = useState("");
  const [tolerance, setTolerance] = useState("1e-9");
  const [alternatives, setAlternatives] = useState("");
  const [maxTrials, setMaxTrials] = useState("8");
  const [seed, setSeed] = useState("0");
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setLoadError(null);
    void api.skillSources().then(({ items }) => {
      if (!mounted) return;
      const taskSources = items.filter((item) => item.task_id === taskId && item.kind !== "qd_trial"
        && item.state === "COMPLETED" && Array.isArray(item.argv));
      setSources(taskSources);
      setSourceId((current) => taskSources.some((item) => item.execution_id === current)
        ? current : taskSources.at(-1)?.execution_id ?? "");
    }).catch((reason: unknown) => {
      if (mounted) setLoadError(reason instanceof Error ? reason.message : "无法读取脚本来源。");
    }).finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, [taskId]);

  const source = sources.find((item) => item.execution_id === sourceId);
  const outputOptions = useMemo(() => source?.artifacts ?? [], [source]);
  const choices = useMemo(() => {
    if (!source) return [];
    return [source.argv ?? [], ...alternatives.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line) => line.split(/\s+/))];
  }, [alternatives, source]);

  const submit = () => {
    setFormError(null);
    try {
      if (!source || !checkerId) throw new Error("选择真实脚本执行来源和独立数值检查器后，才能启用 QD。");
      if (!outputPath || !field.trim()) throw new Error("请选择来源产物并填写 JSON 数值字段。");
      if (!units.trim()) throw new Error("请明确填写检查结果的单位。");
      if (alternatives.split(/\r?\n/).filter((line) => line.trim()).length === 0) throw new Error("至少填写一个备选参数组合；来源原始参数会自动作为基线。");
      if (choices.length > 32 || choices.some((choice) => choice.length === 0 || choice.length > 32
        || choice.some((token) => token.length > 200 || !/^[A-Za-z0-9_.:=+-]+$/.test(token)))) {
        throw new Error("参数组合最多 32 组，每组 1–32 个 token；token 最长 200 字符，只允许字母、数字及 . _ = : + -。");
      }
      if (new Set(choices.map((choice) => JSON.stringify(choice))).size !== choices.length) throw new Error("备选组合不能与基线或其他组合重复。");
      const parsedMaxTrials = Number(maxTrials);
      const parsedSeed = Number(seed);
      if (!Number.isInteger(parsedMaxTrials) || parsedMaxTrials < 1 || parsedMaxTrials > 32) throw new Error("最大试验数需为 1–32 的整数。");
      if (!Number.isSafeInteger(parsedSeed) || parsedSeed < 0 || parsedSeed > 2147483647) throw new Error("随机种子需为 0–2147483647 的整数。");
      const checker: RevisionQDChecker = {
        checker_id: checkerId,
        output_path: outputPath,
        field: field.trim(),
        units: units.trim(),
      };
      if (checkerId === "json_numeric_sum") {
        if (!inputKey) throw new Error("求和检查器需要选择一个输入项。");
        const parsedTolerance = Number(tolerance);
        if (!Number.isFinite(parsedTolerance) || parsedTolerance < 0 || parsedTolerance > 0.001) throw new Error("容差需为 0 到 0.001 之间的有限数值。");
        checker.input_key = inputKey;
        checker.tolerance = parsedTolerance;
      } else {
        const low = minimum.trim() ? Number(minimum) : undefined;
        const high = maximum.trim() ? Number(maximum) : undefined;
        if (low === undefined && high === undefined) throw new Error("范围检查至少需要设置一个边界。");
        if ((low !== undefined && !Number.isFinite(low)) || (high !== undefined && !Number.isFinite(high)) || (low !== undefined && high !== undefined && low > high)) throw new Error("数值上下限无效。");
        checker.minimum = low;
        checker.maximum = high;
      }
      onRun({ execution_id: source.execution_id, checker, search_mode: searchMode, argv_choices: choices, max_trials: parsedMaxTrials, seed: parsedSeed });
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : "QD 参数无效。");
    }
  };

  return (
    <section className="revision-plan qd-revision" aria-label="QD 参数搜索">
      <div className="revision-section-heading"><div><span className="eyebrow">BOUNDED PARAMETER SEARCH</span><h3>QD 参数搜索</h3></div><span className="revision-plan-state">显式配置</span></div>
      <p className="revision-muted">使用本任务真实脚本执行、冻结原目标和独立检查器，在用户给定的参数组合中搜索。检查结果只描述所选数值合同，不证明物理主张。</p>
      {loading ? <p className="revision-muted" role="status">正在读取本任务的脚本执行记录…</p> : null}
      {loadError ? <p className="inline-error" role="alert">{loadError} 普通反馈修订仍可单独启动。</p> : null}
      {!loading && sources.length === 0 ? <p className="revision-warning">本任务没有带有基线参数的真实脚本执行来源，QD 暂不可用；请使用普通反馈修订。</p> : null}
      {source ? (
        <div className="qd-revision-form">
          <label>真实执行来源<select value={sourceId} onChange={(event) => { setSourceId(event.target.value); setOutputPath(""); }} disabled={disabled || busy}>
      {sources.map((item) => <option key={item.execution_id} value={item.execution_id}>{(item.run_id ?? item.trial_id ?? item.execution_id).slice(-8)} · {item.script_path.split("/").at(-1)} · 基线参数 {(item.argv ?? []).join(" ") || "(空)"}</option>)}
          </select></label>
          <label>独立检查器<select aria-label="独立检查器" value={checkerId} onChange={(event) => setCheckerId(event.target.value as CheckerId)} disabled={disabled || busy}>
            <option value="">选择检查器</option><option value="json_numeric_bounds">JSON 数值范围</option><option value="json_numeric_sum">JSON 数值求和</option>
          </select></label>
          <label>搜索方式<select aria-label="搜索方式" value={searchMode} onChange={(event) => setSearchMode(event.target.value as "directed" | "random")} disabled={disabled || busy}>
            <option value="directed">反馈定向变异（模型在白名单内提案）</option>
            <option value="random">随机候选（固定种子、白名单内抽样）</option>
          </select></label>
          {checkerId ? <>
            <label>检查产物<select value={outputPath} onChange={(event) => setOutputPath(event.target.value)} disabled={disabled || busy}><option value="">选择来源中声明的产物</option>{outputOptions.map((path) => <option key={path} value={path}>{path}</option>)}</select></label>
            <label>JSON 数值字段<input value={field} onChange={(event) => setField(event.target.value)} disabled={disabled || busy} placeholder="例如 result.value" /></label>
            <label>单位<input value={units} onChange={(event) => setUnits(event.target.value)} disabled={disabled || busy} placeholder="例如 eV 或 count" /></label>
            {checkerId === "json_numeric_sum" ? <>
              <label>用于求和的输入项<select value={inputKey} onChange={(event) => setInputKey(event.target.value)} disabled={disabled || busy}><option value="">选择输入</option>{source.input_keys.map((key) => <option key={key}>{key}</option>)}</select></label>
              <label>数值容差<input type="number" min="0" max="0.001" step="any" value={tolerance} onChange={(event) => setTolerance(event.target.value)} disabled={disabled || busy} /></label>
            </> : <div className="qd-bound-fields"><label>最小值<input type="number" step="any" value={minimum} onChange={(event) => setMinimum(event.target.value)} disabled={disabled || busy} /></label><label>最大值<input type="number" step="any" value={maximum} onChange={(event) => setMaximum(event.target.value)} disabled={disabled || busy} /></label></div>}
          </> : null}
          <label>备选参数组合 <span className="revision-muted">每行一组，以空格分隔参数 token；基线已自动包含。不会执行命令行。</span>
            <textarea rows={3} value={alternatives} onChange={(event) => setAlternatives(event.target.value)} disabled={disabled || busy} placeholder="0.8 120\n1.0 120" />
          </label>
          <div className="qd-bound-fields"><label>最大试验数<input type="number" min="1" max="32" value={maxTrials} onChange={(event) => setMaxTrials(event.target.value)} disabled={disabled || busy} /></label><label>随机种子<input type="number" min="0" max="2147483647" value={seed} onChange={(event) => setSeed(event.target.value)} disabled={disabled || busy} /></label></div>
          {choices.length ? <p className="revision-muted">冻结的候选数：{choices.length}（含基线：{(source.argv ?? []).join(" ") || "空参数"}）。每个组合均作为普通参数 token 传递，不经过 shell。</p> : null}
          {formError ? <p className="inline-error" role="alert">{formError}</p> : null}
          <div className="revision-plan-actions"><button className="button button-secondary" disabled={disabled || busy || loading || !source || !checkerId} onClick={submit}>{busy ? "正在启动…" : "按以上边界启动 QD"}</button></div>
        </div>
      ) : null}
      <QDProgress events={events} runId={runId} />
    </section>
  );
}

function QDProgress({ events, runId }: { events: WorkbenchEvent[]; runId?: string | null }) {
  const qdEvents = runId
    ? events.filter((event) => event.run_id === runId && ["qd_compiled", "qd_trial", "qd_completed"].includes(event.type))
    : [];
  if (!qdEvents.length) return null;
  return <div className="qd-progress" aria-live="polite"><h4>QD 搜索进度</h4>{qdEvents.map((event) => {
    const payload = event.payload;
    if (event.type === "qd_compiled") return <div key={event.seq}><strong>参数合同已冻结</strong><p>候选数：{Array.isArray((payload.candidate_space as Record<string, unknown> | undefined)?.argv_choices) ? ((payload.candidate_space as Record<string, unknown>).argv_choices as unknown[]).length : "—"} · 源执行：{String(payload.execution_id ?? "—")} · 目标与合同摘要：{String(payload.goal_sha256 ?? "—")} / {String(payload.contract_sha256 ?? "—")}</p></div>;
    if (event.type === "qd_trial") return <div key={event.seq}><strong>试验 {String(payload.trial_id ?? "")} · {String(payload.status ?? "运行中")}</strong><p>父试验：{list(payload.parent_trial_ids)} · 父候选：{list(payload.parent_ids)} → 候选：{String(payload.genome_id ?? "—")} · 变化：{list(payload.changed_fields)} · 验证：{String(payload.qualification ?? "待检查")} · 成本：{formatCost(payload.cost)}</p>{payload.output_refs ? <p>检查产物：{display(payload.output_refs)}</p> : null}{payload.failure_evidence ? <p>未通过依据：{display(payload.failure_evidence)}</p> : null}{payload.outcomes ? <p>检查结果：{display(payload.outcomes)}</p> : null}</div>;
    return <div key={event.seq}><strong>QD 搜索结束 · {String(payload.status ?? "")}</strong><p>未解决问题：{list(payload.unresolved_issue_ids)} · 最佳候选：{String(payload.best_genome_id ?? "—")} · 总成本：{formatCost(payload.cost)}</p></div>;
  })}</div>;
}

function list(value: unknown): string { return Array.isArray(value) ? value.map(String).join("、") || "无" : "—"; }
function display(value: unknown): string { return typeof value === "string" ? value : JSON.stringify(value); }
function formatCost(value: unknown): string { return value === undefined ? "—" : display(value); }
