import { useEffect, useState } from "react";
import { api } from "../../api";

const conditionLabels: Record<string, string> = { BASE: "基础智能体", FEEDBACK_ONLY: "仅反馈返修", FEEDBACK_MEMORY: "反馈与经验", SINGLE_BEST: "单一最佳方法", FULL_QD: "QD 方法库", NO_TOOL_EVOLUTION: "关闭工具进化", NO_VALIDATION_GATE: "关闭验证门槛" };

export function CapabilityLearningPanel({ taskId, sourceRunId }: { taskId: string; sourceRunId?: string }) {
  const [jobs, setJobs] = useState<Array<Record<string, unknown>>>([]);
  const [benchmarks, setBenchmarks] = useState<Array<Record<string, unknown>>>([]);
  const [candidates, setCandidates] = useState(3);
  const [executions, setExecutions] = useState(24);
  const [searchMode, setSearchMode] = useState<"directed" | "random">("directed");
  const [calls, setCalls] = useState(80);
  const [seconds, setSeconds] = useState(900);
  const [conditions, setConditions] = useState(["BASE", "FEEDBACK_ONLY", "FEEDBACK_MEMORY", "SINGLE_BEST", "FULL_QD"]);
  const [reviewJob, setReviewJob] = useState("");
  const [blindCases, setBlindCases] = useState<Array<Record<string, unknown>>>([]);
  const [blindId, setBlindId] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [quality, setQuality] = useState(3);
  const [coverage, setCoverage] = useState(0.5);
  const [severe, setSevere] = useState(0);
  const [repeated, setRepeated] = useState(0);
  const [minutes, setMinutes] = useState(0);
  const [notes, setNotes] = useState("");
  const [reviewSaved, setReviewSaved] = useState("");
  const [repeats, setRepeats] = useState(2);
  const [tokens, setTokens] = useState(150000);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const load = () => api.capabilityLearning(taskId).then((r) => setJobs(r.items));
  useEffect(() => {
    let alive = true;
    const reload = () => api.capabilityLearning(taskId).then((r) => { if (alive) setJobs(r.items); }).catch(() => undefined);
    setJobs([]); setBenchmarks([]); setBlindCases([]); setReviewJob(""); setBlindId(""); setNotes(""); setReviewSaved(""); setError(""); void reload();
    void api.capabilityBenchmarks(taskId).then((r) => { if (alive) setBenchmarks(r.items); }).catch(() => undefined);
    const timer = setInterval(() => { void reload(); void api.capabilityBenchmarks(taskId).then((r) => { if (alive) setBenchmarks(r.items); }).catch(() => undefined); }, 2500);
    return () => { alive = false; clearInterval(timer); };
  }, [taskId]);
  const start = async () => {
    if (!sourceRunId) return;
    setBusy(true); setError("");
    try {
      await api.startCapabilityLearning(taskId, { source_run_id: sourceRunId, max_candidates: candidates, max_executions: executions, search_mode: searchMode });
      await load();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const benchmark = async () => {
    setBusy(true); setError("");
    try {
      await api.startCapabilityBenchmark(taskId, { repeats, max_tokens: tokens, max_model_calls: calls, wall_seconds: seconds, conditions });
      setBenchmarks((await api.capabilityBenchmarks(taskId)).items);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  return <section className="inspector-card" aria-label="自动能力学习">
    <h2>自动能力学习</h2>
    <p className="small-copy">研究交付后自动从真实脚本提炼方法。原错误、新案例和反例通过后启用，科学有效性仍单独评审。没有可检查的执行记录时不会把报告直接当作技能。</p>
    <details><summary>学习预算与搜索设置</summary>
      <label>候选版本上限<input type="number" min={1} max={8} value={candidates} onChange={(e) => setCandidates(Number(e.target.value))} /></label>
      <label>独立执行上限<input type="number" min={4} max={64} value={executions} onChange={(e) => setExecutions(Number(e.target.value))} /></label>
      <label>候选顺序<select value={searchMode} onChange={(e) => setSearchMode(e.target.value as "directed" | "random")}><option value="directed">优先评审修正后的版本</option><option value="random">固定种子的随机版本顺序</option></select></label>
    </details>
    <button className="button button-secondary" disabled={busy || !sourceRunId} onClick={() => void start()}>重新验证并搜索能力版本</button>
    {error && <p role="alert" className="inline-error">{error}</p>}
    {jobs.length === 0 && <p className="small-copy">暂无符合独立检查合同的学习作业。</p>}
    {jobs.map((job) => <details key={String(job.job_task_id)}>
      <summary>能力学习 · {String(job.status)} · {String(job.chain_sha256).slice(0, 12)}</summary>
      {(job.candidates as Array<Record<string, unknown>>).map((method) => <div key={String(method.skill_id)}>
        <p><strong>{String(method.validation_status) === "PASS" ? "方法已验证" : "方法未通过"}</strong> · {String(method.skill_id)}</p>
        <p className="small-copy">流程经验：{method.procedure_id ? "自动启用" : "未启用"} · 科研有效性：未知</p>
        <details><summary>回归、新案例、反例及入档依据</summary><pre>{JSON.stringify(method, null, 2)}</pre></details>
      </div>)}
      {["INTERRUPTED", "FAILED", "PARTIAL"].includes(String(job.status)) && <button className="text-button" onClick={() => void api.resumeRun(String(job.run_id)).then(load).catch((e: Error) => setError(e.message))}>恢复学习作业</button>}
      <button className="text-button" onClick={() => void api.cancelRun(String(job.run_id)).then(load).catch((e: Error) => setError(e.message))}>中止学习</button>
    </details>)}
    <details><summary>冻结对照评测 · 独立预算</summary>
      <p className="small-copy">比较基础、仅反馈、反馈记忆、单一方法与 QD 方法。测试组不参与学习；数值通过率与科研盲评分开，收益不会自动宣称已证实。</p>
      <label>重复次数<input type="number" min={1} max={5} value={repeats} onChange={(e) => setRepeats(Number(e.target.value))} /></label>
      {["BASE", "FEEDBACK_ONLY", "FEEDBACK_MEMORY", "SINGLE_BEST", "FULL_QD", "NO_TOOL_EVOLUTION", "NO_VALIDATION_GATE"].map((condition) => <label key={condition}><input aria-label={condition} type="checkbox" checked={conditions.includes(condition)} onChange={(e) => setConditions((previous) => e.target.checked ? [...previous, condition] : previous.filter((c) => c !== condition))} />{conditionLabels[condition]}（{condition}）</label>)}
      <label>总模型调用上限<input type="number" min={1} max={200} value={calls} onChange={(e) => setCalls(Number(e.target.value))} /></label>
      <label>总时间上限（秒）<input type="number" min={1} max={7200} value={seconds} onChange={(e) => setSeconds(Number(e.target.value))} /></label>
      <label>总 token 上限<input type="number" min={1} max={1000000} value={tokens} onChange={(e) => setTokens(Number(e.target.value))} /></label>
      <button className="button button-secondary" disabled={busy || conditions.length === 0 || !jobs.some((job) => job.status === "SUCCEEDED")} onClick={() => void benchmark()}>启动冻结对照</button>
      {benchmarks.map((job) => <details key={String(job.job_task_id)}><summary>评测 · {String(job.status)}</summary>
        {job.report != null && <><p className="small-copy">作业结束表示评测已执行；案例通过率与科研有效性需分别判断。以下仅是局部计算检查。</p>
          <table><thead><tr><th>条件</th><th>局部检查通过</th><th>Token 消耗</th></tr></thead><tbody>{Object.entries(((job.report as Record<string, unknown>).metrics ?? {}) as Record<string, Record<string, unknown>>).map(([condition, metric]) => <tr key={condition}><td>{conditionLabels[condition] ?? condition}</td><td>{String(metric.machine_pass)} / {String(metric.cases)}</td><td>{Number(metric.tokens).toLocaleString()}</td></tr>)}</tbody></table></>}
        <details><summary>完整评测记录与停止原因</summary><pre>{JSON.stringify(job.report, null, 2)}</pre></details>
        <details><summary>已保存的独立评审 · {Array.isArray(job.scientific_reviews) ? job.scientific_reviews.length : 0}</summary><pre>{JSON.stringify(job.scientific_reviews ?? [], null, 2)}</pre></details>
        {["INTERRUPTED", "FAILED", "PARTIAL"].includes(String(job.status)) && <button className="text-button" onClick={() => void api.resumeRun(String(job.run_id)).catch((e: Error) => setError(e.message))}>恢复冻结评测</button>}
        {job.report != null && <button className="text-button" onClick={() => void api.blindCapabilityCases(taskId, String(job.job_task_id)).then((r) => { setReviewJob(String(job.job_task_id)); setBlindCases(r.items); setBlindId(String(r.items[0]?.blind_id ?? "")); setNotes(""); setReviewSaved(""); }).catch((e: Error) => setError(e.message))}>查看盲评材料</button>}
      </details>)}
      {blindCases.length > 0 && <form onSubmit={(e) => { e.preventDefault(); void api.reviewCapabilityCase(taskId, reviewJob, { blind_id: blindId, reviewer, scientific_quality: quality, goal_coverage: coverage, severe_errors: severe, repeated_errors: repeated, expert_minutes: minutes, notes }).then(async () => { setNotes(""); setError(""); setReviewSaved("独立评审已保存。"); setBenchmarks((await api.capabilityBenchmarks(taskId)).items); }).catch((err: Error) => setError(err.message)); }}>
        <h3>独立专家盲评</h3><p className="small-copy">仅展示输入与答案，不展示算法条件；评分不会被当作实测器件证据。</p>
        {reviewSaved && <p role="status">{reviewSaved}</p>}
        <label>盲评样本<select value={blindId} onChange={(e) => setBlindId(e.target.value)}>{blindCases.map((c) => <option key={String(c.blind_id)}>{String(c.blind_id)}</option>)}</select></label>
        <pre>{JSON.stringify(blindCases.find((c) => c.blind_id === blindId), null, 2)}</pre>
        <label>评委<input required value={reviewer} onChange={(e) => setReviewer(e.target.value)} /></label>
        <label>科研质量（0–5）<input type="number" min={0} max={5} step={0.1} value={quality} onChange={(e) => setQuality(Number(e.target.value))} /></label>
        <label>任务覆盖率（0–1）<input type="number" min={0} max={1} step={0.01} value={coverage} onChange={(e) => setCoverage(Number(e.target.value))} /></label>
        <label>严重错误数<input type="number" min={0} value={severe} onChange={(e) => setSevere(Number(e.target.value))} /></label>
        <label>重复错误数<input type="number" min={0} value={repeated} onChange={(e) => setRepeated(Number(e.target.value))} /></label>
        <label>专家用时（分钟）<input type="number" min={0} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} /></label>
        <label>评审依据<textarea required value={notes} onChange={(e) => setNotes(e.target.value)} /></label><button className="button button-secondary">保存独立评审</button>
      </form>}
    </details>
  </section>;
}
