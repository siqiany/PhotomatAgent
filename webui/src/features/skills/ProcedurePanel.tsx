import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";

import { api, type Artifact, type Procedure, type ProcedureCreate, type RunRecord } from "../../api";

interface Props {
  taskId: string;
  artifacts: Artifact[];
  runs?: RunRecord[];
  disabled?: boolean;
  planId?: string;
  planRunIds?: string[];
}

const TERMINAL_RUNS = new Set(["SUCCEEDED", "PARTIAL", "FAILED", "CANCELLED", "INTERRUPTED"]);

export function ProcedurePanel({ taskId, artifacts, runs = [], disabled = false, planId, planRunIds = [] }: Props) {
  const [procedures, setProcedures] = useState<Procedure[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [artifactId, setArtifactId] = useState("");
  const [instructions, setInstructions] = useState("");
  const [loadingInstructions, setLoadingInstructions] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [scope, setScope] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, Procedure>>({});
  const [reviewNotes, setReviewNotes] = useState<Record<string, string>>({});

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.procedures();
      setProcedures(result.items);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { void reload(); }, [reload]);

  const eligibleArtifacts = useMemo(() => artifacts.filter((artifact) => {
    if (artifact.task_id !== taskId || artifact.run_id === null
      || !["text/markdown", "text/plain"].includes(artifact.media_type)) return false;
    const run = runs.find((item) => item.run_id === artifact.run_id);
    return runs.length === 0 || Boolean(run && TERMINAL_RUNS.has(run.status));
  }), [artifacts, runs, taskId]);
  const selectedArtifact = eligibleArtifacts.find((artifact) => artifact.artifact_id === artifactId);
  const selectedArtifactId = selectedArtifact?.artifact_id;

  useEffect(() => {
    let current = true;
    setInstructions("");
    if (!selectedArtifactId) return () => { current = false; };
    setLoadingInstructions(true);
    setError(null);
    void fetch(`/api/artifacts/${encodeURIComponent(selectedArtifactId)}/content`)
      .then(async (response) => {
        if (!response.ok) throw new Error("无法读取来源产物，请刷新任务后重试。");
        return response.text();
      })
      .then((text) => {
        if (current) {
          if (text.length > 20_000) throw new Error("来源产物超过 20,000 字符；请使用较短的文本产物。");
          setInstructions(text);
        }
      })
      .catch((reason: unknown) => { if (current) setError(messageOf(reason)); })
      .finally(() => { if (current) setLoadingInstructions(false); });
    return () => { current = false; };
  }, [selectedArtifactId]);

  const create = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setNotice(null);
    if (!selectedArtifact || !selectedArtifact.run_id) {
      setError("请选择当前任务中已完成运行的文本产物。");
      return;
    }
    const slug = name.trim();
    const text = instructions.trim();
    if (!/^[a-z][a-z0-9-]{0,59}$/.test(slug)) {
      setError("名称需为小写 slug：以字母开头，只含小写字母、数字和连字符，最多 60 个字符。");
      return;
    }
    if (!description.trim() || description.trim().length > 500) {
      setError("请填写 1–500 个字符的经验描述。");
      return;
    }
    if (!scope.trim() || scope.trim().length > 200) {
      setError("请填写 1–200 个字符的适用范围。");
      return;
    }
    if (!text || text.length > 20_000) {
      setError("流程说明需为 1–20,000 个字符。");
      return;
    }
    const sourcePlan = planId && planRunIds.includes(selectedArtifact.run_id) ? planId : undefined;
    const payload: ProcedureCreate = {
      name: slug,
      description: description.trim(),
      scope: scope.trim(),
      instructions: text,
      task_id: taskId,
      run_id: selectedArtifact.run_id,
      artifact_id: selectedArtifact.artifact_id,
      ...(sourcePlan ? { plan_id: sourcePlan } : {}),
    };
    setBusy(true);
    try {
      const created = await api.createProcedure(payload);
      setProcedures((current) => [...current.filter((item) => item.skill_id !== created.skill_id), created]);
      setDetails((current) => ({ ...current, [created.skill_id]: created }));
      setExpandedId(created.skill_id);
      setReviewNotes((current) => ({ ...current, [created.skill_id]: "" }));
      setNotice("候选已保存，状态仍为 UNKNOWN；请逐条审阅流程说明后再决定是否批准。");
      setName("");
      setDescription("");
      setScope("");
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  };

  const expand = async (procedure: Procedure) => {
    if (expandedId === procedure.skill_id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(procedure.skill_id);
    if (details[procedure.skill_id]) return;
    try {
      const result = await api.procedure(procedure.skill_id);
      setDetails((current) => ({ ...current, [procedure.skill_id]: result }));
    } catch (reason) {
      setError(messageOf(reason));
    }
  };

  const approve = async (procedure: Procedure) => {
    const note = (reviewNotes[procedure.skill_id] ?? "").trim();
    if (!note) {
      setError("填写审阅说明后，才能批准此流程经验。");
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const approved = await api.approveProcedure(procedure.skill_id, note);
      updateProcedure(approved);
      setNotice("经验已批准，可供后续任务检索。它仍是 UNKNOWN 的指导，不是验证通过或执行有效性的证明。");
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  };

  const disable = async (procedure: Procedure) => {
    setBusy(true);
    setError(null);
    try {
      const updated = await api.disableProcedure(procedure.skill_id);
      updateProcedure(updated);
      setNotice("流程经验已停用；此前记录保留。");
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  };

  const updateProcedure = (updated: Procedure) => {
    setProcedures((current) => current.map((item) => item.skill_id === updated.skill_id ? updated : item));
    setDetails((current) => ({ ...current, [updated.skill_id]: updated }));
  };
  return (
    <details className="inspector-card skill-card procedure-card" aria-label="科研流程经验">
      <summary className="card-heading compact-heading">
        <div><span className="eyebrow">APPROVED GUIDANCE</span><h2>科研流程经验</h2></div>
        <span className="pill-count">{procedures.length}</span>
      </summary>
      <p className="small-copy">从本任务运行产物中整理可复用的文字流程。候选和已批准内容都保持 UNKNOWN；批准仅允许后续任务检索，不代表科学验证或方法有效。</p>
      {error ? <div className="inline-error skill-feedback" role="alert">{error}</div> : null}
      {notice ? <div className="inline-notice skill-feedback" role="status">{notice}</div> : null}
      {loading ? <p className="muted small-copy" role="status">正在读取流程经验…</p> : null}
      {!loading && procedures.length === 0 ? <p className="muted small-copy">还没有流程经验记录。</p> : null}
      <div className="skill-list">
        {procedures.map((procedure) => {
          const detail = details[procedure.skill_id];
          return (
            <article className="revision-plan skill-item procedure-item" key={procedure.skill_id}>
              <button type="button" className="revision-section-heading" style={{ width: "100%", padding: 0, border: 0, background: "transparent", color: "inherit", textAlign: "left" }} aria-expanded={expandedId === procedure.skill_id} onClick={() => void expand(procedure)}>
                <span className="skill-item-copy"><strong>{procedure.name}</strong><span>{procedure.description}</span></span>
                <span className={`revision-plan-state${procedure.status === "DISABLED" ? " state-unavailable" : ""}`}>{procedureStatus(procedure.status)}</span>
              </button>
              <p className="revision-muted">验证状态：UNKNOWN · 检索 {procedure.retrieval_count ?? 0} 次（仅表示被检索，不代表有效性）</p>
              {expandedId === procedure.skill_id ? (
                <div className="skill-detail">
                  {procedure.scope ? <p><strong>适用范围</strong>{procedure.scope}</p> : null}
                  <p><strong>检索索引</strong><code>{procedure.index_name}</code></p>
                  {detail?.instructions ? <pre className="procedure-instructions">{detail.instructions}</pre> : <p className="revision-muted">正在读取已保存流程说明…</p>}
                  {procedure.review_note ? <p><strong>审阅记录：</strong>{procedure.review_note}</p> : null}
                  {procedure.status === "CANDIDATE" ? <>
                    <label>审阅说明<textarea aria-label="审阅说明" rows={3} value={reviewNotes[procedure.skill_id] ?? ""} onChange={(event) => setReviewNotes((current) => ({ ...current, [procedure.skill_id]: event.target.value }))} disabled={disabled || busy} placeholder="说明检查了哪些步骤、适用边界和仍需核验之处。" /></label>
                    <button type="button" className="button button-primary" disabled={disabled || busy || !(reviewNotes[procedure.skill_id] ?? "").trim()} onClick={() => void approve(procedure)}>批准并供后续任务检索</button>
                  </> : null}
                  {procedure.status !== "DISABLED" ? <button type="button" className="button button-secondary" disabled={disabled || busy} onClick={() => void disable(procedure)}>停用流程经验</button> : null}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
      <form className="revision-feedback-form procedure-create-form" aria-label="保存科研流程经验候选" onSubmit={(event) => void create(event)}>
        <label>经验来源产物<select aria-label="经验来源产物" value={artifactId} onChange={(event) => setArtifactId(event.target.value)} disabled={disabled || busy}>
          <option value="">选择当前任务的运行产物</option>
          {eligibleArtifacts.map((artifact) => <option key={artifact.artifact_id} value={artifact.artifact_id}>{artifact.filename} · {artifact.run_id?.slice(-8)}</option>)}
        </select></label>
        {!eligibleArtifacts.length ? <p className="revision-warning">本任务没有可用的运行产物；仅接受任务内已结束运行生成的 Markdown 或文本文件，不包括上传附件。</p> : null}
        {selectedArtifact ? <>
          <label>名称 slug<input aria-label="名称 slug" value={name} onChange={(event) => setName(event.target.value)} maxLength={60} disabled={disabled || busy} placeholder="例如 ftir-baseline" /></label>
          <label>经验描述<input aria-label="经验描述" value={description} onChange={(event) => setDescription(event.target.value)} maxLength={500} disabled={disabled || busy} /></label>
          <label>适用范围<input aria-label="适用范围" value={scope} onChange={(event) => setScope(event.target.value)} maxLength={200} disabled={disabled || busy} /></label>
          <label>流程说明（可编辑）<textarea aria-label="流程说明" rows={8} maxLength={20_000} value={instructions} onChange={(event) => setInstructions(event.target.value)} disabled={disabled || busy || loadingInstructions} /></label>
          <p className="revision-muted">{loadingInstructions ? "正在载入来源文本…" : `${instructions.length.toLocaleString()} / 20,000 字符`} · 保存后为 UNKNOWN 候选，需审阅。</p>
          <button type="submit" className="button button-secondary" disabled={disabled || busy || loadingInstructions || !instructions.trim()}>{busy ? "正在保存…" : "保存为候选"}</button>
        </> : null}
      </form>
      <p className="revision-muted">获批经验会在后续任务中由模型通过 tool_search / skill_view 检索。检索次数不代表执行成功或结论有效。</p>
    </details>
  );
}

function procedureStatus(status: Procedure["status"]): string {
  return ({ CANDIDATE: "候选 · 待审阅", APPROVED_PROCEDURE: "已批准 · 可检索", DISABLED: "已停用" })[status];
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : "流程经验操作失败，请稍后重试。";
}
