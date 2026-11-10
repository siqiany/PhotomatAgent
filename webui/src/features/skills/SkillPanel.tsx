import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import { api, type Artifact, type LocalSkill, type RunRef, type SkillCheckerSpec, type SkillExecutionSource, type SkillSourceRef, type SkillUse } from "../../api";

interface Props {
  taskId: string;
  artifacts: Artifact[];
  disabled?: boolean;
  refreshKey?: string;
  onRunQueued?: (reference: RunRef) => void;
  onChanged?: () => void;
}

type CheckerKind = "none" | SkillCheckerSpec["checker_id"];

export function SkillPanel({ taskId, artifacts, disabled = false, refreshKey = "", onRunQueued, onChanged }: Props) {
  const [skills, setSkills] = useState<LocalSkill[]>([]);
  const [sources, setSources] = useState<SkillExecutionSource[]>([]);
  const [invalidatedSources, setInvalidatedSources] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState("");
  const [name, setName] = useState("");
  const [parentSkill, setParentSkill] = useState("");
  const [configurations, setConfigurations] = useState<Record<string, string>>({});
  const [description, setDescription] = useState("");
  const [scope, setScope] = useState("");
  const [checkerKind, setCheckerKind] = useState<CheckerKind>("none");
  const [inputKey, setInputKey] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [field, setField] = useState("");
  const [tolerance, setTolerance] = useState("1e-9");
  const [minimum, setMinimum] = useState("");
  const [maximum, setMaximum] = useState("");
  const [units, setUnits] = useState("unspecified");
  const [expandedSkill, setExpandedSkill] = useState<string | null>(null);
  const [inputSelections, setInputSelections] = useState<Record<string, string>>({});
  const latestReload = useRef(0);

  const reload = useCallback(async () => {
    const requestId = ++latestReload.current;
    setLoading(true);
    setError(null);
    try {
      const [skillResult, sourceResult] = await Promise.all([api.skills(), api.skillSources()]);
      if (requestId !== latestReload.current) return;
      setSkills(skillResult.items);
      const taskSources = sourceResult.items.filter((item) => item.task_id === taskId);
      setSources(taskSources);
      setInvalidatedSources(sourceResult.invalidated_by_task?.[taskId] ?? 0);
      setSelectedSource((current) => taskSources.some((item) => item.execution_id === current)
        ? current
        : taskSources.at(-1)?.execution_id ?? "");
    } catch (reason) {
      if (requestId === latestReload.current) setError(messageOf(reason));
    } finally {
      if (requestId === latestReload.current) setLoading(false);
    }
  }, [taskId]);

  useEffect(() => { void reload(); }, [reload, refreshKey]);

  useEffect(() => {
    if (!disabled) return;
    const timer = window.setInterval(() => {
      void api.skills().then((result) => setSkills(result.items)).catch(() => undefined);
    }, 2500);
    return () => window.clearInterval(timer);
  }, [disabled]);

  const source = sources.find((item) => item.execution_id === selectedSource);
  const currentArtifacts = useMemo(() => artifacts.filter((artifact) => artifact.task_id === taskId), [artifacts, taskId]);

  const buildChecker = (): SkillCheckerSpec | null => {
    if (checkerKind === "none") return null;
    if (checkerKind === "rc_network" || checkerKind === "report_tables" || checkerKind === "fresnel_film") {
      if (!source?.input_keys.includes("data") || !source.artifacts.includes("result.json") || (checkerKind === "report_tables" && !source.input_keys.includes("report"))) {
        throw new Error("独立检查要求 data 输入、result.json 产物；表格检查还需 report 输入。");
      }
      return { checker_id: checkerKind, input_key: "data", output_path: "result.json", field: { rc_network: "samples", report_tables: "mismatches", fresnel_film: "R" }[checkerKind] };
    }
    if (checkerKind === "mwir_noise_budget") {
      if (!source?.input_keys.includes("data") || !source.artifacts.includes("result.json")) {
        throw new Error("MWIR 噪声预算检查要求来源执行包含 data 输入和 result.json 产物。");
      }
      return { checker_id: checkerKind, input_key: "data", output_path: "result.json", field: "detectivity_jones" };
    }
    if (!outputPath.trim() || !field.trim()) throw new Error("请填写检查产物和 JSON 字段。");
    if (checkerKind === "json_numeric_sum") {
      if (!inputKey) throw new Error("请选择求和输入项。");
      const parsedTolerance = Number(tolerance);
      if (!Number.isFinite(parsedTolerance) || parsedTolerance < 0 || parsedTolerance > 0.001) {
        throw new Error("容差需为 0 到 0.001 之间的有限数值。");
      }
      return { checker_id: checkerKind, input_key: inputKey, output_path: outputPath, field: field.trim(), tolerance: parsedTolerance };
    }
    const parsedMinimum = minimum.trim() ? Number(minimum) : undefined;
    const parsedMaximum = maximum.trim() ? Number(maximum) : undefined;
    if (parsedMinimum === undefined && parsedMaximum === undefined) throw new Error("至少设置一个数值边界。");
    if ((parsedMinimum !== undefined && !Number.isFinite(parsedMinimum)) || (parsedMaximum !== undefined && !Number.isFinite(parsedMaximum)) || (parsedMinimum !== undefined && parsedMaximum !== undefined && parsedMinimum > parsedMaximum)) {
      throw new Error("数值边界无效，请检查上下限。");
    }
    return { checker_id: checkerKind, output_path: outputPath, field: field.trim(), minimum: parsedMinimum, maximum: parsedMaximum, units: units.trim() || "unspecified" };
  };

  const create = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!source) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const created = await api.createSkill({
        name: name.trim(), description: description.trim(), scope: scope.trim(),
        source: sourceReference(source),
        checker: buildChecker(),
        ...(parentSkill ? { parent_skill_id: parentSkill } : {}),
      });
      await reload();
      setExpandedSkill(created.skill_id);
      setNotice(created.source_evidence?.status === "PASS"
        ? "候选已保存。请在目标任务绑定新输入并回放，通过后才可作为本地试用方法。"
        : "候选已保存，但独立检查为 UNKNOWN；补充检查证据后才能回放。没有检查器不代表通过。");
      setName("");
      setDescription("");
      setScope("");
      setParentSkill("");
      onChanged?.();
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  };

  const replay = async (skill: LocalSkill) => {
    const keys = skill.input_keys ?? Object.keys(skill.request?.input_refs ?? {});
    const refs: Record<string, string> = {};
    for (const key of keys) {
      const artifactId = inputSelections[`${skill.skill_id}:${key}`];
      if (!artifactId) {
        if (skill.source?.task_id !== taskId) {
          setError(`请为输入“${key}”选择当前任务中的资料。`);
          return;
        }
        continue;
      }
      refs[key] = `artifact:${artifactId}`;
    }
    if (skill.source?.task_id !== taskId && keys.length > 0 && Object.keys(refs).length !== keys.length) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const reference = await api.replaySkill(skill.skill_id, {
        task_id: taskId,
        scope: skill.scope ?? "",
        ...(configurations[skill.skill_id] ? { argv: JSON.parse(configurations[skill.skill_id]) as string[] } : {}),
        ...(Object.keys(refs).length ? { input_refs: refs } : {}),
      });
      onRunQueued?.(reference);
      setNotice(`技能回放已进入后端队列（${reference.run_id}）。运行和检查记录会随本轮过程更新。`);
      await reload();
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  };

  const disableSkill = async (skill: LocalSkill) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await api.disableSkill(skill.skill_id);
      await reload();
      setNotice("技能已停用；既有回放和失败记录仍保留。");
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  };

  const enableSkill = async (skill: LocalSkill) => {
    setBusy(true);
    setError(null);
    try {
      await api.enableSkill(skill.skill_id);
      await reload();
      setNotice("已恢复此固定版本；科学有效性与跨任务收益仍分别判断。");
      onChanged?.();
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="inspector-card skill-card" aria-label="技能复用">
      <div className="card-heading compact-heading">
        <div>
          <span className="eyebrow">LOCAL METHODS</span>
          <h2>技能复用</h2>
        </div>
        <span className="pill-count">{skills.length}</span>
      </div>
      <p className="small-copy">从真实执行记录保存方法候选，在当前任务选择输入并回放。检查只说明规则是否满足，不验证材料科学假设。</p>
      {error ? <div className="inline-error skill-feedback" role="alert">{error}</div> : null}
      {notice ? <div className="inline-notice skill-feedback" role="status">{notice}</div> : null}
      {loading ? <p className="muted small-copy" role="status">正在读取技能记录…</p> : null}
      {!loading && skills.length === 0 ? <p className="muted small-copy">还没有技能候选。完成带有真实脚本执行的科研运行后，可以从执行记录创建候选。</p> : null}

      <div className="skill-list">
        {skills.map((skill) => {
          const evidence = skill.source_evidence?.status ?? "UNKNOWN";
          const canReplay = evidence === "PASS" && !["DISABLED", "REVALIDATION_REQUIRED"].includes(skill.status);
          const inputKeys = skill.input_keys ?? Object.keys(skill.request?.input_refs ?? {});
          return (
            <article className="revision-plan skill-item" key={skill.skill_id}>
              <button className="revision-section-heading" style={{ width: "100%", padding: 0, border: 0, background: "transparent", color: "inherit", textAlign: "left" }} aria-expanded={expandedSkill === skill.skill_id} onClick={() => setExpandedSkill((current) => current === skill.skill_id ? null : skill.skill_id)}>
                <span className="skill-item-copy">
                  <strong>{skill.name ?? "未命名方法"}</strong>
                  <span>{skill.description ?? "本地方法候选"}</span>
                </span>
                <span className={`revision-plan-state${skill.status === "REVALIDATION_REQUIRED" || skill.status === "DISABLED" ? " state-unavailable" : ""}`}>{skillStatusLabel(skill.status)}</span>
              </button>
              {expandedSkill === skill.skill_id ? (
                <div className="skill-detail">
                  {skill.scope ? <p><strong>适用范围</strong>{skill.scope}</p> : null}
                  {skill.interface ? <><label>冻结参数配置<select aria-label="冻结参数配置" value={configurations[skill.skill_id] ?? JSON.stringify(skill.interface.argv_choices[0])} onChange={(event) => setConfigurations((previous) => ({ ...previous, [skill.skill_id]: event.target.value }))}>{skill.interface.argv_choices.map((argv) => <option key={JSON.stringify(argv)} value={JSON.stringify(argv)}>{argv.length ? argv.join(" ") : "默认参数"}</option>)}</select></label><details><summary>输入输出接口、单位与边界</summary><pre>{JSON.stringify(skill.interface, null, 2)}</pre></details></> : null}
                  {skill.version ? <p><strong>固定版本</strong>{skill.version}<br />{skill.parent_skill_id ? `父版本：${skill.parent_skill_id}` : "首个版本"}<br />{skill.contract_sha256 ? `合同摘要：${skill.contract_sha256}` : ""}</p> : null}
                  <p><strong>源检查</strong><CheckerBadge status={evidence} />{skill.source_evidence?.gap ? ` · ${skill.source_evidence.gap}` : ""}</p>
                  {skill.source_evidence?.limitations?.map((limitation) => <p className="revision-warning" key={limitation}>{limitation}</p>)}
                  {skill.status === "REVALIDATION_REQUIRED" ? <p className="revision-warning">技能文件或合同摘要已变化，需重新创建候选并检查。</p> : null}
                  {inputKeys.length > 0 && skill.status !== "DISABLED" && skill.status !== "REVALIDATION_REQUIRED" ? (
                    <fieldset className="revision-artifact-picker" disabled={disabled || busy || !canReplay}>
                      <legend>回放输入</legend>
                      {inputKeys.map((key) => (
                        <label key={key}>{key}
                          <select
                            aria-label={`技能输入 ${key}`}
                            value={inputSelections[`${skill.skill_id}:${key}`] ?? ""}
                            onChange={(event) => setInputSelections((current) => ({ ...current, [`${skill.skill_id}:${key}`]: event.target.value }))}
                          >
                            <option value="">{skill.source?.task_id === taskId ? "使用候选中的原始输入" : "选择当前任务资料"}</option>
                            {currentArtifacts.map((artifact) => <option value={artifact.artifact_id} key={artifact.artifact_id}>{artifact.filename}</option>)}
                          </select>
                        </label>
                      ))}
                      <small>跨任务回放必须选择当前任务资料；附件仅通过任务内 ID 绑定。</small>
                    </fieldset>
                  ) : null}
                  <div className="revision-plan-actions">
                    {canReplay ? <button className="button button-secondary" disabled={disabled || busy} onClick={() => void replay(skill)}>回放到当前任务</button> : null}
                    {skill.status !== "DISABLED" && skill.status !== "REVALIDATION_REQUIRED" ? <button className="text-button danger-text" disabled={disabled || busy} onClick={() => void disableSkill(skill)}>停用</button> : null}
                    {skill.status === "DISABLED" ? <button className="button button-secondary" disabled={disabled || busy} onClick={() => void enableSkill(skill)}>恢复此版本</button> : null}
                  </div>
                  <div className="revision-manual-note" aria-label="回放历史">
                    <strong>回放与复用记录 · {skill.uses.length}</strong>
                    {skill.uses.length === 0 ? <span>尚无回放记录。</span> : skill.uses.map((use, index) => <UseSummary key={use.use_id ?? `${use.task_id}:${index}`} use={use} />)}
                  </div>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>

      <details className="budget-controls skill-create-details">
        <summary>从本任务执行记录创建候选</summary>
        {invalidatedSources > 0 && <p className="muted small-copy" role="status">{invalidatedSources} 条来源执行已失效。请在处理反馈后重新运行脚本，再选择新的来源执行。</p>}
        {sources.length === 0 ? !loading && invalidatedSources === 0 && <p className="muted small-copy">本任务还没有可用的真实脚本执行记录。</p> : (
          <form className="revision-feedback-form skill-create-form" onSubmit={(event) => void create(event)}>
            <label>来源执行<select aria-label="来源执行" value={selectedSource} onChange={(event) => {
              setSelectedSource(event.target.value);
              const next = sources.find((item) => item.execution_id === event.target.value);
              setInputKey(next?.input_keys[0] ?? "");
              setOutputPath(next?.artifacts[0] ?? "");
            }}>
              {sources.map((item) => <option key={item.execution_id} value={item.execution_id}>{(item.trial_id ?? item.run_id ?? item.execution_id).slice(-8)} · {item.script_path.split("/").at(-1)} · {item.state ?? "已记录"}</option>)}
            </select></label>
            <label>技能名称（英文小写、数字、连字符）<input aria-label="技能名称" required pattern="[a-z][a-z0-9-]{1,60}" maxLength={61} value={name} onChange={(event) => setName(event.target.value)} placeholder="numeric-sum" /></label>
            <label>父版本（可选）<select aria-label="父版本" value={parentSkill} onChange={(event) => {
              setParentSkill(event.target.value);
              const parent = skills.find((item) => item.skill_id === event.target.value);
              if (parent?.logical_name) setName(parent.logical_name);
            }}>
              <option value="">创建新方法</option>
              {skills.filter((item) => item.status !== "REVALIDATION_REQUIRED").map((item) => <option key={item.skill_id} value={item.skill_id}>{item.name} · {item.skill_id.slice(-8)}</option>)}
            </select></label>
            <label>方法说明<input aria-label="方法说明" required maxLength={500} value={description} onChange={(event) => setDescription(event.target.value)} placeholder="说明脚本处理步骤" /></label>
            <label>适用范围<input aria-label="适用范围" required maxLength={200} value={scope} onChange={(event) => setScope(event.target.value)} placeholder="例如：结构化数值数据的汇总" /></label>
            <label>独立规则检查<select aria-label="独立规则检查" value={checkerKind} onChange={(event) => {
              const next = event.target.value as CheckerKind;
              setCheckerKind(next);
              if (next === "json_numeric_sum") setInputKey(source?.input_keys[0] ?? "");
              if (next === "mwir_noise_budget") {
                setInputKey("data");
                setOutputPath("result.json");
                setField("detectivity_jones");
              }
              if (next !== "none" && next !== "mwir_noise_budget") setOutputPath(source?.artifacts[0] ?? "");
            }}>
              <option value="none">不设置（证据为 UNKNOWN，不能回放）</option>
              <option value="json_numeric_sum">JSON 数组求和检查</option>
              <option value="json_numeric_bounds">JSON 数值范围检查</option>
              <option value="mwir_noise_budget">中红外噪声预算核验</option>
              <option value="rc_network">被动 RC 电路独立节点求解</option>
              <option value="report_tables">报告本体逐格核验</option>
              <option value="fresnel_film">无损单层膜正入射反射率</option>
            </select></label>
            {checkerKind !== "none" ? <>
              {checkerKind === "rc_network" || checkerKind === "report_tables" || checkerKind === "fresnel_film" ? <>
                <p className="small-copy">固定绑定 data 输入和 result.json 输出。{{ rc_network: "使用独立复数 KCL 节点矩阵核对各频点跨阻，支持最多 8 个节点的被动 RC 网络。", report_tables: "还需 report 输入。直接读取 Markdown 报告表格，核对列、行与每个值。", fresnel_film: "使用独立特征矩阵核对无损单层膜的正入射反射率 R。" }[checkerKind]}通过仅表示数值合同一致，科研结论仍需证据。</p>
              </> : checkerKind === "mwir_noise_budget" ? <>
                <p className="small-copy">固定读取 data 输入及 result.json，核验面积换算、NEP 与比探测率计算。shot_only 会标为噪声证据不足；科研结论保持 UNKNOWN。</p>
                {(!source?.input_keys.includes("data") || !source.artifacts.includes("result.json"))
                  ? <p className="revision-warning">当前来源必须实际执行并记录 data 输入和 result.json 产物，才能使用此检查器。</p>
                  : <p className="small-copy">输入绑定：data · 输出绑定：result.json · 输出字段：area_cm2、nep_w_sqrt_hz、detectivity_jones、assessment、scientific_status</p>}
              </> : <>
              <label>输出产物<select aria-label="检查输出产物" value={outputPath} required onChange={(event) => setOutputPath(event.target.value)}>
                {source?.artifacts.map((path) => <option key={path} value={path}>{path}</option>)}
                {!source?.artifacts.length ? <option value="">没有可检查产物</option> : null}
              </select></label>
              <label>JSON 字段<input aria-label="JSON 字段" required value={field} onChange={(event) => setField(event.target.value)} placeholder="sum" /></label>
              {checkerKind === "json_numeric_sum" ? <>
                <label>对应输入项<select aria-label="求和输入项" value={inputKey} required onChange={(event) => setInputKey(event.target.value)}>
                  {source?.input_keys.map((key) => <option key={key} value={key}>{key}</option>)}
                  {!source?.input_keys.length ? <option value="">没有输入项</option> : null}
                </select></label>
                <label>绝对容差<input aria-label="绝对容差" type="number" min={0} max={0.001} step="any" value={tolerance} onChange={(event) => setTolerance(event.target.value)} /></label>
              </> : <>
                <label>允许最小值（可空）<input aria-label="允许最小值" type="number" step="any" value={minimum} onChange={(event) => setMinimum(event.target.value)} /></label>
                <label>允许最大值（可空）<input aria-label="允许最大值" type="number" step="any" value={maximum} onChange={(event) => setMaximum(event.target.value)} /></label>
                <label>单位<input aria-label="检查单位" value={units} onChange={(event) => setUnits(event.target.value)} placeholder="unspecified" /></label>
                <p className="revision-warning">边界由操作者预先冻结；范围检查不验证科学假设。</p>
              </>}
              </>}
            </> : null}
            <button className="button button-primary" type="submit" disabled={disabled || busy || !source || (checkerKind === "mwir_noise_budget" && (!source.input_keys.includes("data") || !source.artifacts.includes("result.json")))}>{busy ? "正在保存…" : "保存候选"}</button>
          </form>
        )}
      </details>
      <p className="revision-manual-note">本地“可试用”不等于 QD 发布成熟度。失败尝试与产物会保留，停用不会删除历史记录。</p>
    </section>
  );
}

function UseSummary({ use }: { use: SkillUse }) {
  const result = use.checker?.status ?? "UNKNOWN";
  return (
    <div className="skill-use-row">
      <span>{use.code === "OK" ? "脚本已运行" : `脚本状态：${use.code}`}</span>
      <CheckerBadge status={result} />
      {use.artifacts.length ? <span>产物 {use.artifacts.join("、")}</span> : null}
      {use.checker?.gap ? <span>{use.checker.gap}</span> : null}
    </div>
  );
}

function CheckerBadge({ status }: { status: string }) {
  const normalized = status === "PASS" || status === "FAIL" ? status : "UNKNOWN";
  const label = normalized === "PASS" ? "检查通过" : normalized === "FAIL" ? "检查未通过" : "检查未知";
  return <span className={`skill-checker skill-checker-${normalized.toLowerCase()}`} aria-label={normalized} title={normalized}>{label}</span>;
}

function skillStatusLabel(status: LocalSkill["status"]): string {
  const labels: Record<LocalSkill["status"], string> = {
    CANDIDATE: "候选",
    TRIALABLE: "本地可试用",
    REUSED: "已有跨任务复用",
    DISABLED: "已停用",
    REVALIDATION_REQUIRED: "需重新验证",
  };
  return labels[status];
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : "技能操作暂时失败，请稍后重试。";
}

function sourceReference(source: SkillExecutionSource): SkillSourceRef {
  if (source.kind === "qd_trial") {
    if (!source.trial_id) throw new Error("QD 来源缺少 trial ID，无法创建候选。");
    return { kind: "qd_trial", task_id: source.task_id, trial_id: source.trial_id, execution_id: source.execution_id };
  }
  if (!source.run_id) throw new Error("普通运行来源缺少 run ID，无法创建候选。");
  return { kind: "ordinary_run", task_id: source.task_id, run_id: source.run_id, execution_id: source.execution_id };
}
