import { useEffect, useState, type FormEvent } from "react";
import { api, type ResearchTask } from "../../api";

export interface Project {
  project_id: string; name: string; description: string; revision: number;
  archived_at: string | null; tasks: number; tags: string[]; usage?: { known_tokens: number; unsettled_requests: number };
}

export function ProjectControls({ onFilter, refreshKey }: { onFilter: (project: string, query: string, archived: boolean, status: string) => void; refreshKey?: unknown }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState("");
  const [query, setQuery] = useState("");
  const [name, setName] = useState("");
  const [archived, setArchived] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [storage, setStorage] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const load = () => api.information<Project[]>("/projects?archived=true").then((result) => setProjects((result as unknown as { items: Project[] }).items));
  useEffect(() => { void load().catch(() => setError("课题列表暂不可用")); }, [refreshKey]);
  const selected = projects.find((item) => item.project_id === project);
  const create = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim()) return;
    try { await api.information("/projects", { name }); setName(""); await load(); setError(""); }
    catch (reason) { setError(String(reason)); }
  };
  const maintenance = async (operation: string) => {
    setBusy(true); setError("");
    try { setStorage(await api.information(operation === "status" || operation === "diagnostics" ? `/storage/${operation}` : `/storage/${operation}`, operation === "status" || operation === "diagnostics" ? undefined : {})); }
    catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  };
  return <section className="information-panel" aria-label="课题与数据管理">
    <label>研究课题<select aria-label="研究课题" value={project} onChange={(event) => { setProject(event.target.value); onFilter(event.target.value, query, archived, status); }}>
      <option value="">全部课题</option>{projects.map((item) => <option key={item.project_id} value={item.project_id}>{item.name}{item.archived_at ? "（已归档）" : ""} · {item.tasks}</option>)}
    </select></label>
    <form onSubmit={(event) => void create(event)}><input aria-label="新课题名称" placeholder="新课题名称" value={name} onChange={(event) => setName(event.target.value)} /><button type="submit">创建课题</button></form>
    {selected && <p>课题任务 {selected.tasks} · 已知 tokens {selected.usage?.known_tokens ?? 0} · 待结算调用 {selected.usage?.unsettled_requests ?? 0}</p>}
    {selected && <details key={selected.project_id}><summary>课题管理 · {selected.name}</summary><p>{selected.description}</p><input aria-label="修改课题名称" defaultValue={selected.name} onBlur={(event) => { if (event.target.value.trim() && event.target.value !== selected.name) void api.information(`/projects/${selected.project_id}`, { expected_revision: selected.revision, name: event.target.value }, "PATCH").then(load).catch((reason) => setError(String(reason))); }} /><button disabled={selected.project_id === "project_unclassified"} onClick={() => void api.information(`/projects/${selected.project_id}/${selected.archived_at ? "unarchive" : "archive"}`, { expected_revision: selected.revision }).then(load).catch((reason) => setError(String(reason)))}>{selected.archived_at ? "恢复课题" : "归档课题"}</button></details>}
    <form onSubmit={(event) => { event.preventDefault(); onFilter(project, query, archived, status); }}><input aria-label="搜索任务" placeholder="任务关键词" value={query} onChange={(event) => setQuery(event.target.value)} /><button>搜索任务</button></form>
    <select aria-label="任务管理状态" value={status} onChange={(event) => { setStatus(event.target.value); onFilter(project, query, archived, event.target.value); }}><option value="">全部状态</option><option value="OPEN">开展中</option><option value="PAUSED">已暂停</option><option value="DONE">已结束</option></select>
    <label><input type="checkbox" checked={archived} onChange={(event) => { setArchived(event.target.checked); onFilter(project, query, event.target.checked, status); }} />包含归档任务</label>
    <details><summary>研究历史检索</summary><form onSubmit={(event) => { event.preventDefault(); void api.information(`/search?${new URLSearchParams({ q: query, ...(project ? { project_id: project } : {}) })}`).then(setStorage).catch((reason) => setError(String(reason))); }}><input aria-label="搜索研究历史" value={query} onChange={(event) => setQuery(event.target.value)} /><button>检索历史与来源</button></form></details>
    <details><summary>数据管理</summary><div className="information-actions">{["status", "diagnostics", "import", "refresh", "reindex", "export", "backup"].map((operation, index) => <button key={operation} disabled={busy} onClick={() => void maintenance(operation)}>{["存储状态", "检查数据", "导入旧会话", "刷新索引", "重建索引", "导出", "主库备份"][index]}</button>)}</div><p>完整备份及恢复需停止工作台后使用 /data restore --help 查看说明。</p></details>
    {error && <p role="alert">{error}</p>}{storage !== null && <details open><summary>查询结果</summary><pre className="information-result">{JSON.stringify(storage, null, 2)}</pre></details>}
  </section>;
}

export function TaskManagement({ task, onChanged }: { task: ResearchTask; onChanged: () => void }) {
  const [title, setTitle] = useState(task.title);
  const [target, setTarget] = useState("");
  const [tags, setTags] = useState((task.tags ?? []).join(", "));
  const [sourceArtifact, setSourceArtifact] = useState("");
  const [replacedInput, setReplacedInput] = useState("");
  const [projects, setProjects] = useState<Project[]>([]);
  const [checkpoints, setCheckpoints] = useState<unknown>(null);
  const [error, setError] = useState("");
  const mutate = async (path: string, body: unknown, method?: string) => {
    try { await api.information(path, body, method); setError(""); onChanged(); }
    catch (reason) { setError(String(reason)); }
  };
  return <details className="task-objective"><summary>任务管理 · {task.management_status ?? "OPEN"}{task.archived_at ? " · 已归档" : ""}</summary>
    <form onSubmit={(event) => { event.preventDefault(); void mutate(`/tasks/${task.task_id}`, { expected_revision: task.revision, title }, "PATCH"); }}><input aria-label="修改当前任务标题" value={title} onChange={(event) => setTitle(event.target.value)} /><button>保存标题</button></form>
    <label>管理状态<select aria-label="管理状态" value={task.management_status ?? "OPEN"} onChange={(event) => void mutate(`/tasks/${task.task_id}`, { expected_revision: task.revision, management_status: event.target.value }, "PATCH")}><option>OPEN</option><option>PAUSED</option><option>DONE</option></select></label>
    <button onClick={() => void mutate(`/tasks/${task.task_id}/${task.archived_at ? "unarchive" : "archive"}`, { expected_revision: task.revision })}>{task.archived_at ? "恢复任务" : "归档任务"}</button>
    <form onSubmit={(event) => { event.preventDefault(); void mutate(`/tasks/${task.task_id}`, { expected_revision: task.revision, tags: tags.split(/[,，]/).map((tag) => tag.trim()).filter(Boolean) }, "PATCH"); }}><input aria-label="任务标签" value={tags} onChange={(event) => setTags(event.target.value)} placeholder="标签，用逗号分隔" /><button>保存标签</button></form>
    <button onClick={() => void api.information<{ items: Project[] }>("/projects").then((value) => setProjects(value.items)).catch((reason) => setError(String(reason)))}>选择目标课题</button>
    {projects.length > 0 && <><select aria-label="目标课题" value={target} onChange={(event) => setTarget(event.target.value)}><option value="">请选择</option>{projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}</select><button disabled={!target} onClick={() => void mutate(`/tasks/${task.task_id}/move`, { expected_revision: task.revision, project_id: target, target_revision: projects.find((project) => project.project_id === target)?.revision })}>移动任务</button></>}
    <form onSubmit={(event) => { event.preventDefault(); void mutate(`/tasks/${task.task_id}/inputs`, { source_artifact_id: sourceArtifact, ...(replacedInput ? { replaces_artifact_id: replacedInput } : {}) }); }}>
      <input aria-label="引用历史产物" placeholder="来源产物 ID" value={sourceArtifact} onChange={(event) => setSourceArtifact(event.target.value)} />
      <input aria-label="待替换的关联输入" placeholder="失效关联输入 ID（可选）" value={replacedInput} onChange={(event) => setReplacedInput(event.target.value)} />
      <button disabled={!sourceArtifact}>{replacedInput ? "替换失效输入" : "关联为当前任务输入"}</button>
    </form>
    <button onClick={() => void api.information(`/sessions/${task.session_id}/checkpoints`).then(setCheckpoints).catch((reason) => setError(String(reason)))}>查看恢复检查点</button>
    {checkpoints !== null && <pre className="information-result">{JSON.stringify(checkpoints, null, 2)}</pre>}{error && <p role="alert">{error}</p>}
    <p>结束任务属于管理操作，科学结论以独立检查结果为准。</p>
  </details>;
}
