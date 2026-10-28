import { useCallback, useEffect, useState, type FormEvent } from "react";

import { api, type HistorySession, type TaskCreate, type TaskSummary } from "./api";
import { TaskList } from "./features/tasks/TaskList";
import { TaskWorkspace } from "./features/tasks/TaskWorkspace";
import "./styles.css";

type MobilePanel = "tasks" | "workspace" | "details";

export default function App() {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("workspace");
  const [loading, setLoading] = useState(true);
  const [initialLoadAttempt, setInitialLoadAttempt] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistorySession[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [importingSessionId, setImportingSessionId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const refreshTasks = useCallback(async () => {
    const result = await api.tasks();
    setTasks(result.items);
    setSelectedTaskId((current) => {
      const saved = current ?? localStorage.getItem("photomatagent.selectedTask");
      if (saved && result.items.some((task) => task.task_id === saved)) return saved;
      return result.items[0]?.task_id ?? null;
    });
  }, []);

  const retryTaskList = async () => {
    setError(null);
    try {
      await refreshTasks();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务列表刷新失败，请重试。");
    }
  };

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    void api.tasks()
      .then((result) => {
        if (!mounted) return;
        setTasks(result.items);
        const saved = localStorage.getItem("photomatagent.selectedTask");
        setSelectedTaskId(
          saved && result.items.some((task) => task.task_id === saved)
            ? saved
            : result.items[0]?.task_id ?? null,
        );
      })
      .catch((reason: unknown) => {
        if (mounted) setError(reason instanceof Error ? reason.message : "无法连接本地工作台。 ");
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [initialLoadAttempt]);

  const selectTask = (taskId: string) => {
    setSelectedTaskId(taskId);
    localStorage.setItem("photomatagent.selectedTask", taskId);
    setMobilePanel("workspace");
  };

  const createTask = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const body: TaskCreate = {
      title: String(data.get("title") ?? "").trim(),
      goal: String(data.get("goal") ?? "").trim(),
    };
    if (!body.title || !body.goal) return;
    setCreating(true);
    setError(null);
    try {
      const result = await api.createTask(body);
      await refreshTasks();
      selectTask(result.task_id);
      setShowCreate(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务创建失败，请重试。 ");
    } finally {
      setCreating(false);
    }
  };

  const openHistory = async () => {
    setHistoryError(null);
    setHistoryLoading(true);
    setShowHistory(true);
    try {
      const result = await api.history();
      setHistory(result.items);
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : "历史任务暂时无法读取。 ");
    } finally {
      setHistoryLoading(false);
    }
  };

  const importSession = async (sessionId: string) => {
    if (importingSessionId) return;
    setImportingSessionId(sessionId);
    setHistoryError(null);
    try {
      const result = await api.importHistory(sessionId);
      await refreshTasks();
      selectTask(result.task_id);
      setShowHistory(false);
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : "历史任务导入失败。 ");
    } finally {
      setImportingSessionId(null);
    }
  };

  return (
    <div className={`app-shell ${selectedTaskId ? "has-task" : ""}`} data-mobile-panel={mobilePanel}>
      <div className="mobile-panel-switcher" role="tablist" aria-label="工作台面板">
        <button role="tab" aria-selected={mobilePanel === "tasks"} onClick={() => setMobilePanel("tasks")}>任务</button>
        <button role="tab" aria-selected={mobilePanel === "workspace"} onClick={() => setMobilePanel("workspace")}>工作区</button>
        <button role="tab" aria-selected={mobilePanel === "details"} onClick={() => setMobilePanel("details")}>详情</button>
      </div>
      <TaskList
        tasks={tasks}
        selectedTaskId={selectedTaskId}
        onSelect={selectTask}
        onCreate={() => setShowCreate(true)}
        onImportHistory={() => void openHistory()}
        loading={loading}
      />

      {selectedTaskId ? (
        <TaskWorkspace
          key={selectedTaskId}
          taskId={selectedTaskId}
          mobilePanel={mobilePanel}
          onTaskChanged={() => void refreshTasks().catch((reason: unknown) => setError(
            reason instanceof Error ? reason.message : "任务状态刷新失败。",
          ))}
        />
      ) : (
        <main className="welcome-panel" data-workbench-panel="workspace">
          {error ? (
            <div className="inline-error" role="alert">
              {error}
              <button className="text-button" onClick={() => setInitialLoadAttempt((value) => value + 1)}>重试</button>
            </div>
          ) : null}
          <div className="welcome-orbit" aria-hidden="true"><span>◎</span><i>·</i><b>✦</b></div>
          <span className="eyebrow">RESEARCH WORKSPACE</span>
          <h1>让每一轮科研<br />都有迹可循</h1>
          <p>创建一个持续科研任务，将目标、运行过程、资料与结果保存在同一个工作区。</p>
          <button className="button button-primary" onClick={() => setShowCreate(true)}>创建第一个任务 <span aria-hidden="true">↗</span></button>
          <span className="welcome-footnote">本地运行 · 过程可恢复 · 结果保留来源</span>
        </main>
      )}

      {error && selectedTaskId ? (
        <div className="global-toast" role="alert">
          {error}
          <button className="text-button" onClick={() => void retryTaskList()}>重试</button>
          <button onClick={() => setError(null)} aria-label="关闭错误">×</button>
        </div>
      ) : null}

      {showCreate ? (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setShowCreate(false);
        }}>
          <form className="modal-card" role="dialog" aria-modal="true" aria-labelledby="create-title" onSubmit={(event) => void createTask(event)}>
            <div className="modal-topline"><span className="eyebrow">NEW RESEARCH TASK</span><button type="button" className="icon-button" onClick={() => setShowCreate(false)} aria-label="关闭">×</button></div>
            <h2 id="create-title">建立一个持续科研任务</h2>
            <p className="muted">任务目标会贯穿后续运行；具体要求可以在每一轮补充。</p>
            <label htmlFor="task-title">任务名称</label>
            <input id="task-title" name="title" required maxLength={500} placeholder="例如：中波红外探测材料筛选" />
            <label htmlFor="task-goal">研究目标</label>
            <textarea id="task-goal" name="goal" required maxLength={20000} rows={4} placeholder="写下当前已知的研究方向和希望解决的问题；不确定的指标可以注明待定。" />
            <div className="modal-actions"><button type="button" className="button button-subtle" onClick={() => setShowCreate(false)}>返回</button><button className="button button-primary" disabled={creating}>{creating ? "正在创建…" : "创建任务"}</button></div>
          </form>
        </div>
      ) : null}

      {showHistory ? (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setShowHistory(false);
        }}>
          <section className="modal-card" role="dialog" aria-modal="true" aria-labelledby="history-title">
            <div className="modal-topline"><span className="eyebrow">SESSION ARCHIVE</span><button className="icon-button" onClick={() => setShowHistory(false)} aria-label="关闭">×</button></div>
            <h2 id="history-title">导入一段历史会话</h2>
            <p className="muted">历史消息会以只读来源关联到新任务，不会自动升级其中的科学证据。</p>
            {historyError ? <div className="inline-error" role="alert">{historyError}</div> : null}
            {historyLoading ? <p className="muted" role="status">正在读取历史会话…</p> : history.length ? (
              <div className="history-list">
                {history.map((item) => (
                  <button key={item.session_id} className="history-item" disabled={importingSessionId !== null} onClick={() => void importSession(item.session_id)}>
                    <span className="history-icon" aria-hidden="true">◷</span>
                    <span><strong>{item.session_id}</strong><small>{item.provider ?? "未知模型"} · {item.iterations ?? 0} 轮 · {item.tool_calls ?? 0} 次工具调用</small></span>
                    <span>{importingSessionId === item.session_id ? "正在导入…" : "↗"}</span>
                  </button>
                ))}
              </div>
            ) : historyError ? null : <div className="empty-card">当前没有可导入的会话。</div>}
            <div className="modal-actions"><button className="button button-subtle" onClick={() => setShowHistory(false)}>关闭</button></div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
