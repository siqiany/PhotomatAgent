import type { TaskSummary } from "../../api";
import { statusLabel } from "../../components/RunStatus";

interface Props {
  tasks: TaskSummary[];
  selectedTaskId: string | null;
  onSelect: (taskId: string) => void;
  onCreate: () => void;
  onImportHistory: () => void;
  loading: boolean;
}

export function TaskList({
  tasks,
  selectedTaskId,
  onSelect,
  onCreate,
  onImportHistory,
  loading,
}: Props) {
  return (
    <aside className="task-sidebar" data-workbench-panel="tasks" aria-label="科研任务列表">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">P</div>
        <div>
          <strong>PhotomatAgent</strong>
          <span>科研工作台</span>
        </div>
      </div>
      <div className="sidebar-actions">
        <button className="button button-primary button-full" onClick={onCreate}>
          <span aria-hidden="true">＋</span> 新建任务
        </button>
        <button className="button button-subtle button-full" onClick={onImportHistory}>
          导入历史任务
        </button>
      </div>
      <div className="sidebar-section-title">
        <span>我的任务</span>
        <span className="task-count">{tasks.length}</span>
      </div>
      <nav className="task-nav" aria-label="任务">
        {loading && tasks.length === 0 ? <p className="muted sidebar-empty">正在加载任务…</p> : null}
        {!loading && tasks.length === 0 ? (
          <div className="empty-card sidebar-empty">
            <span className="empty-icon" aria-hidden="true">⌕</span>
            <p>还没有科研任务</p>
            <span>新建任务后，研究过程会保存在这里。</span>
          </div>
        ) : null}
        {tasks.map((task) => (
          <button
            key={task.task_id}
            className={`task-nav-item ${task.task_id === selectedTaskId ? "selected" : ""}`}
            onClick={() => onSelect(task.task_id)}
            aria-current={task.task_id === selectedTaskId ? "page" : undefined}
          >
            <span className="task-nav-icon" aria-hidden="true">◫</span>
            <span className="task-nav-copy">
              <strong>{task.title}</strong>
              <span>{statusLabel(task.latest_run?.status)}</span>
            </span>
            <span className="nav-chevron" aria-hidden="true">›</span>
          </button>
        ))}
      </nav>
      <footer className="sidebar-footer">
        <span className="connection-dot" /> 本地工作区已连接
      </footer>
    </aside>
  );
}
