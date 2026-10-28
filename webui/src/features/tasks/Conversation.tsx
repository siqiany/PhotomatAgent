import { useMemo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { RunRecord, WorkbenchEvent } from "../../api";
import { isRunActive, statusLabel } from "../../components/RunStatus";
import { repliesForRun, stopExplanation } from "./conversation";

export function Conversation({ runs, events, activeRunId, activeStatus, onInspect }: {
  runs: RunRecord[]; events: WorkbenchEvent[]; activeRunId: string | null;
  activeStatus: string; onInspect: (runId: string) => void;
}) {
  const grouped = useMemo(() => {
    const byRun = new Map<string, WorkbenchEvent[]>();
    for (const event of events) {
      const group = byRun.get(event.run_id) ?? [];
      group.push(event); byRun.set(event.run_id, group);
    }
    return byRun;
  }, [events]);
  return <section className="conversation" aria-label="任务对话">
    {runs.map((run) => {
      const runEvents = grouped.get(run.run_id) ?? [];
      const replies = repliesForRun(runEvents, run.run_id);
      const final = replies.filter((reply) => reply.final).at(-1);
      const progress = replies.filter((reply) => !reply.final);
      const status = run.run_id === activeRunId ? activeStatus : run.status;
      const active = isRunActive(status);
      const stopped = stopExplanation(runEvents, run.run_id);
      return <article className="conversation-turn" key={run.run_id}>
        <div className="conversation-user">
          {run.request.text.length > 1200 ? <>
            <p className="conversation-request-preview">{run.request.text.slice(0, 300)}…</p>
            <details className="conversation-request-details">
              <summary>展开完整需求</summary><p>{run.request.text}</p>
            </details>
          </> : <p>{run.request.text}</p>}
        </div>
        <div className="conversation-assistant">
          <span className="conversation-avatar" aria-hidden="true">P</span>
          <div className="assistant-body">
            <div className="assistant-heading"><strong>PhotomatAgent</strong><span>{statusLabel(status)}</span></div>
            {progress.length ? <details className="research-progress"><summary>{active ? "正在研究" : "查看研究过程"} · {progress.length} 段记录</summary>
              {progress.map((reply, i) => <div className="markdown-body progress-reply" key={i}><Markdown remarkPlugins={[remarkGfm]}>{reply.text}</Markdown></div>)}
            </details> : null}
            {final ? <div className="markdown-body"><Markdown remarkPlugins={[remarkGfm]}>{final.text}</Markdown></div> :
              <p className="reply-placeholder">{active ? "正在处理你的问题…" : "本轮没有完成最终回复。已有研究过程与产物可以查看。"}</p>}
            {stopped ? <div className="stop-explanation" role="status">{stopped}</div> : null}
            <button type="button" className="text-button turn-inspect" onClick={() => onInspect(run.run_id)}>查看本轮详情</button>
          </div>
        </div>
      </article>;
    })}
  </section>;
}
