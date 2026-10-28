import { useState } from "react";

import type { WorkbenchEvent } from "../../api";

interface Props {
  events: WorkbenchEvent[];
  status: string;
  label: (type: string, payload?: Record<string, unknown>) => string;
  toolResultText: (payload: Record<string, unknown>) => string;
}

const PAGE_SIZE = 25;
const STREAM_FRAGMENTS = new Set(["text_delta", "tool_call_arguments_delta"]);

export function ActivityTimeline({ events, status, label, toolResultText }: Props) {
  const [showAll, setShowAll] = useState(false);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [anchorSeq, setAnchorSeq] = useState<number | null>(null);
  const available = showAll ? events : events.filter((event) => !STREAM_FRAGMENTS.has(event.type));
  const activity = anchorSeq === null ? available : available.filter((event) => event.seq <= anchorSeq);
  const visible = activity.slice(-visibleCount);
  const fragmentCount = events.filter((event) => STREAM_FRAGMENTS.has(event.type)).length;
  const newEventCount = anchorSeq === null ? 0 : events.filter((event) => event.seq > anchorSeq).length;

  return (
    <details className="activity-panel">
      <summary>
        <span className="activity-summary-icon" aria-hidden="true">⌁</span>
        <span><strong>运行过程</strong><small>{events.length ? `${events.length} 条事件 · ${status || "等待中"}` : "尚无运行事件"}</small></span>
        <span className="activity-summary-action">查看过程</span>
      </summary>
      <div className="activity-panel-body">
        {!events.length ? <p className="activity-text">提交本轮需求后，运行过程会显示在这里。</p> : null}
        {!showAll && fragmentCount > 0 ? <p className="activity-fragment-note">已收起 {fragmentCount} 条流式文字或工具参数片段；可以切换到全部事件逐条查看。</p> : null}
        {activity.length > 0 ? (
          <ol className="activity-timeline" aria-label="科研运行进度">
            {visible.map((event) => (
              <li key={event.seq} className="activity-item">
                <span className={`activity-marker marker-${event.type}`} aria-hidden="true">
                  {event.type.startsWith("run_") ? "●" : "·"}
                </span>
                <details className="activity-card">
                  <summary className="activity-card-heading">
                    <strong>{label(event.type, event.payload)}</strong>
                    <span>事件 {event.seq} · 查看详情</span>
                  </summary>
                  {event.type === "tool_started" ? <p className="activity-text">正在调用 {String(event.payload.tool_name ?? "科研工具")}。</p> : null}
                  {event.type === "tool_completed" ? <p className="activity-text">{toolResultText(event.payload)}</p> : null}
                  <pre className="activity-payload">{JSON.stringify(event.payload, null, 2)}</pre>
                </details>
              </li>
            ))}
          </ol>
        ) : null}
        {activity.length > visibleCount ? (
          <button className="text-button activity-load-older" type="button" onClick={() => {
            setAnchorSeq((current) => current ?? available.at(-1)?.seq ?? null);
            setVisibleCount((count) => count + PAGE_SIZE);
          }}>
            查看更早的事件（还有 {activity.length - visibleCount} 条）
          </button>
        ) : null}
        {anchorSeq !== null ? (
          <button className="text-button activity-latest" type="button" onClick={() => { setAnchorSeq(null); setVisibleCount(PAGE_SIZE); }}>
            查看最新事件{newEventCount ? `（${newEventCount} 条新事件）` : ""}
          </button>
        ) : null}
        {fragmentCount > 0 ? (
          <button className="text-button activity-show-all" type="button" onClick={() => { setShowAll((value) => !value); setAnchorSeq(null); setVisibleCount(PAGE_SIZE); }}>
            {showAll ? "只看关键事件" : `查看全部事件（${events.length} 条）`}
          </button>
        ) : null}
      </div>
    </details>
  );
}
