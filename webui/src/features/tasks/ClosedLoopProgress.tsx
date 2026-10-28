import type { WorkbenchEvent } from "../../api";

const TERMINAL_LABELS: Record<string, string> = {
  DELIVERY_COMPLETE: "交付检查完成",
  NEEDS_EVIDENCE: "仍需证据",
  STALLED: "多轮无进展，已停止",
  ROUND_LIMIT: "达到轮数上限",
  BUDGET_EXHAUSTED: "达到运行预算",
  WALL_TIME_EXHAUSTED: "达到运行时间上限",
  EXECUTION_STOPPED: "运行已停止",
};

const STATUS_LABELS: Record<string, string> = { PASS: "通过", FAIL: "未通过", UNKNOWN: "未知" };

export function ClosedLoopProgress({ events }: { events: WorkbenchEvent[] }) {
  const relevant = events.filter((event) => [
    "closed_loop_round_started", "closed_loop_checked", "closed_loop_completed",
    "evolution_guidance_retrieved", "evolution_observation", "evolution_candidate_captured",
  ].includes(event.type));
  if (!relevant.length) return null;

  const completion = [...relevant].reverse().find((event) => event.type === "closed_loop_completed");
  const rounds = new Map<number, { phase?: string; check?: WorkbenchEvent }>();
  for (const event of relevant) {
    if (event.type !== "closed_loop_round_started" && event.type !== "closed_loop_checked") continue;
    const roundIndex = Number(event.payload.round_index);
    if (!Number.isInteger(roundIndex) || roundIndex < 1) continue;
    const current = rounds.get(roundIndex) ?? {};
    if (event.type === "closed_loop_round_started") current.phase = stringValue(event.payload.phase);
    else current.check = event;
    rounds.set(roundIndex, current);
  }

  return (
    <details className="closed-loop-progress">
      <summary>自动修订进度 <span>{completion ? TERMINAL_LABELS[stringValue(completion.payload.status) ?? ""] ?? "运行结束" : `${rounds.size} 轮已记录`}</span></summary>
      <div className="closed-loop-body">
        {[...rounds.entries()].sort(([left], [right]) => left - right).map(([roundIndex, round]) => (
          <section className="closed-loop-round" key={roundIndex}>
            <strong>第 {roundIndex} 轮{round.phase ? ` · ${phaseLabel(round.phase)}` : ""}</strong>
            {round.check ? <>
              <span>交付：{statusLabel(round.check.payload.delivery_status)}　科研证据：{statusLabel(round.check.payload.scientific_status)}</span>
              {Array.isArray(round.check.payload.issues) && round.check.payload.issues.length > 0 ? (
                <ul>{round.check.payload.issues.map((issue, index) => {
                  if (typeof issue !== "object" || issue === null) return null;
                  const item = issue as Record<string, unknown>;
                  return <li key={`${String(item.code ?? "issue")}-${index}`}>{String(item.code ?? "检查项")}: {String(item.detail ?? "需要补充信息")}</li>;
                })}</ul>
              ) : null}
            </> : <span>检查进行中</span>}
          </section>
        ))}
        {completion ? <p className="closed-loop-stop"><strong>停止原因：</strong>{TERMINAL_LABELS[stringValue(completion.payload.status) ?? ""] ?? "运行结束"} · 交付 {statusLabel(completion.payload.delivery_status)} · 科研证据 {statusLabel(completion.payload.scientific_status)}</p> : null}
        {relevant.some((event) => event.type.startsWith("evolution_")) ? <p className="closed-loop-note">已记录流程经验检索或观察事件；检索记录不代表经验已执行或科研验证通过。</p> : null}
        <p className="closed-loop-note">智能体运行状态不会替代独立检查；缺少科研证据时状态显示为“未知”。</p>
      </div>
    </details>
  );
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function statusLabel(value: unknown): string {
  return typeof value === "string" ? STATUS_LABELS[value] ?? "未知" : "未知";
}

function phaseLabel(value: string): string {
  const phases: Record<string, string> = { MAKER: "生成", CHECK: "独立检查", REPAIR: "修订", research: "研究", repair: "自动修订" };
  return phases[value] ?? value.replaceAll("_", " ");
}
