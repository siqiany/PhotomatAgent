import { useCallback, useEffect, useRef, useState } from "react";

import { api, subscribeToTask, type Capability, type RevisionPlan as RevisionPlanRecord, type RevisionQDRequest, type RunRef, type TaskDetail, type WorkbenchEvent } from "../../api";
import { ArtifactPanel } from "../artifacts/ArtifactPanel";
import { isRunActive, RunStatus, statusLabel } from "../../components/RunStatus";
import { FeedbackForm } from "../revisions/FeedbackForm";
import { RevisionComparison } from "../revisions/RevisionComparison";
import { RevisionPlan } from "../revisions/RevisionPlan";
import { QDRevisionPanel } from "../revisions/QDRevisionPanel";
import { SkillPanel } from "../skills/SkillPanel";
import { ProcedurePanel } from "../skills/ProcedurePanel";
import { EvaluationPanel } from "../evaluations/EvaluationPanel";
import { canApplyStatusRefresh, persistedRunStatus, replayedRunStartedStatus } from "./runStatus";
import { budgetFailure, pendingApproval } from "./runDiagnostics";
import { ActivityTimeline } from "./ActivityTimeline";
import { ClosedLoopProgress } from "./ClosedLoopProgress";
import { Conversation } from "./Conversation";

interface SavedRevisionContext {
  planId: string;
  sourceRunId: string;
  revisionRunId: string | null;
}

interface Props {
  taskId: string;
  mobilePanel: "tasks" | "workspace" | "details";
  onTaskChanged: () => void;
}

export function TaskWorkspace({ taskId, mobilePanel, onTaskChanged }: Props) {
  const currentTaskIdRef = useRef(taskId);
  currentTaskIdRef.current = taskId;
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [inspectedRunId, setInspectedRunId] = useState<string | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [allEvents, setAllEvents] = useState<WorkbenchEvent[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const followLatest = useRef(true);
  const [events, setEvents] = useState<WorkbenchEvent[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [runProviders, setRunProviders] = useState<Record<string, string>>({});
  const [statusOverride, setStatusOverride] = useState<string | null>(null);
  const [requestText, setRequestText] = useState("");
  const [runReports, setRunReports] = useState<Record<string, string>>({});
  const [revisionOpen, setRevisionOpen] = useState(false);
  const [revisionFeedback, setRevisionFeedback] = useState("");
  const [revisionPlan, setRevisionPlan] = useState<RevisionPlanRecord | null>(null);
  const revisionPlanRef = useRef<RevisionPlanRecord | null>(null);
  const [revisionHistory, setRevisionHistory] = useState<RevisionPlanRecord[]>([]);
  const [revisionHistoryError, setRevisionHistoryError] = useState<string | null>(null);
  const revisionHistorySelection = useRef(0);
  const [revisionSourceRunId, setRevisionSourceRunId] = useState<string | null>(null);
  const [revisionRunId, setRevisionRunId] = useState<string | null>(null);
  const [revisionRestoring, setRevisionRestoring] = useState(false);
  const [revisionRestoreError, setRevisionRestoreError] = useState<string | null>(null);
  const [revisionBusy, setRevisionBusy] = useState(false);
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [capabilityStatusUnavailable, setCapabilityStatusUnavailable] = useState(false);
  const [historical, setHistorical] = useState<Awaited<ReturnType<typeof api.historicalTask>> | null>(null);
  const [historyUnavailable, setHistoryUnavailable] = useState(false);
  const [loading, setLoading] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [maxModelCalls, setMaxModelCalls] = useState(20);
  const [maxTokens, setMaxTokens] = useState(100_000);
  const [wallSeconds, setWallSeconds] = useState(1800);
  const [autoRevision, setAutoRevision] = useState(true);
  const [evolutionRounds, setEvolutionRounds] = useState(3);
  const [requiredArtifactsText, setRequiredArtifactsText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [approvalSubmitting, setApprovalSubmitting] = useState<string | null>(null);
  const activeRunRef = useRef<string | null>(null);
  const detailRef = useRef<TaskDetail | null>(null);
  const setCurrentRevisionPlan = useCallback((plan: RevisionPlanRecord | null) => {
    revisionPlanRef.current = plan;
    setRevisionPlan(plan);
  }, []);
  const statusRevision = useRef(0);
  const seenSequences = useRef(new Set<number>());
  const refreshRef = useRef(onTaskChanged);

  useEffect(() => {
    if (mobilePanel === "workspace") setInspectorOpen(false);
  }, [mobilePanel]);

  useEffect(() => {
    refreshRef.current = onTaskChanged;
  }, [onTaskChanged]);

  const loadTask = useCallback(async () => {
    const value = await api.task(taskId);
    detailRef.current = value;
    setDetail(value);
    refreshRef.current();
    return value;
  }, [taskId]);

  const refreshRevisionPlanForRun = useCallback(async (runId: string) => {
    const current = revisionPlanRef.current;
    if (!current || ![
      current.source_run_id,
      ...(current.run_attempts ?? []).map((attempt) => attempt.run_id),
    ].includes(runId)) return;
    const updated = await api.revisionPlan(taskId, current.plan_id);
    if (currentTaskIdRef.current !== taskId || revisionPlanRef.current?.plan_id !== current.plan_id) return;
    setCurrentRevisionPlan(updated);
    const latestRunId = updated.run_id ?? null;
    setRevisionRunId(latestRunId);
    saveRevisionContext(taskId, {
      planId: updated.plan_id,
      sourceRunId: updated.source_run_id,
      revisionRunId: latestRunId,
    });
  }, [taskId, setCurrentRevisionPlan]);

  useEffect(() => {
    let mounted = true;
    let source: EventSource | undefined;
    let eventBatch: WorkbenchEvent[] = [];
    let flushTimer: ReturnType<typeof setTimeout> | undefined;
    const queueEvent = (event: WorkbenchEvent) => {
      eventBatch.push(event);
      if (flushTimer !== undefined) return;
      flushTimer = setTimeout(() => {
        flushTimer = undefined;
        const batch = eventBatch;
        eventBatch = [];
        if (!mounted) return;
        setAllEvents((previous) => [...previous, ...batch]);
        setRunReports((previous) => {
          const next = { ...previous };
          for (const item of batch) {
            if (item.type === "text_delta" && typeof item.payload.text === "string") next[item.run_id] = (next[item.run_id] ?? "") + item.payload.text;
          }
          return next;
        });
        const selected = batch.filter((item) => item.run_id === activeRunRef.current);
        if (selected.length) setEvents((previous) => [...previous, ...selected]);
      }, 50);
    };
    setLoading(true);
    setInspectedRunId(null);
    setInspectorOpen(false);
    setDetail(null);
    detailRef.current = null;
    setEvents([]);
    setAllEvents([]);
    setRunReports({});
    setRunProviders({});
    seenSequences.current.clear();
    activeRunRef.current = null;
    setActiveRunId(null);
    setStatusOverride(null);
    setError(null);
    setNotice(null);
    setApprovalSubmitting(null);
    setRevisionOpen(false);
    setRevisionFeedback("");
    setCurrentRevisionPlan(null);
    setRevisionHistory([]);
    setRevisionHistoryError(null);
    revisionHistorySelection.current += 1;
    setRevisionSourceRunId(null);
    setRevisionRunId(null);
    setRevisionRestoring(false);
    setRevisionRestoreError(null);
    setHistorical(null);
    setHistoryUnavailable(false);
    setAutoRevision(true);
    setEvolutionRounds(3);
    setRequiredArtifactsText("");

    void api.task(taskId)
      .then((task) => {
        if (!mounted) return;
        detailRef.current = task;
        setDetail(task);
        const savedRevision = readSavedRevisionContext(taskId);
        setRevisionOpen(Boolean(savedRevision));
        setRevisionSourceRunId(savedRevision?.sourceRunId ?? null);
        setRevisionRunId(savedRevision?.revisionRunId ?? null);
        if (savedRevision) {
          setRevisionRestoring(true);
          setRevisionRestoreError(null);
          void api.revisionPlan(taskId, savedRevision.planId)
            .then((plan) => {
              if (!mounted) return;
              setCurrentRevisionPlan(plan);
              setRevisionFeedback(plan.feedback_text);
              const restored: SavedRevisionContext = {
                planId: plan.plan_id,
                sourceRunId: plan.source_run_id,
                revisionRunId: plan.run_id ?? savedRevision.revisionRunId ?? null,
              };
              setRevisionSourceRunId(restored.sourceRunId);
              setRevisionRunId(restored.revisionRunId);
              saveRevisionContext(taskId, restored);
              setRevisionRestoring(false);
              setRevisionRestoreError(null);
            })
            .catch((reason: unknown) => {
              if (mounted) {
                setRevisionRestoring(false);
                setRevisionRestoreError(messageOf(reason));
              }
            });
        } else {
          setRevisionRestoring(false);
          setRevisionRestoreError(null);
          setCurrentRevisionPlan(null);
          setRevisionFeedback("");
        }
        setRevisionRestoring(true);
        void api.revisionPlans(taskId)
          .then(({ items }) => {
            if (!mounted || currentTaskIdRef.current !== taskId) return;
            setRevisionHistory(items);
            if (!savedRevision && revisionPlanRef.current === null) {
              const latest = items.at(-1);
              if (latest) {
                setCurrentRevisionPlan(latest);
                setRevisionOpen(true);
                setRevisionFeedback(latest.feedback_text);
                setRevisionSourceRunId(latest.source_run_id);
                setRevisionRunId(latest.run_id ?? null);
                saveRevisionContext(taskId, {
                  planId: latest.plan_id,
                  sourceRunId: latest.source_run_id,
                  revisionRunId: latest.run_id ?? null,
                });
              }
              setRevisionRestoring(false);
            }
          })
          .catch((reason: unknown) => {
            if (!mounted || currentTaskIdRef.current !== taskId) return;
            setRevisionHistoryError(messageOf(reason));
            if (!savedRevision) setRevisionRestoring(false);
          });
        const previousResearchRun = [...task.runs].reverse().find((run) => !run.request.host_operation && run.request.budget);
        if (previousResearchRun?.request.budget) {
          setMaxModelCalls(previousResearchRun.request.budget.max_model_calls);
          setMaxTokens(previousResearchRun.request.budget.max_tokens);
          setWallSeconds(previousResearchRun.request.budget.wall_seconds);
        }
        if (previousResearchRun?.request.evolution) {
          setAutoRevision(previousResearchRun.request.evolution.enabled);
          setEvolutionRounds(previousResearchRun.request.evolution.max_rounds);
          setRequiredArtifactsText((previousResearchRun.request.evolution.required_artifacts ?? []).join(", "));
        }
        const latest = task.runs.at(-1) ?? null;
        activeRunRef.current = latest?.run_id ?? null;
        setActiveRunId(latest?.run_id ?? null);
        setStatusOverride(latest?.status ?? null);
        setLoading(false);
        if (task.links.some((link) => link.source_type === "runtime_session")) {
          void api.historicalTask(taskId)
            .then((value) => { if (mounted) setHistorical(value); })
            .catch(() => { if (mounted) setHistoryUnavailable(true); });
        }
        source = subscribeToTask(
          taskId,
          (event) => {
            if (!mounted || seenSequences.current.has(event.seq)) return;
            seenSequences.current.add(event.seq);
            queueEvent(event);
            if (event.type === "model_stream_started" && typeof event.payload.provider === "string") {
              setRunProviders((previous) => ({ ...previous, [event.run_id]: event.payload.provider as string }));
            }
            if (event.type === "run_started" && activeRunRef.current === null) {
              activeRunRef.current = event.run_id;
              setActiveRunId(event.run_id);
              setEvents([]);
            }
            if (event.run_id !== activeRunRef.current) return;
            if (event.type === "approval_resolved") setApprovalSubmitting(null);
            if (event.type === "run_started") {
              statusRevision.current += 1;
              const persisted = detailRef.current?.runs.find((run) => run.run_id === event.run_id)?.status;
              setStatusOverride(replayedRunStartedStatus(persisted));
            }
            if (event.type === "cancel_requested") {
              statusRevision.current += 1;
              setStatusOverride("CANCEL_REQUESTED");
            }
            if (event.type === "run_completed") {
              statusRevision.current += 1;
              const revisionAtRefresh = statusRevision.current;
              void loadTask().then(async (value) => {
                await refreshRevisionPlanForRun(event.run_id);
                const persistedStatus = persistedRunStatus(value, event.run_id);
                if (persistedStatus && canApplyStatusRefresh(revisionAtRefresh, statusRevision.current)) setStatusOverride(persistedStatus);
              }).catch((reason: unknown) => setError(messageOf(reason)));
            }
            if (event.type === "run_failed") {
              statusRevision.current += 1;
              setStatusOverride("FAILED");
              void loadTask().then(() => refreshRevisionPlanForRun(event.run_id)).catch((reason: unknown) => setError(messageOf(reason)));
            }
            if (event.type === "run_cancelled") {
              statusRevision.current += 1;
              setStatusOverride("CANCELLED");
              void loadTask().then(() => refreshRevisionPlanForRun(event.run_id)).catch((reason: unknown) => setError(messageOf(reason)));
            }
            if (event.type === "run_interrupted") {
              statusRevision.current += 1;
              setStatusOverride("INTERRUPTED");
              void loadTask().then(() => refreshRevisionPlanForRun(event.run_id)).catch((reason: unknown) => setError(messageOf(reason)));
            }
          },
          () => setNotice("事件连接暂时中断，浏览器会自动重连并继续读取已保存的过程。"),
        );
      })
      .catch((reason: unknown) => {
        if (mounted) {
          setError(messageOf(reason));
          setLoading(false);
        }
      });

    return () => {
      mounted = false;
      source?.close();
      if (flushTimer !== undefined) clearTimeout(flushTimer);
    };
  }, [taskId, loadTask, refreshRevisionPlanForRun, retryCount]);

  useEffect(() => {
    let mounted = true;
    void api.capabilities()
      .then((result) => {
        if (!mounted) return;
        setCapabilities(result.items);
        setCapabilityStatusUnavailable(false);
      })
      .catch(() => {
        if (!mounted) return;
        setCapabilities([]);
        setCapabilityStatusUnavailable(true);
      });
    return () => { mounted = false; };
  }, [taskId]);

  useEffect(() => {
    const container = scrollRef.current;
    if (container && followLatest.current) container.scrollTop = container.scrollHeight;
  }, [allEvents.length, detail?.runs.length]);

  const inspectRun = (runId: string) => {
    setInspectedRunId(runId);
    setInspectorOpen(true);
  };

  const submitRun = async () => {
    if (!requestText.trim()) return;
    followLatest.current = true;
    setInspectedRunId(null);
    setSubmitting(true);
    setError(null);
    setNotice(null);
    setEvents([]);
    activeRunRef.current = null;
    setActiveRunId(null);
    setStatusOverride("QUEUED");
    try {
      const runRequest: Parameters<typeof api.startRun>[1] = {
        text: requestText.trim(),
        strategy: "standard",
        budget: {
          max_model_calls: maxModelCalls,
          max_tokens: maxTokens,
          wall_seconds: wallSeconds,
        },
        evolution: {
          enabled: autoRevision,
          max_rounds: evolutionRounds,
          patience: 2,
          required_artifacts: [...new Set(requiredArtifactsText.split(/[\n,]/).map((name) => name.trim()).filter(Boolean))],
          checks: [],
          target: null,
        },
      };
      const reference = await api.startRun(taskId, runRequest);
      activeRunRef.current = reference.run_id;
      setActiveRunId(reference.run_id);
      setRequestText("");
      const revisionAtRefresh = statusRevision.current;
      const refreshed = await loadTask();
      const currentRun = refreshed.runs.find((run) => run.run_id === reference.run_id);
      if (currentRun && canApplyStatusRefresh(revisionAtRefresh, statusRevision.current)) setStatusOverride(currentRun.status);
      setNotice("本轮需求已交给当前科研任务，过程由后端持续保存。 ");
    } catch (reason) {
      setError(messageOf(reason));
      setStatusOverride(null);
    } finally {
      setSubmitting(false);
    }
  };

  const cancelRun = async () => {
    if (!activeRunId) return;
    setError(null);
    try {
      const result = await api.cancelRun(activeRunId);
      setStatusOverride(result.status);
      setNotice(result.status === "CANCELLED" ? "本轮已取消。" : "已请求停止，本轮会在安全边界结束。 ");
      await loadTask();
    } catch (reason) {
      setError(messageOf(reason));
    }
  };

  const resolveApproval = async (requestId: string, approved: boolean) => {
    if (!activeRunId || approvalSubmitting) return;
    setApprovalSubmitting(requestId);
    setError(null);
    try {
      await api.resolveApproval(taskId, activeRunId, requestId, approved);
    } catch (reason) {
      setError(messageOf(reason));
      setApprovalSubmitting(null);
    }
  };

  const resumeRun = async () => {
    if (!activeRunId) return;
    setError(null);
    try {
      const reference = await api.resumeRun(activeRunId);
      activeRunRef.current = reference.run_id;
      setActiveRunId(reference.run_id);
      setStatusOverride("QUEUED");
      setEvents([]);
      await loadTask();
      setNotice("已基于原任务创建继续运行。 ");
    } catch (reason) {
      setError(messageOf(reason));
    }
  };

  const uploadFile = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    try {
      await api.upload(taskId, file);
      await loadTask();
      setNotice(`已将 ${file.name} 添加到当前任务。`);
    } catch (reason) {
      setError(messageOf(reason));
    }
  };

  const prepareRevision = async (sourceRunId: string, feedback: string, artifactIds: string[]) => {
    setRevisionBusy(true);
    setError(null);
    try {
      const plan = await api.prepareFeedback(taskId, {
        source_run_id: sourceRunId,
        text: feedback,
        artifact_ids: artifactIds,
        budget: { max_model_calls: 1, max_tokens: maxTokens, wall_seconds: wallSeconds },
      });
      setRevisionFeedback(feedback);
      setRevisionSourceRunId(sourceRunId);
      setCurrentRevisionPlan(plan);
      setRevisionRunId(null);
      saveRevisionContext(taskId, {
        planId: plan.plan_id,
        sourceRunId,
        revisionRunId: null,
      });
      setNotice(plan.status === "READY" ? "意见已整理成修订计划；请检查每一项后再启动补充工作。" : "反馈已保留，但计划暂时无法生成；可以重试。 ");
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setRevisionBusy(false);
    }
  };

  const selectRevisionHistory = async (planId: string) => {
    if (!planId || planId === revisionPlanRef.current?.plan_id) return;
    const requestId = ++revisionHistorySelection.current;
    const requestedTaskId = taskId;
    setRevisionBusy(true);
    setError(null);
    try {
      const plan = await api.revisionPlan(requestedTaskId, planId);
      if (
        currentTaskIdRef.current !== requestedTaskId
        || revisionHistorySelection.current !== requestId
      ) return;
      setCurrentRevisionPlan(plan);
      setRevisionFeedback(plan.feedback_text);
      setRevisionSourceRunId(plan.source_run_id);
      setRevisionRunId(plan.run_id ?? null);
      setRevisionOpen(true);
      saveRevisionContext(requestedTaskId, {
        planId: plan.plan_id,
        sourceRunId: plan.source_run_id,
        revisionRunId: plan.run_id ?? null,
      });
    } catch (reason) {
      if (currentTaskIdRef.current === requestedTaskId && revisionHistorySelection.current === requestId) {
        setError(messageOf(reason));
      }
    } finally {
      if (currentTaskIdRef.current === requestedTaskId && revisionHistorySelection.current === requestId) {
        setRevisionBusy(false);
      }
    }
  };

  const invalidateRevisionPlan = () => {
    setCurrentRevisionPlan(null);
    setRevisionSourceRunId(null);
    setRevisionRunId(null);
    setRevisionRestoreError(null);
    clearSavedRevisionContext(taskId);
  };

  const discardRevisionRestore = () => {
    clearSavedRevisionContext(taskId);
    setCurrentRevisionPlan(null);
    setRevisionSourceRunId(null);
    setRevisionRunId(null);
    setRevisionFeedback("");
    setRevisionRestoreError(null);
    setRevisionOpen(true);
  };

  const executeRevision = async (qd?: RevisionQDRequest) => {
    if (!revisionPlan || revisionPlan.status !== "READY") return;
    setRevisionBusy(true);
    setError(null);
    setEvents([]);
    setStatusOverride("QUEUED");
    try {
      const reference = await api.executeRevision(taskId, revisionPlan.plan_id, {
        strategy: qd ? "qd" : "standard",
        ...(qd ? { qd } : {}),
        budget: {
          max_model_calls: maxModelCalls,
          max_tokens: maxTokens,
          wall_seconds: wallSeconds,
        },
      });
      activeRunRef.current = reference.run_id;
      setActiveRunId(reference.run_id);
      setRevisionRunId(reference.run_id);
      saveRevisionContext(taskId, {
        planId: revisionPlan.plan_id,
        sourceRunId: revisionPlan.source_run_id,
        revisionRunId: reference.run_id,
      });
      setStatusOverride("QUEUED");
      const revisionAtRefresh = statusRevision.current;
      const refreshed = await loadTask();
      const updatedPlan = await api.revisionPlan(taskId, revisionPlan.plan_id);
      setCurrentRevisionPlan(updatedPlan);
      setRevisionRunId(updatedPlan.run_id ?? reference.run_id);
      saveRevisionContext(taskId, {
        planId: updatedPlan.plan_id,
        sourceRunId: updatedPlan.source_run_id,
        revisionRunId: updatedPlan.run_id ?? reference.run_id,
      });
      const currentRun = refreshed.runs.find((run) => run.run_id === reference.run_id);
      if (currentRun && canApplyStatusRefresh(revisionAtRefresh, statusRevision.current)) setStatusOverride(currentRun.status);
      setNotice("已在当前科研任务中启动补充工作；原结果和产物仍保留。 ");
    } catch (reason) {
      setStatusOverride(null);
      setError(messageOf(reason));
    } finally {
      setRevisionBusy(false);
    }
  };

  const acceptSkillReplay = (reference: { task_id: string; run_id: string }) => {
    activeRunRef.current = reference.run_id;
    setActiveRunId(reference.run_id);
    setStatusOverride("QUEUED");
    setEvents([]);
    setError(null);
    setNotice("技能回放已加入当前任务的持久运行队列；本轮可像其他科研运行一样查看或取消。");
    void loadTask().then((value) => {
      const currentRun = value.runs.find((run) => run.run_id === reference.run_id);
      if (currentRun) setStatusOverride(currentRun.status);
    }).catch((reason: unknown) => setError(messageOf(reason)));
  };

  const acceptEvaluationRun = (reference: RunRef) => {
    activeRunRef.current = reference.run_id;
    setActiveRunId(reference.run_id);
    setStatusOverride("QUEUED");
    setEvents([]);
    setError(null);
    setNotice("冻结评测已加入当前任务的后台队列；运行状态和取消操作与其他科研运行共享。 ");
    void loadTask().then((value) => {
      const currentRun = value.runs.find((run) => run.run_id === reference.run_id);
      if (currentRun) setStatusOverride(currentRun.status);
    }).catch((reason: unknown) => setError(messageOf(reason)));
  };

  if (loading) {
    return <div className="workspace-loading" data-workbench-panel="workspace">正在打开科研任务…</div>;
  }
  if (!detail) {
    return (
      <div className="workspace-loading" role="alert">
        <span>{error ?? "任务暂时无法打开。"}</span>
        <button className="button button-secondary" onClick={() => { setError(null); setLoading(true); setRetryCount((value) => value + 1); }}>重试打开</button>
      </div>
    );
  }

  const task = detail.task;
  const selectedRun = detail.runs.find((run) => run.run_id === activeRunId) ?? detail.runs.at(-1) ?? null;
  const displayedRun = selectedRun && statusOverride
    ? { ...selectedRun, status: statusOverride }
    : selectedRun;
  const status = statusOverride ?? selectedRun?.status ?? "";
  const revisionRun = revisionRunId ? detail.runs.find((run) => run.run_id === revisionRunId) : null;
  const revisionRunStatus = revisionRunId === activeRunId && statusOverride
    ? statusOverride
    : revisionRun?.status ?? "";
  const report = events
    .filter((event) => event.type === "text_delta")
    .map((event) => typeof event.payload.text === "string" ? event.payload.text : "")
    .join("") || (activeRunId ? runReports[activeRunId] ?? "" : "");
  const active = isRunActive(status);
  const approval = status === "RUNNING" ? pendingApproval(events, activeRunId) : null;
  const tokenFailure = budgetFailure(events, activeRunId);
  const evolution = capabilities.find((capability) => capability.id === "evolution");
  const acquisition = capabilities.find((capability) => capability.id === "local_tool_acquisition");

  return (
    <section className={`task-workspace ${inspectorOpen ? "inspector-open" : ""}`} data-mobile-panel={mobilePanel} data-workbench-panel="workspace">
      <main className="workspace-main">
        <header className="workspace-header">
          <div className="breadcrumbs"><h1>{task.title}</h1><code className="run-id" data-testid="active-run-id">{activeRunId}</code></div>
          <div className="header-actions">
            <button className="button button-secondary revision-entry" disabled={active || submitting || revisionBusy || revisionRestoring || !detail.runs.some((run) => ["SUCCEEDED", "PARTIAL", "FAILED", "CANCELLED", "INTERRUPTED"].includes(run.status))} onClick={() => setRevisionOpen((open) => !open)}>{revisionOpen ? "收起改进" : "继续改进"}</button>
            <RunStatus run={displayedRun} />
            <button className="button button-subtle" type="button" aria-expanded={inspectorOpen} onClick={() => setInspectorOpen((open) => !open)}>{inspectorOpen ? "收起详情" : "产物与自进化"}</button>
          </div>
        </header>

        <div className="workspace-scroll" ref={scrollRef} onScroll={(event) => { const el = event.currentTarget; followLatest.current = el.scrollHeight - el.scrollTop - el.clientHeight < 100; }}>
          <details className="task-objective"><summary>研究目标与改进</summary><p>{task.goal}</p>

          </details>

          {revisionOpen ? (
            <section className="revision-workbench" aria-label="继续改进工作区">
              <div className="revision-intro">
                <span className="eyebrow">FEEDBACK → REVISION</span>
                <h2>根据意见继续研究</h2>
                <p>选择同一任务中的一轮结果，写下具体意见。旧运行不会被覆盖，新的运行会保留自己的过程和产物。</p>
                {revisionHistory.length ? (
                  <label>历史修订计划
                    <select
                      aria-label="历史修订计划"
                      value={revisionPlan?.plan_id ?? ""}
                      onChange={(event) => void selectRevisionHistory(event.target.value)}
                      disabled={revisionBusy || revisionRestoring}
                    >
                      <option value="">选择一份历史计划</option>
                      {revisionHistory.map((plan) => <option key={plan.plan_id} value={plan.plan_id}>{revisionPlanLabel(plan)}</option>)}
                    </select>
                  </label>
                ) : null}
              </div>
              {revisionRestoring ? <p className="revision-muted" role="status">正在从服务端恢复上次的修订计划…</p> : null}
              {revisionHistoryError ? <p className="revision-warning" role="status">历史修订计划暂时无法读取；仍可提交新的反馈。</p> : null}
              {revisionRestoreError ? (
                <div className="revision-restore-error" role="alert">
                  <span>无法读取上次的修订计划；重试会保留恢复信息，重新开始只清除此标签页的恢复链接。</span>
                  <div>
                    <button className="text-button" onClick={() => setRetryCount((value) => value + 1)}>重试恢复</button>
                    <button className="text-button" onClick={discardRevisionRestore}>重新开始反馈</button>
                  </div>
                </div>
              ) : null}
              <FeedbackForm
                runs={detail.runs}
                artifacts={detail.artifacts}
                disabled={active || submitting || revisionBusy || revisionRestoring || Boolean(revisionRestoreError)}
                submitting={revisionBusy}
                feedback={revisionFeedback}
                preferredSourceRunId={revisionPlan?.source_run_id ?? revisionSourceRunId}
                preferredArtifactIds={(revisionPlan?.source_artifacts ?? []).map((artifact) => artifact.artifact_id)}
                preservePreferredArtifactSelection={Boolean(revisionPlan)}
                onFeedbackChange={setRevisionFeedback}
                onInvalidatePlan={invalidateRevisionPlan}
                onPrepare={(sourceRunId, feedback, artifactIds) => void prepareRevision(sourceRunId, feedback, artifactIds)}
              />
              {revisionPlan ? (
                <RevisionPlan
                  plan={revisionPlan}
                  busy={revisionBusy}
                  runDisabled={active || submitting}
                  onRun={() => void executeRevision()}
                  onClarify={(clarification) => {
                    const updated = `${revisionFeedback}\n\n用户补充说明：${clarification}`;
                    const artifactIds = (revisionPlan.source_artifacts ?? []).map((artifact) => artifact.artifact_id);
                    void prepareRevision(revisionPlan.source_run_id, updated, artifactIds);
                  }}
                  onRetry={() => void prepareRevision(
                    revisionPlan.source_run_id,
                    revisionPlan.feedback_text,
                    (revisionPlan.source_artifacts ?? []).map((artifact) => artifact.artifact_id),
                  )}
                />
              ) : null}
              {revisionPlan?.status === "READY" ? (
                <QDRevisionPanel
                  taskId={taskId}
                  disabled={active || submitting || revisionBusy || revisionRestoring}
                  busy={revisionBusy}
                  events={events}
                  runId={revisionRunId}
                  onRun={(qd) => void executeRevision(qd)}
                />
              ) : null}
              {revisionPlan && revisionRunId ? (
                <RevisionComparison
                  plan={revisionPlan}
                  taskId={task.task_id}
                  revisionRunId={revisionRunId}
                  before={revisionSourceRunId ? runReports[revisionSourceRunId] ?? "" : ""}
                  after={runReports[revisionRunId] ?? ""}
                  sourceProvider={revisionSourceRunId ? runProviders[revisionSourceRunId] ?? null : null}
                  revisionProvider={runProviders[revisionRunId] ?? null}
                  revisionStatus={statusLabel(revisionRunStatus)}
                  revisionActive={isRunActive(revisionRunStatus)}
                />
              ) : null}
            </section>
          ) : null}

          {detail.links.map((link) => link.source_type === "runtime_session" ? (
            <section className="history-source-card" key={`${link.source_type}:${link.source_id}`} aria-label="导入历史来源">
              <div className="history-source-heading">
                <strong>已导入的历史会话</strong>
                <span>
                  {historical?.source.current === false || historical?.source.messages_current === false
                    ? "来源文件已变化"
                    : historical?.source.messages_current === null
                      ? "历史快照未验证"
                      : "只读来源"}
                </span>
              </div>
              <div className="history-source-meta">
                <span>会话 ID <code>{link.source_id}</code></span>
                <span>导入摘要 <code>{link.source_digest.slice(0, 16)}…</code></span>
              </div>
              {historyUnavailable ? <p className="activity-text">历史会话当前无法读取；新运行仍可独立继续。</p> : null}
              {historical ? (
                historical.source.messages_current !== true ? (
                  <p className="activity-text">
                    {historical.source.messages_current === false
                      ? "历史对话快照与导入时不一致，已隐藏变更内容。"
                      : "此会话导入时未保存快照摘要，无法验证对话来源；为避免混淆，历史消息已隐藏。"}
                  </p>
                ) : <details className="history-messages">
                  <summary>查看历史对话（只读）<span>{historical.messages.length} 条</span></summary>
                  {!historical.messages.length ? <p className="activity-text">此会话没有可显示的持久对话快照。</p> : null}
                  {historical.messages.map((message, index) => (
                    <div className="history-message" key={`${index}:${message.role}`}>
                      <strong>{message.role === "user" ? "用户" : "智能体"}</strong>
                      <p>{message.text}</p>
                    </div>
                  ))}
                  {historical.messages_truncated ? <p className="activity-text">历史内容已截断，仅显示有界预览。</p> : null}
                </details>
              ) : null}
            </section>
          ) : null)}

          {detail.runs.length ? <Conversation runs={detail.runs} events={allEvents} activeRunId={activeRunId} activeStatus={status} onInspect={inspectRun} /> : <div className="conversation-welcome"><strong>今天想研究什么？</strong><span>描述问题，智能体将寻找证据、调用工具并记录结果。</span></div>}

          <ActivityTimeline key={activeRunId ?? "no-run"} events={events} status={statusLabel(status)} label={eventLabel} toolResultText={toolResultText} />
          <ClosedLoopProgress events={events} />

          {approval ? (
            <section className="interrupted-card" aria-label="工具审批">
              <div>
                <strong>需要确认工具调用：{approval.toolName}</strong>
                <span>{approval.reason}</span>
                <pre>{approval.argumentsPreview}</pre>
              </div>
              <div>
                <button className="button button-secondary" disabled={approvalSubmitting !== null} onClick={() => void resolveApproval(approval.requestId, false)}>拒绝</button>
                <button className="button button-primary" disabled={approvalSubmitting !== null} onClick={() => void resolveApproval(approval.requestId, true)}>允许一次</button>
              </div>
            </section>
          ) : null}

          {tokenFailure ? (
            <div className="inline-error workspace-error" role="alert">
              本轮 Token 预算不足，下一次模型请求尚未发出。上限 {tokenFailure.totalLimit.toLocaleString()}，
              已计入 {tokenFailure.consumed.toLocaleString()}，剩余 {tokenFailure.remaining.toLocaleString()}；
              下一次输入的保守上界约 {tokenFailure.inputUpperBound.toLocaleString()} tokens。
              本轮预算累计计算模型输入和输出，每次请求还会重新发送工作上下文。可以提高 Token 上限后发起新一轮运行。
            </div>
          ) : null}

          {error ? <div className="inline-error workspace-error" role="alert">{error}</div> : null}
          {notice ? <div className="inline-notice" role="status">{notice}</div> : null}
          {active ? (
            <div className="running-strip">
              <span className="spinner" aria-hidden="true" />
              <span>后端正在继续本轮科研任务，关闭浏览器不会停止运行。</span>
              <button className="text-button danger-text" onClick={() => void cancelRun()}>请求取消</button>
            </div>
          ) : null}
          {status === "INTERRUPTED" && selectedRun?.request.host_operation?.operation === "evaluation" ? (
            <div className="interrupted-card">
              <div><strong>评测进程已中断</strong><span>已保存的模型调用成本和已有检查报告仍可查看；为避免重复计费，不会自动重跑冻结评测。</span></div>
            </div>
          ) : status === "INTERRUPTED" ? (
            <div className="interrupted-card">
              <div><strong>这轮运行已中断</strong><span>可以检查外部计算状态后继续。</span></div>
              <button className="button button-secondary" onClick={() => void resumeRun()}>继续运行</button>
            </div>
          ) : null}

        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void submitRun();
          }}
        >
          <label htmlFor="run-request">本轮需求</label>
          <textarea
            id="run-request"
            aria-label="本轮需求"
            value={requestText}
            onChange={(event) => setRequestText(event.target.value)}
            onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (!active && !submitting) { followLatest.current = true; void submitRun(); } } }}
            placeholder="描述本轮要解决的问题、要比较的材料或需要补充的证据…"
            rows={2}
            disabled={active || submitting}
          />
          <details className="budget-controls">
            <summary>本轮预算与自动修订 <span>{maxModelCalls} 次模型调用 · {maxTokens.toLocaleString()} tokens · {wallSeconds} 秒</span></summary>
            <div className="budget-inputs">
              <label>模型调用上限<input aria-label="模型调用上限" type="number" min={1} value={maxModelCalls} onChange={(event) => setMaxModelCalls(Math.max(1, Number(event.target.value) || 1))} disabled={active || submitting} /></label>
              <label>Token 上限<input aria-label="Token 上限" type="number" min={1} value={maxTokens} onChange={(event) => setMaxTokens(Math.max(1, Number(event.target.value) || 1))} disabled={active || submitting} /></label>
              <label>运行时间上限（秒）<input aria-label="运行时间上限（秒）" type="number" min={1} value={wallSeconds} onChange={(event) => setWallSeconds(Math.max(1, Number(event.target.value) || 1))} disabled={active || submitting} /></label>
            </div>
            <div className="evolution-controls">
              <label className="evolution-toggle"><input aria-label="启用自动修订" type="checkbox" checked={autoRevision} onChange={(event) => setAutoRevision(event.target.checked)} disabled={active || submitting} /> 自动检查并修订（默认开启）</label>
              <label>最多修订轮数<input aria-label="最多修订轮数" type="number" min={1} max={8} value={evolutionRounds} onChange={(event) => setEvolutionRounds(Math.min(8, Math.max(1, Number(event.target.value) || 1)))} disabled={!autoRevision || active || submitting} /></label>
              <label className="evolution-artifacts">必须交付的文件名<input aria-label="必须交付的文件名" value={requiredArtifactsText} onChange={(event) => setRequiredArtifactsText(event.target.value)} disabled={active || submitting} placeholder="例如 report.md, data.csv" /></label>
            </div>
            <p>自动修订沿用本轮累计预算。多个文件名用逗号分隔；科研结论仍需独立证据检查。</p>
            <p>Token 上限累计计算本轮全部自动修订的模型输入和输出；请求前估计输入用量，获得真实用量后校准后续估计，余量不足时优先交付已有结果。估计并非精确计费上限。运行时间限制涵盖本轮研究和自动修订；到期停止本地运行，但不会取消已提交的外部科学计算作业。</p>
          </details>
          <div className="composer-footer">
            <div className="composer-tools">
              <label className="button button-subtle attach-button">
                <input
                  aria-label="添加任务附件"
                  type="file"
                  accept=".txt,.md,.csv,.json,.png,.jpg,.jpeg"
                  onChange={(event) => {
                    void uploadFile(event.currentTarget.files?.[0]);
                    event.currentTarget.value = "";
                  }}
                />
                <span aria-hidden="true">⌕</span> 添加资料
              </label>
              <span className="composer-hint">Enter 发送 · Shift+Enter 换行</span>
            </div>
            <button className="button button-primary submit-button" disabled={active || submitting || !requestText.trim()}>
              {submitting ? "正在提交…" : "提交给智能体"}
              <span aria-hidden="true">↗</span>
            </button>
          </div>
        </form>
      </main>

      <div className="workspace-inspector skill-inspector" style={{ padding: 0 }}>
        <div className="inspector-toolbar"><strong>产物与自进化</strong><button className="text-button" onClick={() => setInspectorOpen(false)}>收起详情</button></div>
        {inspectedRunId && inspectedRunId !== activeRunId ? <button className="text-button" onClick={() => setInspectedRunId(null)}>返回最新一轮</button> : null}
        {inspectedRunId && inspectedRunId !== activeRunId ? <ActivityTimeline events={allEvents.filter((event) => event.run_id === inspectedRunId)} status={statusLabel(detail.runs.find((run) => run.run_id === inspectedRunId)?.status)} label={eventLabel} toolResultText={toolResultText} /> : null}
        <ArtifactPanel
          report={inspectedRunId ? runReports[inspectedRunId] ?? "" : report}
          artifacts={detail.artifacts}
          taskId={task.task_id}
          runId={inspectedRunId ?? activeRunId}
          status={detail.runs.find((run) => run.run_id === (inspectedRunId ?? activeRunId))?.status ?? status}
          error={null}
        />
        <SkillPanel
          taskId={task.task_id}
          artifacts={detail.artifacts}
          disabled={active || submitting || revisionBusy}
          refreshKey={detail.runs.map((run) => `${run.run_id}:${run.status}`).join("|")}
          onRunQueued={acceptSkillReplay}
        />
        <ProcedurePanel
          taskId={task.task_id}
          artifacts={detail.artifacts}
          runs={detail.runs}
          disabled={active || submitting || revisionBusy}
          planId={revisionPlan?.plan_id}
          planRunIds={revisionPlan ? [
            revisionPlan.source_run_id,
            ...(revisionPlan.run_attempts ?? []).map((attempt) => attempt.run_id),
          ] : []}
        />
        <EvaluationPanel
          key={task.task_id}
          taskId={task.task_id}
          artifacts={detail.artifacts}
          disabled={active || submitting || revisionBusy}
          onRunQueued={acceptEvaluationRun}
        />
        {acquisition?.available ? (
          <section className="acquisition-stages" aria-label="本地工具流程">
            <strong>工具获取流程</strong>
            <span>发现 {acquisitionStage(events, "capability_search")} → 安装 {acquisitionStage(events, "capability_install")} → 执行 {acquisitionStage(events, "script_run")} → 科研检查（待证据验证）</span>
            <small>本地开发模式在主机进程运行，没有容器隔离；仅支持 Python wheel 和其 CLI 入口。安装或执行成功不等于科研结论通过检查。</small>
          </section>
        ) : null}
      </div>

      <div className="capability-strip" aria-label="功能状态">
        <span className="capability-indicator" />
        <span>自进化功能：</span>
        <span>{evolution?.available ? "可用" : evolution?.reason ?? (capabilityStatusUnavailable ? "状态暂不可读" : "后续阶段接入")}</span>
        <span className="capability-divider" />
        <span>新工具获取：</span>
        <span>{acquisition?.available ? "可用" : acquisition?.reason ?? (capabilityStatusUnavailable ? "状态暂不可读" : "后续阶段接入")}</span>
      </div>
    </section>
  );
}

function eventLabel(type: string, payload?: Record<string, unknown>): string {
  const tool = payload?.underlying_tool ?? payload?.tool_name;
  if (type === "tool_started" || type === "tool_completed") {
    const stage: Record<string, string> = {
      capability_search: "发现 Python 工具",
      capability_inspect: "核对工具与依赖",
      capability_install: "安装任务工具",
      script_run: "执行任务脚本",
    };
    if (typeof tool === "string" && stage[tool]) {
      return `${stage[tool]}${type === "tool_started" ? "中" : "完成"}`;
    }
  }
  const labels: Record<string, string> = {
    run_started: "智能体开始工作",
    loop_started: "开始分析本轮需求",
    text_delta: "研究进展",
    tool_started: "调用科研工具",
    tool_completed: "科研工具返回",
    run_completed: "本轮运行结束",
    run_failed: "本轮运行失败",
    run_cancelled: "本轮运行已取消",
    run_interrupted: "本轮运行已中断",
    evaluation_completed: "冻结对照评测已保存",
    skill_use: "技能回放与独立检查完成",
    provider_failed: "模型服务返回错误",
    model_request_started: "正在请求模型",
    model_response_completed: "模型响应完成",
    budget_updated: "运行预算更新",
    approval_pending: "等待工具审批",
    approval_resolved: "工具审批已处理",
  };
  return labels[type] ?? type.replaceAll("_", " ");
}

function toolResultText(payload: Record<string, unknown>): string {
  const name = payload.underlying_tool ?? payload.tool_name;
  if (name !== "capability_search" && name !== "capability_inspect" && name !== "capability_install" && name !== "script_run") {
    return `${String(name ?? "科研工具")} 已返回结果。`;
  }
  const raw = payload.output;
  if (typeof raw !== "string") return "工具返回了结果。";
  try {
    const result = JSON.parse(raw) as Record<string, unknown>;
    const code = typeof result.code === "string" ? result.code : "已返回";
    const backend = typeof result.backend === "string" ? ` · ${result.backend}` : "";
    return `${code}${backend}。请继续检查产物和科研证据。`;
  } catch {
    return raw.slice(0, 160);
  }
}

function acquisitionStage(events: WorkbenchEvent[], tool: string): string {
  const matching = events.filter((event) => (event.payload.underlying_tool ?? event.payload.tool_name) === tool);
  const last = matching.at(-1);
  if (!last) return "（待开始）";
  if (last.type === "tool_started") return "（进行中）";
  if (last.type === "tool_completed") {
    try {
      const result = JSON.parse(String(last.payload.output ?? "")) as Record<string, unknown>;
      return result.code === "OK" ? "（完成）" : "（需处理）";
    } catch {
      return "（已返回）";
    }
  }
  return "（待确认）";
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : "请求暂时未能完成，请重试。";
}

function revisionContextKey(taskId: string): string {
  return `photomatagent:revision-context:${taskId}`;
}

function revisionPlanLabel(plan: RevisionPlanRecord): string {
  const attempts = plan.run_attempts ?? [];
  const latest = attempts.at(-1);
  const cost = typeof plan.compiler_cost?.model_calls === "number"
    ? `${plan.compiler_cost.model_calls} 次模型调用`
    : "成本未知";
  return `来源 ${plan.source_run_id.slice(-8)} · 修订运行 ${latest?.status ?? "未启动"} · ${cost} · ${plan.feedback_text.slice(0, 64)}`;
}

function readSavedRevisionContext(taskId: string): SavedRevisionContext | null {
  try {
    const value: unknown = JSON.parse(window.sessionStorage.getItem(revisionContextKey(taskId)) ?? "null");
    if (!value || typeof value !== "object") return null;
    const saved = value as Partial<SavedRevisionContext>;
    if (typeof saved.planId !== "string") return null;
    if (typeof saved.sourceRunId !== "string") return null;
    if (saved.revisionRunId !== null && typeof saved.revisionRunId !== "string") return null;
    return saved as SavedRevisionContext;
  } catch {
    return null;
  }
}

function saveRevisionContext(taskId: string, value: SavedRevisionContext): void {
  try {
    window.sessionStorage.setItem(revisionContextKey(taskId), JSON.stringify(value));
  } catch {
    // The server remains authoritative if tab-scoped recovery storage is unavailable.
  }
}

function clearSavedRevisionContext(taskId: string): void {
  try {
    window.sessionStorage.removeItem(revisionContextKey(taskId));
  } catch {
    // Clearing tab-scoped recovery state is best effort.
  }
}
