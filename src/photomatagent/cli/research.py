"""Terminal interface to the same persistent research service as the Web UI."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from photomatagent.application.models import (
    AutomaticFeedbackRequest, BudgetSpec, EvolutionOptions, RunRequest, TaskCreate,
)

research_app = typer.Typer(help="与 Web 共用任务、专家返修和能力库。先启动科研工作台服务。", no_args_is_help=True)
console = Console()
_ACTIVE = {"QUEUED", "RUNNING", "WAITING_INPUT", "CANCEL_REQUESTED"}


class WorkbenchClient:
    """HTTP adapter only; never builds a runtime, registry or second task store."""

    def __init__(self, url: str = "http://127.0.0.1:8765") -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise typer.BadParameter("服务地址需要是 http(s) URL，不含凭据、查询或片段。")
        self.url = url.rstrip("/")

    def request(self, path: str, body: Any = None, *, key: str | None = None, method: str | None = None) -> Any:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Idempotency-Key"] = key or uuid4().hex
        request = Request(self.url + "/api" + path, headers=headers, method=method,
                          data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
        try:
            with urlopen(request, timeout=1900) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                detail = json.load(error).get("error", {})
                message = f"{detail.get('code', error.code)}: {detail.get('message', '请求失败')}"
            except (ValueError, AttributeError):
                message = f"HTTP {error.code}"
            console.print(message, markup=False)
            raise typer.Exit(1) from error
        except (URLError, TimeoutError, OSError) as error:
            console.print("无法连接科研工作台，或请求响应超时。先运行 bash scripts/start-workbench.sh --no-browser。", markup=False)
            if key:
                console.print(f"操作可能已被服务器接收；重试请保留 --request-key {key}，先查看任务记录。", markup=False)
            raise typer.Exit(1) from error

    def upload_history(self, task_id: str, content: str) -> Any:
        """Persist the complete extraction source as a hash-bound task attachment."""
        data = content.encode("utf-8")
        if len(data) > 50 * 1024 * 1024:
            raise ValueError("历史附件超过 50 MiB，请选择较小的来源任务。")
        boundary = "photomat-" + uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="history.json"\r\n'
                'Content-Type: application/json\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
        request = Request(self.url + "/api/tasks/" + quote(task_id, safe="") + "/attachments", data=body,
                          headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise ValueError("历史附件上传失败；未启动技能提炼，可查看已创建任务后重试。") from error

    def task(self, task_id: str) -> Any:
        return self.request(f"/tasks/{quote(task_id, safe='')}?view=summary")

    def events(self, task_id: str, after: int = 0) -> list[dict[str, Any]]:
        cursor = after
        events: list[dict[str, Any]] = []
        while True:
            page = self.request(f"/tasks/{quote(task_id, safe='')}/event-log?after={cursor}&limit=1000")
            events.extend(page)
            if len(page) < 1000:
                return events
            cursor = page[-1]["seq"]


def _show(detail: dict[str, Any], *, full: bool = False) -> None:
    if full:
        console.print(json.dumps(detail, ensure_ascii=False, indent=2), markup=False)
        return
    task = detail["task"]
    console.print(f"{task['title']} · {task['task_id']}", markup=False)
    for run in detail["runs"]:
        console.print(f"  {run['run_id']} · {run['status']}", markup=False)
    progress = detail.get("evolution", {}).get("latest") or {}
    if progress:
        console.print(f"交付检查：{progress.get('delivery_status', 'UNKNOWN')} · 科研证据：{progress.get('scientific_status', 'UNKNOWN')} · 停止原因：{progress.get('status', 'UNKNOWN')}", markup=False)
    for artifact in detail["artifacts"]:
        console.print(f"  产物 {artifact['filename']} · {artifact['artifact_id']}", markup=False)
    console.print("交付检查通过不等于科学结论已验证。查看过程：research events；完整记录：research show --details。", markup=False)


def final_response_text(rows: list[dict[str, Any]], run_id: str) -> str:
    """Only completed, tool-free responses; later repair rounds supersede drafts."""
    current = ""
    final = ""
    for event in rows:
        if event["run_id"] != run_id:
            continue
        payload = event["payload"]
        if event["type"] in {"closed_loop_round_started", "loop_started"}:
            current = final = ""
        elif event["type"] == "model_stream_started":
            current = ""
        elif event["type"] == "text_delta":
            current += str(payload.get("text", ""))
        elif event["type"] == "model_response_completed":
            final = current if payload.get("tool_call_count") == 0 and payload.get("finish_reason") == "stop" else ""
            current = ""
    if not final:
        compact = next((event for event in reversed(rows) if event["run_id"] == run_id
                        and event["type"] in {"context_compaction_completed", "context_compaction_failed", "context_compaction_skipped"}
                        and event["payload"].get("trigger") == "manual"), None)
        if compact:
            payload = compact["payload"]
            if compact["type"] == "context_compaction_completed":
                return f"上下文压缩完成：估计 {payload['tokens_before']} → {payload['tokens_after']} tokens；完整历史保留。"
            return f"上下文压缩未完成或已跳过：{payload.get('reason', 'UNKNOWN')}；原历史保留。"
    return final


@research_app.command("compact")
def compact_context(
    task_id: str,
    max_tokens: int = typer.Option(100_000, min=1),
    max_model_calls: int = typer.Option(16, min=1),
    wall_seconds: int = typer.Option(300, min=1),
    request_key: str | None = typer.Option(None, "--request-key"),
    url: str = "http://127.0.0.1:8765",
) -> None:
    """压缩任务工作上下文，保留完整持久历史。"""
    from photomatagent.application.models import ContextCompactRequest
    key = request_key or uuid4().hex
    body = ContextCompactRequest(budget=BudgetSpec(max_tokens=max_tokens, max_model_calls=max_model_calls, wall_seconds=wall_seconds))
    console.print(f"压缩请求键：{key}；摘要调用计入本次独立预算。", markup=False)
    result = WorkbenchClient(url).request(f"/tasks/{quote(task_id, safe='')}/context/compact", body.model_dump(mode="json"), key=key)
    console.print(f"已排队：{result['run_id']}。使用 research watch {task_id} 查看结果。", markup=False)


@research_app.command("list")
def list_tasks(url: str = typer.Option("http://127.0.0.1:8765", "--url")) -> None:
    table = Table("任务", "标题", "最近运行")
    for item in WorkbenchClient(url).request("/tasks")["items"]:
        table.add_row(item["task_id"], item["title"], (item.get("latest_run") or {}).get("status", "未开始"))
    console.print(table)


@research_app.command()
def start(
    goal: str = typer.Argument(...),
    title: str | None = typer.Option(None, "--title"),
    max_tokens: int = typer.Option(100_000, min=1),
    max_model_calls: int = typer.Option(20, min=1),
    wall_seconds: int = typer.Option(1800, min=1),
    required_artifact: list[str] = typer.Option([], "--required-artifact"),
    url: str = typer.Option("http://127.0.0.1:8765", "--url"),
) -> str:
    client = WorkbenchClient(url)
    task = TaskCreate(title=title or goal.strip()[:100], goal=goal)
    request = RunRequest(text=goal, budget=BudgetSpec(max_tokens=max_tokens, max_model_calls=max_model_calls, wall_seconds=wall_seconds),
                         evolution=EvolutionOptions(required_artifacts=required_artifact))
    task_id = client.request("/tasks", task.model_dump(mode="json"))["task_id"]
    console.print(f"任务：{task_id}", markup=False)
    run = client.request(f"/tasks/{task_id}/runs", request.model_dump(mode="json"))
    console.print(f"已启动：{run['run_id']}。在 Web 可查看同一任务；终端使用 research watch {task_id}。", markup=False)
    return str(task_id)


@research_app.command()
def show(task_id: str, details: bool = False, url: str = "http://127.0.0.1:8765") -> None:
    client = WorkbenchClient(url)
    detail = client.task(task_id)
    _show(detail, full=details)
    if not details and detail["runs"] and detail["runs"][-1]["status"] not in _ACTIVE:
        response = final_response_text(client.events(task_id), detail["runs"][-1]["run_id"])
        if response:
            console.print(response, markup=False)


@research_app.command()
def feedback(
    task_id: str, text: str,
    source_run: str | None = typer.Option(None, "--source-run"),
    max_tokens: int = typer.Option(100_000, min=1),
    max_model_calls: int = typer.Option(20, min=1),
    wall_seconds: int = typer.Option(1800, min=1),
    compiler_tokens: int = typer.Option(65_536, min=1),
    request_key: str | None = typer.Option(None, "--request-key"),
    url: str = "http://127.0.0.1:8765",
) -> None:
    client = WorkbenchClient(url)
    detail = client.task(task_id)
    runs = [run for run in detail["runs"] if (run.get("request", {}).get("host_operation") or {}).get("operation") != "context_compaction"]
    if request_key and source_run is None:
        raise typer.BadParameter("使用 --request-key 重试时请明确 --source-run，避免绑定到新一轮结果。")
    if not request_key and any(run["status"] in _ACTIVE for run in runs):
        raise typer.BadParameter("当前任务仍在运行，请完成或中止后提交评审。")
    if source_run is None:
        if not runs:
            raise typer.BadParameter("任务尚无运行结果。")
        source_run = runs[-1]["run_id"]
    latest = {item["filename"]: item["artifact_id"] for item in detail["artifacts"] if item["run_id"] == source_run}
    body = AutomaticFeedbackRequest(
        source_run_id=source_run, text=text, artifact_ids=list(latest.values()),
        budget=BudgetSpec(max_model_calls=1, max_tokens=compiler_tokens, wall_seconds=wall_seconds),
        revision_budget=BudgetSpec(max_model_calls=max_model_calls, max_tokens=max_tokens, wall_seconds=wall_seconds),
    )
    key = request_key or uuid4().hex
    console.print(f"请求键：{key}。意见整理与返修使用分别列出的预算。", markup=False)
    result = client.request(f"/tasks/{quote(task_id, safe='')}/feedback/auto", body.model_dump(mode="json"), key=key)
    console.print(f"计划：{result['plan']['plan_id']} · {result['status']}", markup=False)
    if result.get("run"):
        console.print(f"已自动启动返修：{result['run']['run_id']}", markup=False)
    if result.get("diagnostic"):
        console.print(result["diagnostic"], markup=False)


@research_app.command()
def events(task_id: str, details: bool = False, url: str = "http://127.0.0.1:8765") -> None:
    rows = WorkbenchClient(url).events(task_id)
    if details:
        console.print(json.dumps(rows, ensure_ascii=False, indent=2), markup=False)
    else:
        for row in rows:
            if row["type"] in {"run_started", "closed_loop_checked", "closed_loop_completed", "approval_pending", "run_completed", "run_failed"}:
                console.print(f"{row['seq']} · {row['run_id']} · {row['type']} · {json.dumps(row['payload'], ensure_ascii=False)}", markup=False)
        console.print("使用 --details 查看全部事件内容。", markup=False)


@research_app.command()
def watch(task_id: str, url: str = "http://127.0.0.1:8765") -> None:
    client = WorkbenchClient(url)
    previous = ""
    cursor = 0
    pending: dict[str, dict[str, Any]] = {}
    try:
        while True:
            detail = client.task(task_id)
            state = json.dumps([(r["run_id"], r["status"]) for r in detail["runs"]])
            if state != previous:
                _show(detail)
                previous = state
            active = [r for r in detail["runs"] if r["status"] in _ACTIVE]
            if not active:
                if detail["runs"]:
                    response = final_response_text(client.events(task_id), detail["runs"][-1]["run_id"])
                    if response:
                        console.print(response, markup=False)
                return
            for event in client.events(task_id, cursor):
                cursor = event["seq"]
                if event["run_id"] not in {r["run_id"] for r in active}:
                    continue
                if event["type"] == "approval_pending":
                    pending[event["payload"]["request_id"]] = event
                elif event["type"] == "approval_resolved":
                    pending.pop(event["payload"]["request_id"], None)
            pending = {key: event for key, event in pending.items() if event["run_id"] in {r["run_id"] for r in active}}
            if pending:
                for request_id, event in pending.items():
                    console.print(f"等待执行审批：{json.dumps(event['payload'], ensure_ascii=False)}", markup=False)
                    console.print(f"research approve {task_id} {event['run_id']} {request_id} --allow 或 --deny；也可在 Web 处理。", markup=False)
                return
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("已停止查看，后台任务继续；可用 research cancel 中止任务。", markup=False)


@research_app.command()
def cancel(run_id: str, url: str = "http://127.0.0.1:8765") -> None:
    console.print(WorkbenchClient(url).request(f"/runs/{quote(run_id, safe='')}/cancel", {}), markup=False)


@research_app.command()
def resume(run_id: str, request_key: str | None = None, url: str = "http://127.0.0.1:8765") -> None:
    console.print(WorkbenchClient(url).request(f"/runs/{quote(run_id, safe='')}/resume", {}, key=request_key), markup=False)


@research_app.command()
def approve(task_id: str, run_id: str, request_id: str, allow: bool = typer.Option(..., "--allow/--deny"), url: str = "http://127.0.0.1:8765") -> None:
    path = f"/tasks/{quote(task_id, safe='')}/runs/{quote(run_id, safe='')}/approvals/{quote(request_id, safe='')}"
    console.print(WorkbenchClient(url).request(path, {"approved": allow}), markup=False)


@research_app.command("abilities")
def abilities(url: str = "http://127.0.0.1:8765") -> None:
    client = WorkbenchClient(url)
    console.print("可执行方法", markup=False)
    console.print(json.dumps(client.request("/skills"), ensure_ascii=False, indent=2), markup=False)
    console.print("流程经验（保留原可靠性与审批来源）", markup=False)
    console.print(json.dumps(client.request("/procedures"), ensure_ascii=False, indent=2), markup=False)


@research_app.command("capture-method")
def capture_method(contract: Path, url: str = "http://127.0.0.1:8765") -> None:
    """保存真实脚本执行的方法版本，JSON 合同可指定 parent_skill_id。"""
    from photomatagent.application.skills import SkillCreate
    try:
        payload = SkillCreate.model_validate_json(contract.read_text())
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"方法合同无效：{exc}") from exc
    console.print(json.dumps(WorkbenchClient(url).request("/skills", payload.model_dump(mode="json")), ensure_ascii=False, indent=2), markup=False)


@research_app.command("method-sources")
def method_sources(url: str = "http://127.0.0.1:8765") -> None:
    console.print(json.dumps(WorkbenchClient(url).request("/skills/sources"), ensure_ascii=False, indent=2), markup=False)


@research_app.command("replay-method")
def replay_method(ability_id: str, contract: Path, request_key: str = typer.Option(...), url: str = "http://127.0.0.1:8765") -> None:
    """绑定当前任务资料，在与 Web 相同的队列回放固定版本。"""
    from photomatagent.application.skills import SkillReplay
    try:
        payload = SkillReplay.model_validate_json(contract.read_text())
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"回放合同无效：{exc}") from exc
    console.print(WorkbenchClient(url).request(f"/skills/{quote(ability_id, safe='')}/replay", payload.model_dump(mode="json"), key=request_key), markup=False)


@research_app.command()
def chat(task: str | None = None, url: str = "http://127.0.0.1:8765") -> None:
    """Input → autonomous delivery → expert feedback → autonomous repair."""
    client = WorkbenchClient(url)
    if task is None:
        goal = typer.prompt("科研问题").strip()
        task = start(goal, None, 100_000, 20, 1800, [], url)
    console.print("自动研究完成后输入评审意见即可返修。:quit 退出；:watch 查看运行；:events 查看详细过程。", markup=False)
    watch(task, url)
    while True:
        try:
            text = typer.prompt("评审意见").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if text == ":quit":
            return
        if text == ":watch":
            watch(task, url)
            continue
        if text == ":events":
            events(task, True, url)
            continue
        if not text:
            continue
        if any(run["status"] in _ACTIVE for run in client.task(task)["runs"]):
            console.print("任务仍在运行，请先完成或处理执行审批；使用 :watch 查看。", markup=False)
            continue
        feedback(task, text, None, 100_000, 20, 1800, 65_536, None, url)
        watch(task, url)


@research_app.command("disable-ability")
def disable_ability(ability_id: str, kind: str = "procedure", url: str = "http://127.0.0.1:8765") -> None:
    if kind not in {"procedure", "script"}:
        raise typer.BadParameter("--kind 只能是 procedure 或 script。")
    group = "procedures" if kind == "procedure" else "skills"
    console.print(WorkbenchClient(url).request(f"/{group}/{quote(ability_id, safe='')}/disable", {}), markup=False)


@research_app.command("enable-method")
def enable_method(ability_id: str, url: str = "http://127.0.0.1:8765") -> None:
    """恢复已经自身回放通过的固定脚本版本，不继承父版本检查。"""
    console.print(WorkbenchClient(url).request(f"/skills/{quote(ability_id, safe='')}/enable", {}), markup=False)


@research_app.command('learn')
def learn(task_id: str, source_run: str | None = None, max_candidates: int = 3, max_executions: int = 24, search_mode: str = 'directed', url: str = 'http://127.0.0.1:8765') -> None:
    """查看自动能力学习；指定来源运行时启动独立验证和版本搜索。"""
    client = WorkbenchClient(url)
    path = f'/tasks/{quote(task_id, safe="")}/capability-learning'
    console.print(client.request(path, {'source_run_id': source_run, 'max_candidates':max_candidates, 'max_executions':max_executions, 'search_mode':search_mode} if source_run else None), markup=False)


@research_app.command('benchmark')
def benchmark(task_id: str, start_run: bool = False, repeats: int = 2, max_tokens: int = 150000, max_model_calls: int = 80, conditions: str = 'BASE,FEEDBACK_ONLY,FEEDBACK_MEMORY,SINGLE_BEST,FULL_QD', url: str = 'http://127.0.0.1:8765') -> None:
    """查看冻结对照；--start-run 在同一服务启动独立预算评测。"""
    client=WorkbenchClient(url)
    path=f'/tasks/{quote(task_id, safe="")}/capability-benchmarks'
    console.print(client.request(path, {'repeats':repeats,'max_tokens':max_tokens,'max_model_calls':max_model_calls,'conditions':conditions.split(',')} if start_run else None),markup=False)


@research_app.command('blind-cases')
def blind_cases(task_id: str, job_task_id: str, url: str = 'http://127.0.0.1:8765') -> None:
    console.print(WorkbenchClient(url).request(f'/tasks/{quote(task_id,safe="")}/capability-benchmarks/{quote(job_task_id,safe="")}/blind-cases'),markup=False)


@research_app.command('review-case')
def review_case(task_id: str, job_task_id: str, review_file: Path, url: str = 'http://127.0.0.1:8765') -> None:
    from photomatagent.application.capability_benchmark import ScientificReviewRequest
    body=ScientificReviewRequest.model_validate_json(review_file.read_text()).model_dump()
    console.print(WorkbenchClient(url).request(f'/tasks/{quote(task_id,safe="")}/capability-benchmarks/{quote(job_task_id,safe="")}/reviews',body),markup=False)
