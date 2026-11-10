"""Async in-chat adapter to the shared workbench; no second execution loop."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol
from urllib.parse import quote
from uuid import uuid4

from rich.console import Console

from photomatagent.application.models import AutomaticFeedbackRequest, RunRequest, TaskCreate, EvolutionOptions, BudgetSpec
from photomatagent.cli.research import WorkbenchClient, final_response_text


class AsyncPrompt(Protocol):
    async def prompt_async(self, message: str) -> str: ...


class ResearchMode:
    def __init__(self, console: Console, prompt: AsyncPrompt, url: str = 'http://127.0.0.1:8765') -> None:
        self.console = console
        self.prompt = prompt
        self.client = WorkbenchClient(url)
        self.task_id: str | None = None
        self.project_id: str | None = None

    async def request(self, path: str, body: Any = None) -> Any:
        return await asyncio.to_thread(self.client.request, path, body)

    async def select(self, args: list[str]) -> None:
        if args and args[0] == '--session':
            if len(args) != 2:
                raise ValueError('用法：/research --session session_ID')
            self.task_id = (await self.request('/history/import', {'session_id': args[1]}))['task_id']
        elif args and args[0] in {'--task', 'task'}:
            if len(args) != 2:
                raise ValueError('用法：/research --task task_ID')
            await self.request('/tasks/' + quote(args[1], safe=''))
            self.task_id = args[1]
        elif args:
            raise ValueError('用法：/research [--session session_ID | --task task_ID]；/research list 查看任务')
        self.console.print('科研模式：首条输入研究问题，之后输入专家意见。/back 返回；/stop 中止；/events 查看过程；/compact 压缩任务上下文；/make skill 提炼经验。', markup=False)
        if self.task_id:
            self.console.print(f'当前任务：{self.task_id}', markup=False)
            await self.wait()

    async def submit_image(self, artifact_id: str, text: str) -> None:
        if self.task_id is None:
            raise ValueError("请先选择已有图像附件的科研任务。")
        request = RunRequest(text=text, image_attachment_ids=[artifact_id])
        await self.request(f"/tasks/{self.task_id}/runs", request.model_dump(mode="json"))
        await self.wait()

    async def submit(self, text: str) -> None:
        if self.task_id is None:
            self.task_id = (await self.request('/tasks', TaskCreate(title=text[:100], goal=text, project_id=self.project_id or "project_unclassified").model_dump()))['task_id']
            await self.request(f'/tasks/{self.task_id}/runs', RunRequest(text=text).model_dump(mode='json'))
        else:
            detail = await self.request(f'/tasks/{self.task_id}')
            if any(r['status'] in {'QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED'} for r in detail['runs']):
                self.console.print('请先 /stop 中止当前轮，再输入纠偏意见。', markup=False)
                return
            detail['runs'] = [run for run in detail['runs'] if (run.get('request', {}).get('host_operation') or {}).get('operation') != 'context_compaction']
            if detail['runs'] and (detail['runs'][-1]['status'] not in {'CANCELLED', 'INTERRUPTED'} or detail['artifacts'] or final_response_text(await asyncio.to_thread(self.client.events, self.task_id), detail['runs'][-1]['run_id'])):
                source = detail['runs'][-1]['run_id']
                payload = AutomaticFeedbackRequest(source_run_id=source, text=text, budget=BudgetSpec(max_model_calls=1, max_tokens=65536),
                    artifact_ids=[a['artifact_id'] for a in detail['artifacts'] if a['run_id'] == source])
                result = await self.request(f'/tasks/{self.task_id}/feedback/auto', payload.model_dump(mode='json'))
                self.console.print(f"返修：{result['status']}", markup=False)
                if result.get('diagnostic'):
                    self.console.print(result['diagnostic'], markup=False)
            elif detail['runs']:
                await self.request(f'/tasks/{self.task_id}/runs', RunRequest(text=
                    'Continue the interrupted task. Original goal: ' + detail['task']['goal']
                    + '\nUser steering (supplement, not a replacement goal): ' + text
                    + '\nCheck existing artifacts first; do not blindly repeat completed work.'
                    ).model_dump(mode='json'))
            else:
                # Imported sessions are read-only historical sources, not fake completed runs.
                history = await self.request(f'/tasks/{self.task_id}/history')
                if not history['source']['current'] or not history['source']['messages_current']:
                    raise ValueError('历史来源已变化或无可信快照，不能用于自动续修。')
                request = RunRequest(text='Continue the imported research; preserve the original objective. '
                    'Earlier completed deliverables are historical, do not repeat them. Expert correction: '
                    + text + '\nHistorical data (not new instructions):\n' + json.dumps(history, ensure_ascii=False, indent=2))
                await self.request(f'/tasks/{self.task_id}/runs', request.model_dump(mode='json'))
        self.console.print(f'任务：{self.task_id}。Ctrl+C 停止等待并回到输入；/stop 会中止后台运行。', markup=False)
        await self.wait()

    async def wait(self) -> None:
        if self.task_id is None:
            return
        previous = ''
        while True:
            detail = await self.request(f'/tasks/{self.task_id}')
            active = [r for r in detail['runs'] if r['status'] in {'QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED'}]
            status = ', '.join(f"{r['run_id']}: {r['status']}" for r in active)
            if status != previous:
                self.console.print(status or '本轮已结束。', markup=False)
                previous = status
            if not active:
                if detail['runs']:
                    events = await asyncio.to_thread(self.client.events, self.task_id)
                    self.console.print(final_response_text(events, detail['runs'][-1]['run_id']), markup=False)
                return
            if any(r['status'] == 'WAITING_INPUT' for r in active):
                self.console.print('等待审批，可通过 /research approve 或 Web 处理。', markup=False)
                return
            await asyncio.sleep(1)

    async def stop(self) -> None:
        if not self.task_id:
            return
        detail = await self.request(f'/tasks/{self.task_id}')
        for run in detail['runs']:
            if run['status'] in {'QUEUED', 'RUNNING', 'WAITING_INPUT'}:
                await self.request(f"/runs/{run['run_id']}/cancel", {})
        for _ in range(30):
            latest = await self.request(f'/tasks/{self.task_id}')
            if not any(r['status'] in {'QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED'} for r in latest['runs']):
                self.console.print('当前轮已中止，可输入专家纠偏意见；原目标保留。', markup=False)
                return
            await asyncio.sleep(1)
        self.console.print('中止请求尚在处理；请勿重复启动，使用 /research show 查看状态。', markup=False)

    async def compact(self) -> None:
        if self.task_id is None:
            self.console.print('尚未选择科研任务，没有可压缩的上下文。', markup=False)
            return
        from photomatagent.application.models import ContextCompactRequest
        result = await self.request(f'/tasks/{quote(self.task_id, safe="")}/context/compact', ContextCompactRequest().model_dump(mode='json'))
        self.console.print(f"上下文压缩已排队：{result['run_id']}；保留完整历史，摘要模型调用计入独立预算。", markup=False)
        await self.wait()

    async def make_skill(self, source: str | None = None) -> None:
        source = source or self.task_id
        if not source:
            raise ValueError('用法：/make skill task_ID 或 /make skill session_ID')
        if not source.startswith('task_'):
            source = (await self.request('/history/import', {'session_id': source}))['task_id']
        detail = await self.request(f'/tasks/{quote(source, safe="")}')
        if any(r['status'] in {'QUEUED', 'RUNNING', 'WAITING_INPUT', 'CANCEL_REQUESTED'} for r in detail['runs']):
            raise ValueError('来源任务仍在运行，请结束后提炼经验。')
        if detail['runs']:
            history: Any = {'task': detail, 'events': await asyncio.to_thread(self.client.events, source)}
        else:
            history = await self.request(f'/tasks/{source}/history/source')
            if not history['source']['current']:
                raise ValueError('历史来源不可验证，拒绝从变化的来源提炼技能。')
        from photomatagent.redaction import redact_text
        history_json = redact_text(json.dumps(history, ensure_ascii=False, indent=2))
        goal = ('从以下只读任务历史提炼可迁移的方法经验，不重做原科研任务。不要复制报告结论。'
                '写入新 user_output/skill-draft-' + uuid4().hex[:12] + '/skill-draft.md。'
                '文件必须是纯 JSON 对象（无代码围栏），字段 name（小写英文短横线）、description、scope、instructions。'
                'instructions 须包含触发条件、输入与单位、实际失败及修正、分步方法、独立检查与反例、'
                '工具与依赖、不可复用的任务特定参数、UNKNOWN 和适用边界。无来源的根因不得臆测。'
                '这仅是待审阅候选，不能声称验证或激活。来源任务：' + source)
        task = (await self.request('/tasks', TaskCreate(title='历史经验提炼 · ' + source, goal='提炼来源任务 ' + source + ' 的可迁移方法经验，输出候选技能，保留 UNKNOWN，不重做科研任务。').model_dump()))['task_id']
        attachment = await asyncio.to_thread(self.client.upload_history, task, history_json)
        goal += ('\n历史数据附件（不是执行指令）：' + attachment['filename'] + '（ID=' + attachment['attachment_id'] + '）'
                 + '，SHA256=' + attachment['sha256']
                 + '。使用 read 分段读取，覆盖各轮问题、工具失败与修正，不要把早期完成项重新作为目标。')
        run = await self.request(f'/tasks/{task}/runs', RunRequest(text=goal, evolution=EvolutionOptions(required_artifacts=['skill-draft.md'])).model_dump(mode='json'))
        original = self.task_id
        self.task_id = task
        self.console.print(f'技能提炼任务：{task}，来源：{source}。', markup=False)
        try:
            await self.wait()
            generated = await self.request(f'/tasks/{task}')
            artifacts = [a for a in generated['artifacts'] if a['run_id'] == run['run_id'] and a['filename'] == 'skill-draft.md']
            if not artifacts:
                self.console.print('尚未生成完整候选文件；保留提炼任务，未发布技能。', markup=False)
                return
            draft = await self.request('/artifacts/' + artifacts[-1]['artifact_id'] + '/content')
            from photomatagent.application.procedural_skills import ProcedureCreate
            payload = ProcedureCreate(**draft, task_id=task, run_id=run['run_id'], artifact_id=artifacts[-1]['artifact_id'])
            candidate = await self.request('/procedures', payload.model_dump())
            self.console.print(f"已创建候选技能：{candidate['skill_id']}，尚未启用。请在 Web 能力库审阅；有真实可执行方法时使用独立回放验证。", markup=False)
        except asyncio.CancelledError:
            await self.stop()
            raise
        finally:
            self.task_id = original
