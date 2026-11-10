"""Durable capability learning, trusted transfer tests and existing QD archive.

Host orchestration runs frozen scripts through LocalSkillService, never imports
candidate code. Research results and local method qualification stay separate.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field
from photomatagent.application.capability_cases import SUITE_VERSION, cases
from photomatagent.application.errors import ServiceError
from photomatagent.application.models import RunRequest, TaskCreate
from photomatagent.application.plugin_contracts import interface_for
from photomatagent.application.procedural_skills import ProcedureCreate, ProcedureService
from photomatagent.application.skills import SkillCreate, SkillReplay, CheckerSpec, OrdinaryRunSource
from photomatagent.capability_acquisition.models import ExecutionRecord, ExecutionRequest
from photomatagent.capability_acquisition.sandbox import _safe_output_file, _sha256_file
from photomatagent.errors import ToolExecutionError
from photomatagent.workspace import Workspace
if TYPE_CHECKING:
    from photomatagent.application.service import WorkbenchService


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def immutable(path: Path, value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ServiceError('CAPABILITY_RECORD_CHANGED', '冻结记录发生变化。')
    else:
        with path.open('xb') as stream:
            stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


class CapabilityEvolutionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    source_run_id: str
    max_candidates: int = Field(default=3, ge=1, le=8)
    max_executions: int = Field(default=24, ge=4, le=64)
    wall_seconds: float = Field(default=300, gt=0, le=1800, allow_inf_nan=False)
    search_mode: Literal['directed', 'random'] = 'directed'


class CapabilityEvolutionService:
    def __init__(self, workbench: WorkbenchService) -> None:
        self.host = workbench
        self.store = workbench.store
        self.workspace = Workspace(workbench.workspace_root)
        self.procedures = ProcedureService(workbench.workspace_root)
        workbench.manager.register_host_operation('capability_evolution', self.execute)
        workbench.manager.register_completion_hook(self.after_research)

    def chain(self, task_id: str) -> dict[str, Any]:
        task = self.store.task(task_id)
        plans = self.store.revision_plans_for_task(task_id)
        journal = self.store.events_after(task_id, 0)
        trace=[]
        for event in journal:
            name=event.payload.get('underlying_tool') or event.payload.get('tool_name')
            if event.type=='tool_failed' or (event.type=='tool_completed' and name in {'capability_search','capability_inspect','capability_install','script_run','skill_run'}):
                raw=str(event.payload.get('output') or event.payload.get('error') or '')
                trace.append({'run_id':event.run_id,'seq':event.seq,'event':event.type,'tool':name,
                              'observation_sha256':hashlib.sha256(raw.encode()).hexdigest(),'diagnostic_excerpt':raw[:1000]})
        value = {'task_id': task_id, 'goal': task.goal,
                 'feedback': [p.model_dump(mode='json') for p in plans],
                 'runs': [r.model_dump(mode='json') for r in self.store.runs_for_task(task_id) if r.request.host_operation is None],
                 'artifacts': [{k: row[k] for k in ('artifact_id', 'run_id', 'filename', 'sha256', 'size_bytes')} for row in self.store.artifacts_for_task(task_id)],
                 'observations': [e.payload for e in journal if e.type in {'evolution_observation', 'evolution_skill_executed', 'closed_loop_completed'}],
                 'execution_trace':trace,
                 'root_cause_status': 'HYPOTHESIS_ONLY', 'scientific_status': 'UNKNOWN'}
        return value | {'chain_sha256': digest(value)}

    def sources(self, task_id: str, run_id: str) -> list[dict[str, Any]]:
        run = self.store.run(run_id)
        if run.task_id != task_id or run.status not in {'SUCCEEDED', 'PARTIAL', 'FAILED'} or run.request.host_operation is not None:
            raise ServiceError('CAPABILITY_SOURCE_INVALID', '需要本任务已结束的真实科研运行。')
        if run.result_invalidation_ids:
            return []
        found = []
        for source in self.host.skill_service.sources():
            if source['task_id'] != task_id or source['run_id'] != run_id or source['state'] != 'COMPLETED':
                continue
            if 'data' not in source['input_keys'] or 'result.json' not in source['artifacts']:
                continue
            root = self.workspace.resolve(f".photomatagent/executions/{source['execution_id']}")
            try:
                data = json.loads((root / 'input-data').read_text())
                kind = 'report_tables' if isinstance(data, dict) and 'tables' in data and 'report' in source['input_keys'] else 'rc_network' if isinstance(data, dict) and {'nodes', 'components', 'frequencies_hz'}.issubset(data) else None
                if isinstance(data, dict) and data.get('contract') == 'lossless_normal_incidence_film':
                    kind = 'fresnel_film'
                if kind:
                    from photomatagent.application.reproduction import ReproductionService

                    reproduction = ReproductionService(
                        self.workspace, None, scratch=f'tmp/workbench/{task_id}',
                    ).qualification_source(source['execution_id'])
                    expected_inputs = {'data', 'report'} if kind == 'report_tables' else {'data'}
                    if reproduction and set(source['input_keys']) != expected_inputs:
                        continue
                    found.append(source | {'checker_id': kind} | ({'reproduction': reproduction} if reproduction else {}))
            except (OSError, ValueError, ToolExecutionError):
                continue
        return found

    async def after_research(self, run_id: str) -> None:
        run = self.store.run(run_id)
        if run.request.host_operation is not None or not run.request.evolution.enabled:
            return
        if run.status not in {'SUCCEEDED', 'PARTIAL', 'FAILED'}:
            return
        if self.sources(run.task_id, run_id):
            await self.start(run.task_id, CapabilityEvolutionRequest(source_run_id=run_id), 'automatic:' + run_id)

    async def start(self, task_id: str, payload: CapabilityEvolutionRequest, request_key: str) -> dict[str, str]:
        # Source identity remains frozen on retries even when later runs append to the task.
        prior = next((link for link in self.store.links_for_task(task_id)
                      if link.source_type == 'capability_request' and link.source_id == payload.source_run_id + ':' + request_key), None)
        if prior:
            jobs = self.list_for_task(task_id)
            record = next(j for j in jobs if j['job_task_id'] == prior.source_digest)
            if record['request'] != payload.model_dump():
                raise ServiceError('IDEMPOTENCY_CONFLICT', '重复请求的演化配置发生变化。', status_code=409)
            return {'task_id': record['job_task_id'], 'run_id': record['run_id']}
        selected = self.sources(task_id, payload.source_run_id)
        if not selected:
            raise ServiceError('CAPABILITY_SOURCE_UNAVAILABLE', '没有满足独立电路/报告检查合同且适用当前版本的真实脚本执行。')
        chain = self.chain(task_id)
        identity = {'request': payload.model_dump(), 'source_task_id': task_id, 'sources': selected,
                    'chain_sha256': chain['chain_sha256'], 'suite_version': SUITE_VERSION}
        job_task = self.store.create_historical_task(request=TaskCreate(title='能力学习 · ' + self.store.task(task_id).title[:100],
            goal='Validate narrow reusable methods without changing the research contract.'),
            source_type='capability_job', source_id=task_id + ':' + request_key, source_digest=digest(identity))
        root = self.workspace.resolve(f'.photomatagent/capability-evolution/{job_task}', must_exist=False)
        immutable(root / 'chain.json', chain)
        immutable(root / 'request.json', identity)
        request = RunRequest(text='独立能力验证与经验提炼', strategy='capability_evolution',
            host_operation={'operation': 'capability_evolution', 'source_task_id': task_id, 'payload': payload.model_dump(), 'job_digest': digest(identity)})
        ref = self.store.reserve_run(job_task, request, 'capability:' + request_key)
        # The link table has a globally unique source identity; use it to recover jobs.
        self.store.link_source(task_id, 'capability_job_result', job_task, digest(identity))
        self.store.link_source(task_id, 'capability_request', payload.source_run_id + ':' + request_key, job_task)
        await self.host.manager.enqueue(ref.run_id)
        self.store.append_event_for_run(payload.source_run_id, 'capability_learning_queued', {'job_task_id': job_task, 'run_id': ref.run_id})
        return ref.model_dump()

    def list_for_task(self, task_id: str) -> list[dict[str, Any]]:
        self.store.task(task_id)
        results = []
        for link in self.store.links_for_task(task_id):
            if link.source_type != 'capability_job_result':
                continue
            job = link.source_id
            runs = self.store.runs_for_task(job)
            root = self.workspace.resolve(f'.photomatagent/capability-evolution/{job}')
            identity = json.loads((root / 'request.json').read_text())
            latest = runs[-1] if runs else None
            records = [e.payload for e in self.store.events_after(job, 0) if e.type == 'capability_candidate_result']
            records = [record | {'validation': json.loads(self.workspace.resolve(str(record['validation_path'])).read_text())} for record in records]
            results.append({'job_task_id': job, 'run_id': latest.run_id if latest else '', 'status': latest.status if latest else 'UNKNOWN',
                            'request': identity['request'], 'chain_sha256': identity['chain_sha256'], 'candidates': records,
                            'scientific_status': 'UNKNOWN'})
        return results

    async def execute(self, run: Any) -> None:
        operation = run.request.host_operation
        if not operation or operation.get('operation') != 'capability_evolution':
            raise ValueError('CAPABILITY_JOB_INVALID')
        root = self.workspace.resolve(f'.photomatagent/capability-evolution/{run.task_id}')
        identity = json.loads((root / 'request.json').read_text())
        if digest(identity) != operation['job_digest']:
            raise ValueError('CAPABILITY_JOB_CHANGED')
        payload = CapabilityEvolutionRequest.model_validate(identity['request'])
        if self.store.run(payload.source_run_id).result_invalidation_ids:
            raise ValueError('CAPABILITY_SOURCE_REVALIDATION_REQUIRED')
        chain = json.loads((root / 'chain.json').read_text())
        if chain['chain_sha256'] != identity['chain_sha256'] or digest({k: v for k, v in chain.items() if k != 'chain_sha256'}) != chain['chain_sha256']:
            raise ValueError('CAPABILITY_CHAIN_CHANGED')
        started = time.monotonic()
        receipts: dict[tuple[object, object], dict[str, Any]] = {}
        for event in self.store.events_after(run.task_id, 0):
            if event.run_id == run.run_id and event.type == 'capability_validation_case':
                key = (event.payload['skill_id'], event.payload['id'])
                if key in receipts:
                    raise ValueError('CAPABILITY_CASE_RECEIPT_DUPLICATE')
                receipts[key] = event.payload
        spent_before = sum(float(row['wall_seconds']) for row in receipts.values())
        executed = sum(1 for method in self.host.skill_service.list_skills() for use in method.get('uses', []) if use.get('task_id') == run.task_id and use.get('execution'))
        admitted = 0
        selected = []
        for source in identity['sources'][:payload.max_candidates]:
            kind = source['checker_id']
            name = 'research-' + kind.replace('_', '-')
            prior = [item for item in self.host.skill_service.list_skills() if item.get('logical_name') == name and item.get('status') != 'REVALIDATION_REQUIRED']
            parent = prior[-1]['skill_id'] if prior else None
            candidate = self.host.skill_service.create(SkillCreate(name=name, description='Frozen scientific method learned from a recorded research execution',
                scope=kind, source=OrdinaryRunSource(task_id=identity['source_task_id'], run_id=payload.source_run_id, execution_id=source['execution_id']),
                checker=CheckerSpec.model_validate({'checker_id': kind, 'input_key': 'data', 'output_path': 'result.json',
                    'field': {'rc_network': 'samples', 'report_tables': 'mismatches', 'fresnel_film': 'R'}[kind]}),
                interface=interface_for(kind, source.get('argv', [])), parent_skill_id=parent),
                capture_key=digest({'source': source, 'chain': identity['chain_sha256'], 'qualification_job':run.task_id}))
            if candidate.get('reproduction') != source.get('reproduction'):
                raise ValueError('CAPABILITY_SOURCE_CHANGED')
            selected.append(candidate)
        # Version replacement is a bounded real mutation, not an invented parameter score.
        if payload.search_mode == 'random':
            import random
            random.Random(int(identity['chain_sha256'][:16], 16)).shuffle(selected)
        variants = list(selected)
        for item in self.host.skill_service.list_skills():
            if len(variants) >= payload.max_candidates:
                break
            if item.get('checker') and item['checker']['checker_id'] in {c['checker']['checker_id'] for c in selected} and item['skill_id'] not in {c['skill_id'] for c in variants} and item.get('status') not in {'DISABLED', 'REVALIDATION_REQUIRED'}:
                variants.append(self.host.skill_service.read(item['skill_id']))
        for candidate in variants[:payload.max_candidates]:
            kind = candidate['checker']['checker_id']
            suite = cases(kind)
            output = root / f"validation-{candidate['skill_id']}.json"
            def recover(case: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
                receipt = receipts.get((candidate['skill_id'], case['id']))
                if receipt is None or receipt != cached | {'skill_id': candidate['skill_id']}:
                    raise ValueError('CAPABILITY_CASE_RECEIPT_MISSING_OR_CHANGED')
                self._verify_case_receipt(run, candidate, case, cached)
                return cached

            if output.exists():
                report = json.loads(output.read_text())
                report_cases = report['cases']
                if len(report_cases) > len(suite):
                    raise ValueError('CAPABILITY_VALIDATION_CHANGED')
                for case, row in zip(suite, report_cases):
                    if row.get('use_id'):
                        recover(case, row)
                    elif (row.get('passed') is not False or row.get('diagnostic') not in {
                        'VALIDATION_BUDGET_EXHAUSTED', 'VALIDATION_TIMEOUT',
                    } or (row['id'], row['role']) != (case['id'], case['role'])):
                        raise ValueError('CAPABILITY_VALIDATION_CHANGED')
                if report['wall_seconds'] != sum(float(row.get('wall_seconds', 0)) for row in report_cases):
                    raise ValueError('CAPABILITY_VALIDATION_CHANGED')
            else:
                results = []
                for case in suite:
                    partial = root / candidate['skill_id'] / (case['id'] + '.json')
                    saved = receipts.get((candidate['skill_id'], case['id']))
                    if partial.exists():
                        results.append(recover(case, json.loads(partial.read_text())))
                        continue
                    if saved is not None:
                        row = {key: value for key, value in saved.items() if key != 'skill_id'}
                        results.append(recover(case, row))
                        immutable(partial, row)
                        continue
                    if executed >= payload.max_executions or spent_before + time.monotonic() - started >= payload.wall_seconds:
                        results.append({'id': case['id'], 'role': case['role'], 'passed': False, 'diagnostic': 'VALIDATION_BUDGET_EXHAUSTED'})
                        break
                    case_root = self.workspace.resolve(f'tmp/workbench/{run.task_id}/{candidate["skill_id"]}/{case["id"]}', must_exist=False)
                    refs = {}
                    for key, data in case['inputs'].items():
                        path = case_root / key
                        immutable(path, data) if key != 'report' else self._text(path, data)
                        refs[key] = self.workspace.relative(path)
                    executed += 1
                    case_started = time.monotonic()
                    try:
                        use = await asyncio.wait_for(self.host.skill_service.replay(candidate['skill_id'], SkillReplay(task_id=run.task_id, scope=candidate['scope'], input_refs=refs), run_id=run.run_id),
                                                     timeout=max(0.01, payload.wall_seconds - spent_before - (time.monotonic() - started)))
                        results.append({'id': case['id'], 'role': case['role'], 'passed': self._case_passed(case, use),
                                        'use_id': use['use_id'], 'code': use['code'], 'checker': use['checker'],
                                        'contract_sha256': candidate['contract_sha256'], 'case_sha256': digest(case),
                                        'use_sha256': digest({key: value for key, value in use.items() if key != 'use_sha256'}),
                                        'wall_seconds': time.monotonic() - case_started})
                        self._verify_case_receipt(run, candidate, case, results[-1])
                    except TimeoutError:
                        results.append({'id': case['id'], 'role': case['role'], 'passed': False, 'diagnostic': 'VALIDATION_TIMEOUT', 'wall_seconds': time.monotonic() - case_started})
                        break
                    self.store.append_event_for_run(run.run_id, 'capability_validation_case', results[-1] | {'skill_id': candidate['skill_id']})
                    immutable(partial, results[-1])
                report = {'skill_id': candidate['skill_id'], 'contract_sha256': candidate['contract_sha256'],
                          'suite_version': SUITE_VERSION, 'suite_sha256': digest(suite), 'chain_sha256': identity['chain_sha256'],
                          'status': 'PASS' if len(results) == len(suite) and all(r['passed'] for r in results) else 'FAIL',
                          'cases': results, 'wall_seconds': sum(float(r.get('wall_seconds',0)) for r in results), 'scientific_status': 'UNKNOWN'}
                immutable(output, report)
            qualified = (
                [(case['id'], case['role']) for case in report['cases']]
                == [(case['id'], case['role']) for case in suite]
                and all(case['passed'] is True and case.get('use_id') for case in report['cases'])
            )
            if (report['skill_id'] != candidate['skill_id']
                    or report['contract_sha256'] != candidate['contract_sha256']
                    or report['suite_sha256'] != digest(suite) or report['suite_version'] != SUITE_VERSION
                    or report['chain_sha256'] != identity['chain_sha256']
                    or report['status'] != ('PASS' if qualified else 'FAIL')):
                raise ValueError('CAPABILITY_VALIDATION_CHANGED')
            procedure = None
            with self.store.database.lock:
                if self.store.run(payload.source_run_id).result_invalidation_ids:
                    raise ValueError('CAPABILITY_SOURCE_REVALIDATION_REQUIRED')
                if qualified and not candidate.get('reproduction'):
                    procedure = self._distill(candidate, report, output, chain, identity['source_task_id'], payload.source_run_id)
                decision = self._archive(candidate, report, output, run.task_id)
                if qualified and candidate.get('reproduction'):
                    if decision['action'] not in {'INSERT', 'REPLACE', 'KEEP'}:
                        raise ValueError('REPRODUCTION_ARCHIVE_REJECTED')
                    self.host.skill_service.record_reproduction_qualification(candidate['skill_id'], decision['trial_id'])
                    procedure = self._distill(candidate, report, output, chain, identity['source_task_id'], payload.source_run_id)
            if qualified:
                admitted += 1
            else:
                self.host.skill_service.disable(candidate['skill_id'])
            self.store.append_event_for_run(run.run_id, 'capability_candidate_result', {
                'skill_id': candidate['skill_id'], 'plugin_id': candidate.get('plugin_id'), 'validation_status': report['status'],
                'procedure_id': procedure['skill_id'] if procedure else None, 'archive': decision,
                'validation_path': self.workspace.relative(output), 'scientific_status': 'UNKNOWN'})
        self.store.append_event_for_run(run.run_id, 'capability_learning_completed', {'executions': executed, 'wall_seconds': spent_before + time.monotonic() - started,
            'model_calls': 0, 'tokens': 0, 'admitted': admitted, 'scientific_status': 'UNKNOWN'})
        self.store.append_event_for_run(run.run_id, 'loop_completed', {'reason': 'completed' if admitted else 'unresolved'})

    def _case_passed(self, case: dict[str, Any], use: dict[str, Any]) -> bool:
        checked = use['checker']
        code = use['code']
        if case['expected'] == 'REJECT':
            record = use.get('execution') or {}
            return (code == 'SKILL_SCHEMA_REJECTED'
                    or (code == 'EXECUTION_FAILED' and record.get('state') == 'FAILED'
                        and isinstance(record.get('exit_code'), int) and record['exit_code'] > 0
                        and record.get('reason_code') is None)
                    or (code == 'OK' and self._declared_rejection(use)))
        if case['expected'] == 'AUDIT_FAIL':
            return (code == 'OK' and checked.get('status') == 'FAIL'
                    and checked.get('mismatches') != ['candidate audit differs from host parsing'])
        return code == 'OK' and checked.get('status') == 'PASS'

    def _verify_case_receipt(self, run: Any, candidate: dict[str, Any], case: dict[str, Any], row: dict[str, Any]) -> None:
        uses = [use for use in self.host.skill_service.uses(candidate['skill_id'])
                if use.get('use_id') == row.get('use_id')]
        if len(uses) != 1:
            raise ValueError('CAPABILITY_REPLAY_RECEIPT_MISSING_OR_DUPLICATE')
        use = uses[0]
        if (use['code'] == 'SKILL_USE_CHANGED' or digest(use) != row.get('use_sha256')
                or row.get('contract_sha256') != candidate['contract_sha256']
                or row.get('case_sha256') != digest(case)
                or (row['id'], row['role']) != (case['id'], case['role'])
                or use['contract_sha256'] != candidate['contract_sha256']
                or (use['skill_id'], use['task_id'], use['run_id']) != (candidate['skill_id'], run.task_id, run.run_id)
                or (row['code'], row['checker']) != (use['code'], use['checker'])
                or row['passed'] is not self._case_passed(case, use)):
            raise ValueError('CAPABILITY_REPLAY_RECEIPT_CHANGED')
        expected = {
            key: hashlib.sha256((data if key == 'report' else
                json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)).encode()).hexdigest()
            for key, data in case['inputs'].items()
        }
        if not use.get('execution'):
            # A host-side schema rejection is evidence only for the exact input checked.
            if row['passed'] and (case['expected'] != 'REJECT' or use['code'] != 'SKILL_SCHEMA_REJECTED'
                                  or use.get('presented_input_sha256') != expected):
                raise ValueError('CAPABILITY_EXECUTION_RECEIPT_MISSING')
            return
        record = ExecutionRecord.model_validate(use['execution'])
        root = self.workspace.resolve(f'.photomatagent/executions/{record.execution_id}')
        provenance = json.loads((root / 'request.json').read_bytes())
        request = ExecutionRequest.model_validate(provenance['request'])
        original = ExecutionRequest.model_validate(candidate['request'])
        if original.source_tree:
            expected.update({key: candidate['files'][f'input-{key}'] for key in original.source_tree.inputs})
        if (ExecutionRecord.model_validate_json((root / 'record.json').read_bytes()) != record
                or request.execution_id != record.execution_id or request.trial_id != use['use_id']
                or request.genome_sha256 != candidate['contract_sha256'] or request.argv != original.argv
                or request.source_tree != original.source_tree or request.input_sha256 != expected
                or provenance['input_sha256'] != expected
                or request.script_digest != candidate['files']['script.py']
                or record.script_sha256 != request.script_digest
                or record.environment_sha256 != candidate['environment_sha256']):
            raise ValueError('CAPABILITY_EXECUTION_BINDING_CHANGED')
        inputs = {}
        for key, sha in expected.items():
            path = _safe_output_file(root, f'input-{key}')
            if path is None or _sha256_file(path) != sha:
                raise ValueError('CAPABILITY_EXECUTION_INPUT_CHANGED')
            inputs[key] = path
        if use['code'] == 'OK':
            output_root = self.workspace.resolve(f"user_output/{run.task_id}/skill-uses/{use['use_id']}")
            checked = self.host.skill_service._check(CheckerSpec.model_validate(candidate['checker']), inputs, output_root, record)
            if checked != use['checker']:
                raise ValueError('CAPABILITY_CHECKER_RESULT_CHANGED')

    @staticmethod
    def _text(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != value:
            raise ValueError('FROZEN_CASE_CHANGED')
        if not path.exists():
            path.write_text(value)

    def _declared_rejection(self, use: dict[str, Any]) -> bool:
        for relative in use.get('artifacts', []):
            if relative.endswith('/result.json'):
                try:
                    output = json.loads(self.workspace.resolve(relative).read_text())
                    return output.get('status') in {'UNKNOWN', 'NOT_APPLICABLE', 'REJECTED'} and output.get('scientific_status') == 'UNKNOWN'
                except (ValueError, OSError):
                    pass
        return False

    def _distill(self, candidate: dict[str, Any], report: dict[str, Any], output: Path, chain: dict[str, Any], task_id: str, source_run: str) -> dict[str, Any] | None:
        kind = candidate['checker']['checker_id']
        issues = [{'plan_id': plan['plan_id'], 'issue_id': issue['issue_id'],
                   'problem': issue['problem'], 'repair_actions': issue['actions'],
                   'acceptance_test': issue.get('acceptance_test'),
                   'historical_status': issue['status'], 'causal_status': 'UNESTABLISHED'}
                  for plan in chain['feedback'] for issue in plan.get('issues', [])]
        import difflib
        parent_id = candidate.get('parent_skill_id')
        changes = ''
        if parent_id:
            old = (self.host.skill_service._directory(parent_id) / 'script.py').read_text()
            new = (self.host.skill_service._directory(candidate['skill_id']) / 'script.py').read_text()
            changes = '\n'.join(difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile=parent_id, tofile=candidate['skill_id']))[:6000]
        lessons = {'expert_issues': issues, 'observed_code_changes': changes or 'No earlier frozen script version available.',
                   'execution_trace':chain['execution_trace'],
                   'tested_cases': [{'id': c['id'], 'role': c['role'], 'passed': c['passed']} for c in report['cases']],
                   'issue_closure': 'Host case tests qualify this narrow method; expert scientific issues are not automatically closed.',
                   'causal_status': 'UNESTABLISHED'}
        instructions = (f"Trigger: a {kind} computation or report consistency task. Preconditions: inspect the frozen interface, units and limits of {candidate['skill_id']}; "
            f"scope is exactly {candidate['scope']}. Call skill_run with this fixed skill_id and newly bound task input_refs. "
            "Do not reuse source fixture values. Inspect execution and deterministic checker separately. "
            "For passive RC graphs, validate positive SI elements, grounding and supported nodes; KCL reference checks transimpedance samples. "
            "For reports, bind both structured tables and the actual report body; missing tables, columns and numeric mismatches must be surfaced. "
            "For a lossless single film, bind real refractive indices and nanometre thickness and wavelength; the host checks normal-incidence reflectance only. "
            "If input is outside scope, a tool fails, or checker is UNKNOWN/FAIL, stop promoting conclusions, preserve diagnostics and choose another validated method or gather evidence. "
            "The historical root cause is only a hypothesis unless separately supported; mathematical tests do not establish device performance. "
            f"Source chain: {chain['chain_sha256']}; feedback rounds: {len(chain['feedback'])}; exact plugin contract: {candidate['contract_sha256']}.")
        instructions += '\n\nIndependently executed regression, transfer and negative cases:\n' + json.dumps(lessons['tested_cases'],ensure_ascii=False)
        instructions += '\n\nHistorical repairs (bounded excerpt; observations, not proven causes):\n' + json.dumps(lessons, ensure_ascii=False)[:8000]
        # Candidate source must be a real artifact or expert plan; script use is separately bound in provenance.
        artifact = next((row for row in chain['artifacts'] if row['run_id'] == source_run), None)
        plans = [p for p in chain['feedback'] if p.get('source_run_id') == source_run or source_run in [a.get('run_id') for a in p.get('run_attempts', [])]]
        if not artifact and not plans:
            if candidate.get('reproduction'):
                return None
            raise ServiceError('CAPABILITY_PROCEDURE_SOURCE_MISSING', '已验证方法缺少报告/评审来源；保持脚本方法，不发布流程经验。')
        item = self.procedures.create(ProcedureCreate(name='validated-' + kind.replace('_', '-'),
            description='Executable research procedure with independent regression, transfer and negative checks', scope=kind,
            instructions=instructions, task_id=task_id, run_id=source_run,
            artifact_id=artifact['artifact_id'] if artifact else None, plan_id=plans[-1]['plan_id'] if plans else None),
            provenance={'source': 'validated_multiround_chain', 'chain_sha256': chain['chain_sha256'],
                        'plugin_contract_sha256': candidate['contract_sha256'], 'plugin_id': candidate['skill_id'],
                        'lessons': lessons,
                        'independent_validation': {'path': self.workspace.relative(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()},
                        'scientific_validity': 'UNKNOWN', 'human_approval': 'NOT_CLAIMED'}, capture_key=digest({'contract': candidate['contract_sha256'], 'chain': chain['chain_sha256']}))
        return self.procedures.activate_validated(item['skill_id'])

    def _archive(self, candidate: dict[str, Any], report: dict[str, Any], output: Path, job_task: str) -> dict[str, Any]:
        from photomatagent.application.capability_archive import archive_capability
        return archive_capability(self.workspace, candidate, report, output, job_task)
