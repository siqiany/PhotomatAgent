"""Frozen held-out capability comparisons through the same AgentRuntime factory."""
from __future__ import annotations
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field
from photomatagent.application.capability_cases import cases
from photomatagent.application.capability_evolution import digest, immutable
from photomatagent.application.errors import ServiceError
from photomatagent.application.frozen_methods import FrozenEvaluationSkills
from photomatagent.application.models import RunRequest, TaskCreate, BudgetSpec
from photomatagent.application.runtime_factory import RuntimeFactory
from photomatagent.application.skills import CHECKERS, CheckerSpec
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.context_engine import ContextEngineConfig
from photomatagent.skills.loader import SkillLoader
from photomatagent.runtime.permissions import PolicyRule, PermissionDecision, PermissionResult
if TYPE_CHECKING:
    from photomatagent.application.service import WorkbenchService

Condition = Literal['BASE', 'FEEDBACK_ONLY', 'FEEDBACK_MEMORY', 'SINGLE_BEST', 'FULL_QD', 'NO_TOOL_EVOLUTION', 'NO_VALIDATION_GATE']


class CapabilityBenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    conditions: list[Condition] = Field(default=['BASE','FEEDBACK_ONLY','FEEDBACK_MEMORY','SINGLE_BEST','FULL_QD'], min_length=1, max_length=7)
    repeats: int = Field(default=2, ge=1, le=5)
    max_model_calls: int = Field(default=80, ge=1, le=200)
    max_tokens: int = Field(default=150000, ge=1, le=1000000)
    wall_seconds: float = Field(default=900, gt=0, le=7200, allow_inf_nan=False)


class ScientificReviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    blind_id: str = Field(pattern=r'^blind_[0-9a-f]{24}$')
    reviewer: str = Field(min_length=1, max_length=100)
    scientific_quality: float = Field(ge=0, le=5, allow_inf_nan=False)
    goal_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    severe_errors: int = Field(ge=0, le=1000)
    repeated_errors: int = Field(ge=0, le=1000)
    expert_minutes: float = Field(ge=0, le=10000, allow_inf_nan=False)
    notes: str = Field(min_length=1, max_length=5000)


class _FrozenComputationPolicy(PolicyRule):
    def __init__(self, task_id: str, *, learned: bool) -> None:
        self.prefix=f'tmp/workbench/{task_id}/'
        super().__init__(rules={'read':PermissionDecision.ALLOW,'calculator':PermissionDecision.ALLOW,'write':PermissionDecision.ALLOW,
            'script_run':PermissionDecision.ALLOW,'tool_search':PermissionDecision.ALLOW,'tool_describe':PermissionDecision.ALLOW,'tool_call':PermissionDecision.ALLOW,
            'skill_run':PermissionDecision.ALLOW if learned else PermissionDecision.DENY,'skill_search':PermissionDecision.ALLOW if learned else PermissionDecision.DENY}, default=PermissionDecision.DENY)
    async def check(self, tool_name: str, arguments: dict[str,object]) -> PermissionResult:
        if tool_name=='write':
            path=arguments.get('path')
            if not isinstance(path,str) or not path.startswith(self.prefix) or '..' in Path(path).parts:
                return PermissionResult(PermissionDecision.DENY,'Frozen evaluation writes only current-case scripts/data; immutable inputs and methods cannot be changed.')
            # Input fixtures are read-only to model requests.
            if Path(path).name in {'data','report'}:
                return PermissionResult(PermissionDecision.DENY,'Frozen case input cannot be changed.')
        return await super().check(tool_name,arguments)


class CapabilityBenchmarkService:
    def __init__(self, host: WorkbenchService) -> None:
        self.host = host
        host.manager.register_host_operation('capability_benchmark', self.execute)

    async def start(self, task_id: str, payload: CapabilityBenchmarkRequest, request_key: str) -> dict[str, str]:
        self.host.store.task(task_id)
        previous=next((link for link in self.host.store.links_for_task(task_id) if link.source_type=='capability_benchmark_request' and link.source_id==task_id+':'+request_key),None)
        if previous:
            root=self.host.skill_service.workspace.resolve(f'.photomatagent/capability-benchmark/{previous.source_digest}')
            manifest=json.loads((root/'manifest.json').read_text())
            if manifest['request']!=payload.model_dump():
                raise ServiceError('IDEMPOTENCY_CONFLICT','重复请求的冻结评测配置发生变化。',status_code=409)
            latest=self.host.store.runs_for_task(previous.source_digest)[-1]
            return {'task_id':previous.source_digest,'run_id':latest.run_id}
        methods: list[str] = []
        scores: dict[str, float] = {}
        archive_keys: set[str] = set()
        for source_job in self.host.capability_evolution.list_for_task(task_id):
            for c in source_job['candidates']:
                if (entry := c['archive'].get('active_entry')):
                    archive_keys.add(entry['archive_key'])
                if c['validation_status'] == 'PASS':
                    methods.append(c['skill_id'])
                    report=json.loads(self.host.skill_service.workspace.resolve(c['validation_path']).read_text())
                    scores[c['skill_id']]=float(report['wall_seconds'])
        methods = sorted(set(methods), key=lambda key: (scores[key], key))
        if not methods:
            raise ServiceError('BENCHMARK_METHOD_UNAVAILABLE', '先完成至少一个真实方法的独立验证。')
        selected = [self.host.skill_service.read(s) for s in methods]
        condition_methods={'SINGLE_BEST':methods[:], 'FULL_QD':methods[:]}
        from photomatagent.scientific.evolution.qd.store import QDStore
        qd=QDStore(self.host.skill_service.workspace, trial_factory=lambda event, genome: (_ for _ in ()).throw(ValueError('READ_ONLY')))
        repertoire=[]
        for key in sorted(archive_keys):
            for entry in qd.load_archive_entries(key):
                genome=qd.load_genome(entry.genome_id)
                archived_method_id=str(genome.steps[0].parameters['skill_id'])
                if self.host.skill_service.read(archived_method_id)['source_evidence']['status']=='PASS':repertoire.append(archived_method_id)
        condition_methods['FULL_QD']=list(dict.fromkeys(repertoire))
        # Freeze ablation pools before examining held-out inputs. Ungated means
        # removal of development-suite qualification only; runtime safety and
        # the independent host scoring checker still apply.
        ungated=[]; parents=[]
        for source_job in self.host.capability_evolution.list_for_task(task_id):
            for record in source_job['candidates']:
                method=self.host.skill_service.read(record['skill_id'])
                if method['source_evidence']['status']=='PASS': ungated.append(method['skill_id'])
                parent=method.get('parent_skill_id')
                if parent and self.host.skill_service.read(parent)['source_evidence']['status']=='PASS':parents.append(parent)
        condition_methods['NO_VALIDATION_GATE']=list(dict.fromkeys(ungated))
        condition_methods['NO_TOOL_EVOLUTION']=list(dict.fromkeys(parents))
        all_methods=list(dict.fromkeys(methods+ungated+parents+repertoire))
        selected = [self.host.skill_service.read(s) for s in all_methods]
        families = {m['checker']['checker_id'] for m in selected}
        dataset = {kind: [c for c in cases(kind, holdout=True) if c['expected']=='PASS'] for kind in sorted(families)}
        options = self.host.manager.runtime_options
        provider = str(options.get('provider', 'fake')); model = options.get('model')
        from photomatagent.scientific.evolution.qd.evaluation_runtime import evaluation_software_sha256
        manifest = {'source_task_id':task_id,'request':payload.model_dump(), 'dataset':dataset,'dataset_sha256':digest(dataset),
                    'condition_methods':condition_methods, 'methods':[{ 'skill_id':m['skill_id'],'contract_sha256':m['contract_sha256'],'checker_sha256':m.get('checker_sha256')} for m in selected],
                    'provider':provider,'model':model,'software_sha256':evaluation_software_sha256(),
                    'runtime_context_config':ContextEngineConfig(output_recovery_attempts=1).model_dump(mode='json'),
                    'development_groups':['capability-development-v1'], 'test_groups':['capability-holdout-v1'],
                    'scientific_review':'PENDING_HUMAN_BLIND_REVIEW',
                    'base_toolset':'read/write current-case Python scripts; same frozen dependency environments; calculator; script_run via AgentRuntime; no acquisition tools',
                    'execution_isolation':'Backend-dependent: local_dev is host execution, not an OS sandbox. Podman isolation must be separately validated.',
                    'selection_basis':'development PASS then lowest measured development execution time; heldout never selects',
                    'memory':[self.host.capability_evolution.procedures.read(p['skill_id'])['instructions'] for p in self.host.capability_evolution.procedures.list_skills() if p['status']=='VALIDATED_PROCEDURE' and p['task_id']==task_id],
                    'condition_notes':{'NO_TOOL_EVOLUTION':'Frozen parent method; unavailable if no independently source-checked parent exists.',
                                       'NO_VALIDATION_GATE':'Development-suite admission removed only in the isolated evaluation; mandatory host checker and runtime checks remain. If the pool is identical, this ablation is degenerate.'}}
        job = self.host.store.create_historical_task(request=TaskCreate(title='冻结能力评测 · '+self.host.store.task(task_id).title[:100], goal='Controlled capability evaluation; empirical scientific claims require human review.'),
            source_type='capability_benchmark', source_id=task_id+':'+request_key,source_digest=digest(manifest))
        root=self.host.skill_service.workspace.resolve(f'.photomatagent/capability-benchmark/{job}',must_exist=False)
        immutable(root/'manifest.json',manifest)
        run=self.host.store.reserve_run(job,RunRequest(text='冻结科研能力对照',strategy='capability_benchmark',budget=BudgetSpec(max_tokens=payload.max_tokens,max_model_calls=payload.max_model_calls,wall_seconds=max(1,int(payload.wall_seconds))),host_operation={'operation':'capability_benchmark','manifest_sha256':digest(manifest)}),'benchmark:'+request_key)
        self.host.store.link_source(task_id,'capability_benchmark_result',job,digest(manifest))
        self.host.store.link_source(task_id,'capability_benchmark_request',task_id+':'+request_key,job)
        if self.host.store.run(run.run_id).status=='QUEUED':
            await self.host.manager.enqueue(run.run_id)
        return run.model_dump()

    def list_for_task(self, task_id: str) -> list[dict[str, Any]]:
        result=[]
        for link in self.host.store.links_for_task(task_id):
            if link.source_type!='capability_benchmark_result':continue
            runs=self.host.store.runs_for_task(link.source_id)
            root=self.host.skill_service.workspace.resolve(f'.photomatagent/capability-benchmark/{link.source_id}')
            report=json.loads((root/'report.json').read_text()) if (root/'report.json').exists() else None
            reviews=[json.loads(p.read_text()) for p in sorted((root/'reviews').glob('*.json'))]
            result.append({'job_task_id':link.source_id,'run_id':runs[-1].run_id,'status':runs[-1].status,'report':report,'scientific_reviews':reviews})
        return result

    def blind_cases(self, task_id: str, job_task_id: str) -> list[dict[str, Any]]:
        if not any(link.source_type=='capability_benchmark_result' and link.source_id==job_task_id for link in self.host.store.links_for_task(task_id)):
            raise ServiceError('BENCHMARK_NOT_FOUND', '评测不属于当前任务。', status_code=404)
        root=self.host.skill_service.workspace.resolve(f'.photomatagent/capability-benchmark/{job_task_id}')
        report=json.loads((root/'report.json').read_text())
        manifest=json.loads((root/'manifest.json').read_text())
        items=[]
        for row in report['results']:
            answer=root/'cases'/row['case_id']/'answer.json'
            case=next(c for c in manifest['dataset'][row['family']] if row['case_id'].endswith('-'+c['id']))
            items.append({'blind_id':'blind_'+hashlib.sha256((job_task_id+row['case_id']).encode()).hexdigest()[:24],
                          'inputs':case['inputs'],'answer':json.loads(answer.read_text()) if answer.exists() else None,
                          'scientific_status':'UNKNOWN'})
        return sorted(items,key=lambda item:item['blind_id'])

    def review(self, task_id: str, job_task_id: str, payload: ScientificReviewRequest, request_key: str) -> dict[str, Any]:
        if not any(c['blind_id']==payload.blind_id for c in self.blind_cases(task_id,job_task_id)):
            raise ServiceError('BLIND_CASE_NOT_FOUND', '盲评编号无效。', status_code=404)
        root=self.host.skill_service.workspace.resolve(f'.photomatagent/capability-benchmark/{job_task_id}')
        record=payload.model_dump() | {'source':'HUMAN_EXPERT_REPORT','scientific_measurement':'NOT_CLAIMED'}
        immutable(root/'reviews'/(hashlib.sha256(request_key.encode()).hexdigest()+'.json'),record)
        return record

    async def execute(self, run: Any) -> None:
        root=self.host.skill_service.workspace.resolve(f'.photomatagent/capability-benchmark/{run.task_id}')
        manifest=json.loads((root/'manifest.json').read_text())
        from photomatagent.scientific.evolution.qd.evaluation_runtime import evaluation_software_sha256
        if digest(manifest)!=run.request.host_operation['manifest_sha256'] or manifest['software_sha256']!=evaluation_software_sha256():
            raise ValueError('FROZEN_BENCHMARK_CHANGED')
        payload=CapabilityBenchmarkRequest.model_validate(manifest['request'])
        methods=[self.host.skill_service.read(m['skill_id']) for m in manifest['methods']]
        if any(m['contract_sha256']!=f['contract_sha256'] for m,f in zip(methods,manifest['methods'],strict=True)):
            raise ValueError('FROZEN_METHOD_CHANGED')
        total_cases=len(set(payload.conditions))*payload.repeats*sum(len(c) for c in manifest['dataset'].values())
        case_tokens=max(1,payload.max_tokens//total_cases);case_calls_limit=max(1,payload.max_model_calls//total_cases)
        spent_before=sum(float(row.get('wall_seconds',0)) for path in root.glob('*.json') if 'case_id' in (row:=json.loads(path.read_text())))
        started=time.monotonic(); calls=0; tokens=0; results=[]
        for condition in dict.fromkeys(payload.conditions):
            for repeat in range(payload.repeats):
                for kind,suite in manifest['dataset'].items():
                    family=[m for m in methods if m['checker']['checker_id']==kind]
                    for case in suite:
                        case_key=f'{condition}-{repeat}-{kind}-{case["id"]}'
                        saved=root/(case_key+'.json')
                        if saved.exists():
                            row=json.loads(saved.read_text()); results.append(row);calls+=row['model_calls'];tokens+=row['tokens'];continue
                        if calls>=payload.max_model_calls or tokens>=payload.max_tokens or spent_before+time.monotonic()-started>=payload.wall_seconds:
                            break
                        case_root=root/'cases'/case_key
                        case_root.mkdir(parents=True,exist_ok=True)
                        task_id='task_'+hashlib.sha256(case_key.encode()).hexdigest()[:32]
                        run_id='run_'+hashlib.sha256((run.task_id+case_key).encode()).hexdigest()[:32]
                        refs={}; inputs={}
                        for key,data in case['inputs'].items():
                            path=case_root/f'tmp/workbench/{task_id}/{key}'
                            path.parent.mkdir(parents=True,exist_ok=True)
                            path.write_text(data if key=='report' else json.dumps(data))
                            refs[key]=str(path.relative_to(case_root));inputs[key]=path
                        input_digests={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in inputs.items()}
                        condition_ids=manifest['condition_methods'].get(condition,[])
                        allowed=[m for m in family if m['skill_id'] in condition_ids]
                        if condition=='SINGLE_BEST':allowed=allowed[:1]
                        if condition in {'NO_TOOL_EVOLUTION','NO_VALIDATION_GATE'} and not allowed:
                            row={'case_id':case_key,'condition':condition,'repeat':repeat,'family':kind,'machine_status':'UNKNOWN',
                                 'runtime_stop':'CONDITION_UNAVAILABLE','model_calls':0,'tokens':0,'estimated_tokens':False,
                                 'wall_seconds':0.,'diagnostic':manifest['condition_notes'][condition], 'scientific_status':'UNKNOWN','expert_review':'PENDING'}
                            immutable(saved,row);results.append(row);continue
                        frozen=FrozenEvaluationSkills(case_root,self.host.skill_service,[m['skill_id'] for m in allowed], allow_unqualified=condition=='NO_VALIDATION_GATE',environment_method_ids=[m['skill_id'] for m in family])
                        generic=frozen.generic_sandbox(task_id,run_id)
                        permissions=_FrozenComputationPolicy(task_id,learned=bool(allowed))
                        # Separate context/workspace and permissions per case.
                        # local_dev itself is not an OS security boundary.
                        budget=BudgetState(max_iterations=min(case_calls_limit,payload.max_model_calls-calls),max_total_tokens=min(case_tokens,payload.max_tokens-tokens))
                        runtime,_=RuntimeFactory.build(provider=manifest['provider'],model=manifest['model'],workspace_root=case_root,
                            capability_task_id=task_id,capability_run_id=run_id,execution_services={'local_skill_service':frozen,'capability_sandbox_service':generic},
                            evaluation_isolation=True,permission_policy=permissions,budget=budget,log_events=False,
                            context_engine_config=ContextEngineConfig.model_validate(manifest.get('runtime_context_config',{})),
                            context_builder=ContextBuilder(SkillLoader(case_root/'frozen-skill-index', sources=[], include_published=False)))
                        prompt=f'Frozen research computation {kind}. Bound input refs: {refs}. Actual frozen input values: {json.dumps(case['inputs'])}. Return only JSON under the checker contract, scientific_status UNKNOWN. '
                        environments=[{'environment_id':e.environment.environment_id,'requirements':list(e.plan.direct_requirements)} for e in frozen.frozen_backends.values()]
                        prompt+=f'All conditions may use calculator or write Python scripts under tmp/workbench/{task_id}/ then script_run with environment_id from {environments}, input_refs={refs}; write result.json in current working directory. Dependencies are already installed; no install/network or input modification is permitted. '
                        if allowed:
                            bindings=[{'skill_id':m['skill_id'],'scope':m['scope'],'contract_sha256':m['contract_sha256']} for m in allowed]
                            prompt+=f'Use a frozen method through tool_call -> skill_run. Available exact bindings: {bindings}; input_refs={refs}. Select by applicability; if a method fails preserve the diagnosis and choose another allowed version within budget. '
                        if condition=='FEEDBACK_MEMORY':
                            prompt+='Actual frozen prior validated procedures (guidance only; no executable tool in this condition): '+json.dumps(manifest['memory'])[:6000]+' '
                        if condition=='NO_VALIDATION_GATE':
                            prompt+='Ablation: prior validation guidance is withheld. Host evaluation checks remain mandatory. '
                        prompt+='RC output: samples[{frequency_hz,real_ohm,imag_ohm}]. Table output: mismatches[]. Do not invent device performance.'
                        cancelled=False
                        text='';stop='UNKNOWN';pending=0;case_calls=0;cost_tokens=0;local_started=time.monotonic();verdict='UNKNOWN';diagnostic=None
                        try:
                            async with asyncio.timeout(max(.01,payload.wall_seconds-spent_before-(time.monotonic()-started))):
                                for attempt in range(2 if condition in {'FEEDBACK_ONLY','FEEDBACK_MEMORY'} else 1):
                                    current=''
                                    async for event in runtime.run(prompt):
                                        data=event.model_dump(mode="json")
                                        if event.kind in {'model_request_started','model_response_completed'}:
                                            self.host.store.append_event_for_run(run.run_id,event.kind,data | {'case_id':case_key})
                                        self.host.store.append_event_for_run(run.run_id,'benchmark_runtime_event',{'case_id':case_key,'type':event.kind,'payload':data})
                                        if event.kind=='model_request_started':pending+=1;case_calls+=1
                                        elif event.kind=='model_response_completed':
                                            pending=max(0,pending-1)
                                            text=current if data.get('tool_call_count')==0 and data.get('finish_reason')=='stop' else ''
                                            current=''
                                        elif event.kind=='text_delta':current+=str(data.get('text',''))
                                        elif event.kind=='loop_completed':stop=str(data.get('reason','UNKNOWN'))
                                    document=None
                                    used=[u for m in allowed for u in frozen.uses(m['skill_id']) if u.get('task_id')==task_id and u.get('code')=='OK']
                                    if used:
                                        result_path=next((p for p in used[-1]['artifacts'] if p.endswith('/result.json')),None)
                                        if result_path:document=json.loads((case_root/result_path).read_text())
                                    if document is None:
                                        for execution in reversed(list(generic._records.values())):
                                            if execution.state=='COMPLETED' and execution.exit_code==0:
                                                output=case_root/'tmp/capability-executions'/execution.execution_id/'result.json'
                                                artifact=next((a for a in execution.artifacts if a.path=='result.json'),None)
                                                if artifact and output.is_file() and hashlib.sha256(output.read_bytes()).hexdigest()==artifact.sha256:
                                                    document=json.loads(output.read_text());break
                                    if document is None and text:
                                        cleaned=text.strip()
                                        if cleaned.startswith('```'):
                                            cleaned='\n'.join(cleaned.splitlines()[1:-1])
                                        try:document=json.loads(cleaned)
                                        except ValueError:document=None
                                    if document is not None:
                                        output=case_root/'answer.json';output.write_text(json.dumps(document))
                                        if any(not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=input_digests[key] for key,path in inputs.items()):
                                            checked={'status':'FAIL','diagnostic':'FROZEN_INPUT_CHANGED'}
                                            diagnostic='FROZEN_INPUT_CHANGED'
                                        else:
                                            checked=CHECKERS[kind](CheckerSpec.model_validate({'checker_id':kind,'output_path':'result.json','field':'unused'}),inputs,output)
                                        verdict=checked['status']
                                    if verdict=='PASS':break
                                    prompt=f'Repair the prior answer {text[:3000]}. Actual deterministic checker status: {verdict}; diagnostics: {checked if document is not None else "no parseable answer"}. Same frozen inputs: {json.dumps(case["inputs"])}. Return corrected JSON. Evidence absent remains UNKNOWN.'
                        except asyncio.CancelledError:
                            cancelled=True;diagnostic='CANCELLED'
                        except Exception as error:
                            diagnostic=type(error).__name__+': '+str(error)[:500]
                        finally:
                            cost_tokens=budget.total_tokens
                            if pending:cost_tokens=max(cost_tokens,min(case_tokens,payload.max_tokens-tokens))
                            close=getattr(runtime._model,'aclose',None)
                            if close:await close()
                        row={'case_id':case_key,'condition':condition,'repeat':repeat,'family':kind,'machine_status':verdict,
                             'runtime_stop':stop,'model_calls':case_calls,'tokens':cost_tokens,'estimated_tokens':bool(pending),
                             'wall_seconds':time.monotonic()-local_started,'diagnostic':diagnostic,'scientific_status':'UNKNOWN','expert_review':'PENDING'}
                        immutable(saved,row);results.append(row);calls+=case_calls;tokens+=cost_tokens
                        if cancelled:raise asyncio.CancelledError
        report={'manifest_sha256':digest(manifest),'software_sha256':manifest['software_sha256'],'model':{'provider':manifest['provider'],'model':manifest['model']},
                'dataset_sha256':manifest['dataset_sha256'],'results':results,'model_calls':calls,'tokens':tokens,'wall_seconds':spent_before+time.monotonic()-started,
                'condition_methods':manifest['condition_methods'],'condition_notes':manifest['condition_notes'],
                'per_case_caps':{'tokens':case_tokens,'model_calls':case_calls_limit},
                'metrics':{condition:{'cases':len(rows:=[r for r in results if r['condition']==condition]),'machine_pass':sum(r['machine_status']=='PASS' for r in rows),
                                      'tokens':sum(r['tokens'] for r in rows),'scientific_quality':'PENDING_BLIND_EXPERT_REVIEW'} for condition in payload.conditions},
                'scientific_status':'UNKNOWN','effect_claim':'NOT_ESTABLISHED','status':'EXECUTION_ERRORS' if any(r['diagnostic'] and r['runtime_stop']!='CONDITION_UNAVAILABLE' for r in results) else 'COMPLETE' if len(results)==total_cases else 'BUDGET_LIMITED'}
        immutable(root/'report.json',report)
        self.host.store.append_event_for_run(run.run_id,'capability_benchmark_completed',report)
        self.host.store.append_event_for_run(run.run_id,'loop_completed',{'reason':'completed' if report['status']=='COMPLETE' else 'unresolved'})
