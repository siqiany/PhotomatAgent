"""Record actually executed method/version workflows in the authoritative QD store."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from photomatagent.scientific.evolution.qd.archive import QDArchive
from photomatagent.scientific.evolution.qd.models import (ArtifactDigest, EvaluationContract, ProposalRecord, WorkflowGenome, TrialRecord, ValidationRecord, CheckOutcome)
from photomatagent.scientific.evolution.qd.store import QDStore
from photomatagent.scientific.evolution.qd.validation import CHECKER_VERSION, canonical_contract_sha256
from photomatagent.workspace import Workspace


def archive_capability(workspace: Workspace, candidate: dict[str, Any], report: dict[str, Any], output: Path, job_task: str) -> dict[str, Any]:
    kind = candidate['checker']['checker_id']
    suite = report['suite_sha256']
    contract = EvaluationContract(contract_id='capability-' + kind, task_family=kind, version='1',
        input_schema=(candidate.get('interface') or {}).get('input_schema', {'type': 'object'}),
        output_schema=(candidate.get('interface') or {}).get('output_schema', {'type': 'object'}), checks=(),
        quality_metric='independently_checked_case_fraction', quality_direction='maximize', quality_tolerance=0.0,
        evaluation_suite_sha256=suite, seed_set=(0,), descriptor_config={'method': kind, 'cost': 'wall_seconds'})
    layout = candidate.get('request', {}).get('source_tree') or {}
    source_names = ['script.py', *(f'input-{key}' for key in sorted(layout.get('inputs', ())))]
    script_refs = []
    for name in source_names:
        path = workspace.resolve(f".photomatagent/local-skills/{candidate['skill_id']}/{name}")
        expected = candidate['files'][name]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('CAPABILITY_ARCHIVE_SOURCE_CHANGED')
        script_refs.append(ArtifactDigest(path=workspace.relative(path), sha256=expected, size_bytes=path.stat().st_size,
                                          media_type='text/x-python' if name == 'script.py' else 'application/octet-stream'))
    genome = WorkflowGenome.from_behavior(genome_id='g_' + candidate['contract_sha256'][:32], task_family=kind,
        steps=[{'step_id': 'execute', 'kind': 'registered_tool', 'binding_id': 'skill_run', 'parameters': {'skill_id': candidate['skill_id'], 'contract_sha256': candidate['contract_sha256'], 'scope': candidate['scope']},
                'assumptions': tuple((candidate.get('interface') or {}).get('assumptions', []))},
               {'step_id': 'verify', 'kind': 'registered_tool', 'binding_id': 'trusted_checker', 'depends_on': ('execute',), 'parameters': candidate['checker']}],
        entrypoint='execute', script_refs=script_refs, environment_ref=candidate['environment_sha256'],
        applicability=(candidate['scope'],), recovery=('Preserve UNKNOWN; reject qualification on a failed required check.',))
    contract_sha = canonical_contract_sha256(contract)
    event_id = 'e_' + hashlib.sha256((job_task + candidate['contract_sha256']).encode()).hexdigest()[:32]
    trial_id = 't_' + hashlib.sha256(event_id.encode()).hexdigest()[:32]
    attempt = f'.photomatagent/capability-evolution/{job_task}/qd/{trial_id}'
    destination = workspace.resolve(attempt + '/validation.json', must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != output.read_bytes():
        raise ValueError('CAPABILITY_ARCHIVE_OUTPUT_CHANGED')
    if not destination.exists():
        destination.write_bytes(output.read_bytes())
    output_ref = ArtifactDigest(path=workspace.relative(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), size_bytes=destination.stat().st_size, media_type='application/json')
    def factory(event: str, genome_id: str) -> TrialRecord:
        return TrialRecord(trial_id=trial_id, event_id=event, genome_id=genome_id, contract_sha256=contract_sha,
            input_digests=(), environment_sha256=candidate['environment_sha256'], seed_set=(0,), status='RESERVED',
            session_id=job_task, attempt_dir=attempt, runtime_trace_ref='', cost={}, output_refs=())
    store = QDStore(workspace, trial_factory=factory)
    proposal = ProposalRecord(proposal_id='p_' + trial_id[2:], parent_ids=('g_' + candidate['parent_contract_sha256'][:32],) if candidate.get('parent_contract_sha256') else (), operator='replace_verified_plugin_version',
        feedback_refs=('chain_' + report['chain_sha256'][:32],), change_summary='Use the exact method version with mandatory independent regression/transfer/negative checks.',
        genome_id=genome.genome_id, model_provenance={
            'source_execution': candidate['source'], 'model_generated_candidate': True, 'checker_host_owned': True,
            **({'reproduction': candidate['reproduction']} if candidate.get('reproduction') else {}),
        })
    archive = QDArchive(store, contract, method_class_by_binding={'skill_run': kind, 'trusted_checker': kind})
    archive.stage_genome(genome, proposal)
    existing = store.trials_for_event(event_id)
    if existing:
        if len(existing) != 1:
            raise ValueError('CAPABILITY_ARCHIVE_TRIAL_CONFLICT')
        trial = existing[0]
        reservation = factory(event_id, genome.genome_id)
        if trial.model_copy(update={'status': 'RESERVED', 'cost': {}, 'output_refs': ()}) != reservation:
            raise ValueError('CAPABILITY_ARCHIVE_TRIAL_CHANGED')
        if trial.status in {'COLLECTED', 'VALIDATED'} and (trial.output_refs != (output_ref,)
                or trial.cost != {'wall_seconds': float(report['wall_seconds']), 'model_calls': 0.0, 'tokens': 0.0}):
            raise ValueError('CAPABILITY_ARCHIVE_OUTCOME_CHANGED')
    else:
        trial = store.reserve_trial(event_id, genome.genome_id)
    if trial.status == 'RESERVED':
        trial = store.start_trial(trial.trial_id)
    if trial.status == 'RUNNING':
        trial = store.commit_trial_outcome(trial.model_copy(update={'status': 'COLLECTED', 'output_refs': (output_ref,), 'cost': {'wall_seconds': float(report['wall_seconds']), 'model_calls': 0.0, 'tokens': 0.0}}))
    outcomes = tuple(CheckOutcome(check_id='case-' + case['id'], verdict='PASS' if case['passed'] else 'FAIL',
        evidence_refs=(output_ref.sha256,), reason_code='HOST_CASE_CHECKED') for case in report['cases'])
    bundle = hashlib.sha256(json.dumps([output_ref.model_dump(mode='json')], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    validation = ValidationRecord(validation_id='v_' + trial_id[2:], trial_id=trial.trial_id, genome_id=genome.genome_id,
        genome_sha256=genome.content_sha256, bundle_sha256=bundle, contract_sha256=contract_sha, suite_sha256=suite,
        checker_version=CHECKER_VERSION, qualification='PASS' if report['status'] == 'PASS' else 'FAIL', outcomes=outcomes,
        quality=float(sum(c['passed'] for c in report['cases']) / max(1, len(report['cases']))), cost=trial.cost,
        provenance={'verification_report': output_ref.sha256, 'chain_sha256': report['chain_sha256'], 'scientific_status': 'UNKNOWN'},
        seed_set=(0,), runtime_samples_seconds=(float(report['wall_seconds']),))
    if trial.status in {'COLLECTED', 'VALIDATED'}:
        store.commit_validation(validation)
    trial = store.load_trial(trial.trial_id)
    result = archive.consider(trial=trial, validation=validation, genome=genome, contract=contract)
    active_entry = result.new_entry or result.old_entry
    return {'action': result.action, 'reason': result.reason, 'genome_id': genome.genome_id, 'trial_id': trial.trial_id,
            'active_entry': active_entry.model_dump(mode='json') if active_entry else None,
            'quality_scope': 'mathematical_case_contract_only', 'scientific_status': 'UNKNOWN'}
