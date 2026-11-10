"""Domain method candidates, separate from software and execution records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Literal, TYPE_CHECKING

from photomatagent.capability_acquisition.models import ExecutionRequest, MethodNeed, StrictModel
from photomatagent.capability_acquisition.web_sources import tool_result
from photomatagent.errors import ToolExecutionError
from photomatagent.persistence.atomic import atomic_write
from photomatagent.scientific.applicability import assess_method, domain_exclusions, with_domain_input
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.workspace import Workspace

if TYPE_CHECKING:
    from photomatagent.capability_acquisition.discovery import CapabilityDiscoveryService
    from photomatagent.capability_acquisition.repository_sources import RepositorySearchService
    from photomatagent.capability_acquisition.web_sources import WebSearchService
    from photomatagent.scientific.capabilities.literature.federated import FederatedLiterature
    from photomatagent.scientific.state import ScientificState


class MethodCandidate(StrictModel):
    schema_version: Literal[1] = 1
    method_id: str
    name: str
    required_inputs: tuple[str, ...]
    limits: tuple[str, ...]
    tool_names: tuple[str, ...] = ()
    verification: str


def _method(
    method_id: str, name: str, inputs: tuple[str, ...], limits: tuple[str, ...],
    verification: str, *tools: str,
) -> MethodCandidate:
    return MethodCandidate(method_id=method_id, name=name, required_inputs=inputs,
                           limits=limits, verification=verification, tool_names=tools)


_CATALOG: tuple[tuple[MethodCandidate, tuple[str, ...]], ...] = (
    (_method("transfer-matrix", "Transfer matrix for stratified films",
             ("complex refractive index spectra", "film thickness", "incidence and polarization"),
             ("Laterally homogeneous layers", "Use a coherence model appropriate to each layer"),
             "Compare R/T with a Fresnel interface and check energy balance."),
     ("thin film", "thinfilm", "multilayer", "transfer matrix", "spectrum", "薄膜", "多层", "光谱", "涂层")),
    (_method("fdtd-thin-film", "FDTD thin-film optical response",
             ("optical constants", "film thickness", "wavelength range"),
             ("Check discretization and dispersive material support", "Requires an enabled solver"),
             "Check convergence with resolution and a reference optical stack.", "optics.meep_thinfilm"),
     ("thin film", "thinfilm", "fdtd", "multilayer", "薄膜", "多层", "光谱")),
    (_method("effective-mass-confinement", "Effective-mass quantum confinement",
             ("dot size", "effective masses", "dielectric constant", "bulk band gap"),
             ("Approximation for near-parabolic bands", "Check strong confinement and band nonparabolicity"),
             "Compare against a measured size series or higher-fidelity calculation.", "qd.brus_transition_energy"),
     ("confinement", "quantum dot", "brus", "band gap", "量子点", "禁闭", "带隙")),
    (_method("band-bowing", "Composition-dependent alloy band bowing",
             ("composition", "endpoint gaps", "bowing parameter"),
             ("Require compatible phase, temperature and gap definition", "Stay inside the fitted alloy domain"),
             "Compare against reported points in the same alloy phase.", "alloy.bandgap_bowing"),
     ("bowing", "alloy", "composition", "合金", "组分")),
    (_method("kp-heterostructure", "Multiband k.p for bulk and planar heterostructures",
             ("material parameters", "layer sequence", "boundary conditions"),
             ("Require an enabled solver and model parameters for the material system",
              "The kdotpy adapter does not solve zero-dimensional dot confinement"),
             "Check the selected basis and compare a bulk or well limit.", "kp.run_kdotpy"),
     ("k.p", "kdotpy", "heterostructure", "quantum well", "异质结构", "量子阱")),
    (_method("carrier-transport", "Carrier transport model comparison",
             ("carrier-density or current data", "temperature", "sample geometry"),
             ("Different transport regimes can fit the same sparse data", "Require condition-matched data"),
             "Compare against independent temperature or field dependence.", "transport.analyze"),
     ("transport", "mobility", "ligand", "输运", "迁移率", "配体")),
    (_method("film-process-controls", "Film deposition and interface process controls",
             ("material and substrate", "process equipment", "thermal budget"),
             ("Transfer requires explicit substrate and interface assumptions", "Use independent replicates"),
             "Compare morphology, phase and electrical controls across replicate batches."),
     ("deposition", "anneal", "interface", "process", "ligand exchange", "配体交换", "工艺", "退火", "沉积", "界面")),
    (_method("structure-relaxation", "Structure relaxation with an interatomic potential",
             ("initial structure", "elements", "checkpoint"),
             ("Require valid element and structure coverage", "Energy alone does not establish stability"),
             "Check convergence and compare with reference structures or an independent method.",
             "chgnet.screen", "chgnet.relax"),
     ("relaxation", "structure", "geometry", "结构", "弛豫")),
    (_method("structure-generation", "Candidate composition or crystal generation",
             ("supported chemical system", "checkpoint or formula prior", "target conditions"),
             ("Check the generator's training and conditioning domain",
              "Generated candidates require separate structural and property validation"),
             "Verify composition and structure identity before selecting an independent checker.",
             "generation.vae_formula", "generation.mattergen"),
     ("candidate generation", "mattergen", "vae", "crystal generation", "结构生成", "组分生成")),
    (_method("detector-noise", "Condition-matched photodetector noise and responsivity",
             ("measured noise spectrum", "responsivity", "active area", "bias and temperature"),
             ("Do not substitute shot-noise estimates for measured noise", "Use the same sample and bandwidth"),
             "Recompute detectivity from independently measured noise and responsivity."),
     ("detector", "responsivity", "noise", "detectivity", "探测", "噪声", "响应率")),
)


def _matches(term: str, query: str) -> bool:
    if term.isascii():
        return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", query) is not None
    return term in query


def evaluate_method_candidate(
    need: MethodNeed, candidate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply hard constraints and annotate gaps without inferring applicability."""
    record = candidate.get("record")
    if isinstance(record, dict):
        identifier = candidate["method_id"]
        metadata = record
        inputs: list[str] = []
    else:
        metadata = candidate["method"]
        identifier = metadata["method_id"]
        inputs = list(candidate.get("replay_input_keys", ()))
    name = metadata.get("name") or metadata.get("repository") or metadata.get("normalized_name") or metadata.get("title")
    exclusion = {"method_id": identifier}
    excluded = {value.strip().casefold() for value in need.exclusions}
    if excluded.intersection(value.strip().casefold() for value in (identifier, name) if isinstance(value, str)):
        return candidate, exclusion | {"reason": "METHOD_EXPLICITLY_EXCLUDED"}

    domain = assess_method(identifier, need.regimes)
    if domain.status == "UNSUPPORTED":
        return candidate, exclusion | {
            "reason": "SCIENTIFIC_REGIME_INCOMPATIBLE", "regimes": list(domain.incompatible_regimes),
            "reasons": list(domain.reasons),
        }

    required = metadata.get("required_inputs", ())
    if isinstance(required, dict):
        schema = required.get("input_schema")
        required = schema.get("required", ()) if isinstance(schema, dict) else ()
    if isinstance(required, (list, tuple)):
        inputs.extend(value for value in required if isinstance(value, str) and value not in inputs)
    fixed = {method_input_identity(value) for value in need.fixed_unavailable_inputs}
    forbidden = [value for value in inputs if method_input_identity(value) in fixed]
    if forbidden:
        return candidate, exclusion | {"reason": "REQUIRED_INPUT_FIXED_UNAVAILABLE", "inputs": forbidden}
    unavailable = {method_input_identity(value) for value in need.unavailable_inputs}
    blocked = [value for value in inputs if method_input_identity(value) in unavailable]

    unknown = {"", "unknown", "noassertion"}
    license_hint = str(metadata.get("license_hint") or metadata.get("license") or "unknown").strip()
    allowed = {value.strip().casefold() for value in need.allowed_licenses}
    license_unknown = license_hint.casefold() in unknown
    if allowed and not license_unknown and license_hint.casefold() not in allowed:
        return candidate, exclusion | {"reason": "LICENSE_NOT_ALLOWED", "license_hint": license_hint}

    platforms = metadata.get("platforms")
    reported = [
        value for value in platforms if isinstance(value, str) and value.strip().casefold() not in unknown
    ] if isinstance(platforms, list) else []
    if (need.required_platform and reported
            and need.required_platform.strip().casefold() not in {value.strip().casefold() for value in reported}):
        return candidate, exclusion | {"reason": "PLATFORM_NOT_SUPPORTED", "platforms": reported}

    known = {method_input_identity(value) for value in need.known_inputs}
    missing = [value for value in inputs if method_input_identity(value) not in known | unavailable]
    unconfirmed = []
    if need.required_platform and not reported:
        unconfirmed.append("platform")
    if allowed and license_unknown:
        unconfirmed.append("license")
    existing = candidate.get("comparison")
    comparison = {
        **(existing if isinstance(existing, dict) else {}),
        "known_input_count": len(inputs) - len(missing) - len(blocked),
        "license_hint": license_hint,
        "platform_to_verify": need.required_platform,
        "reported_platforms": reported or "UNKNOWN",
        "unconfirmed_constraints": unconfirmed,
    }
    return candidate | {
        "applicability": "UNKNOWN", "comparison": comparison,
        "unconfirmed_inputs": missing, "unavailable_inputs": blocked,
        "execution_blocker": "REQUIRED_INPUT_UNAVAILABLE" if blocked else None,
        "conditional": bool(blocked or missing or unconfirmed),
    }, None


def search_methods(need: MethodNeed, limit: int = 5) -> dict[str, Any]:
    query = re.sub(r"[-\s]+", " ", " ".join((need.query, need.operation or "", need.scale or "", *need.desired_outputs)).casefold())
    matches = []
    excluded_methods = []
    for candidate, terms in _CATALOG:
        matched = [term for term in terms if _matches(term, query)]
        if matched:
            row, rejection = evaluate_method_candidate(need, {
                "method": candidate.model_dump(mode="json"), "matched_terms": matched, "applicability": "UNKNOWN",
                "comparison": {"task_match": "catalog_terms", "evidence_scope": "method_seed",
                               "execution": "UNVERIFIED", "cost": "UNKNOWN", "reference_check": candidate.verification},
                "sources": [{"kind": "local_method_catalog", "id": candidate.method_id}],
            })
            if rejection:
                excluded_methods.append(rejection)
                continue
            matches.append(((len(matched), -len(row["unconfirmed_inputs"])), row))
    matches.sort(key=lambda item: item[0], reverse=True)
    return {
        "status": "searched" if matches else "incompatible" if excluded_methods else "no_match",
        "need": need.model_dump(mode="json"),
        "candidates": [item[1] for item in matches[:limit]],
        "source_status": {"local_method_catalog": "OK"},
        "excluded": excluded_methods,
    }


_ACADEMIC = ("crossref", "openalex", "semantic_scholar", "zenodo")
_EXTERNAL = ("web", "github", "pypi", *_ACADEMIC)
_INPUT_ALIASES = {"膜厚": "film thickness", "薄膜厚度": "film thickness", "膜层厚度": "film thickness",
                  "波长范围": "wavelength range", "光学常数": "optical constants", "样品几何": "sample geometry",
                  "温度": "temperature", "载流子有效质量": "effective masses", "量子点尺寸": "dot size"}


def method_input_identity(value: str) -> str:
    return _INPUT_ALIASES.get(value.strip(), " ".join(value.casefold().split()))


def _lead_id(source: str, record: dict[str, Any]) -> str:
    return "lead_" + hashlib.sha256(json.dumps([source, record], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:32]


class MethodDiscoveryService:
    def __init__(self, workspace: Workspace | None = None, *, web: WebSearchService | None = None,
                 repositories: RepositorySearchService | None = None, literature: FederatedLiterature | None = None,
                 packages: CapabilityDiscoveryService | None = None) -> None:
        self.workspace, self.web, self.repositories = workspace, web, repositories
        self.literature, self.packages = literature, packages
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._leads: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _archive(self) -> tuple[str, list[dict[str, Any]]]:
        from photomatagent.application.errors import ServiceError
        from photomatagent.application.skills import LocalSkillService
        from photomatagent.scientific.evolution.qd.store import QDStore, QDStoreError

        if self.workspace is None:
            return "NOT_CONFIGURED", []
        try:
            archives = self.workspace.resolve(".photomatagent/qd/archives", must_exist=False)
            if not archives.is_dir():
                return "EMPTY", []
            store = QDStore(self.workspace, trial_factory=lambda *_: (_ for _ in ()).throw(ValueError("READ_ONLY")))
            directories = sorted(archives.iterdir())[:128]
        except (OSError, ValueError, QDStoreError, ToolExecutionError):
            return "ARCHIVE_UNAVAILABLE", []
        skills = LocalSkillService(self.workspace.root)
        rows = []
        unavailable = False
        for directory in directories:
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                for entry in store.load_archive_entries(directory.name):
                    trial = store.load_trial(entry.trial_ids[-1])
                    if not trial.attempt_dir.startswith(".photomatagent/capability-evolution/"):
                        continue
                    check = store.load_validation(trial.trial_id, entry.validation_ids[-1])
                    genome = store.load_genome(entry.genome_id)
                    method = next(step for step in genome.steps if step.binding_id == "skill_run")
                    skill_id = str(method.parameters["skill_id"])
                    skill = skills.read(skill_id)
                    if (check.qualification != "PASS" or not check.outcomes
                            or any(row.verdict != "PASS" for row in check.outcomes)
                            or skill["contract_sha256"] != method.parameters["contract_sha256"]
                            or check.genome_sha256 != genome.content_sha256
                            or (skills._directory(skill_id) / "disabled").exists()
                            or not trial.output_refs):
                        continue
                    report_ref = trial.output_refs[0]
                    raw = self.workspace.resolve(report_ref.path).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != report_ref.sha256:
                        raise ValueError("ARCHIVE_REPORT_CHANGED")
                    report = json.loads(raw)
                    if not all(case.get("use_sha256") for case in report["cases"]):
                        unavailable = True
                        continue
                    request = ExecutionRequest.model_validate(skill["request"])
                    rows.append({
                        "method": {"method_id": skill_id, "name": skill["name"], "required_inputs": skill.get("interface", {}),
                                   "limits": genome.applicability, "tool_names": ["skill_run"], "verification": skill["checker"]},
                        "replay_input_keys": list(request.replay_input_keys),
                        "applicability": "UNKNOWN",
                        "comparison": {"task_match": "archived_scope", "evidence_scope": "independent_host_suite",
                                       "execution": "REPLAYABLE", "cost": dict(trial.cost)},
                        "sources": [{"kind": "capability_archive", "trial_id": trial.trial_id,
                                     "validation_id": check.validation_id, "report_sha256": report_ref.sha256,
                                     "contract_sha256": skill["contract_sha256"], "provenance": skill.get("reproduction")}],
                    })
            except (OSError, ValueError, KeyError, StopIteration, ToolExecutionError, QDStoreError, ServiceError):
                unavailable = True
        return ("PARTIAL_OR_LEGACY" if unavailable else "OK"), rows

    async def search(self, need: MethodNeed, limit: int, sources: tuple[str, ...] = ()) -> dict[str, Any]:
        result = search_methods(need, limit)
        archive_status, archived = await asyncio.to_thread(self._archive)
        result["source_status"]["capability_archive"] = archive_status
        tokens = set(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", need.query.casefold()))
        archived_matches = []
        archived_excluded = []
        for row in archived:
            searchable = json.dumps(row["method"], ensure_ascii=False)
            score = len(tokens.intersection(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", searchable.casefold())))
            if score:
                evaluated, rejection = evaluate_method_candidate(need, row)
                if rejection:
                    archived_excluded.append(rejection)
                else:
                    archived_matches.append((score, evaluated))
        archived_matches.sort(key=lambda pair: (-pair[0], pair[1]["method"]["method_id"]))
        local = result["candidates"]
        if archived_matches:
            archived_rows = [row for _, row in archived_matches]
            result["candidates"] = (archived_rows[:1] + local[:1] + archived_rows[1:] + local[1:])[:limit]
        result["excluded"].extend(archived_excluded[:limit])
        if sources:
            network = await self._external(need, limit, sources)
            result["implementations"] = network["implementations"]
            result["literature_evidence_leads"] = network["literature_evidence_leads"]
            result["source_status"].update(network["source_status"])
            result["excluded"].extend(network["excluded"])
        if result["candidates"] or result.get("implementations") or result.get("literature_evidence_leads"):
            result["status"] = "searched"
        elif result["excluded"]:
            result["status"] = "incompatible"
        elif sources and not any(result["source_status"].get(source) == "OK" for source in sources):
            result["status"] = "source_unavailable"
        return result

    async def _external(self, need: MethodNeed, limit: int, sources: tuple[str, ...]) -> dict[str, Any]:
        fingerprint = json.dumps([need.model_dump(mode="json"), sources, limit], sort_keys=True, ensure_ascii=False)
        if (cached := self._cache.get(fingerprint)) and cached[0] > time.monotonic():
            result = deepcopy(cached[1])
            self._remember_leads(result)
            return result
        async def fetch(source: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
            if source == "pypi" and self.packages:
                data = await self.packages.search(need.query, limit=limit)
                return {source: data.code}, [item.model_dump(mode="json") for item in data.candidates]
            provider = self.web if source == "web" else self.repositories if source == "github" else None
            if provider:
                result = await provider.search(need.query, limit)
                return {source: str(result.get("diagnostic", "OK"))}, result.get("hits", [])
            if source == "academic" and self.literature:
                result = await self.literature.search(need.query, tuple(s for s in sources if s in _ACADEMIC), limit)
                return result["source_status"], result["hits"]
            return {s: "NOT_CONFIGURED" for s in sources if s in _ACADEMIC} if source == "academic" else {source: "NOT_CONFIGURED"}, []

        routes = tuple(s for s in sources if s not in _ACADEMIC)
        if any(s in _ACADEMIC for s in sources):
            routes += ("academic",)
        collected = await asyncio.gather(*(fetch(source) for source in routes), return_exceptions=True)
        status: dict[str, str] = {}
        implementations: list[tuple[int, dict[str, Any]]] = []
        papers: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for source, response in zip(routes, collected, strict=True):
            if isinstance(response, BaseException):
                for name in (tuple(s for s in sources if s in _ACADEMIC) if source == "academic" else (source,)):
                    status[name] = "SOURCE_UNAVAILABLE"
                continue
            source_status, hits = response
            status.update(source_status)
            source_rank = 0
            for hit in hits:
                identifier = _lead_id(source, hit)
                lead = {"method_id": identifier, "source": source, "record": hit, "applicability": "UNKNOWN",
                        "comparison": {"task_match": "source_search", "evidence_scope": "metadata_only",
                                       "execution": "UNVERIFIED", "cost": "UNKNOWN",
                                       "version": hit.get("version") or "UNKNOWN",
                                       "default_branch_hint": hit.get("default_branch") or "UNKNOWN",
                                       "last_upstream_update": hit.get("pushed_at") or "UNKNOWN",
                                       "archived_upstream": hit.get("archived"),
                                       "reference_case": "UNVERIFIED"},
                        "verify": ["Identify the method, exact version and conditions before using it.",
                                   "Check required inputs, platform, license and reference cases."]}
                lead, rejection = evaluate_method_candidate(need, lead)
                if rejection:
                    rejected.append(rejection)
                    continue
                if source == "academic":
                    papers.append(lead)
                else:
                    implementations.append((source_rank, lead))
                    source_rank += 1
        # Keep the first result from every available implementation source before adding more.
        implementations.sort(key=lambda item: item[0])
        result = {"implementations": [lead for _, lead in implementations[:limit]], "literature_evidence_leads": papers[:limit],
                  "source_status": status, "excluded": rejected[:limit]}
        self._cache[fingerprint] = (time.monotonic() + (120 if implementations or papers else 15), deepcopy(result))
        if len(self._cache) > 32:
            self._cache.popitem(last=False)
        self._remember_leads(result)
        return result

    def _remember_leads(self, result: dict[str, Any]) -> None:
        for lead in result["implementations"] + result["literature_evidence_leads"]:
            self._leads[lead["method_id"]] = lead
            self._leads.move_to_end(lead["method_id"])
            if self.workspace is not None:
                relative = f".photomatagent/method-discovery/{lead['method_id']}.json"
                try:
                    path = self.workspace.resolve(relative, must_exist=False)
                    payload = json.dumps(lead, ensure_ascii=False).encode()
                    if self.workspace.relative(path) == relative and len(payload) <= 65536:
                        atomic_write(path, payload)
                except (OSError, ValueError, ToolExecutionError):
                    pass
        while len(self._leads) > 256:
            self._leads.popitem(last=False)

    async def inspect(self, identifier: str) -> dict[str, Any] | None:
        for candidate, _ in _CATALOG:
            if candidate.method_id == identifier:
                return {"method": candidate.model_dump(mode="json"), "applicability": "UNKNOWN",
                        "incompatible_regimes": sorted(domain_exclusions(identifier)),
                        "domain_exclusions": domain_exclusions(identifier),
                        "sources": [{"kind": "local_method_catalog", "id": identifier}]}
        if re.fullmatch(r"lead_[0-9a-f]{32}", identifier):
            if identifier in self._leads:
                return self._leads[identifier]
            if self.workspace is not None:
                try:
                    relative = f".photomatagent/method-discovery/{identifier}.json"
                    path = self.workspace.resolve(relative)
                    if self.workspace.relative(path) == relative and path.stat().st_size <= 65536:
                        snapshot = json.loads(path.read_bytes())
                        if (isinstance(snapshot.get("record"), dict)
                                and snapshot.get("method_id") == identifier == _lead_id(snapshot["source"], snapshot["record"])):
                            return snapshot
                except (OSError, ValueError, KeyError, AttributeError, ToolExecutionError):
                    pass
            return None
        _, entries = await asyncio.to_thread(self._archive)
        return next((row for row in entries if row["method"]["method_id"] == identifier), None)


class MethodSearchTool(Tool):
    name = "method_search"
    description = "Compare a research need against local methods and verified capability archives. Discover external leads with method_search_external."
    short_description = "Discover materials research methods."
    namespace = "research"
    source = "builtin"
    exposure = ToolExposure.DIRECT
    tags = ("methods", "materials", "discovery")
    input_schema = {
        "type": "object", "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 500},
            "known_inputs": {"type": "array", "maxItems": 20, "items": {"type": "string", "maxLength": 200}},
            "exclusions": {"type": "array", "maxItems": 20, "items": {"type": "string", "maxLength": 200}},
            "unavailable_inputs": {"type": "array", "maxItems": 20, "items": {"type": "string", "maxLength": 200}},
            "fixed_unavailable_inputs": {"type": "array", "maxItems": 20, "items": {"type": "string", "maxLength": 200},
                                         "description": "Inputs the task forbids obtaining or varying."},
            "allowed_licenses": {"type": "array", "maxItems": 20, "items": {"type": "string", "maxLength": 200}},
            "required_platform": {"type": "string", "minLength": 1, "maxLength": 200},
            "operation": {"type": "string", "minLength": 1, "maxLength": 200},
            "scale": {"type": "string", "minLength": 1, "maxLength": 200},
            "desired_outputs": {"type": "array", "maxItems": 12, "items": {"type": "string", "minLength": 1, "maxLength": 200}},
            "conditions": {"type": "array", "maxItems": 12, "items": {"type": "string", "minLength": 1, "maxLength": 200}},
            "limit": {"type": "integer", "minimum": 1, "maximum": 8},
        }, "required": ["query"], "additionalProperties": False,
    }

    input_schema = with_domain_input(input_schema)

    def __init__(self, service: MethodDiscoveryService | None = None, state: ScientificState | None = None) -> None:
        self.service = service or MethodDiscoveryService()
        self.state = state

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        need = MethodNeed.model_validate({key: value for key, value in arguments.items() if key != "limit"})
        return await self._search(need, int(arguments.get("limit", 5)))

    async def _search(self, need: MethodNeed, limit: int, sources: tuple[str, ...] = ()) -> ToolResult:
        from photomatagent.scientific.process import MethodNeedRecord

        data = await self.service.search(need, limit, sources)
        result = tool_result(data)
        if self.state is not None:
            result.state_updates.append(MethodNeedRecord(
                goal=self.state.goal, need=need, origin=self.name, basis="Model-requested search",
                status=data["status"],
            ))
        return result


class MethodSearchExternalTool(MethodSearchTool):
    name = "method_search_external"
    description = "Compare local methods and query selected external sources for implementation and literature leads."
    short_description = "Find methods and authorized public-source leads."
    exposure = ToolExposure.DEFERRED
    input_schema = {
        **MethodSearchTool.input_schema,
        "properties": {**MethodSearchTool.input_schema["properties"],
                       "sources": {"type": "array", "minItems": 1, "maxItems": len(_EXTERNAL),
                                   "items": {"type": "string", "enum": list(_EXTERNAL)}}},
        "required": ["query", "sources"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        need = MethodNeed.model_validate({key: value for key, value in arguments.items() if key not in {"limit", "sources"}})
        sources = tuple(dict.fromkeys(s for s in arguments["sources"] if s in _EXTERNAL))
        return await self._search(need, int(arguments.get("limit", 5)), sources)


class MethodInspectTool(Tool):
    name = "method_inspect"
    description = "Inspect a materials method returned by method_search, including limits and validation checks."
    short_description = "Inspect one materials method."
    namespace = "research"
    source = "builtin"
    exposure = ToolExposure.DEFERRED
    tags = ("methods", "materials", "applicability")
    input_schema = {"type": "object", "properties": {"method_id": {"type": "string", "maxLength": 100},
                                                     "need": MethodNeed.model_json_schema()},
                    "required": ["method_id"], "additionalProperties": False}

    def __init__(self, service: MethodDiscoveryService | None = None) -> None:
        self.service = service or MethodDiscoveryService()

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        if result := await self.service.inspect(arguments["method_id"]):
            if "need" in arguments:
                result, rejection = evaluate_method_candidate(MethodNeed.model_validate(arguments["need"]), result)
                if rejection:
                    return tool_result(result | {"applicability": "UNSUPPORTED", "exclusion": rejection})
            return tool_result(result)
        return tool_result({"diagnostic": "METHOD_NOT_FOUND"}, error=True)
