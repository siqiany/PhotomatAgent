"""Build hash-bound candidate skill bundles from independently validated QD trials."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from photomatagent.scientific.evolution.qd.models import (
    ArtifactDigest, SkillPackage, TrialOutcome, TrialRecord, ValidationRecord,
    WorkflowGenome, canonical_genome_sha256,
)
from photomatagent.workspace import Workspace


LockLoader = Callable[[str], bytes]
ReplayRunner = Callable[..., Awaitable[TrialOutcome]]
_MAX_FIXTURE_BYTES = 1_000_000
_MAX_LOCK_BYTES = 2_000_000


class SkillPackageBuilder:
    """Stage and independently replay candidates; never publish them."""

    def __init__(self, *, workspace: Workspace, qd_store: Any, lock_loader: LockLoader) -> None:
        self.workspace = workspace
        self.qd_store = qd_store
        self.lock_loader = lock_loader
        self.root = workspace.root / ".photomatagent" / "qd" / "skill-candidates"
        self._ensure_directory(self.root)

    def stage(
        self, *, trial_id: str, description: str, applicability: tuple[str, ...],
        replay_input: Mapping[str, Any], replay_expected: Mapping[str, Any],
        experience_refs: tuple[str, ...],
    ) -> SkillPackage:
        if (not description or len(description) > 500 or not applicability
                or len(applicability) > 10
                or any(not item or len(item) > 100 or "\n" in item or "\r" in item for item in applicability)):
            raise ValueError("SKILL_CANDIDATE_INVALID")
        if _contains_validation_claim(description):
            raise ValueError("VALIDATION_CLAIM_FORBIDDEN")
        if _contains_absolute_path(description) or any(_contains_absolute_path(item) for item in applicability):
            raise ValueError("PRIVATE_PATH_IN_SKILL_DOC")
        if _contains_embedded_secret(description) or any(_contains_embedded_secret(item) for item in applicability):
            raise ValueError("SKILL_CREDENTIAL_DETECTED")
        trial: TrialRecord = self.qd_store.load_trial(trial_id)
        validation: ValidationRecord = self.qd_store.load_current_validation(trial_id)
        genome: WorkflowGenome = self.qd_store.load_genome(trial.genome_id)
        if (trial.status != "VALIDATED" or validation.qualification != "PASS"
                or validation.trial_id != trial.trial_id
                or validation.genome_sha256 != genome.content_sha256
                or validation.genome_id != genome.genome_id
                or genome.content_sha256 != canonical_genome_sha256(genome)
                or trial.environment_sha256 != genome.environment_ref):
            raise ValueError("SKILL_SOURCE_TRIAL_NOT_VALIDATED")
        if _contains_absolute_path(genome.model_dump_json()):
            raise ValueError("PRIVATE_PATH_IN_SKILL_DOC")
        if _contains_embedded_secret(genome.model_dump_json()):
            raise ValueError("SKILL_CREDENTIAL_DETECTED")
        if not _valid_identifier_tuple(experience_refs):
            raise ValueError("SKILL_EXPERIENCE_REFS_INVALID")

        try:
            lock = self.lock_loader(genome.environment_ref)
        except Exception as exc:
            raise ValueError("SKILL_LOCK_UNAVAILABLE") from exc
        _validate_lock(lock)
        input_bytes = _bounded_json(replay_input)
        expected_bytes = _bounded_json(replay_expected)

        stage_name = "candidate_" + hashlib.sha256(
            f"{trial.trial_id}\0{genome.content_sha256}\0{description}".encode()
        ).hexdigest()[:24]
        target = self.candidate_path(stage_name)
        if target.exists():
            raise FileExistsError("SKILL_CANDIDATE_EXISTS")
        temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=self.root))
        try:
            self._write_payload(
                temporary, trial=trial, validation=validation, genome=genome,
                description=description, applicability=applicability,
                lock=lock, replay_input=input_bytes, replay_expected=expected_bytes,
                experience_refs=experience_refs,
            )
            self._write_manifest(temporary, status="DRAFT")
            file_digests = self._file_digests(temporary)
            bundle_sha = _bundle_sha256(file_digests)
            package = SkillPackage(
                skill_id=stage_name, version="0.1.0", genome_id=genome.genome_id,
                bundle_sha256=bundle_sha, description=description,
                applicability=applicability, file_digests=file_digests,
                validation_refs=(validation.validation_id,), experience_refs=experience_refs,
                status="DRAFT",
            )
            (temporary / "candidate.json").write_text(
                package.model_dump_json(indent=2) + "\n", encoding="utf-8",
            )
            os.replace(temporary, target)
            return package
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    async def replay(self, candidate_id: str, *, runner: ReplayRunner) -> SkillPackage:
        package = self.load_candidate(candidate_id)
        if package.status != "DRAFT":
            raise ValueError("SKILL_CANDIDATE_STATE_INVALID")
        trial = self.qd_store.load_trial(_candidate_source_trial(self.candidate_path(candidate_id)))
        genome = self.qd_store.load_genome(package.genome_id)
        outcome = await runner(
            package_path=self.candidate_path(candidate_id), genome=genome, source_trial=trial,
        )
        validation = outcome.validation
        if (validation is None or outcome.trial.status != "VALIDATED"
                or validation.qualification != "PASS"
                or outcome.trial.trial_id == trial.trial_id
                or outcome.trial.genome_id != genome.genome_id
                or validation.genome_sha256 != genome.content_sha256
                or validation.genome_id != genome.genome_id
                or outcome.trial.environment_sha256 != genome.environment_ref
                or validation.contract_sha256 != trial.contract_sha256
                or validation.suite_sha256 != self.qd_store.load_current_validation(trial.trial_id).suite_sha256
                or outcome.trial.seed_set != trial.seed_set
                or outcome.trial.resource_profile != trial.resource_profile):
            raise ValueError("SKILL_REPLAY_FAILED")
        # The replay runner must persist through QDStore/Coordinator before its
        # result can become a credential. Re-load instead of trusting its return.
        persisted_trial = self.qd_store.load_trial(outcome.trial.trial_id)
        persisted_validation = self.qd_store.load_current_validation(outcome.trial.trial_id)
        if (persisted_trial != outcome.trial or persisted_validation != validation
                or persisted_validation.validation_id != validation.validation_id):
            raise ValueError("SKILL_REPLAY_RECORD_MISMATCH")

        package_root = self.candidate_path(candidate_id)
        replay_record = {
            "trial_id": persisted_trial.trial_id,
            "validation_id": persisted_validation.validation_id,
            "genome_sha256": genome.content_sha256,
            "bundle_sha256": package.bundle_sha256,
            "contract_sha256": persisted_validation.contract_sha256,
            "suite_sha256": persisted_validation.suite_sha256,
        }
        _write_new_or_same(package_root / "replay-evidence.json", _canonical_json(replay_record) + b"\n")
        self._write_manifest(package_root, status="REPLAY_PASSED")
        digests = self._file_digests(package_root)
        updated = package.model_copy(update={
            "bundle_sha256": _bundle_sha256(digests),
            "file_digests": digests,
            "status": "REPLAY_PASSED",
            "replay_validation_ref": persisted_validation.validation_id,
        })
        (package_root / "candidate.json").write_text(
            updated.model_dump_json(indent=2) + "\n", encoding="utf-8",
        )
        return updated

    def extend_candidate(
        self, candidate_id: str, *, source_trial_id: str,
        experience_refs: tuple[str, ...],
    ) -> SkillPackage:
        """Create a new, replay-invalidated bundle with another reviewed trial link.

        Only trials for the exact same frozen Genome, environment, contract,
        suite, seed protocol, and resource profile can contribute provenance.
        """
        package = self.load_candidate(candidate_id)
        if package.status not in ("DRAFT", "REPLAY_PASSED"):
            raise ValueError("SKILL_CANDIDATE_STATE_INVALID")
        if not _valid_identifier_tuple(experience_refs):
            raise ValueError("SKILL_EXPERIENCE_REFS_INVALID")
        source_root = self.candidate_path(candidate_id)
        genome = self.qd_store.load_genome(package.genome_id)
        new_trial: TrialRecord = self.qd_store.load_trial(source_trial_id)
        new_validation: ValidationRecord = self.qd_store.load_current_validation(source_trial_id)
        original_trial: TrialRecord = self.qd_store.load_trial(_candidate_source_trial(source_root))
        original_validation: ValidationRecord = self.qd_store.load_current_validation(original_trial.trial_id)
        if (new_trial.status != "VALIDATED" or new_validation.qualification != "PASS"
                or new_validation.trial_id != new_trial.trial_id
                or new_validation.genome_id != genome.genome_id
                or new_validation.genome_sha256 != genome.content_sha256
                or new_trial.genome_id != genome.genome_id
                or new_trial.environment_sha256 != genome.environment_ref
                or new_trial.contract_sha256 != original_trial.contract_sha256
                or new_validation.suite_sha256 != original_validation.suite_sha256
                or new_trial.seed_set != original_trial.seed_set
                or new_trial.resource_profile != original_trial.resource_profile):
            raise ValueError("SKILL_SOURCE_TRIAL_INCOMPATIBLE")
        for ref in genome.script_refs:
            self._verified_trial_file(new_trial, ref)

        try:
            provenance = json.loads((source_root / "provenance.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("SKILL_PROVENANCE_INVALID") from exc
        source_trials = provenance.get("source_trials")
        if not isinstance(source_trials, list) or any(not isinstance(item, dict) for item in source_trials):
            raise ValueError("SKILL_PROVENANCE_INVALID")
        if any(item.get("trial_id") == source_trial_id for item in source_trials):
            raise ValueError("SKILL_SOURCE_TRIAL_DUPLICATE")
        source_trials.append({
            "trial_id": new_trial.trial_id,
            "validation_id": new_validation.validation_id,
            "genome_sha256": genome.content_sha256,
            "environment_sha256": genome.environment_ref,
        })
        source_trials.sort(key=lambda item: str(item["trial_id"]))
        provenance["source_trials"] = source_trials
        provenance["validation_ids"] = sorted({
            *provenance.get("validation_ids", [package.validation_refs[0]]),
            new_validation.validation_id,
        })
        provenance["experience_refs"] = sorted({*package.experience_refs, *experience_refs})

        name = "candidate_" + hashlib.sha256(
            f"{package.bundle_sha256}\0{source_trial_id}\0{','.join(provenance['experience_refs'])}".encode()
        ).hexdigest()[:24]
        target = self.candidate_path(name)
        if target.exists():
            raise FileExistsError("SKILL_CANDIDATE_EXISTS")
        temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=self.root))
        try:
            shutil.copytree(
                source_root, temporary, dirs_exist_ok=True, symlinks=False,
                ignore=lambda directory, names: (
                    {"candidate.json", "manifest.json", "replay-evidence.json"} & set(names)
                    if Path(directory) == source_root else set()
                ),
            )
            (temporary / "provenance.json").write_bytes(_canonical_json(provenance) + b"\n")
            self._write_manifest(temporary, status="DRAFT")
            digests = self._file_digests(temporary)
            updated = SkillPackage(
                skill_id=name,
                version=f"0.1.{len(provenance['validation_ids'])}",
                genome_id=package.genome_id, bundle_sha256=_bundle_sha256(digests),
                description=package.description, applicability=package.applicability,
                file_digests=digests,
                validation_refs=tuple(provenance["validation_ids"]),
                experience_refs=tuple(provenance["experience_refs"]), status="DRAFT",
            )
            (temporary / "candidate.json").write_text(
                updated.model_dump_json(indent=2) + "\n", encoding="utf-8",
            )
            os.replace(temporary, target)
            return updated
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def load_candidate(self, candidate_id: str) -> SkillPackage:
        path = self.candidate_path(candidate_id) / "candidate.json"
        self._reject_symlink(path)
        package = SkillPackage.model_validate_json(path.read_text(encoding="utf-8"))
        if package.skill_id != candidate_id:
            raise ValueError("SKILL_CANDIDATE_ID_MISMATCH")
        self.verify_bundle(package)
        return package

    def verify_bundle(self, package: SkillPackage) -> None:
        root = self.candidate_path(package.skill_id)
        actual = self._file_digests(root)
        if actual != package.file_digests or _bundle_sha256(actual) != package.bundle_sha256:
            raise ValueError("APPROVAL_BUNDLE_MISMATCH")

    def verify_replay(self, package: SkillPackage) -> None:
        if package.status not in ("REPLAY_PASSED", "ELIGIBLE", "PUBLISHED") or not package.replay_validation_ref:
            raise ValueError("SKILL_REPLAY_REQUIRED")
        root = self.candidate_path(package.skill_id)
        try:
            evidence = json.loads((root / "replay-evidence.json").read_text(encoding="utf-8"))
            replay_trial_id = evidence["trial_id"]
            validation: ValidationRecord = self.qd_store.load_current_validation(replay_trial_id)
            trial: TrialRecord = self.qd_store.load_trial(replay_trial_id)
            original: TrialRecord = self.qd_store.load_trial(_candidate_source_trial(root))
            genome: WorkflowGenome = self.qd_store.load_genome(package.genome_id)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("SKILL_REPLAY_CREDENTIAL_INVALID") from exc
        if (evidence.get("validation_id") != package.replay_validation_ref
                or validation.validation_id != package.replay_validation_ref
                or evidence.get("genome_sha256") != genome.content_sha256
                or validation.qualification != "PASS"
                or validation.trial_id != trial.trial_id
                or validation.genome_id != genome.genome_id
                or validation.genome_sha256 != genome.content_sha256
                or validation.contract_sha256 != original.contract_sha256
                or validation.suite_sha256 != self.qd_store.load_current_validation(original.trial_id).suite_sha256
                or trial.status != "VALIDATED" or trial.environment_sha256 != genome.environment_ref
                or trial.genome_id != genome.genome_id or trial.trial_id == original.trial_id
                or trial.seed_set != original.seed_set
                or trial.resource_profile != original.resource_profile):
            raise ValueError("SKILL_REPLAY_CREDENTIAL_INVALID")

    def candidate_path(self, candidate_id: str) -> Path:
        if not re.fullmatch(r"candidate_[0-9a-f]{24}", candidate_id):
            raise ValueError("SKILL_CANDIDATE_ID_INVALID")
        path = self.root / candidate_id
        if path.is_symlink():
            raise ValueError("SKILL_CANDIDATE_PATH_INVALID")
        if not self.workspace.contains(path.resolve(strict=False)):
            raise ValueError("SKILL_CANDIDATE_PATH_INVALID")
        return path

    def _write_payload(
        self, root: Path, *, trial: TrialRecord, validation: ValidationRecord,
        genome: WorkflowGenome, description: str, applicability: tuple[str, ...],
        lock: bytes, replay_input: bytes, replay_expected: bytes,
        experience_refs: tuple[str, ...],
    ) -> None:
        (root / "tests").mkdir()
        (root / "scripts").mkdir()
        script_targets: dict[str, ArtifactDigest] = {}
        for ref in genome.script_refs:
            source = self._verified_trial_file(trial, ref)
            source_bytes = source.read_bytes()
            if _contains_embedded_secret(source_bytes.decode("utf-8", errors="ignore")):
                raise ValueError("SKILL_CREDENTIAL_DETECTED")
            name = Path(ref.path).name
            if name in script_targets:
                raise ValueError("SKILL_SCRIPT_NAME_CONFLICT")
            script_targets[name] = ref
            target = root / "scripts" / name
            target.write_bytes(source_bytes)
        if not script_targets:
            raise ValueError("SKILL_SCRIPT_MISSING")
        (root / "workflow.json").write_bytes(_canonical_json(genome.model_dump(mode="json")) + b"\n")
        (root / "requirements.lock").write_bytes(lock)
        (root / "tests" / "replay-input.json").write_bytes(replay_input)
        (root / "tests" / "replay-expected.json").write_bytes(replay_expected)
        script_bundle = _bundle_sha256(tuple(
            ArtifactDigest(path=ref.path, sha256=ref.sha256, size_bytes=ref.size_bytes,
                           media_type=ref.media_type)
            for ref in sorted(genome.script_refs, key=lambda item: item.path)
        ))
        provenance = {
            "source_trial_id": trial.trial_id,
            "source_trials": [{
                "trial_id": trial.trial_id,
                "validation_id": validation.validation_id,
                "genome_sha256": genome.content_sha256,
                "environment_sha256": genome.environment_ref,
            }],
            "validation_id": validation.validation_id,
            "genome_id": genome.genome_id,
            "genome_sha256": genome.content_sha256,
            "environment_sha256": genome.environment_ref,
            "script_bundle_sha256": script_bundle,
            "source_artifacts": [item.sha256 for item in trial.output_refs],
            "lock_sha256": hashlib.sha256(lock).hexdigest(),
            "experience_refs": list(experience_refs),
        }
        (root / "provenance.json").write_bytes(_canonical_json(provenance) + b"\n")
        (root / "SKILL.md").write_text(
            _skill_document(genome, description, applicability), encoding="utf-8",
        )

    def _verified_trial_file(self, trial: TrialRecord, ref: ArtifactDigest) -> Path:
        source_ref = f"{trial.attempt_dir}/{ref.path}"
        path = self.workspace.resolve(source_ref)
        current = self.workspace.root
        for part in Path(source_ref).parts:
            current /= part
            self._reject_symlink(current)
        if (not path.is_file() or path.stat().st_size != ref.size_bytes
                or _sha256_file(path) != ref.sha256):
            raise ValueError("ARTIFACT_HASH_MISMATCH")
        return path

    def _write_manifest(self, root: Path, *, status: str) -> None:
        files = self._file_digests(root, exclude={"manifest.json", "candidate.json"})
        payload = {
            "schema_version": 1,
            "status": status,
            "files": [item.model_dump(mode="json") for item in files],
        }
        (root / "manifest.json").write_bytes(_canonical_json(payload) + b"\n")

    def _file_digests(self, root: Path, *, exclude: set[str] | None = None) -> tuple[ArtifactDigest, ...]:
        excluded = exclude or {"candidate.json"}
        digests: list[ArtifactDigest] = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("SKILL_BUNDLE_SYMLINK")
            if not path.is_file() or path.relative_to(root).as_posix() in excluded:
                continue
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            digests.append(ArtifactDigest(
                path=relative, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), media_type=_media_type(path),
            ))
        return tuple(digests)

    def _ensure_directory(self, path: Path) -> None:
        current = self.workspace.root
        for part in path.relative_to(self.workspace.root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("SKILL_CANDIDATE_PATH_INVALID")
            current.mkdir(exist_ok=True)
        if not self.workspace.contains(path.resolve()):
            raise ValueError("SKILL_CANDIDATE_PATH_INVALID")

    @staticmethod
    def _reject_symlink(path: Path) -> None:
        if path.is_symlink():
            raise ValueError("SKILL_BUNDLE_SYMLINK")


def _skill_document(
    genome: WorkflowGenome, description: str, applicability: tuple[str, ...],
) -> str:
    if _contains_absolute_path(description) or any(_contains_absolute_path(item) for item in applicability):
        raise ValueError("PRIVATE_PATH_IN_SKILL_DOC")
    name = "qd-" + re.sub(r"[^a-z0-9-]+", "-", genome.genome_id.lower()).strip("-")[:48]
    description_line = " ".join(description.split()).replace("\"", "'")
    paths = ", ".join(f"scripts/{Path(ref.path).name}" for ref in genome.script_refs)
    return (
        "---\n"
        f"name: {name}\n"
        f"description: \"{description_line}\"\n"
        "category: materials-research\n"
        "tags: [qd, workflow]\n"
        "---\n\n"
        f"# {description_line}\n\n"
        "## When to use\n\n"
        + "; ".join(applicability)
        + "\n\n## Inputs and outputs\n\n"
        + f"Input roles: {', '.join(sorted({role for step in genome.steps for role in step.input_roles}) or ('none',))}.\n\n"
        + f"Output roles: {', '.join(sorted({role for step in genome.steps for role in step.output_roles}) or ('none',))}.\n\n"
        + f"## Entry point\n\nRun the frozen workflow through its registered runtime; bundled scripts: {paths}.\n\n"
        + "## Validation and fallback\n\nUse the included replay fixture and frozen requirements lock. Stop if inputs, units, or dependencies differ from the workflow contract; fall back to the approved baseline.\n"
    )


def _validate_lock(lock: bytes) -> None:
    if len(lock) > _MAX_LOCK_BYTES:
        raise ValueError("SKILL_LOCK_TOO_LARGE")
    try:
        text = lock.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("SKILL_LOCK_INVALID") from exc
    if _contains_absolute_path(text) or "\x00" in text:
        raise ValueError("SKILL_LOCK_INVALID")
    if "://" in text or re.search(r"(?:^|\s)(?:-e|--editable|file:)", text):
        raise ValueError("SKILL_LOCK_INVALID")
    if text.strip() and not re.search(r"--hash=sha256:[0-9a-fA-F]{64}", text):
        raise ValueError("SKILL_LOCK_UNHASHED")
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "==" not in line and "--hash=sha256:" not in line and not line.rstrip().endswith("\\"):
            raise ValueError("SKILL_LOCK_INVALID")


def _bounded_json(value: Mapping[str, Any]) -> bytes:
    try:
        result = _canonical_json(value) + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("SKILL_REPLAY_FIXTURE_INVALID") from exc
    if len(result) > _MAX_FIXTURE_BYTES:
        raise ValueError("SKILL_REPLAY_FIXTURE_TOO_LARGE")
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _bundle_sha256(digests: tuple[ArtifactDigest, ...]) -> str:
    payload = [item.model_dump(mode="json") for item in sorted(digests, key=lambda item: item.path)]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_type(path: Path) -> str:
    if path.suffix == ".py":
        return "text/x-python"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix in {".md", ".lock"}:
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def _contains_absolute_path(value: str) -> bool:
    return re.search(r"(?:^|[\s=:'\"])(?:/[A-Za-z0-9_.-]+)+|\b[A-Za-z]:[\\/]|\\\\[^\\\s]+", value) is not None


def _contains_validation_claim(value: str) -> bool:
    return re.search(r"\b(?:PASS|passed|validated|validation succeeded|verified)\b", value, re.IGNORECASE) is not None


def _contains_embedded_secret(value: str) -> bool:
    return (
        re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", value) is not None
        or re.search(r"\bAKIA[0-9A-Z]{16}\b", value) is not None
        or re.search(r"\bBearer\s+[A-Za-z0-9._~-]{16,}", value, re.IGNORECASE) is not None
        or re.search(
            r"\b(?:api[_-]?key|access[_-]?token|password|client_secret)\s*=\s*['\"][^'\"]{8,}['\"]",
            value, re.IGNORECASE,
        ) is not None
    )


def _valid_identifier_tuple(values: tuple[str, ...]) -> bool:
    return len(values) <= 100 and len(set(values)) == len(values) and all(
        re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value) for value in values
    )


def _candidate_source_trial(path: Path) -> str:
    try:
        provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
        trial_id = provenance["source_trial_id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("SKILL_PROVENANCE_INVALID") from exc
    if not isinstance(trial_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", trial_id):
        raise ValueError("SKILL_PROVENANCE_INVALID")
    return trial_id


def _write_new_or_same(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("SKILL_REPLAY_RECORD_MISMATCH")
