"""Explicit bundle approval and maturity-gated immutable skill publication."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from photomatagent.scientific.evolution.experience import (
    ExperiencePromotionError, ExperienceRecord, promote_experience,
)
from photomatagent.scientific.evolution.qd.models import (
    PublicationApproval, ReviewedTrialLink, SkillPackage,
)
from photomatagent.scientific.evolution.qd.skill_package import SkillPackageBuilder
from photomatagent.workspace import Workspace


ExperienceLoader = Callable[[str, str], ExperienceRecord]
ReviewedLinks = Callable[[str], tuple[ReviewedTrialLink, ...]]
ReviewedLinkVerifier = Callable[[ReviewedTrialLink, SkillPackage], None]


class SkillPublisher:
    """Publish only the reviewed bundle approved by a user-facing host path."""

    def __init__(
        self, *, workspace: Workspace, package_builder: SkillPackageBuilder,
        experience_loader: ExperienceLoader, reviewed_links: ReviewedLinks,
        verify_reviewed_link: ReviewedLinkVerifier,
    ) -> None:
        self.workspace = workspace
        self.package_builder = package_builder
        self.experience_loader = experience_loader
        self.reviewed_links = reviewed_links
        self.verify_reviewed_link = verify_reviewed_link
        self.root = workspace.root / ".photomatagent" / "qd"
        self.approval_root = self.root / "publication-approvals"
        self.published_root = self.root / "published-skills"
        self._ensure_directory(self.approval_root)
        self._ensure_directory(self.published_root)
        self.manifest_path = self.published_root / "active-manifest.json"
        self.lock_path = self.published_root / ".manifest.lock"

    def approve(self, candidate_id: str, *, actor: str) -> PublicationApproval:
        package = self.package_builder.load_candidate(candidate_id)
        if package.status != "REPLAY_PASSED":
            raise ValueError("MATURITY_INSUFFICIENT")
        self.package_builder.verify_replay(package)
        self._verify_experience_links(package)
        if not actor or len(actor) > 200 or "\n" in actor:
            raise ValueError("APPROVAL_ACTOR_INVALID")
        approval = PublicationApproval(
            approval_id=f"approval_{uuid4().hex}", actor=actor,
            approved_at=datetime.now(timezone.utc).isoformat(),
            bundle_sha256=package.bundle_sha256, scope=f"candidate:{candidate_id}",
        )
        self._write_approval(approval)
        return approval

    def publish(self, *, candidate_id: str, approval_id: str) -> SkillPackage:
        package = self.package_builder.load_candidate(candidate_id)
        approval = self._load_approval(approval_id)
        if (approval.scope != f"candidate:{candidate_id}"
                or approval.bundle_sha256 != package.bundle_sha256):
            raise ValueError("APPROVAL_BUNDLE_MISMATCH")
        if package.status != "REPLAY_PASSED":
            raise ValueError("MATURITY_INSUFFICIENT")
        self.package_builder.verify_replay(package)
        self._verify_experience_links(package)

        source = self.package_builder.candidate_path(candidate_id)
        destination = self.published_root / package.skill_id / package.bundle_sha256
        self._ensure_directory(destination.parent)
        if destination.exists():
            if not self._published_bundle_matches(destination, package):
                raise ValueError("APPROVAL_BUNDLE_MISMATCH")
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=destination.parent))
            shutil.rmtree(temporary)
            try:
                shutil.copytree(
                    source, temporary, symlinks=False,
                    ignore=lambda directory, names: (
                        {"candidate.json"} & set(names)
                        if Path(directory) == source else set()
                    ),
                )
                copied_digests = self.package_builder._file_digests(temporary)
                if copied_digests != package.file_digests:
                    raise ValueError("APPROVAL_BUNDLE_MISMATCH")
                self.package_builder.verify_bundle(package)
                os.replace(temporary, destination)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        published = package.model_copy(update={"status": "PUBLISHED"})
        self._activate(published)
        return published

    def revoke(self, *, skill_id: str, reason: str, actor: str) -> None:
        if (not re.fullmatch(r"candidate_[0-9a-f]{24}", skill_id)
                or not reason.strip() or len(reason) > 1000 or not actor or len(actor) > 200):
            raise ValueError("SKILL_REVOCATION_INVALID")
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
                    raise ValueError("SKILL_NOT_ACTIVE")
                try:
                    manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ValueError("SKILL_MANIFEST_INVALID") from exc
                skills = manifest.get("skills") if isinstance(manifest, dict) else None
                if not isinstance(skills, dict) or skill_id not in skills:
                    raise ValueError("SKILL_NOT_ACTIVE")
                record = skills.pop(skill_id)
                self._ensure_directory(self.root / "revocations")
                audit = {
                    "skill_id": skill_id, "bundle_sha256": record.get("bundle_sha256"),
                    "actor": actor, "reason": reason.strip(),
                    "revoked_at": datetime.now(timezone.utc).isoformat(),
                }
                audit_path = self.root / "revocations" / f"{skill_id}-{uuid4().hex}.json"
                with audit_path.open("xb") as stream:
                    stream.write(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode() + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                fd, temporary = tempfile.mkstemp(prefix=".revoke-", dir=self.published_root)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, self.manifest_path)
                except BaseException:
                    Path(temporary).unlink(missing_ok=True)
                    raise
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _verify_experience_links(self, package: SkillPackage) -> tuple[ExperienceRecord, ...]:
        if not package.experience_refs:
            raise ValueError("MATURITY_INSUFFICIENT")
        links = self.reviewed_links(package.bundle_sha256)
        if not links or any(link.bundle_sha256 != package.bundle_sha256 for link in links):
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        experience_keys = {(link.experience_evolution_id, link.experience_id) for link in links}
        experiences: dict[tuple[str, str], ExperienceRecord] = {}
        for key in experience_keys:
            experience = self.experience_loader(*key)
            if (experience.experience_id != key[1]
                    or experience.maturity != "REUSABLE_SKILL"
                    or not experience.user_approved_for_reuse
                    or experience.experience_id not in package.experience_refs):
                raise ValueError("MATURITY_INSUFFICIENT")
            self._verify_experience_chain(experience)
            experiences[key] = experience

        linked_observations: set[tuple[str, str, str, float | None]] = set()
        linked_groups: set[str] = set()
        for link in links:
            self.verify_reviewed_link(link, package)
            linked_experience = experiences.get((link.experience_evolution_id, link.experience_id))
            if linked_experience is None:
                raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
            for comparison_id in link.comparison_ids:
                evidence = [item for item in linked_experience.observations if item.comparison_id == comparison_id]
                if len(evidence) != 1 or evidence[0].task_group_id != link.task_group_id:
                    raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
                linked_observations.add((link.experience_id, comparison_id, link.task_group_id, evidence[0].reward))
                linked_groups.add(link.task_group_id)
        required = {
            (experience.experience_id, observation.comparison_id, observation.task_group_id, observation.reward)
            for experience in experiences.values()
            for observation in experience.observations
        }
        if len(linked_groups) < 5 or not required.issubset(linked_observations):
            raise ValueError("MATURITY_INSUFFICIENT")
        return tuple(experiences.values())

    def _verify_experience_chain(self, current: ExperienceRecord) -> None:
        while current.maturity != "OBSERVATION":
            if current.base_experience_id is None or current.previous_maturity is None:
                raise ValueError("MATURITY_INSUFFICIENT")
            try:
                base = self.experience_loader(current.evolution_id, current.base_experience_id)
                reconstructed = promote_experience(
                    base, to=current.maturity, evidence=current.observations,
                    user_approved=current.user_approved_for_reuse,
                )
            except (ExperiencePromotionError, ValueError) as exc:
                raise ValueError("MATURITY_INSUFFICIENT") from exc
            if reconstructed != current or base.maturity != current.previous_maturity:
                raise ValueError("MATURITY_INSUFFICIENT")
            current = base

    def _write_approval(self, approval: PublicationApproval) -> None:
        path = self.approval_root / f"{approval.approval_id}.json"
        payload = approval.model_dump_json(indent=2).encode("utf-8") + b"\n"
        try:
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise ValueError("APPROVAL_ID_CONFLICT") from exc

    def _load_approval(self, approval_id: str) -> PublicationApproval:
        if not re.fullmatch(r"approval_[0-9a-f]{32}", approval_id):
            raise ValueError("APPROVAL_NOT_FOUND")
        path = self.approval_root / f"{approval_id}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("APPROVAL_NOT_FOUND")
        try:
            approval = PublicationApproval.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("APPROVAL_INVALID") from exc
        if approval.approval_id != approval_id:
            raise ValueError("APPROVAL_INVALID")
        return approval

    def _activate(self, package: SkillPackage) -> None:
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                manifest: dict[str, object] = {"schema_version": 1, "skills": {}}
                if self.manifest_path.exists():
                    if self.manifest_path.is_symlink():
                        raise ValueError("SKILL_MANIFEST_UNSAFE")
                    try:
                        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        raise ValueError("SKILL_MANIFEST_INVALID") from exc
                skills = manifest.get("skills")
                if not isinstance(skills, dict):
                    raise ValueError("SKILL_MANIFEST_INVALID")
                skills[package.skill_id] = {
                    "version": package.version,
                    "bundle_sha256": package.bundle_sha256,
                    "relative_path": f"{package.skill_id}/{package.bundle_sha256}",
                    "status": "ACTIVE",
                }
                descriptor, temporary_name = tempfile.mkstemp(prefix=".active-", dir=self.published_root)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary_name, self.manifest_path)
                finally:
                    Path(temporary_name).unlink(missing_ok=True)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _published_bundle_matches(self, root: Path, package: SkillPackage) -> bool:
        try:
            if any(path.is_symlink() for path in root.rglob("*")):
                return False
            return self.package_builder._file_digests(root) == package.file_digests
        except (OSError, ValueError):
            return False

    def _ensure_directory(self, path: Path) -> None:
        current = self.workspace.root
        for part in path.relative_to(self.workspace.root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("SKILL_PUBLISH_PATH_INVALID")
            current.mkdir(exist_ok=True)
        if not self.workspace.contains(path.resolve()):
            raise ValueError("SKILL_PUBLISH_PATH_INVALID")
