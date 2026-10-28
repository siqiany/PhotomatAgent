"""Cross-check QD artifacts against reviewed evolution records before reuse."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from photomatagent.scientific.evolution.qd.models import (
    SkillPackage, ValidationRecord, WorkflowGenome, ReviewedTrialLink,
    canonical_genome_sha256,
)
from photomatagent.scientific.evolution.qd.skill_package import SkillPackageBuilder
from photomatagent.workspace import Workspace


class QDExperienceBridge:
    """Persist links only after reloading each QD and expert-review source."""

    def __init__(self, *, workspace: Workspace, qd_store: Any, evolution_store: Any,
                 package_builder: SkillPackageBuilder) -> None:
        self.workspace = workspace
        self.qd_store = qd_store
        self.evolution_store = evolution_store
        self.package_builder = package_builder
        self.root = workspace.root / ".photomatagent" / "qd" / "reviewed-trial-links"
        self._ensure_directory(self.root)

    def link_trial(
        self, *, trial_id: str, evolution_id: str, episode_version: str,
        comparison_ids: tuple[str, ...], experience_id: str,
        package: SkillPackage, experience_evolution_id: str | None = None,
    ) -> ReviewedTrialLink:
        self.package_builder.verify_bundle(package)
        trial = self.qd_store.load_trial(trial_id)
        validation: ValidationRecord = self.qd_store.load_current_validation(trial_id)
        genome: WorkflowGenome = self.qd_store.load_genome(trial.genome_id)
        if (trial.status != "VALIDATED" or validation.qualification != "PASS"
                or validation.trial_id != trial.trial_id
                or validation.genome_id != genome.genome_id
                or validation.genome_sha256 != genome.content_sha256
                or package.genome_id != genome.genome_id
                ):
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        if (genome.content_sha256 != canonical_genome_sha256(genome)
                or trial.environment_sha256 != genome.environment_ref):
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        for ref in trial.output_refs:
            self._verified_episode_artifact(ref.path, ref.sha256)

        task = self.evolution_store.load_task(evolution_id)
        episode = self.evolution_store.load_episode(evolution_id, episode_version)
        if (episode.status != "COMPLETED" or episode.runtime_session_id != trial.session_id
                or episode.artifact is None or episode.episode_id not in task.episode_ids
                or not comparison_ids):
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        binding = (
            episode.qd_binding.model_dump(mode="json")
            if episode.qd_binding is not None else None
        )
        script_bundle_sha = _script_bundle_sha256(genome)
        expected_binding = {
            "qd_task_id": trial.event_id,
            "trial_id": trial.trial_id,
            "genome_id": genome.genome_id,
            "genome_sha256": genome.content_sha256,
            "environment_sha256": genome.environment_ref,
            "script_bundle_sha256": script_bundle_sha,
        }
        if binding != expected_binding:
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        package_root = self.package_builder.candidate_path(package.skill_id)
        try:
            bundled_genome = WorkflowGenome.model_validate_json(
                (package_root / "workflow.json").read_text(encoding="utf-8")
            )
            provenance = json.loads((package_root / "provenance.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE") from exc
        contributing_trials = provenance.get("source_trials")
        if (bundled_genome != genome or not isinstance(contributing_trials, list)
                or not any(
                    isinstance(item, dict) and item.get("trial_id") == trial_id
                    and item.get("validation_id") == validation.validation_id
                    and item.get("genome_sha256") == genome.content_sha256
                    and item.get("environment_sha256") == genome.environment_ref
                    for item in contributing_trials
                )
                or validation.validation_id not in package.validation_refs
                or provenance.get("genome_sha256") != genome.content_sha256
                or provenance.get("environment_sha256") != genome.environment_ref
                or provenance.get("script_bundle_sha256") != script_bundle_sha):
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        artifact = self._verified_episode_artifact(episode.artifact.path, episode.artifact.sha256)
        matches = [item for item in trial.output_refs if item.sha256 == episode.artifact.sha256]
        if len(matches) != 1 or artifact.stat().st_size != episode.artifact.size_bytes:
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")

        exp_evolution_id = experience_evolution_id or evolution_id
        experience = self.evolution_store.load_experience(exp_evolution_id, experience_id)
        if experience.experience_id not in self.evolution_store.load_task(exp_evolution_id).experience_ids:
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        evidence_by_comparison = {item.comparison_id: item for item in experience.observations}
        for comparison_id in comparison_ids:
            if comparison_id not in task.comparison_ids:
                raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
            comparison = self.evolution_store.load_comparison(evolution_id, comparison_id)
            evidence = evidence_by_comparison.get(comparison_id)
            if (comparison.current_version != episode_version
                    or comparison.phase != "POST_FEEDBACK"
                    or comparison.reward is None or comparison.expert_utility_delta is None
                    or evidence is None or evidence.task_group_id != task.task_group_id
                    or evidence.reward != comparison.reward):
                raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")

        link = ReviewedTrialLink(
            qd_task_id=trial.event_id, trial_id=trial.trial_id,
            genome_id=genome.genome_id, genome_sha256=genome.content_sha256,
            bundle_sha256=package.bundle_sha256, evolution_id=evolution_id,
            episode_version=episode_version, runtime_session_id=trial.session_id,
            artifact_sha256=episode.artifact.sha256,
            comparison_ids=tuple(sorted(comparison_ids)), experience_id=experience_id,
            experience_evolution_id=exp_evolution_id,
            task_group_id=task.task_group_id, environment_sha256=genome.environment_ref,
            script_bundle_sha256=script_bundle_sha,
        )
        self._persist(link)
        return link

    def verify_link(self, link: ReviewedTrialLink, package: SkillPackage) -> None:
        current = self.link_trial(
            trial_id=link.trial_id, evolution_id=link.evolution_id,
            episode_version=link.episode_version, comparison_ids=link.comparison_ids,
            experience_id=link.experience_id, experience_evolution_id=link.experience_evolution_id,
            package=package,
        )
        if current != link:
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")

    def links_for_bundle(self, bundle_sha256: str) -> tuple[ReviewedTrialLink, ...]:
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
            raise ValueError("BUNDLE_HASH_INVALID")
        links: list[ReviewedTrialLink] = []
        if not self.root.exists():
            return ()
        for path in sorted(self.root.glob("*.json")):
            self._reject_symlink(path)
            try:
                link = ReviewedTrialLink.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("EXPERIENCE_LINK_CORRUPT") from exc
            if link.bundle_sha256 == bundle_sha256:
                links.append(link)
        return tuple(links)

    def _persist(self, link: ReviewedTrialLink) -> None:
        key = hashlib.sha256(
            f"{link.trial_id}\0{link.bundle_sha256}\0{link.evolution_id}\0{link.episode_version}".encode()
        ).hexdigest()
        path = self.root / f"{key}.json"
        payload = link.model_dump_json(indent=2).encode("utf-8") + b"\n"
        try:
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("EXPERIENCE_LINK_CONFLICT")

    def _verified_episode_artifact(self, relative: str, expected_sha256: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        current = self.workspace.root
        for part in path.parts:
            current = current / part
            self._reject_symlink(current)
        resolved = self.workspace.resolve(relative)
        if not resolved.is_file() or _sha256_file(resolved) != expected_sha256:
            raise ValueError("EXPERIENCE_LINK_UNAVAILABLE")
        return resolved

    def _ensure_directory(self, path: Path) -> None:
        current = self.workspace.root
        for part in path.relative_to(self.workspace.root).parts:
            current = current / part
            self._reject_symlink(current)
            current.mkdir(exist_ok=True)
        if not self.workspace.contains(path.resolve()):
            raise ValueError("EXPERIENCE_LINK_PATH_INVALID")

    @staticmethod
    def _reject_symlink(path: Path) -> None:
        if path.is_symlink():
            raise ValueError("EXPERIENCE_LINK_PATH_INVALID")


def _script_bundle_sha256(genome: WorkflowGenome) -> str:
    payload = [ref.model_dump(mode="json") for ref in sorted(genome.script_refs, key=lambda item: item.path)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
