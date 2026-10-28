"""Read-only adapter for explicitly published, hash-verified QD skills."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class PublishedSkillSource:
    """Resolve only active immutable bundles and revalidate them on every use."""

    source_name = "qd-published"
    priority = 50

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "active-manifest.json"
        self.usage_path = self.root.parent / "skill-usage.jsonl"

    def load_index(self) -> list[Any]:
        from photomatagent.skills.loader import SkillIndexEntry, _parse_frontmatter, _parse_tags

        entries: list[SkillIndexEntry] = []
        for skill_id, record, bundle_root, file_digests in self._active_bundles():
            skill_file = bundle_root / "SKILL.md"
            if "SKILL.md" not in file_digests:
                continue
            metadata = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
            name = metadata.get("name", "")
            if not name:
                continue
            entries.append(SkillIndexEntry(
                name=name,
                description=metadata.get("description", ""),
                category=metadata.get("category", ""),
                tags=_parse_tags(metadata.get("tags", "")),
                path=skill_file,
                source=self.source_name,
                license="",
                priority=self.priority,
                bundle_sha256=str(record["bundle_sha256"]),
            ))
        return entries

    def view(self, name: str, path: str | None = None) -> tuple[str, str, str]:
        for entry in self.load_index():
            if entry.name != name:
                continue
            relative = "SKILL.md" if path is None else _safe_relative_path(path)
            active = self._bundle_for_path(entry.path)
            files = active[2]
            if relative not in files:
                raise OSError("published skill file is not in the verified bundle")
            target = active[1] / relative
            if _sha256_file(target) != files[relative]["sha256"]:
                raise OSError("published skill file changed")
            self.record_event({
                "stage": "retrieved" if path is None else "reference_retrieved",
                "skill": name, "bundle_sha256": entry.bundle_sha256,
            })
            self.record_event({
                "stage": "loaded" if path is None else "reference_loaded",
                "skill": name, "bundle_sha256": entry.bundle_sha256,
                "path": relative,
            })
            return target.read_text(encoding="utf-8"), relative, entry.bundle_sha256 or ""
        raise KeyError(f"unknown active published skill: {name}")

    def authorize_execution(self, script_path: str) -> str | None:
        """Revalidate active publication state before a script enters the sandbox."""
        candidate = Path(script_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        published_prefix = Path(".photomatagent/qd/published-skills")
        if candidate.parts[:len(published_prefix.parts)] != published_prefix.parts:
            return None
        try:
            resolved = (self.root.parent.parent.parent / candidate).resolve(strict=True)
            entries = self.load_index()
        except (OSError, ValueError):
            raise ValueError("PUBLISHED_SKILL_UNAVAILABLE") from None
        for entry in entries:
            try:
                bundle = self._bundle_for_path(entry.path)
                relative_root = bundle[1].resolve()
                relative = resolved.relative_to(relative_root).as_posix()
            except (OSError, ValueError):
                continue
            try:
                file_record = bundle[2].get(relative)
                if file_record is None or _sha256_file(resolved) != file_record["sha256"]:
                    continue
                return entry.name
            except OSError:
                continue
        raise ValueError("PUBLISHED_SKILL_UNAVAILABLE")

    def record_execution(
        self, script_path: str, *, code: str, execution_id: str | None,
        expected_skill: str | None = None,
    ) -> str | None:
        """Attribute a real sandbox run only when its script remains active."""
        skill = self.authorize_execution(script_path)
        if skill is None or (expected_skill is not None and skill != expected_skill):
            return None
        resolved = (self.root.parent.parent.parent / script_path).resolve(strict=True)
        for entry in self.load_index():
            if entry.name == skill:
                bundle = self._bundle_for_path(entry.path)
                relative = resolved.relative_to(bundle[1].resolve()).as_posix()
                self.record_event({
                    "stage": "script_executed", "skill": entry.name,
                    "bundle_sha256": entry.bundle_sha256, "path": relative,
                    "execution_id": execution_id, "outcome": code,
                    "succeeded": code == "OK",
                })
                return entry.name
        return None

    def record_event(self, event: dict[str, Any]) -> None:
        self._ensure_safe_directory(self.usage_path.parent)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.usage_path, flags, 0o600)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            raise

    def _active_bundles(self) -> list[tuple[str, dict[str, Any], Path, dict[str, dict[str, Any]]]]:
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            return []
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        skills = manifest.get("skills") if isinstance(manifest, dict) else None
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(skills, dict):
            return []
        result = []
        for skill_id, record in skills.items():
            if not isinstance(skill_id, str) or not isinstance(record, dict):
                continue
            bundle_sha = record.get("bundle_sha256")
            relative = record.get("relative_path")
            if (not isinstance(bundle_sha, str) or len(bundle_sha) != 64
                    or any(char not in "0123456789abcdef" for char in bundle_sha)
                    or relative != f"{skill_id}/{bundle_sha}"):
                continue
            try:
                bundle_root = self._safe_bundle_root(relative)
                digests = self._verify_bundle(bundle_root, bundle_sha)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            result.append((skill_id, record, bundle_root, digests))
        return result

    def _bundle_for_path(self, skill_file: Path) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
        for _, _, root, files in self._active_bundles():
            if (root / "SKILL.md").resolve() == skill_file.resolve():
                return self.root, root, files
        raise KeyError("published skill was revoked or changed")

    def _safe_bundle_root(self, relative: str) -> Path:
        rel = _safe_relative_path(relative)
        current = self.root
        for part in PurePosixPath(rel).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("published bundle contains symlink")
        resolved = current.resolve(strict=True)
        resolved.relative_to(self.root)
        if not resolved.is_dir():
            raise ValueError("published bundle is missing")
        return resolved

    def _verify_bundle(self, root: Path, expected_bundle_sha: str) -> dict[str, dict[str, Any]]:
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("bundle manifest missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("files") if isinstance(manifest, dict) else None
        if (not isinstance(manifest, dict) or manifest.get("status") not in {"REPLAY_PASSED", "PUBLISHED"}
                or not isinstance(rows, list)):
            raise ValueError("bundle manifest invalid")
        files: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("bundle manifest invalid")
            relative = _safe_relative_path(row.get("path"))
            path = root / relative
            current = root
            for part in PurePosixPath(relative).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("published bundle contains symlink")
            if (relative in files or not path.is_file()
                    or path.stat().st_size != row.get("size_bytes")
                    or _sha256_file(path) != row.get("sha256")):
                raise ValueError("published bundle hash mismatch")
            files[relative] = row
        manifest_bytes = manifest_path.read_bytes()
        files["manifest.json"] = {
            "path": "manifest.json", "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size_bytes": len(manifest_bytes), "media_type": "application/json",
        }
        payload = [files[key] for key in sorted(files)]
        bundle_sha = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        if bundle_sha != expected_bundle_sha:
            raise ValueError("published bundle hash mismatch")
        all_paths = list(root.rglob("*"))
        if any(path.is_symlink() for path in all_paths):
            raise ValueError("published bundle contains symlink")
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in all_paths if path.is_file() and path.name != "candidate.json"
        }
        if actual_paths != set(files):
            raise ValueError("published bundle contents changed")
        return files

    def _ensure_safe_directory(self, directory: Path) -> None:
        # Do not create absent publication roots while recording telemetry.
        if directory.is_symlink() or not directory.is_dir():
            raise OSError("published usage directory unavailable")
        directory.resolve(strict=True).relative_to(self.root.parent.parent)


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid bundle path")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts
            or path.as_posix() != value):
        raise ValueError("invalid bundle path")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
