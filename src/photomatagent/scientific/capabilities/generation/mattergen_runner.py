"""Safe runner for the isolated MatterGen executable.

MatterGen has a dependency stack which is intentionally kept outside the main
PhotomatAgent environment.  This module owns the narrow boundary between the
two environments: it builds an argv list (never a shell command), constrains
the output directory to the workspace's ``user_output/mattergen`` tree, and
normalizes MatterGen's ZIP archive into a deterministic manifest.
"""

from __future__ import annotations

import json
import math
import os
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from photomatagent.workspace import Workspace


MatterGenPretrainedName = Literal["dft_band_gap", "chemical_system"]

MAX_CANDIDATES = 32
MAX_GUIDANCE_FACTOR = 20.0
MAX_SEED = 2**31 - 1
MAX_PROCESS_OUTPUT_CHARS = 4_000
MAX_CIF_BYTES = 20_000_000


@dataclass(frozen=True, slots=True)
class MatterGenRunSpec:
    """Immutable, validated parameters for one MatterGen generation run."""

    output_dir: Path
    pretrained_name: MatterGenPretrainedName
    candidate_count: int
    target_band_gap_eV: float | None
    chemical_system: str | None
    guidance_factor: float
    seed: int

    def __post_init__(self) -> None:
        output_dir = Path(self.output_dir).expanduser()
        object.__setattr__(self, "output_dir", output_dir)

        if not isinstance(self.pretrained_name, str) or self.pretrained_name not in {
            "dft_band_gap",
            "chemical_system",
        }:
            raise ValueError(
                "pretrained_name must be 'dft_band_gap' or 'chemical_system'"
            )
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or not 1 <= self.candidate_count <= MAX_CANDIDATES
        ):
            raise ValueError(
                f"candidate_count must be between 1 and {MAX_CANDIDATES}"
            )
        if isinstance(self.guidance_factor, bool) or not math.isfinite(
            float(self.guidance_factor)
        ) or not 0.0 <= float(self.guidance_factor) <= MAX_GUIDANCE_FACTOR:
            raise ValueError(
                f"guidance_factor must be between 0 and {MAX_GUIDANCE_FACTOR}"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= MAX_SEED
        ):
            raise ValueError(f"seed must be between 0 and {MAX_SEED}")

        if self.target_band_gap_eV is not None:
            if isinstance(self.target_band_gap_eV, bool) or not math.isfinite(
                float(self.target_band_gap_eV)
            ) or float(self.target_band_gap_eV) <= 0:
                raise ValueError("target_band_gap_eV must be a positive finite number")
        chemical_system = (
            self.chemical_system.strip()
            if isinstance(self.chemical_system, str)
            else self.chemical_system
        )
        object.__setattr__(self, "chemical_system", chemical_system)

        if self.pretrained_name == "dft_band_gap":
            if self.target_band_gap_eV is None:
                raise ValueError(
                    "dft_band_gap mode requires target_band_gap_eV"
                )
            if chemical_system:
                raise ValueError(
                    "dft_band_gap mode cannot also condition on chemical_system"
                )
        else:
            if not chemical_system:
                raise ValueError(
                    "chemical_system mode requires chemical_system"
                )
            if self.target_band_gap_eV is not None:
                raise ValueError(
                    "chemical_system mode cannot also condition on target_band_gap_eV"
                )

    def conditioning_properties(self) -> dict[str, float | str]:
        """Return only the property supported by the selected checkpoint."""

        if self.pretrained_name == "dft_band_gap":
            assert self.target_band_gap_eV is not None
            return {"dft_band_gap": float(self.target_band_gap_eV)}
        assert self.chemical_system is not None
        return {"chemical_system": self.chemical_system}

    def manifest_parameters(self) -> dict[str, Any]:
        """Return stable scalar parameters used to decide manifest reuse."""

        return {
            "pretrained_name": self.pretrained_name,
            "candidate_count": self.candidate_count,
            "target_band_gap_eV": (
                float(self.target_band_gap_eV)
                if self.target_band_gap_eV is not None
                else None
            ),
            "chemical_system": self.chemical_system,
            "guidance_factor": float(self.guidance_factor),
            "seed": self.seed,
        }


class MatterGenRunner:
    """Invoke and normalize a configured MatterGen executable."""

    def __init__(
        self,
        executable: str = "mattergen-generate",
        workspace: Workspace | Path | str | None = None,
        *,
        hf_home: str | Path | None = None,
        timeout_seconds: float = 3_600.0,
        config: Any | None = None,
    ) -> None:
        if config is not None:
            executable = getattr(config, "mattergen_executable", executable)
            if hf_home is None:
                hf_home = getattr(config, "mattergen_hf_home", None)
            if timeout_seconds == 3_600.0:
                timeout_seconds = getattr(
                    config, "mattergen_timeout_seconds", timeout_seconds
                )
        if not str(executable).strip():
            raise ValueError("MatterGen executable must not be empty")
        if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0:
            raise ValueError("MatterGen timeout_seconds must be positive and finite")
        self.executable = str(executable)
        self.workspace = (
            workspace
            if isinstance(workspace, Workspace)
            else Workspace(workspace or Path.cwd())
        )
        self.hf_home = (
            Path(hf_home).expanduser().resolve() if hf_home is not None else None
        )
        self.timeout_seconds = float(timeout_seconds)

    def build_command(self, spec: MatterGenRunSpec) -> list[str]:
        """Build MatterGen's argv without interpolation through a shell."""

        properties = _format_properties(spec.conditioning_properties())
        return [
            self.executable,
            str(spec.output_dir),
            f"--pretrained_name={spec.pretrained_name}",
            f"--batch_size={spec.candidate_count}",
            "--num_batches=1",
            f"--properties_to_condition_on={properties}",
            f"--diffusion_guidance_factor={_format_float(spec.guidance_factor)}",
            f"--seed={spec.seed}",
        ]

    def run(self, spec: MatterGenRunSpec) -> Path:
        """Run MatterGen, extract CIFs, and return a workspace-local manifest."""

        output_dir = self._validate_output_dir(spec.output_dir)
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists() and not self.workspace.contains(
            manifest_path.resolve()
        ):
            raise ValueError("MatterGen manifest path is outside workspace")
        if self._reusable_manifest(manifest_path, spec):
            return manifest_path

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)
        command = self.build_command(
            MatterGenRunSpec(
                output_dir=output_dir,
                pretrained_name=spec.pretrained_name,
                candidate_count=spec.candidate_count,
                target_band_gap_eV=spec.target_band_gap_eV,
                chemical_system=spec.chemical_system,
                guidance_factor=spec.guidance_factor,
                seed=spec.seed,
            )
        )
        environment = os.environ.copy()
        if self.hf_home is not None:
            self.hf_home.mkdir(parents=True, exist_ok=True)
            environment["HF_HOME"] = str(self.hf_home)
        environment["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=environment,
                cwd=str(output_dir),
                shell=False,
            )
            if completed.returncode:
                detail = _bounded_process_text(
                    getattr(completed, "stdout", None),
                    getattr(completed, "stderr", None),
                )
                raise RuntimeError(
                    f"MatterGen command failed with exit code {completed.returncode}"
                    + (f": {detail}" if detail else "")
                )
        except subprocess.TimeoutExpired as exc:
            detail = _bounded_process_text(exc.stdout, exc.stderr)
            raise TimeoutError(
                f"MatterGen timed out after {self.timeout_seconds:g}s"
                + (f": {detail}" if detail else "")
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = _bounded_process_text(exc.stdout, exc.stderr)
            raise RuntimeError(
                f"MatterGen command failed with exit code {exc.returncode}"
                + (f": {detail}" if detail else "")
            ) from exc
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"MatterGen executable not found: {self.executable}"
            ) from exc

        archive = output_dir / "generated_crystals_cif.zip"
        resolved_archive = archive.resolve(strict=False)
        if not self.workspace.contains(resolved_archive):
            raise ValueError("MatterGen archive path is outside workspace")
        if not resolved_archive.is_file():
            raise FileNotFoundError(
                f"MatterGen archive not found: {self.workspace.relative(archive)}"
            )
        candidates = self._extract_cifs(resolved_archive, output_dir, spec)
        manifest = self._manifest_payload(spec, output_dir, candidates)
        _atomic_json_replace(manifest_path, manifest)
        return manifest_path

    def _validate_output_dir(self, output_dir: Path) -> Path:
        try:
            resolved = self.workspace.resolve(str(output_dir), must_exist=False)
        except Exception as exc:
            raise ValueError(f"MatterGen output is outside workspace: {output_dir}") from exc
        mattergen_root = (self.workspace.user_output_dir / "mattergen").resolve()
        if resolved != mattergen_root and mattergen_root not in resolved.parents:
            raise ValueError(
                "MatterGen output must be below workspace user_output/mattergen: "
                f"{output_dir}"
            )
        return resolved

    def _reusable_manifest(
        self, manifest_path: Path, spec: MatterGenRunSpec
    ) -> bool:
        if not manifest_path.is_file():
            return False
        try:
            if not self.workspace.contains(manifest_path.resolve()):
                return False
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return False
            recorded = raw.get("run_spec")
            if not isinstance(recorded, dict):
                recorded = {
                    key: raw.get(key)
                    for key in spec.manifest_parameters()
                }
            expected = spec.manifest_parameters()
            if not _same_parameters(recorded, expected):
                return False
            candidates = raw.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                return False
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    return False
                candidate_path = candidate.get("structure_path")
                if not isinstance(candidate_path, str):
                    return False
                resolved = self.workspace.resolve(candidate_path, must_exist=True)
                if resolved.suffix.lower() != ".cif":
                    return False
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _extract_cifs(
        self, archive: Path, output_dir: Path, spec: MatterGenRunSpec
    ) -> list[dict[str, Any]]:
        try:
            with zipfile.ZipFile(archive) as handle:
                members = sorted(handle.infolist(), key=lambda member: member.filename)
                candidates: list[dict[str, Any]] = []
                for member in members:
                    _validate_zip_member(member)
                    if not member.filename.lower().endswith(".cif"):
                        continue
                    if member.file_size > MAX_CIF_BYTES:
                        raise ValueError(
                            f"MatterGen CIF is too large: {member.filename}"
                        )
                    payload = handle.read(member)
                    if not payload.strip():
                        continue
                    if len(candidates) >= spec.candidate_count:
                        break
                    number = len(candidates) + 1
                    output_path = output_dir / f"candidate-{number:04d}.cif"
                    _atomic_bytes_replace(output_path, payload)
                    candidates.append(
                        {
                            "candidate_id": f"mattergen-{number:04d}",
                            "archive_member": member.filename,
                            "structure_path": str(output_path),
                            "relative_path": self.workspace.relative(output_path),
                        }
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError(f"MatterGen archive is not a valid ZIP: {archive}") from exc
        if not candidates:
            raise RuntimeError("MatterGen archive contained no non-empty CIF files")
        return candidates

    def _manifest_payload(
        self,
        spec: MatterGenRunSpec,
        output_dir: Path,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "manifest_version": 1,
            "backend": "mattergen",
            "validation_status": "UNVALIDATED_GENERATED_STRUCTURE",
            "pretrained_name": spec.pretrained_name,
            "properties_to_condition_on": spec.conditioning_properties(),
            "run_spec": spec.manifest_parameters(),
            "output_relative_dir": self.workspace.relative(output_dir),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }


def _format_properties(properties: dict[str, float | str]) -> str:
    # Fire accepts a Python-literal dictionary.  repr is deterministic for the
    # two scalar values we support and keeps the whole expression in one argv
    # item, including a chemical-system string containing shell punctuation.
    key, value = next(iter(properties.items()))
    return "{" + repr(key) + ":" + repr(value) + "}"


def _format_float(value: float) -> str:
    return format(float(value), ".12g")


def _bounded_process_text(stdout: Any, stderr: Any) -> str:
    pieces: list[str] = []
    for label, value in (("stdout", stdout), ("stderr", stderr)):
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        if text:
            pieces.append(f"{label}: {text[:MAX_PROCESS_OUTPUT_CHARS]}")
    return " | ".join(pieces)[:MAX_PROCESS_OUTPUT_CHARS]


def _validate_zip_member(member: zipfile.ZipInfo) -> None:
    name = member.filename.replace("\\", "/")
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"MatterGen archive member escapes output: {member.filename}")
    mode = (member.external_attr >> 16) & 0xFFFF
    if mode and stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
        raise ValueError(f"MatterGen archive member is not a regular file: {member.filename}")


def _same_parameters(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            try:
                if actual_value is None or not math.isclose(
                    float(str(actual_value)), expected_value, rel_tol=0, abs_tol=1e-12
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _atomic_bytes_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json_replace(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_bytes_replace(path, encoded)
