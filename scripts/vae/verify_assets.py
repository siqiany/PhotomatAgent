#!/usr/bin/env python3
"""Verify committed VAE training and runtime assets against SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPOSITORY_ROOT / "data" / "photoelectric_vae" / "asset_manifest.json"
)


class AssetVerificationError(ValueError):
    """One or more VAE assets failed the manifest contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify packaged VAE assets and available source archives."
    )
    parser.add_argument(
        "--require-source-archives",
        action="store_true",
        help="fail when externally distributed raw training archives are absent",
    )
    return parser.parse_args()


def verify_manifest(
    manifest_path: Path,
    repository_root: Path,
    *,
    require_source_archives: bool = False,
) -> str:
    """Verify one manifest and return a bounded human-readable summary."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    required_verified = 0
    external_verified = 0
    external_missing = 0
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = repository_root / relative
        is_external_source = entry.get("availability") == "external_source"
        if not path.is_file():
            if is_external_source and not require_source_archives:
                external_missing += 1
                continue
            label = "missing external source archive" if is_external_source else "missing"
            failures.append(f"{label}: {relative}")
            continue
        size = path.stat().st_size
        actual_hash = sha256(path)
        if size != int(entry["size"]):
            failures.append(
                f"size mismatch: {relative} expected={entry['size']} actual={size}"
            )
        if actual_hash != entry["sha256"]:
            failures.append(
                f"hash mismatch: {relative} expected={entry['sha256']} "
                f"actual={actual_hash}"
            )
        if is_external_source:
            external_verified += 1
        else:
            required_verified += 1
    if failures:
        raise AssetVerificationError("\n".join(failures))
    summary = f"verified {required_verified} required VAE asset files"
    if external_verified:
        summary += f"; verified {external_verified} external source archives"
    if external_missing:
        summary += (
            f"; {external_missing} external source archives not present "
            "(use --require-source-archives for strict reproduction checks)"
        )
    return summary


def main() -> None:
    args = parse_args()
    try:
        summary = verify_manifest(
            MANIFEST_PATH,
            REPOSITORY_ROOT,
            require_source_archives=args.require_source_archives,
        )
    except AssetVerificationError as exc:
        raise SystemExit(str(exc)) from exc
    print(summary)


if __name__ == "__main__":
    main()
