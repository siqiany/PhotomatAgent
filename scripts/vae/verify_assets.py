#!/usr/bin/env python3
"""Verify committed VAE training and runtime assets against SHA-256 hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPOSITORY_ROOT / "data" / "photoelectric_vae" / "asset_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
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
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified {len(manifest['files'])} VAE asset files")


if __name__ == "__main__":
    main()
