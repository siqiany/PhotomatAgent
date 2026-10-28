"""Persistent reservation ledger for bounded QD operations."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from photomatagent.scientific.evolution.qd.store import QDStoreConflictError, QDStoreCorruptError, QDStorePathError
from photomatagent.workspace import Workspace


class BudgetConflictError(RuntimeError):
    """An operation conflicts with a reservation or exceeds its budget."""


@dataclass(frozen=True, slots=True)
class BudgetAmounts:
    tokens: int = 0
    wall_seconds: float = 0.0
    trials: int = 0
    installs: int = 0
    download_bytes: int = 0
    model_calls: int = 0
    proposals: int = 0

    def __post_init__(self) -> None:
        for key in ("tokens", "trials", "installs", "download_bytes", "model_calls", "proposals"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid budget amount: {key}")
        if isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float)) or not math.isfinite(self.wall_seconds) or self.wall_seconds < 0:
            raise ValueError("invalid budget amount: wall_seconds")

    def to_dict(self) -> dict[str, int | float]:
        return {key: getattr(self, key) for key in _FIELDS}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BudgetAmounts:
        return cls(**value)


_ZERO = BudgetAmounts()
_FIELDS = ("tokens", "wall_seconds", "trials", "installs", "download_bytes", "model_calls", "proposals")
_MAX_BYTES = 1_000_000
_AMOUNT_FIELDS = frozenset(_FIELDS)


class BudgetLedger:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.root = workspace.root / ".photomatagent" / "qd"
        self._ensure_dir(self.root)
        self.path = self.root / "budget.json"
        self.lock_path = self.root / "budget.lock"

    def reserve(self, operation_id: str, limits: BudgetAmounts) -> None:
        if not operation_id or len(operation_id) > 200 or not isinstance(limits, BudgetAmounts):
            raise ValueError("invalid budget reservation")
        with self._locked():
            data = self._load()
            prior = data["operations"].get(operation_id)
            if prior is not None:
                if BudgetAmounts.from_dict(prior["reserved"]).to_dict() != limits.to_dict():
                    raise BudgetConflictError("operation ID has a different reservation")
                return
            data["operations"][operation_id] = {"reserved": limits.to_dict(), "actual": None, "started": False}
            self._write(data)

    def reserve_with_cap(self, operation_id: str, limits: BudgetAmounts, cap: BudgetAmounts,
                         *, scope: str | None = None) -> None:
        """Atomically reserve usage only if cumulative reservations stay within cap."""
        if (not operation_id or len(operation_id) > 200 or not isinstance(limits, BudgetAmounts)
                or not isinstance(cap, BudgetAmounts)):
            raise ValueError("invalid budget reservation")
        if scope is not None and (not scope or len(scope) > 200):
            raise ValueError("invalid budget scope")
        with self._locked():
            data = self._load()
            prior = data["operations"].get(operation_id)
            if prior is not None:
                if BudgetAmounts.from_dict(prior["reserved"]).to_dict() != limits.to_dict():
                    raise BudgetConflictError("operation ID has a different reservation")
                return
            total = {key: 0 for key in _FIELDS}
            for existing_id, operation in data["operations"].items():
                if scope is not None and not existing_id.startswith(scope):
                    continue
                amount = operation["actual"] if operation["actual"] is not None else operation["reserved"]
                for key in _FIELDS:
                    total[key] += amount.get(key, 0)
            if any(total[key] + getattr(limits, key) > getattr(cap, key) for key in _FIELDS):
                raise BudgetConflictError("reservation exceeds event budget cap")
            data["operations"][operation_id] = {"reserved": limits.to_dict(), "actual": None, "started": False}
            self._write(data)

    def settle(self, operation_id: str, actual: BudgetAmounts) -> None:
        if not isinstance(actual, BudgetAmounts):
            raise ValueError("actual must be BudgetAmounts")
        with self._locked():
            data = self._load()
            prior = data["operations"].get(operation_id)
            if prior is None:
                raise BudgetConflictError("operation has no persisted reservation")
            if prior["actual"] is not None:
                if prior["actual"] != actual.to_dict():
                    raise BudgetConflictError("operation already settled with different actual usage")
                return
            reserved = BudgetAmounts.from_dict(prior["reserved"])
            if any(getattr(actual, key) > getattr(reserved, key) for key in _FIELDS):
                raise BudgetConflictError("actual usage exceeds reservation")
            prior["started"] = True
            prior["actual"] = actual.to_dict()
            self._write(data)

    def begin(self, operation_id: str) -> bool:
        """Durably claim a reserved operation once; ambiguous starts are never replayed."""
        with self._locked():
            data = self._load()
            operation = data["operations"].get(operation_id)
            if operation is None:
                raise BudgetConflictError("operation has no persisted reservation")
            if operation["started"] or operation["actual"] is not None:
                return False
            operation["started"] = True
            self._write(data)
            return True

    def reservation(self, operation_id: str) -> BudgetAmounts | None:
        """Return an operation's durable original reservation, if present."""
        with self._locked():
            operation = self._load()["operations"].get(operation_id)
            return BudgetAmounts.from_dict(operation["reserved"]) if operation is not None else None

    def snapshot(self, *, scope: str | None = None) -> BudgetAmounts:
        with self._locked():
            data = self._load()
            total = {key: 0 for key in _FIELDS}
            for operation_id, operation in data["operations"].items():
                if scope is not None and not operation_id.startswith(scope):
                    continue
                amounts = operation["actual"] if operation["actual"] is not None else operation["reserved"]
                for key in _FIELDS:
                    total[key] += amounts.get(key, 0)
            return BudgetAmounts.from_dict(total)

    def _load(self) -> dict[str, Any]:
        self._safe(self.path)
        if not self.path.exists():
            return {"revision": 0, "operations": {}}
        try:
            if self.path.stat().st_size > _MAX_BYTES:
                raise ValueError("budget ledger too large")
            value = json.loads(self.path.read_text())
            if (not isinstance(value, dict) or set(value) != {"revision", "operations", "sha256"}
                    or isinstance(value["revision"], bool) or not isinstance(value["revision"], int)
                    or value["revision"] < 1 or not isinstance(value["operations"], dict)
                    or not isinstance(value["sha256"], str)):
                raise ValueError("invalid ledger envelope")
            payload = {"revision": value["revision"], "operations": value["operations"]}
            if hashlib.sha256(_canonical_json(payload)).hexdigest() != value["sha256"]:
                raise ValueError("ledger content digest mismatch")
            for operation_id, operation in value["operations"].items():
                if (not isinstance(operation_id, str) or not operation_id or len(operation_id) > 200
                        or not isinstance(operation, dict)
                        or set(operation) not in ({"reserved", "actual"}, {"reserved", "actual", "started"})):
                    raise ValueError("invalid operation record")
                if "started" not in operation:
                    operation["started"] = operation["actual"] is not None
                if not isinstance(operation["started"], bool):
                    raise ValueError("invalid operation started flag")
                _validate_amount_dict(operation["reserved"])
                if operation["actual"] is not None:
                    actual = _validate_amount_dict(operation["actual"])
                    reserved = _validate_amount_dict(operation["reserved"])
                    if any(getattr(actual, key) > getattr(reserved, key) for key in _FIELDS):
                        raise ValueError("actual usage exceeds reservation")
            return value
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise QDStoreCorruptError(f"invalid budget ledger: {exc}") from exc

    def _write(self, value: dict[str, Any]) -> None:
        expected_revision = value.get("revision", 0)
        current = self._load()
        if current["revision"] != expected_revision:
            raise BudgetConflictError("stale budget ledger revision")
        value["revision"] = expected_revision + 1
        content = {"revision": value["revision"], "operations": value["operations"]}
        value["sha256"] = hashlib.sha256(_canonical_json(content)).hexdigest()
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        if len(payload.encode()) > _MAX_BYTES:
            raise BudgetConflictError("budget ledger capacity exceeded")
        fd, name = tempfile.mkstemp(prefix=".budget-", dir=self.root)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._safe(self.path)
            os.replace(name, self.path)
            dfd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            Path(name).unlink(missing_ok=True)

    def _locked(self):
        class Lock:
            def __init__(inner, ledger: BudgetLedger) -> None:
                inner.ledger = ledger
            def __enter__(inner):
                inner.ledger._safe(inner.ledger.lock_path)
                inner.fd = os.open(inner.ledger.lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
                fcntl.flock(inner.fd, fcntl.LOCK_EX)
                return inner
            def __exit__(inner, *args: object) -> None:
                fcntl.flock(inner.fd, fcntl.LOCK_UN)
                os.close(inner.fd)
        return Lock(self)

    def _ensure_dir(self, path: Path) -> None:
        current = self.workspace.root
        for part in path.relative_to(self.workspace.root).parts:
            current = current / part
            if current.is_symlink():
                raise QDStorePathError("symlink in budget path")
            current.mkdir(exist_ok=True)

    def _safe(self, path: Path) -> None:
        if not self.workspace.contains(path):
            raise QDStorePathError("budget path outside workspace")
        current = self.workspace.root
        for part in path.relative_to(current).parts:
            current = current / part
            if current.is_symlink():
                raise QDStorePathError("symlink in budget path")


__all__ = ["BudgetAmounts", "BudgetLedger", "BudgetConflictError"]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _validate_amount_dict(value: Any) -> BudgetAmounts:
    legacy_fields = _AMOUNT_FIELDS - {"model_calls", "proposals"}
    if not isinstance(value, dict) or not legacy_fields.issubset(value) or not set(value).issubset(_AMOUNT_FIELDS):
        raise ValueError("invalid budget amount fields")
    return BudgetAmounts.from_dict(value)
