"""Typed errors shared by workbench application services."""

from __future__ import annotations


class ServiceError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        run_id: str | None = None,
        retryable: bool = False,
        next_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.run_id = run_id
        self.retryable = retryable
        self.next_action = next_action


__all__ = ["ServiceError"]
