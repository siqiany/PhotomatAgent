"""Small, predictable exception hierarchy for runtime boundaries."""

from __future__ import annotations


class PhotomatAgentError(Exception):
    pass


class ProviderError(PhotomatAgentError):
    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class ToolError(PhotomatAgentError):
    pass


class ToolValidationError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


class PermissionDenied(ToolError):
    pass
