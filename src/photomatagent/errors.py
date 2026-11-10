"""Small, predictable exception hierarchy for runtime boundaries."""

from __future__ import annotations


class PhotomatAgentError(Exception):
    pass


class ProviderError(PhotomatAgentError):
    def __init__(self, provider: str, message: str, *, code: str = "PROVIDER_ERROR") -> None:
        self.provider = provider
        self.code = code
        super().__init__(f"{provider}: {message}")


class ToolError(PhotomatAgentError):
    pass


class ToolValidationError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


class PermissionDenied(ToolError):
    pass
