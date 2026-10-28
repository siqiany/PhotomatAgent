"""Application services shared by the CLI and Web entry points."""

from photomatagent.application.runtime_factory import (
    InteractionStrategy,
    RuntimeFactory,
)

__all__ = ["InteractionStrategy", "RuntimeFactory"]
