"""FakeModelProvider: a deterministic model for tests and demos.

Two modes:

* scripted: a fixed list of responses is consumed one call at a time.
* auto: first call requests ``mock.run_calculation``; once a tool result is
  visible, the next call returns a final summary referencing that result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from photomatagent.models.base import ModelProvider
from photomatagent.models.types import ModelResponse, ToolCall
from photomatagent.runtime.state import Message
from photomatagent.tools.base import Tool

DEFAULT_MATERIAL = "GaAs"


@dataclass
class FakeResponse:
    """One scripted model response."""

    text: str = ""
    tool_calls: list[ToolCall] | None = None

    def to_response(self) -> ModelResponse:
        return ModelResponse(
            text=self.text,
            tool_calls=self.tool_calls or [],
            finish_reason="tool_calls" if self.tool_calls else "stop",
        )


def scripted_tool_call(name: str, arguments: dict[str, object]) -> FakeResponse:
    return FakeResponse(tool_calls=[ToolCall(name=name, arguments=arguments)])


def _extract_material(goal: str) -> str:
    """Naive extraction: 'investigate material X' / 'material X' -> X."""
    match = re.search(r"material\s+([A-Za-z0-9_\-]+)", goal)
    if match:
        return match.group(1)
    return DEFAULT_MATERIAL


class FakeModelProvider(ModelProvider):
    name = "fake"

    def __init__(self, responses: list[FakeResponse] | None = None, *, auto: bool = False) -> None:
        self._responses = list(responses or [])
        self.auto = auto

    def set_responses(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResponse:
        if self._responses:
            return self._responses.pop(0).to_response()
        if self.auto:
            return self._auto_response(messages)
        return ModelResponse(text="(fake: no response scripted)", finish_reason="stop")

    def _auto_response(self, messages: list[Message]) -> ModelResponse:
        user_messages = [m for m in messages if m.role == "user"]
        goal = user_messages[-1].content if user_messages else ""
        material = _extract_material(goal)
        has_tool_result = any(m.role == "tool" for m in messages)
        if has_tool_result:
            return ModelResponse(
                text=(
                    f"The mock calculation for {material} suggests a direct band gap "
                    f"of 0.31 eV. This is a placeholder result from the mock backend; "
                    "validate with a real VASP run before drawing conclusions."
                ),
                finish_reason="stop",
            )
        return ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    name="mock.run_calculation",
                    arguments={
                        "material": material,
                        "calculation_type": "band_structure",
                    },
                )
            ],
            finish_reason="tool_calls",
        )
