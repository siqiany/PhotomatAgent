from __future__ import annotations

import pytest

from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import ToolError
from photomatagent.tools.calculator import CalculatorTool
from photomatagent.tools.echo import EchoTool
from photomatagent.tools.mock_calculation import MockCalculationTool
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.scientific_state_inspect import ScientificStateInspectTool


@pytest.mark.asyncio
async def test_echo_roundtrip():
    result = await EchoTool().execute({"text": "hello"})
    assert result.output == "hello"


@pytest.mark.asyncio
async def test_calculator_evaluates_arithmetic():
    result = await CalculatorTool().execute({"expression": "2.5 * (3 + 4)"})
    assert result.data["value"] == 17.5


@pytest.mark.asyncio
async def test_calculator_rejects_unsafe_expression():
    tool = CalculatorTool()
    with pytest.raises(ToolError):
        await tool.execute({"expression": "__import__('os').system('echo hi')"})
    with pytest.raises(ToolError):
        await tool.execute({"expression": "import os"})


def test_registry_validation():
    registry = ToolRegistry()
    registry.register(EchoTool())
    assert registry.get("echo").name == "echo"
    with pytest.raises(ValueError):
        registry.register(EchoTool())
    with pytest.raises(KeyError):
        registry.get("nope")


def test_registry_rejects_bad_arguments():
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ToolError):
        registry.validate_arguments("echo", {})
    with pytest.raises(ToolError):
        registry.validate_arguments("echo", {"text": "x", "extra": 1})


@pytest.mark.asyncio
async def test_mock_calculation_tool():
    tool = MockCalculationTool()
    result = await tool.execute({"material": "GaAs", "calculation_type": "band_structure"})
    assert result.data["results"]["band_gap"] == 0.31
    assert len(result.state_updates) == 2  # CalculationRecord + Evidence


@pytest.mark.asyncio
async def test_state_inspect_tool():
    state = ScientificState()
    state.goal = "understand GaAs"
    tool = ScientificStateInspectTool(state)
    result = await tool.execute({"section": "all"})
    assert "Goal: understand GaAs" in result.output
