"""calculator tool: safe evaluation of simple arithmetic expressions."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from photomatagent.tools.base import Tool, ToolError, ToolResult

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ToolError(f"unsupported expression node: {type(node).__name__}")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a simple arithmetic expression (e.g. '2.5 * (3 + 4)')."
    input_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }

    async def execute(self, arguments: dict) -> ToolResult:
        expression = arguments["expression"]
        try:
            tree = ast.parse(expression, mode="eval")
            value = _eval(tree)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"could not evaluate expression: {exc}") from exc
        return ToolResult(
            output=f"{expression} = {value}",
            data={"expression": expression, "value": value},
        )
