"""
tools.py
=========
Lightweight tool layer the harness can route to *before* falling back to
retrieval. Motivating case: a strictly retrieval-grounded RAG pipeline has
no way to correctly answer "what's 2+2" from a small factual-QA corpus --
the digit "2" may even appear verbatim in an unrelated passage (e.g. "...
inflation ... around 2 percent annually"), which can fool a purely lexical
off-topic guard into treating the query as in-corpus. The right fix is not
to force retrieval to cover everything; it's to give the harness a real
tool call for the class of query retrieval was never meant to answer --
exactly the "tool calls" part of the orchestration requirement.

CalculatorTool uses Python's `ast` module to parse and evaluate arithmetic
expressions safely -- NOT `eval()` on raw input. Only a fixed set of
numeric AST node types are permitted; anything else (names, calls,
attribute access, comprehensions, etc.) is rejected before evaluation.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

# A query is routed to the calculator only if, once a leading question
# phrase is stripped, what's left is *purely* an arithmetic expression --
# this keeps the tool narrow and prevents it from swallowing genuine
# corpus questions that merely happen to contain a number (e.g. "what
# currency does Japan use" is untouched; "what's 2+2" is not).
_LEADING_PHRASES = re.compile(
    r"^\s*(what'?s|what is|calculate|compute|solve|evaluate)\s*[:\-]?\s*", re.IGNORECASE
)
_ARITH_ONLY_RE = re.compile(r"^[\s0-9+\-*/^().]+\??$")

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class UnsafeExpressionError(Exception):
    pass


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise UnsafeExpressionError(f"Disallowed expression node: {type(node).__name__}")


@dataclass
class ToolResult:
    tool: str
    handled: bool
    output: str | None = None
    detail: dict | None = None


class CalculatorTool:
    name = "calculator"

    def matches(self, query: str) -> str | None:
        """Returns the stripped arithmetic expression if this query is a
        pure arithmetic query the calculator should own, else None."""
        stripped = _LEADING_PHRASES.sub("", query).strip().rstrip("?").strip()
        if stripped and _ARITH_ONLY_RE.match(stripped) and any(c.isdigit() for c in stripped):
            return stripped
        return None

    def run(self, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return ToolResult(self.name, True, output=str(result), detail={"expression": expression})
        except (SyntaxError, UnsafeExpressionError, ZeroDivisionError) as e:
            return ToolResult(self.name, False, detail={"error": str(e)})


class ToolRouter:
    """Tries each registered tool's `matches()` in order; the first match
    handles the query via `run()` instead of going through retrieval."""

    def __init__(self):
        self.tools = [CalculatorTool()]

    def try_route(self, query: str) -> tuple[str, ToolResult] | None:
        for tool in self.tools:
            expr = tool.matches(query)
            if expr is not None:
                return tool.name, tool.run(expr)
        return None
