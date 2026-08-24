"""Built-in agent tools: calculator, clock, and per-chat persistent notes."""

from __future__ import annotations

import ast
import asyncio
import json
import math
import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import context
from ..exceptions import WaAgentError
from .base import Tool, tool_from_callable

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _SafeEvaluator(ast.NodeVisitor):
    MAX_POWER = 10_000

    def visit(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise WaAgentError(f"Operator '{op_type.__name__}' is not allowed")
            left = self.visit(node.left)
            right = self.visit(node.right)
            if op_type is ast.Pow and abs(right) > self.MAX_POWER:
                raise WaAgentError("Exponent too large")
            try:
                result = _BIN_OPS[op_type](left, right)
            except ZeroDivisionError as exc:
                raise WaAgentError("Division by zero") from exc
            if isinstance(result, complex):
                raise WaAgentError("Result is not a real number")
            return float(result)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise WaAgentError(f"Unary operator '{op_type.__name__}' is not allowed")
            return float(_UNARY_OPS[op_type](self.visit(node.operand)))
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Name) or node.keywords:
                raise WaAgentError("Only simple function calls are allowed")
            fn = _SAFE_FUNCTIONS.get(func.id)
            if fn is None:
                raise WaAgentError(f"Function '{func.id}' is not available")
            args = [self.visit(a) for a in node.args]
            return float(fn(*args))
        if isinstance(node, ast.Name):
            value = _SAFE_CONSTANTS.get(node.id)
            if value is None:
                raise WaAgentError(f"Unknown name '{node.id}'")
            return value
        raise WaAgentError(f"Expression element '{type(node).__name__}' is not allowed")


_SAFE_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
}

_SAFE_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports + - * / // % **, parentheses, and functions like sqrt(x),
    sin(x), log(x), factorial(n). Constants: pi, e.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise WaAgentError(f"Invalid expression: {exc}") from exc
    value = _SafeEvaluator().visit(tree)
    if not math.isfinite(value):
        raise WaAgentError("Result is not a finite number")
    formatted = f"{value:.12g}"
    if "." in formatted and "e" not in formatted.lower():
        formatted = formatted.rstrip("0").rstrip(".")
    return f"{expression.strip()} = {formatted}"


def current_datetime(timezone: str = "UTC") -> str:
    """Current date/time. Pass an IANA zone like 'Africa/Cairo' or 'UTC'."""
    from zoneinfo import ZoneInfo

    tz = None
    for candidate in (timezone, "UTC"):
        try:
            tz = ZoneInfo(candidate)
            break
        except Exception:  # noqa: BLE001
            continue
    now = datetime.now(tz)
    human = now.strftime("%A, %d %B %Y %H:%M:%S %Z")
    return f"{now.isoformat()} ({human})"


class _NoteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, list[dict[str, str]]]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, list[dict[str, str]]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    async def add(self, chat: str, note: str) -> int:
        async with self._lock:
            data = self._load()
            bucket = data.setdefault(chat, [])
            bucket.append({"note": note, "at": datetime.now().isoformat(timespec="seconds")})
            self._save(data)
            return len(bucket)

    async def list(self, chat: str) -> list[dict[str, str]]:
        async with self._lock:
            return list(self._load().get(chat, ()))

    async def clear(self, chat: str) -> int:
        async with self._lock:
            data = self._load()
            removed = len(data.pop(chat, ()))
            self._save(data)
            return removed


def _current_chat() -> str:
    jid = context.current_chat_jid.get()
    return jid or "global"


def create_builtin_tools(data_dir: Path) -> list[Tool]:
    store = _NoteStore(Path(data_dir) / "notes.json")

    async def remember_note(note: str) -> str:
        count = await store.add(_current_chat(), note)
        return f"Noted. {count} note(s) stored for this chat."

    async def recall_notes() -> str:
        notes = await store.list(_current_chat())
        if not notes:
            return "No notes saved for this chat yet."
        lines = [f"{i}. [{n['at']}] {n['note']}" for i, n in enumerate(notes, 1)]
        return "\n".join(lines)

    async def clear_notes() -> str:
        removed = await store.clear(_current_chat())
        return f"Deleted {removed} note(s)."

    tools = [
        tool_from_callable(calculate),
        tool_from_callable(current_datetime),
        Tool(
            name="remember_note",
            description=(
                "Save a note for the current chat so it can be recalled later "
                "(use it whenever the user says 'remember that ...')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The information to remember"}
                },
                "required": ["note"],
            },
            handler=remember_note,
        ),
        Tool(
            name="recall_notes",
            description=(
                "List every note previously saved for this chat "
                "(useful when the user asks 'what did I tell you?')."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=recall_notes,
        ),
        Tool(
            name="clear_notes",
            description="Delete all notes saved for this chat.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=clear_notes,
        ),
    ]
    return tools
