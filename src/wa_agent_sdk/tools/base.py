"""Tool abstraction: auto JSON-schema generation and async execution."""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, get_args, get_origin, get_type_hints

from ..exceptions import WaAgentError

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass(slots=True)
class Tool:
    """A callable tool exposed to the LLM with a JSON-schema signature."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    async def run(self, arguments: dict[str, Any]) -> str:
        try:
            result = self.handler(**arguments)
            if inspect.isawaitable(result):
                result = await result
        except TypeError as exc:
            return f"Error: bad arguments for '{self.name}': {exc}"
        except WaAgentError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced to the model on purpose
            return f"Error: {type(exc).__name__}: {exc}"
        if result is None:
            return "OK"
        if isinstance(result, (dict, list, tuple)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)


def _annotation_to_json_type(annotation: Any) -> str:
    if annotation in _JSON_TYPES:
        return _JSON_TYPES[annotation]
    origin = get_origin(annotation)
    if annotation is Any or origin is None:
        return "string"
    if origin in (list, typing.List):
        return "array"
    if origin in (dict, typing.Dict):
        return "object"
    if origin in (UnionType, typing.Union):
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            resolved = _annotation_to_json_type(arg)
            if resolved != "string":
                return resolved
        return "string"
    if isinstance(annotation, type) and issubclass(annotation, (str, int, float, bool)):
        return _JSON_TYPES.get(annotation, "string")
    return "string"


def _parse_docstring(fn: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    doc = inspect.getdoc(fn) or ""
    if not doc:
        return "", {}
    sections = doc.split("\n\n")
    description = sections[0].strip().replace("\n", " ")
    params: dict[str, str] = {}
    for section in sections[1:]:
        lines = section.strip().splitlines()
        if lines and lines[0].strip().lower() in ("args:", "arguments:", "parameters:", "params:"):
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    name, desc = line.split(":", 1)
                    params[name.strip()] = desc.strip()
    return description, params


def tool_from_callable(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    try:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
    except (TypeError, NameError):  # builtins without introspectable signatures
        raise WaAgentError(f"Tool '{fn!r}' must be a Python function with a typed signature")

    doc_desc, param_docs = _parse_docstring(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, str)
        prop: dict[str, Any] = {"type": _annotation_to_json_type(annotation)}
        if pname in param_docs:
            prop["description"] = param_docs[pname]
        elif annotation is str and pname in ("query", "question", "input", "text", "note"):
            prop.setdefault("description", pname.replace("_", " "))
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            default = param.default
            if isinstance(default, (str, int, float, bool)):
                prop["default"] = default
        properties[pname] = prop

    resolved_name = name or fn.__name__
    resolved_desc = (
        description
        or doc_desc
        or f"Execute the {resolved_name.replace('_', ' ')} tool."
    )
    return Tool(
        name=resolved_name,
        description=resolved_desc,
        parameters={"type": "object", "properties": properties, "required": required},
        handler=fn,
    )


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator turning a typed function into a :class:`Tool`."""

    def deco(f: Callable[..., Any]) -> Tool:
        return tool_from_callable(f, name=name, description=description)

    return deco(fn) if fn is not None else deco


class ToolRegistry:
    """A mutable set of tools available to an agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, item: Tool | Callable[..., Any], **kwargs: Any) -> Tool:
        resolved: Tool = item if isinstance(item, Tool) else tool_from_callable(item, **kwargs)
        self._tools[resolved.name] = resolved
        return resolved

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def items(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools)

    @classmethod
    def from_iterable(cls, tools: list[Tool | Callable[..., Any]]) -> "ToolRegistry":
        registry = cls()
        for t in tools:
            registry.register(t)
        return registry

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
