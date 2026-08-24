"""Tool layer: registry, document parsing, image prep, built-in tools."""

from .base import Tool, ToolRegistry, tool, tool_from_callable
from .builtin import calculate, create_builtin_tools, current_datetime
from .documents import ParsedDocument, format_document_block, parse_document
from .media import detect_image_mime, prepare_image

__all__ = [
    "Tool",
    "ToolRegistry",
    "tool",
    "tool_from_callable",
    "create_builtin_tools",
    "calculate",
    "current_datetime",
    "parse_document",
    "ParsedDocument",
    "format_document_block",
    "prepare_image",
    "detect_image_mime",
]
