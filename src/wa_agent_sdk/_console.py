"""Console helpers — Windows pipes default to cp1252 and choke on glyphs."""

from __future__ import annotations

import sys


def force_utf8_stdio() -> None:
    """Make stdout/stderr tolerate any Unicode regardless of redirection."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - cosmetic hardening must never raise
            pass
