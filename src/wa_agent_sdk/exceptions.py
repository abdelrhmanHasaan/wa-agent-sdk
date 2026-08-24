"""Exception hierarchy for the WhatsApp Agent SDK."""

from __future__ import annotations


class WaAgentError(Exception):
    """Base class for every SDK error."""


class BridgeError(WaAgentError):
    """The Node/Baileys bridge failed to start or communicate."""


class NodeNotFoundError(BridgeError):
    """Node.js or npm is not available on PATH."""


class BridgeInstallError(BridgeError):
    """npm install of bridge dependencies failed."""


class BridgeNotRunningError(BridgeError):
    """An operation was attempted while the bridge is stopped."""


class QRTimeoutError(BridgeError):
    """No one scanned the pairing QR within the allotted time."""


class NotLinkedError(BridgeError):
    """An operation requiring an authenticated session ran before linking."""


class ProviderError(WaAgentError):
    """An LLM provider returned an error."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class ProviderAuthError(ProviderError):
    """Authentication with the LLM provider failed (bad/missing API key)."""


class ToolExecutionError(WaAgentError):
    """A tool raised while executing; the text is surfaced to the model."""


class UnsupportedDocumentError(WaAgentError):
    """The attached document type has no parser registered."""


class MediaError(WaAgentError):
    """Downloading or preparing media failed."""
