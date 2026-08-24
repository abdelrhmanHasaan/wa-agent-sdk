"""wa-agent-sdk — build WhatsApp AI agents on your own number, no Meta approval.

Connects through a Baileys (WhatsApp Web multi-device) bridge: pair by scanning
a QR code once, then every incoming message is answered by the LLM provider you
configure. Documents (PDF/DOCX/MD/…) are parsed automatically and images are
sent to vision-capable models.

Quick start:
    from wa_agent_sdk import WhatsAppAgent, LLMConfig

    agent = WhatsAppAgent(
        llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-..."),
    )
    agent.run()
"""

from ._version import __version__
from .client import WhatsAppAgent, agent_from_config
from .config import AgentConfig, LLMConfig, known_providers
from .exceptions import (
    BridgeError,
    MediaError,
    NotLinkedError,
    ProviderAuthError,
    ProviderError,
    QRTimeoutError,
    UnsupportedDocumentError,
    WaAgentError,
)
from .llm.base import BaseChatProvider, ChatMessage, ChatResult, ToolCall, image_block, text_block
from .llm.factory import create_provider, register_provider
from .memory import ConversationMemory
from .models import IncomingMessage, MediaType, SentReceipt
from .router import AgentRouter, Route, TriggerBoard
from .routing import (
    ModelEndpoint,
    ProviderRouter,
    QueryProfile,
    Strategy,
    Tier,
    UsageTracker,
    profile_query,
)
from .safety import SafetyManager
from .scheduler import CampaignReport, Job, Scheduler
from .tools import Tool, ToolRegistry, parse_document, prepare_image, tool

__all__ = [
    "WhatsAppAgent",
    "agent_from_config",
    "LLMConfig",
    "AgentConfig",
    "known_providers",
    "register_provider",
    "create_provider",
    "BaseChatProvider",
    "ChatMessage",
    "ChatResult",
    "ToolCall",
    "text_block",
    "image_block",
    "ConversationMemory",
    "IncomingMessage",
    "SentReceipt",
    "MediaType",
    "Tool",
    "ToolRegistry",
    "tool",
    "parse_document",
    "prepare_image",
    "AgentRouter",
    "Route",
    "TriggerBoard",
    "ProviderRouter",
    "ModelEndpoint",
    "Strategy",
    "Tier",
    "QueryProfile",
    "profile_query",
    "UsageTracker",
    "SafetyManager",
    "Scheduler",
    "Job",
    "CampaignReport",
    "WaAgentError",
    "BridgeError",
    "ProviderError",
    "ProviderAuthError",
    "QRTimeoutError",
    "NotLinkedError",
    "UnsupportedDocumentError",
    "MediaError",
    "__version__",
]
