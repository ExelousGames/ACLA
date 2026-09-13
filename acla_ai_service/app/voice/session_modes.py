"""Voice chatbot session-mode routing.

The frontend sends ``session_context.session_mode`` during the voice handshake.
Each mode maps to one startup behavior document in
``app/external_knowledge_base/agent_behaviors``.
"""

DEFAULT_CHATBOT_SESSION_MODE = "front_desk"

SESSION_MODE_AGENT_BEHAVIORS = {
    "front_desk": "front_desk",
    "live": "live",
    "recorded": "recorded",
    "user_summary": "user_summary",
}

VALID_CHATBOT_SESSION_MODES = frozenset(SESSION_MODE_AGENT_BEHAVIORS)


def normalize_chatbot_session_mode(value: object) -> str | None:
    """Return a canonical session mode, accepting display-label variants."""
    if value is None:
        return DEFAULT_CHATBOT_SESSION_MODE
    mode = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return mode if mode in VALID_CHATBOT_SESSION_MODES else None


__all__ = [
    "DEFAULT_CHATBOT_SESSION_MODE",
    "SESSION_MODE_AGENT_BEHAVIORS",
    "VALID_CHATBOT_SESSION_MODES",
    "normalize_chatbot_session_mode",
]
