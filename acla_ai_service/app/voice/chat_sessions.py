"""Process-local reconnectable chat session storage for voice conversations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import uuid
from typing import Any, Dict, List, Optional


Message = Dict[str, Any]


class ChatSessionError(Exception):
    """A policy violation while attaching to a chat session."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class ChatSessionSnapshot:
    """A detached copy of the registry state for one chat session."""

    chat_session_id: str
    user_id: str
    committed_history: List[Message] = field(default_factory=list)
    active: bool = False


@dataclass
class _ChatSession:
    user_id: str
    committed_history: List[Message] = field(default_factory=list)
    active: bool = False


class ChatSessionRegistry:
    """Own process-local chat history and single-socket attachment state."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _ChatSession] = {}

    def create_attached(self, user_id: str) -> ChatSessionSnapshot:
        """Create a new active session owned by ``user_id``."""
        chat_session_id = str(uuid.uuid4())
        while chat_session_id in self._sessions:
            chat_session_id = str(uuid.uuid4())

        session = _ChatSession(user_id=user_id, active=True)
        self._sessions[chat_session_id] = session
        return self._snapshot(chat_session_id, session)

    def resume_attached(
        self,
        chat_session_id: str,
        user_id: str,
    ) -> ChatSessionSnapshot:
        """Atomically attach to an inactive session after policy checks."""
        session = self._sessions.get(chat_session_id)
        if session is None:
            raise ChatSessionError(
                "ChatSessionNotFound",
                "The requested chat session does not exist in this AI-service process.",
            )
        if session.user_id != user_id:
            raise ChatSessionError(
                "ChatSessionOwnerMismatch",
                "The requested chat session belongs to a different user.",
            )
        if session.active:
            raise ChatSessionError(
                "ChatSessionAlreadyActive",
                "The requested chat session already has an active connection.",
            )

        session.active = True
        return self._snapshot(chat_session_id, session)

    def detach(
        self,
        chat_session_id: str,
        committed_history: Optional[List[Message]] = None,
    ) -> None:
        """Detach the transport, optionally replacing committed history."""
        session = self._sessions.get(chat_session_id)
        if session is None:
            return
        if committed_history is not None:
            session.committed_history = deepcopy(committed_history)
        session.active = False

    def get(self, chat_session_id: str) -> Optional[ChatSessionSnapshot]:
        """Return a copy of current session state, if present."""
        session = self._sessions.get(chat_session_id)
        if session is None:
            return None
        return self._snapshot(chat_session_id, session)

    @staticmethod
    def _snapshot(
        chat_session_id: str,
        session: _ChatSession,
    ) -> ChatSessionSnapshot:
        return ChatSessionSnapshot(
            chat_session_id=chat_session_id,
            user_id=session.user_id,
            committed_history=deepcopy(session.committed_history),
            active=session.active,
        )


_CHAT_SESSION_REGISTRY = ChatSessionRegistry()


def get_chat_session_registry() -> ChatSessionRegistry:
    """Return the process-wide reconnectable chat session registry."""
    return _CHAT_SESSION_REGISTRY
