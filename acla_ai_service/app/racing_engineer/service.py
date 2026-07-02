"""
AI Service for natural language processing and conversation
"""

from typing import Dict, Any, Optional
import asyncio
import logging
from openai import AsyncOpenAI
from app.chat_llm import resolve_chat_llm_config

LOGGER = logging.getLogger(__name__)


class AIService:
    """Service for AI-powered analysis and conversation.

    Chat uses the configured remote provider from ``CHAT_LLM_PROVIDER``.
    Both supported providers use the same ``AsyncOpenAI`` client; only
    base_url / api_key / model differ.
    """

    def __init__(self, chat_llm_provider: Optional[str] = None):
        llm_config = resolve_chat_llm_config(chat_llm_provider)
        self.llm_client = AsyncOpenAI(**llm_config.openai_client_kwargs())
        self.chat_model = llm_config.model

    async def _execute_function(self, function_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute server-side racing-engineer knowledge tools.
        
        FUNCTION OUTPUT SEPARATION:
        ┌─────────────────────────────────────────────────────────────────┐
        │                    Function Return Format                       │
        │                                                                 │
        │  {                                                              │
        │    # Regular keys → Sent to OpenAI for final answer            │
        │    "status": "success",                                         │
        │    "message": "Operation completed",                            │
        │                                                                 │
        │    # Keys starting with _ → Side products for external use     │
        │    "_guidance_enabled": true,                                   │
        │    "_prediction_result": {...},                                 │
        │    "_track_corner_data": {...},                                │
        │    "_skip_openai_processing": true                             │
        │  }                                                             │
        └─────────────────────────────────────────────────────────────────┘
        """
        try:
            # ── Racing-engineer server-side tools ──────────────────────────
            if function_name == "explain_label":
                return await self._explain_label_impl(
                    label_id=str(arguments.get("label_id") or "").strip(),
                )
            if function_name == "get_track_knowledge":
                return await self._get_track_knowledge_impl(
                    track=str(arguments.get("track") or "").strip(),
                    corner=(str(arguments.get("corner")).strip()
                            if arguments.get("corner") else None),
                )
            if function_name == "search_racing_knowledge":
                return await self._search_racing_knowledge_impl(
                    query=str(arguments.get("query") or "").strip(),
                    top_k=arguments.get("top_k"),
                )

            print(f"[ERROR] Unknown function: {function_name}")
            return {"error": f"Unknown function: {function_name}"}

        except Exception as e:
            return {"error": f"Function {function_name} execution failed: {str(e)}"}

    # ------------------------------------------------------------------
    # Phase 1 racing-engineer tool implementations
    # ------------------------------------------------------------------

    async def _explain_label_impl(self, label_id: str) -> Dict[str, Any]:
        """Fetch the racing-engineer concept doc for one action label.

        Accepts either a raw id ("MSP44") — typically classifier output —
        or a natural name ("Oversteering at entry"). Internally resolves to
        the canonical human name via ``LABEL_MAPPING`` and looks up the
        slugged file under ``app/external_knowledge_base/labels/``. Ids are
        never used to address files.
        """
        from app.shared.labels import LABEL_MAPPING, LABEL_NAME_TO_ID

        if not label_id:
            return {"error": "label_id is required"}

        # Normalise: input may be a raw id (classifier output) or a name.
        # Convert to the canonical human name via LABEL_MAPPING; the corpus
        # is keyed by that name (filename stem), never by id.
        normalised_id = label_id if label_id in LABEL_MAPPING else LABEL_NAME_TO_ID.get(label_id, label_id)
        name = LABEL_MAPPING.get(normalised_id, label_id)

        try:
            from app.external_knowledge_base import label as _label_lookup
            entry = _label_lookup(name)
        except Exception:
            entry = None

        if entry is None:
            return {
                "name": name,
                "definition": (
                    "Concept doc not authored yet. Racing-engineer corpus "
                    "does not have this label. Rely on your base-model knowledge of "
                    f"'{name}' for now."
                ),
            }

        result = {
            "name": entry.get("name", name),
            "definition": entry.get("definition", ""),
        }
        solution = entry.get("solution")
        if solution:
            result["solution"] = solution
        return result

    async def _get_track_knowledge_impl(
        self, track: str, corner: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Keyed lookup over the racing-engineer ``tracks/`` corpus.

        Returns ``{error, available_tracks}`` if the track id isn't known,
        so the LLM can recover by either retrying with the right id or
        falling back to ``search_racing_knowledge``.
        """
        if not track:
            return {"error": "track is required"}
        try:
            from app.external_knowledge_base import track as _track_lookup
            entry = _track_lookup(track, corner=corner)
        except Exception as exc:
            return {"error": f"track lookup failed: {exc}"}
        if entry is None:
            try:
                from app.external_knowledge_base import _load_category
                available = sorted(_load_category("tracks").keys())
            except Exception:
                available = []
            return {"error": f"track '{track}' not in corpus", "available_tracks": available}
        return entry

    async def _search_racing_knowledge_impl(
        self, query: str, top_k: Any = None,
    ) -> Dict[str, Any]:
        """RAG search over the racing-engineer ``knowledge/`` corpus.

        Runs in a worker thread so the SentenceTransformer encode call
        (CPU-bound, can take ~50ms) doesn't block the event loop.
        """
        if not query:
            return {"error": "query is required"}
        # Coerce top_k — LLM may send "5" as string or skip it entirely.
        k: Optional[int] = None
        if top_k is not None:
            try:
                k = int(top_k)
            except (TypeError, ValueError):
                k = None
        try:
            from app.external_knowledge_base import search as _kb_search
            hits = await asyncio.to_thread(_kb_search, query, k)
        except Exception as exc:
            LOGGER.exception("search_racing_knowledge failed")
            return {"error": f"knowledge search failed: {exc}"}
        return {"query": query, "hits": hits}
