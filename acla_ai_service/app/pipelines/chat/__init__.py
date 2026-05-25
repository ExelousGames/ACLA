"""
AI Service for natural language processing and conversation
"""

from typing import Dict, Any, Optional, List
import asyncio
import logging
from openai import AsyncOpenAI
from app.infra.config import settings

LOGGER = logging.getLogger(__name__)


class AIService:
    """Service for AI-powered analysis and conversation.

    Phase 1: the canonical chat backend is the local llama-server sidecar
    (GGUF model configured via settings.llama_model_*), called through an
    AsyncOpenAI client pointed at settings.llama_server_url. The legacy
    OpenAI client is kept only as an emergency rollback path, selected
    via the `LLM_PROVIDER=openai` env var.
    """

    def __init__(self):
        # Local llama-server (canonical chat backend going forward).
        # llama-server does not authenticate, but the OpenAI client refuses
        # to construct without an api_key — any non-empty string works.
        self.llama_client = AsyncOpenAI(
            base_url=settings.llama_server_url,
            api_key="not-needed",
        )

        # Legacy OpenAI client. Kept for rollback (LLM_PROVIDER=openai) and
        # for /query/health reporting. Constructed lazily so a missing
        # api_key during normal llama-mode operation isn't an error.
        self.openai_client = (
            AsyncOpenAI(api_key=settings.openai_api_key)
            if settings.openai_api_key else None
        )

        # Pick the active chat client based on settings.llm_provider.
        # Default = "llama". Set LLM_PROVIDER=openai in env to revert.
        if settings.llm_provider == "openai":
            if not self.openai_client:
                raise RuntimeError(
                    "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set"
                )
            self.llm_client = self.openai_client
            self.chat_model = "gpt-4o"
            self.llm_provider = "openai"
        else:
            self.llm_client = self.llama_client
            self.chat_model = settings.llama_model_name
            self.llm_provider = "llama"

    async def _execute_function(self, function_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the called function to retrieve data from local AI models and telemetry systems
        
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
            if function_name == "analyze_telemetry":
                return await self._composite_analyze_scope(
                    scope=arguments.get("scope") or {},
                    conn=(context or {}).get("_conn"),
                )
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

    @property
    def segment_classifier(self):
        """Lazy-loaded singleton — defers PyTorch + model load until first use."""
        svc = getattr(self, "_segment_classifier_instance", None)
        if svc is None:
            from app.ml.segment_classifier.service import SegmentClassifierService
            svc = SegmentClassifierService()
            self._segment_classifier_instance = svc
        return svc

    async def _classify_segment_impl(self, telemetry_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run :py:meth:`SegmentClassifierService.predict_segment` over rows.

        Returns labels with both raw ids (under ``_label_ids`` — side product)
        and natural-language names (under ``labels`` — what the LLM sees) so
        the LLM never speaks codes aloud.
        """
        if not telemetry_rows:
            return {"labels": [], "_label_ids": []}
        import asyncio as _asyncio
        import pandas as _pd
        from app.domain.labels import LABEL_MAPPING

        def _run() -> List[str]:
            df = _pd.DataFrame(telemetry_rows)
            return list(self.segment_classifier.predict_segment(df) or [])

        try:
            label_ids = await _asyncio.to_thread(_run)
        except Exception as exc:
            svc = self.segment_classifier
            present = {
                p.name: p.exists()
                for p in (svc.model_path, svc.mlb_path, svc.scaler_path)
            }
            print(
                f"[classifier-debug] models_directory={svc.models_directory} "
                f"expected_files={present}",
                flush=True,
            )
            return {"error": f"classifier failed: {exc}"}

        names = [LABEL_MAPPING.get(lid, lid) for lid in label_ids]
        return {"labels": names, "_label_ids": label_ids}

    async def _explain_label_impl(self, label_id: str) -> Dict[str, Any]:
        """Phase 1 stub. Phase 2 populates this against the racing-engineer
        Markdown corpus at ``app/skills/racing_engineer/labels/<ID>.md``.

        Accepts either a raw id ("MS44") or a natural name ("Oversteering at
        entry") — looks the natural name up in ``LABEL_NAME_TO_ID`` first.
        """
        from app.domain.labels import LABEL_MAPPING, LABEL_NAME_TO_ID

        if not label_id:
            return {"error": "label_id is required"}

        # Normalise: prefer raw id; fall back to name → id lookup.
        normalised = label_id if label_id in LABEL_MAPPING else LABEL_NAME_TO_ID.get(label_id, label_id)
        name = LABEL_MAPPING.get(normalised, label_id)

        try:
            from app.skills.racing_engineer import label as _label_lookup
            entry = _label_lookup(normalised)
        except Exception:
            entry = None

        if entry is None:
            return {
                "name": name,
                "definition": (
                    "Concept doc not authored yet — racing-engineer corpus "
                    "ships in Phase 2. Rely on your base-model knowledge of "
                    f"'{name}' for now."
                ),
                "_label_id": normalised,
            }

        return {
            "name": entry.get("name", name),
            "definition": entry.get("definition", ""),
            "engineer_interpretation": entry.get("engineer_interpretation", ""),
            "remedies": entry.get("remedies", []),
            "_label_id": normalised,
        }

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
            from app.skills.racing_engineer import track as _track_lookup
            entry = _track_lookup(track, corner=corner)
        except Exception as exc:
            return {"error": f"track lookup failed: {exc}"}
        if entry is None:
            try:
                from app.skills.racing_engineer import _load_category
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
            from app.skills.racing_engineer import search as _kb_search
            hits = await asyncio.to_thread(_kb_search, query, k)
        except Exception as exc:
            LOGGER.exception("search_racing_knowledge failed")
            return {"error": f"knowledge search failed: {exc}"}
        return {"query": query, "hits": hits}

    async def _composite_analyze_scope(self, scope: Dict[str, Any], conn: Any) -> Dict[str, Any]:
        """Canonical analyze flow for any QueryScope shape.

        Relays the server-internal ``_get_telemetry_for_scope`` frontend
        handler to fetch rows for the scope, classifies in-process, then
        resolves each detected label against the racing-engineer corpus.
        Rows never re-enter the LLM context — only the labels do.
        """
        if not isinstance(scope, dict) or "type" not in scope:
            return {"error": "scope must be an object with a 'type' field"}
        return await self._composite_analyze(
            conn=conn,
            frontend_tool="_get_telemetry_for_scope",
            frontend_args={"scope": scope},
            scope_summary={"scope": scope},
        )

    async def _composite_analyze(
        self,
        *,
        conn: Any,
        frontend_tool: str,
        frontend_args: Dict[str, Any],
        scope_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Shared chain backing the analyze_* composites.

        relay the named frontend tool → unwrap ``rows`` → classify → look
        up each detected label in the racing-engineer corpus → return the
        bundled payload. Errors at any step return ``{"error": ...}`` so
        the LLM can verbalize cleanly via the system prompt's "if the
        link is down, say so" rule.
        """
        if conn is None:
            return {"error": "no_connection_bound"}

        from app.voice.tool_relay import get_relay
        relay = get_relay()

        telemetry_resp = await relay.dispatch(conn, frontend_tool, frontend_args)
        if not isinstance(telemetry_resp, dict) or "error" in telemetry_resp:
            return {
                "error": (telemetry_resp or {}).get("error", "telemetry_unavailable"),
            }

        # `result` fallback: tool_relay wraps non-dict frontend returns
        # (e.g. a bare list) as {"result": [...]} — accept that shape too.
        _result = telemetry_resp.get("result")
        rows = (
            telemetry_resp.get("rows")
            or telemetry_resp.get("telemetry_rows")
            or (_result if isinstance(_result, list) else [])
        )
        if not rows:
            return {
                "telemetry_summary": {"rows": 0, **scope_summary},
                "labels": [],
            }

        classify_result = await self._classify_segment_impl(rows)
        if "error" in classify_result:
            return classify_result

        from app.domain.labels import LABEL_NAME_TO_ID

        labels_out: List[Dict[str, Any]] = []
        for name in classify_result.get("labels", []):
            label_id = LABEL_NAME_TO_ID.get(name, name)
            entry = await self._explain_label_impl(label_id)
            labels_out.append({
                "name": entry.get("name", name),
                "definition": entry.get("definition", ""),
                "engineer_interpretation": entry.get("engineer_interpretation", ""),
                "remedies": entry.get("remedies", []),
            })

        return {
            "telemetry_summary": {"rows": len(rows), **scope_summary},
            "labels": labels_out,
            "_label_ids": classify_result.get("_label_ids", []),
        }
