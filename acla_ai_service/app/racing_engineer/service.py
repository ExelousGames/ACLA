"""
AI Service for natural language processing and conversation
"""

from typing import Dict, Any, Optional, List
import asyncio
import logging
from openai import AsyncOpenAI
from app.infra.config import settings

LOGGER = logging.getLogger(__name__)


_EVENT_TYPES = {"CORNER", "STRAIGHT", "CRASHED", "OVERTAKE"}
_EVENT_WHICH = {"last", "current"}
_LAP_KEYWORDS = {"current", "last"}


def _validate_scope(scope: Any) -> Optional[str]:
    """Validate a QueryScope object received from the LLM.

    The frontend-owned JSON Schema (QUERY_SCOPE_SCHEMA in ai-command-registry.ts)
    is a flat object with a `type` enum — it doesn't encode the per-type field
    coupling because Groq llama-3.3-70b can't reliably emit oneOf+const
    discriminated unions. This validator enforces the coupling server-side
    and returns a human-readable error string so the LLM can retry with a
    corrected call. Returns None if valid.
    """
    if not isinstance(scope, dict):
        return "scope must be an object"
    t = scope.get("type")
    if t == "now":
        return None
    if t == "last_seconds":
        sec = scope.get("seconds")
        if not isinstance(sec, (int, float)) or sec <= 0:
            return "scope.type='last_seconds' requires positive 'seconds' (number)"
        return None
    if t == "event":
        if scope.get("eventType") not in _EVENT_TYPES:
            return f"scope.type='event' requires 'eventType' in {sorted(_EVENT_TYPES)}"
        if scope.get("which") not in _EVENT_WHICH:
            return f"scope.type='event' requires 'which' in {sorted(_EVENT_WHICH)}"
        return None
    if t == "lap":
        lap = scope.get("lap")
        if not (lap in _LAP_KEYWORDS or isinstance(lap, int)):
            return "scope.type='lap' requires 'lap' as 'current', 'last', or an integer"
        return None
    if t == "range":
        start, end = scope.get("start"), scope.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            return "scope.type='range' requires integer 'start' and 'end'"
        if start >= end:
            return "scope.type='range' requires start < end"
        return None
    return "scope.type must be one of: now, last_seconds, event, lap, range"


class AIService:
    """Service for AI-powered analysis and conversation.

    Backend is chosen by whether ``HOSTED_LLM_BASE_URL`` is set:
      * unset → local llama-server sidecar (GGUF model configured via
        ``settings.llama_model_*``).
      * set   → any OpenAI-compatible hosted endpoint (Groq, Cerebras,
        Together, Fireworks, OpenRouter, …). Also requires
        ``HOSTED_LLM_API_KEY`` + ``HOSTED_LLM_MODEL``.

    Both paths use the same ``AsyncOpenAI`` client; only base_url / api_key /
    model differ.
    """

    def __init__(self):
        if settings.hosted_llm_base_url:
            missing = [
                name for name, val in (
                    ("HOSTED_LLM_API_KEY", settings.hosted_llm_api_key),
                    ("HOSTED_LLM_MODEL", settings.hosted_llm_model),
                ) if not val
            ]
            if missing:
                raise RuntimeError(
                    f"HOSTED_LLM_BASE_URL is set; also requires {', '.join(missing)}"
                )
            self.llm_client = AsyncOpenAI(
                base_url=settings.hosted_llm_base_url,
                api_key=settings.hosted_llm_api_key,
            )
            self.chat_model = settings.hosted_llm_model
        else:
            # llama-server does not authenticate, but the OpenAI client
            # refuses to construct without an api_key — any non-empty string
            # works.
            self.llm_client = AsyncOpenAI(
                base_url=settings.llama_server_url,
                api_key="not-needed",
            )
            self.chat_model = settings.llama_model_name

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
            if function_name == "classify_live_section":
                return await self._classify_live_section_impl(
                    conn=(context or {}).get("_conn"),
                    section_id=str(arguments.get("section_id") or "").strip(),
                    section_name=(str(arguments.get("section_name")).strip()
                                  if arguments.get("section_name") else None),
                    lap=arguments.get("lap", "last"),
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
        """Shared chatbot model hub singleton."""
        from app.ml.model_hub import get_segment_classifier

        return get_segment_classifier()

    @property
    def opportunity_forecaster(self):
        """Shared chatbot model hub singleton."""
        from app.ml.model_hub import get_opportunity_forecaster

        return get_opportunity_forecaster()

    @property
    def expert_imitation_learning(self):
        """Shared chatbot model hub singleton."""
        from app.ml.model_hub import get_expert_imitation_learning

        return get_expert_imitation_learning()

    @property
    def tire_grip_analysis(self):
        """Shared chatbot model hub singleton."""
        from app.ml.model_hub import get_tire_grip_analysis

        return get_tire_grip_analysis()

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
        from app.shared.labels import LABEL_MAPPING

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

    async def _composite_analyze_scope(self, scope: Dict[str, Any], conn: Any) -> Dict[str, Any]:
        """Canonical analyze flow for any QueryScope shape.

        Relays the server-internal ``_get_telemetry_for_scope`` frontend
        handler to fetch rows for the scope, classifies in-process, then
        resolves each detected label against the racing-engineer corpus.
        Rows never re-enter the LLM context — only the labels do.
        """
        err = _validate_scope(scope)
        if err is not None:
            return {"error": err}
        return await self._composite_analyze(
            conn=conn,
            frontend_tool="_get_telemetry_for_scope",
            frontend_args={"scope": scope},
            scope_summary={"scope": scope},
        )

    async def _classify_live_section_impl(
        self,
        *,
        conn: Any,
        section_id: str,
        section_name: Optional[str],
        lap: Any,
    ) -> Dict[str, Any]:
        """Classify one live track-section window and record it in the frontend.

        This is the live analyst bridge: the LLM sees only this compact tool,
        while raw section rows travel over the internal frontend relay and are
        consumed here by the segment classifier.
        """
        if not section_id and not section_name:
            return {"error": "section_id or section_name is required"}

        args: Dict[str, Any] = {"lap": lap or "last"}
        if section_id:
            args["section_id"] = section_id
        if section_name:
            args["section_name"] = section_name

        return await self._composite_analyze(
            conn=conn,
            frontend_tool="_get_live_section_telemetry",
            frontend_args=args,
            scope_summary={
                "live_section": {
                    "section_id": section_id,
                    "section_name": section_name,
                    "lap": lap or "last",
                },
            },
            record_live_classification=True,
        )

    async def _composite_analyze(
        self,
        *,
        conn: Any,
        frontend_tool: str,
        frontend_args: Dict[str, Any],
        scope_summary: Dict[str, Any],
        record_live_classification: bool = False,
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

        from app.shared.labels import LABEL_MAPPING, LABEL_NAME_TO_ID

        labels_out: List[Dict[str, Any]] = []
        for name in classify_result.get("labels", []):
            label_id = LABEL_NAME_TO_ID.get(name, name)
            entry = await self._explain_label_impl(label_id)
            labels_out.append({
                "name": entry.get("name", name),
                "definition": entry.get("definition", ""),
                **({"solution": entry["solution"]} if entry.get("solution") else {}),
            })

        payload: Dict[str, Any] = {
            "telemetry_summary": {"rows": len(rows), **scope_summary},
            "labels": labels_out,
            "_label_ids": classify_result.get("_label_ids", []),
        }

        if record_live_classification:
            label_ids = [str(label_id) for label_id in classify_result.get("_label_ids", [])]
            parent_labels = [
                label_id for label_id in label_ids
                if label_id in {"MSP", "MSR", "EA", "RM", "PS", "O", "OD"}
            ]
            child_labels = [
                LABEL_MAPPING.get(label_id, label_id)
                for label_id in label_ids
                if label_id not in {"MSP", "MSR", "EA", "RM", "PS", "O", "OD"}
                and not label_id.startswith("ST")
            ]
            mistake_count = sum(
                1 for label_id in label_ids
                if label_id in {"MSP", "MSR"} or label_id.startswith(("MSP", "MSR"))
            )
            expert_count = sum(
                1 for label_id in label_ids
                if label_id == "EA" or label_id.startswith("EA")
            )
            severity = round(min(5.0, mistake_count + (0.5 if "RM" in parent_labels else 0.0)), 3)
            confidence = round(min(0.95, 0.45 + (0.08 * len(label_ids)) + min(len(rows), 200) / 1000), 3)

            section = telemetry_resp.get("section") if isinstance(telemetry_resp.get("section"), dict) else {}
            lap_value = telemetry_resp.get("lap")
            record_args = {
                "section_id": section.get("id") or frontend_args.get("section_id"),
                "section_name": section.get("name") or frontend_args.get("section_name"),
                "lap": lap_value if lap_value is not None else frontend_args.get("lap"),
                "start_sample_idx": telemetry_resp.get("startSampleIdx", telemetry_resp.get("start_sample_idx", 0)),
                "end_sample_idx": telemetry_resp.get("endSampleIdx", telemetry_resp.get("end_sample_idx", 0)),
                "mistake_count": mistake_count,
                "expert_adherence_count": expert_count,
                "severity": severity,
                "confidence": confidence,
                "parent_label": LABEL_MAPPING.get(parent_labels[0], parent_labels[0]) if parent_labels else None,
                "child_labels": child_labels,
                "telemetry_stats": _live_section_stats(rows),
            }

            from app.voice.tool_relay import get_relay
            recorded = await get_relay().dispatch(
                conn,
                "_record_live_section_classification",
                record_args,
            )
            if isinstance(recorded, dict) and "error" in recorded:
                payload["recording_error"] = recorded.get("error")
            else:
                payload["classification"] = (
                    recorded.get("classification") if isinstance(recorded, dict) else record_args
                )
                payload["focus"] = recorded.get("focus") if isinstance(recorded, dict) else None
                payload["comparison"] = recorded.get("comparison") if isinstance(recorded, dict) else None

        return payload


def _live_section_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Small LLM-safe telemetry summary for live section comparison."""
    aliases = {
        "speed": ("Physics_speed_kmh", "Physics_speed", "speed_kmh", "speed"),
        "brake": ("Physics_brake", "brake", "brake_pressure"),
        "throttle": ("Physics_gas", "Physics_throttle", "throttle", "gas"),
        "steer": ("Physics_steer_angle", "Physics_steer", "steer", "steering"),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, keys in aliases.items():
        values: List[float] = []
        for row in rows:
            for key in keys:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
                    break
        if values:
            out[name] = {
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "avg": round(sum(values) / len(values), 3),
            }
    return out
