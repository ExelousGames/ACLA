"""
AI Service for natural language processing and conversation
"""

import base64
from typing import Dict, Any, AsyncIterator, Optional, List
import json
import asyncio
import logging
from openai import AsyncOpenAI
from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService
from app.infra.config import settings
from app.integrations.backend.client import BackendService
from app.voice import get_kokoro_service
from app.voice.sentence_streamer import SentenceStreamer
from app.voice.stream_events import (
    event_audio,
    event_done,
    event_error,
    event_token,
    event_tool_end,
    event_tool_start,
)

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

        self.backend_service = BackendService()
        self.telemetryMLService = Full_dataset_TelemetryMLService()
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Tool schemas advertised to the LLM (OpenAI function-calling format).

        Mirrored from :func:`app.voice.pipecat_pipeline._build_tool_schemas`
        — voice and any future HTTP-style path must advertise the same
        capabilities so the LLM behaves consistently across surfaces. Tools
        whose execution lives on the frontend (relayed over the voice WS)
        are listed here too; they're routed at dispatch time by the voice
        tool handler, not by ``_execute_function``.
        """
        _SCOPE_DESC = (
            "One of: {type:'last_seconds', seconds:N}, "
            "{type:'event', eventType:'CORNER'|'CRASHED'|'OVERTAKE', which:'last'|'current'}, "
            "{type:'lap', lap:'current'|'last'|N}, "
            "{type:'range', start:N, end:N}"
        )
        return [
            # ── Telemetry surface (two tools, same scope language) ─────────
            {
                "name": "analyze_telemetry",
                "description": (
                    "Classify driving actions over a scope and return the "
                    "engineer concept for each. Server fetches rows, runs "
                    "the classifier, bundles labels — rows never enter "
                    "this conversation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "object", "description": _SCOPE_DESC},
                    },
                    "required": ["scope"],
                },
            },
            {
                "name": "query_telemetry_metric",
                "description": (
                    "Aggregate a telemetry metric over a scope. Field "
                    "groups: speed, throttle, brake, gear, steering, rpm, "
                    "tyre_pressure, tyre_temp, brake_temp, tyre_slip, "
                    "g_force, suspension, fuel, lap_delta, position, "
                    "race_position."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Field group names or raw Physics_* field names.",
                        },
                        "scope": {"type": "object", "description": _SCOPE_DESC},
                        "reduce": {
                            "type": "string",
                            "enum": ["avg", "min", "max", "stats"],
                            "description": "avg/min/max=single value, stats={avg,min,max,stddev}.",
                        },
                    },
                    "required": ["fields", "scope", "reduce"],
                },
            },
            # ── Concept lookup ─────────────────────────────────────────────
            {
                "name": "explain_label",
                "description": (
                    "Return the racing-engineer concept (definition, "
                    "engineer interpretation, remedies) for one action "
                    "label, by id (e.g. 'MS44') or natural name."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label_id": {"type": "string"},
                    },
                    "required": ["label_id"],
                },
            },
            # ── Frontend-relayed (dispatched at the voice tool handler) ───
            {
                "name": "get_session_info",
                "description": "Return current track / car / user identity from the live session.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "start_per_turn_coaching",
                "description": (
                    "Activate the background monitoring agent that pushes an "
                    "observation after every completed corner."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "stop_per_turn_coaching",
                "description": "Stop the per-corner monitoring agent.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_event_log",
                "description": (
                    "Search the session event log for racing events (corners, crashes, overtakes). "
                    "Use to find when things happened before querying telemetry around them."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "eventType": {
                            "type": "string",
                            "enum": ["CORNER", "STRAIGHT", "CRASHED", "OVERTAKE"],
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["last", "last_n", "lap_current", "lap_last", "all"],
                        },
                        "n": {"type": "integer", "description": "For last_n scope."},
                    },
                    "required": ["eventType", "scope"],
                },
            },
            {
                "name": "get_next_corner",
                "description": "Return the next corner ahead of the car on the current track.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_telemetry_schema",
                "description": "List available telemetry field group names and raw Physics_* field names.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
    
    async def process_natural_language_query(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Process natural language queries about racing data and function calling
        
        PROCESS FLOW DIAGRAM:
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │                           TWO-STAGE AI PROCESSING                           │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        ┌─────────────────┐    ┌─────────────────────────────────────────────────────┐
        │   User Prompt   │───▶│                STAGE 1: ANALYSIS                   │
        └─────────────────┘    │                                                     │
                               │  ┌─────────────────┐                               │
                               │  │   OpenAI GPT-4  │  Examines prompt &            │
                               │  │   First Query   │  decides what functions       │
                               │  │                 │  to call (if any)             │
                               │  └─────────┬───────┘                               │
                               │            │                                       │
                               │            ▼                                       │
                               │  ┌─────────────────┐                               │
                               │  │ Function Calls? │                               │
                               │  │   (tool_calls)  │                               │
                               │  └─────┬─────┬─────┘                               │
                               └────────┼─────┼─────────────────────────────────────┘
                                       NO    YES
                                        │     │
                               ┌────────▼─────▼─────────────────────────────────────┐
                               │                STAGE 2: EXECUTION                  │
                    ┌──────────┤                                                     │
                    │          │  ┌─────────────────┐    ┌─────────────────────────┐│
             Direct │          │  │  Execute Each   │    │    Function Results     ││
             Answer │          │  │    Function     │───▶│                         ││
                    │          │  │                 │    │  • Data for OpenAI      ││
                    │          │  └─────────┬───────┘    │  • Side Products (_*)   ││
                    │          │            │            │    - _guidance_enabled  ││
                    │          │            ▼            │    - _prediction_result ││
                    │          │  ┌─────────────────┐    │    - _track_corner_data ││
                    │          │  │   OpenAI GPT-4  │    │                         ││
                    │          │  │  Second Query   │◀───┤  Filter: Only send     ││
                    │          │  │  (Final Answer) │    │  non-underscore data    ││
                    │          │  └─────────┬───────┘    │  to OpenAI              ││
                    │          └────────────┼────────────┴─────────────────────────┘│
                    │                       │                                       │
                    │                       ▼                                       │
                    │          ┌─────────────────────────────────────────────────────┐
                    └─────────▶│                 FINAL RESULT                        │
                               │                                                     │
                               │  {                                                  │
                               │    "answer": "OpenAI's final response",            │
                               │    "side_products": {                              │
                               │      "track_detail_for_guide": {                 │
                               │        "_guidance_enabled": true,                  │
                               │        "_prediction_result": {...},               │
                               │        "_track_corner_data": {...}                │
                               │      }                                             │
                               │    },                                              │
                               │    "context": {...},                              │
                               │    "messages": [...]                              │
                               │  }                                                 │
                               └─────────────────────────────────────────────────────┘
        
        KEY CONCEPTS:
        • Stage 1: OpenAI analyzes user intent and decides what functions to call
        • Stage 2: Functions execute and produce TWO types of outputs:
          - Regular data: Sent to OpenAI for generating the final answer
          - Side products (prefixed with _): Returned to caller for external use
        • Final result contains both OpenAI's answer AND all side products
        
        Args:
            prompt: The user's natural language query
            context: Optional context information (track, car, etc.)
            conversation_history: Optional previous conversation messages to maintain context
        """
        if not self.llm_client:
            raise Exception(
                f"LLM client is not available (provider={self.llm_provider})."
            )
        
        try:
            # Prepare context information and conversation history
            try:
                # Lists ONLY the tools that actually exist in get_available_functions()
                # and gives concrete trigger phrases for each.
                track_name = context.get('track_name', 'Unknown') if context else 'Unknown'
                car_name = context.get('car_name', 'Unknown') if context else 'Unknown'
                system_message = {
                    "role": "system",
                    "content": (
                        "You are a racing telemetry coach for sim racing. Stay in character.\n"
                        f"Current session: track={track_name}, car={car_name}\n"
                        "\n"
                        "Available tools (call only when the user asks something the tool answers):\n"
                        "- check_car_limit(session_id, data_types?): "
                        "Use when the user asks whether they are pushing the car to its limit, "
                        "or about speed/braking/steering optimality.\n"
                        "- track_detail_for_guide(track_name): "
                        "Use when the user asks to be guided on the track or for a driving guide. "
                        "After calling it, return a JSON object with keys "
                        "throttle_guidance, brake_guidance, steering_guidance — each a list of "
                        "exactly 4 short technique sentences. No prose around the JSON.\n"
                        "\n"
                        "Trigger phrases:\n"
                        "- \"guide me on track\" / \"help me drive this track\" / \"start guiding\" "
                        "→ call track_detail_for_guide.\n"
                        "- \"am I at the limit\" / \"is this fast enough\" / \"how is my braking\" "
                        "→ call check_car_limit.\n"
                        "- General racing questions (\"how should I brake at high speed?\") "
                        "→ answer directly in 2-3 sentences, do not call tools.\n"
                        "\n"
                        "Ground all answers in data the tools return. Be concise unless the user "
                        "asks for detail."
                    ),
                }
                
                # Build messages array - use provided conversation history or start fresh
                if conversation_history and len(conversation_history) > 0:
                    # Use provided conversation history and add the new user message
                    messages = conversation_history.copy()
                    messages.append({"role": "user", "content": prompt})
                else:
                    # Start fresh conversation
                    messages = [
                        system_message,
                        {"role": "user", "content": prompt}
                    ]
                    
            except Exception as e:
                print(f"[ERROR] Failed to prepare context messages: {str(e)}")
                raise Exception(f"Context preparation failed: {str(e)}") from e
            
            # STEP 1: Send initial query to OpenAI to determine what functions to call
            try:
                tools = [{"type": "function", "function": func} for func in self.get_available_functions()]
            except Exception as e:
                print(f"[ERROR] Failed to prepare tools: {str(e)}")
                raise Exception(f"Tools preparation failed: {str(e)}") from e

            try:
                response = await self.llm_client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=500,
                )
            except Exception as e:
                print(f"[ERROR] LLM call failed ({self.llm_provider} / {self.chat_model}): {str(e)}")
                raise Exception(
                    f"LLM call failed ({self.llm_provider} / {self.chat_model}): {str(e)}"
                ) from e
            
            try:
                message = response.choices[0].message
            except Exception as e:
                print(f"[ERROR] Failed to parse OpenAI response: {str(e)}")
                raise Exception(f"OpenAI response parsing failed: {str(e)}") from e
            
            # STEP 2: If OpenAI decides to call functions, execute them to get data from local AI models or perform actions
            if message.tool_calls:
                print(f"[DEBUG] OpenAI decided to call {len(message.tool_calls)} function(s)")

                # Add the assistant message with tool calls to conversation history (once, not per tool call)
                try:
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to append assistant message to history: {str(e)}")
                    # Continue processing even if message history fails
                    pass
                
                # Store side products from function executions
                side_products = {}
                
                try:
                    for tool_call in message.tool_calls:
                        try:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                        except Exception as e:
                            # Store parse error in side_products for debugging
                            side_products["parse_error"] = {
                                "_error": f"Function call parsing failed: {str(e)}",
                                "_function": "parse_error",
                                "_arguments": {}
                            }
                            continue
                        
                        print(f"[DEBUG] Executing function: {function_name} with args: {function_args}")
                        
                        # Execute the function to get data or perform actions
                        try:
                            result = await self._execute_function(function_name, function_args, context)
                            
                            # Check if function has side products (anything that should be returned to caller)
                            if isinstance(result, dict):
                                # Extract data for OpenAI (excluding side products)
                                openai_data = {k: v for k, v in result.items() if not k.startswith('_')}
                                
                                # Extract side products (keys starting with _)
                                function_side_products = {k: v for k, v in result.items() if k.startswith('_')}
                                if function_side_products:
                                    side_products[function_name] = function_side_products
                                
                                # Use filtered data for OpenAI
                                result_for_openai = openai_data if openai_data else result
                            else:
                                result_for_openai = result
                                
                        except Exception as e:
                            print(f"[ERROR] Function execution failed for {function_name}: {str(e)}")
                            result = {"error": f"Function execution failed: {str(e)}"}
                            result_for_openai = result
                        
                        # Add the tool result to conversation history (use filtered data for OpenAI)
                        try:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result_for_openai)
                            })
                        except Exception as e:
                            # Continue processing even if message history fails
                            pass
                except Exception as e:
                    raise Exception(f"Function execution loop failed: {str(e)}") from e
                
                print(f"[DEBUG] All functions executed, sending results back to LLM for final response")
                # STEP 3: Send function results back to the LLM for final comprehensive response
                try:
                    final_response = await self.llm_client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1500,
                    )
                except Exception as e:
                    raise Exception(
                        f"Final LLM call failed ({self.llm_provider} / {self.chat_model}): {str(e)}"
                    ) from e
                
                try:
                    final_answer = final_response.choices[0].message.content
                    
                    # Add the final assistant response to the conversation
                    messages.append({
                        "role": "assistant",
                        "content": final_answer
                    })
                    
                    # Prepare final result with OpenAI answer and all side products
                    result = {
                        "answer": final_answer,
                        "context": context,
                        "messages": messages,  # Return updated conversation for external management
                    }
                    
                    # Add side products from function executions to the final result
                    if side_products:
                        result["side_products"] = side_products
                        
                    return result
                except Exception as e:
                    print(f"[ERROR] Failed to format final response: {str(e)}")
                    raise Exception(f"Final response formatting failed: {str(e)}") from e
            
            # If no functions were called, return the direct response
            try:
                direct_answer = message.content
                
                # Add the assistant response to the conversation
                messages.append({
                    "role": "assistant",
                    "content": direct_answer
                })
                
                result = {
                    "answer": direct_answer,
                    "context": context,
                    "messages": messages,  # Return updated conversation for external management
                }
                
                # No side products since no functions were called, but maintain consistent structure
                # (side_products will be empty/absent)
                
                return result
            except Exception as e:
                print(f"[ERROR] Failed to format direct response: {str(e)}")
                raise Exception(f"Direct response formatting failed: {str(e)}") from e
            
        except Exception as e:

            # Return error response with call stack information
            raise RuntimeError({
                "error": f"Natural language query processing failed: {str(e)}",
                "error_type": type(e).__name__,
                "processing_steps": [
                    "1. Error occurred during natural language query processing",
                    f"2. Error type: {type(e).__name__}",
                    f"3. Error message: {str(e)}"
                ],
                "context": context
            })

    # ------------------------------------------------------------------
    # Phase 2.5 — Streaming natural-language query
    # ------------------------------------------------------------------

    async def stream_natural_language_query(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Streaming variant of `process_natural_language_query`.

        Yields pre-formatted SSE event strings (see voice.stream_events).
        The caller wraps these in a FastAPI `StreamingResponse` with
        media_type="text/event-stream".

        Flow:
            Stage 1 (non-streaming, fast): decide whether to call a tool.
            Tool path:
                emit tool_start → execute → emit tool_end → stream stage 2
                with token + sentence-chunked audio events.
            No-tool path:
                emit the stage 1 content as a single token event, then
                synthesize each sentence and emit audio events.
            Always:
                terminate with a `done` event carrying answer + side_products
                + messages so the frontend can run post-response hooks.

            On any unhandled exception: emit one `error` event and return.
        """
        try:
            async for chunk in self._stream_impl(prompt, context, conversation_history):
                yield chunk
        except Exception as exc:
            LOGGER.exception("stream_natural_language_query failed")
            yield event_error(str(exc), error_type=type(exc).__name__)

    async def _stream_impl(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]],
    ) -> AsyncIterator[str]:
        if not self.llm_client:
            yield event_error(
                f"LLM client is not available (provider={self.llm_provider})",
                error_type="LLMUnavailable",
            )
            return

        # --- Build messages (shared with process_natural_language_query) ---
        track_name = context.get("track_name", "Unknown") if context else "Unknown"
        car_name = context.get("car_name", "Unknown") if context else "Unknown"
        system_message = {
            "role": "system",
            "content": (
                "You are a racing telemetry coach for sim racing. Stay in character.\n"
                f"Current session: track={track_name}, car={car_name}\n"
                "\n"
                "Available tools (call only when the user asks something the tool answers):\n"
                "- check_car_limit(session_id, data_types?): "
                "Use when the user asks whether they are pushing the car to its limit, "
                "or about speed/braking/steering optimality.\n"
                "- track_detail_for_guide(track_name): "
                "Use when the user asks to be guided on the track or for a driving guide. "
                "After calling it, return a JSON object with keys "
                "throttle_guidance, brake_guidance, steering_guidance — each a list of "
                "exactly 4 short technique sentences. No prose around the JSON.\n"
                "\n"
                "Trigger phrases:\n"
                "- \"guide me on track\" / \"help me drive this track\" / \"start guiding\" "
                "→ call track_detail_for_guide.\n"
                "- \"am I at the limit\" / \"is this fast enough\" / \"how is my braking\" "
                "→ call check_car_limit.\n"
                "- General racing questions (\"how should I brake at high speed?\") "
                "→ answer directly in 2-3 sentences, do not call tools.\n"
                "\n"
                "Ground all answers in data the tools return. Be concise unless the user "
                "asks for detail."
            ),
        }

        if conversation_history and len(conversation_history) > 0:
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [system_message, {"role": "user", "content": prompt}]

        tools = [
            {"type": "function", "function": func}
            for func in self.get_available_functions()
        ]

        # --- Stage 1: non-streaming tool routing ---
        # Keep this non-streaming so we have the complete tool_call (if any)
        # before deciding how to handle the response. Stage 1 is fast (<500ms).
        try:
            stage1 = await self.llm_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=500,
            )
        except Exception as exc:
            yield event_error(
                f"Stage 1 LLM call failed: {exc}",
                error_type=type(exc).__name__,
            )
            return

        stage1_msg = stage1.choices[0].message
        side_products: Dict[str, Any] = {}

        # --- Branch A: no tool — emit stage 1 content directly with TTS ---
        if not stage1_msg.tool_calls:
            answer_text = stage1_msg.content or ""
            messages.append({"role": "assistant", "content": answer_text})

            # Single token event with the full answer; client renders immediately.
            # (Token-by-token streaming on the no-tool path would require
            # stream=True in stage 1, which complicates tool_call detection.
            # The audio still streams sentence-by-sentence below.)
            if answer_text:
                yield event_token(answer_text)
                async for audio_evt in self._synth_text_to_audio_events(answer_text):
                    yield audio_evt

            yield event_done(
                answer=answer_text,
                side_products=side_products if side_products else None,
                context=context,
                messages=messages,
            )
            return

        # --- Branch B: tool call(s) — execute, then stream stage 2 ---
        LOGGER.debug("Stream: stage 1 produced %d tool_call(s)", len(stage1_msg.tool_calls))

        messages.append({
            "role": "assistant",
            "content": stage1_msg.content,
            "tool_calls": stage1_msg.tool_calls,
        })

        for tool_call in stage1_msg.tool_calls:
            try:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments or "{}")
            except Exception as exc:
                yield event_tool_end(
                    name=getattr(tool_call.function, "name", "unknown"),
                    ok=False,
                    error=f"Argument parse failed: {exc}",
                )
                continue

            yield event_tool_start(fn_name, fn_args)

            try:
                result = await self._execute_function(fn_name, fn_args, context)
                if isinstance(result, dict):
                    public = {k: v for k, v in result.items() if not k.startswith("_")}
                    private = {k: v for k, v in result.items() if k.startswith("_")}
                    if private:
                        side_products[fn_name] = private
                    tool_payload = public if public else result
                else:
                    tool_payload = result
                yield event_tool_end(fn_name, ok=True)
            except Exception as exc:
                LOGGER.exception("Tool %s failed", fn_name)
                tool_payload = {"error": str(exc)}
                yield event_tool_end(fn_name, ok=False, error=str(exc))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_payload),
            })

        # Stage 2: final answer, streamed token-by-token with audio per sentence.
        accumulated_answer: List[str] = []
        try:
            stage2_stream = await self.llm_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                stream=True,
            )
        except Exception as exc:
            yield event_error(
                f"Stage 2 LLM call failed: {exc}",
                error_type=type(exc).__name__,
            )
            return

        streamer = SentenceStreamer(min_words=6)
        async for chunk in stage2_stream:
            try:
                delta_text = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                continue
            if not delta_text:
                continue

            accumulated_answer.append(delta_text)
            yield event_token(delta_text)

            streamer.feed(delta_text)
            for sentence in list(streamer.drain_sentences()):
                async for audio_evt in self._synth_sentence_to_audio_event(sentence):
                    yield audio_evt

        # Flush any trailing partial sentence at end-of-stream.
        for sentence in list(streamer.flush()):
            async for audio_evt in self._synth_sentence_to_audio_event(sentence):
                yield audio_evt

        final_answer = "".join(accumulated_answer)
        messages.append({"role": "assistant", "content": final_answer})

        yield event_done(
            answer=final_answer,
            side_products=side_products if side_products else None,
            context=context,
            messages=messages,
        )

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    async def _synth_text_to_audio_events(self, text: str) -> AsyncIterator[str]:
        """Split `text` into sentences and emit an `audio` event per sentence.

        Used on the no-tool path where we already have the full answer.
        """
        streamer = SentenceStreamer(min_words=6)
        streamer.feed(text)
        # Force trailing partial through too — we already have the whole text.
        for sentence in list(streamer.drain_sentences()):
            async for evt in self._synth_sentence_to_audio_event(sentence):
                yield evt
        for sentence in list(streamer.flush()):
            async for evt in self._synth_sentence_to_audio_event(sentence):
                yield evt

    async def _synth_sentence_to_audio_event(self, sentence: str) -> AsyncIterator[str]:
        """Synthesize one sentence via Kokoro and yield one `audio` event.

        Failures are logged and swallowed — the user still sees the text,
        just without spoken audio for that sentence.
        """
        try:
            kokoro = await get_kokoro_service()
            wav_bytes = await kokoro.synthesize(sentence)
            wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
            yield event_audio(sentence, wav_b64)
        except Exception as exc:
            LOGGER.warning("Kokoro synth failed for sentence: %s", exc)
            # No event emitted — let the text stream continue without audio for this sentence.

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

            # ── Legacy ──────────────────────────────────────────────────────
            if function_name == "track_detail_for_guide":
                return await self.track_detail_for_guide()
            elif function_name == "compare_sessions":
                return await self.backend_service.compare_sessions(
                    arguments.get("session_ids"),
                    arguments.get("comparison_metrics", ["lap_times"])
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

    async def track_detail_for_guide(self, trackName: str = None) -> Dict[str, Any]:
        
        try:
            # Retrieve the active track corner analysis model
            response = await self.backend_service.getCompleteActiveModelData(modelType='track_corner_analysis')

            track_corner_payload = response.modelData
            if not track_corner_payload:
                raise Exception("No model data found in the response")

            # Run prediction using the assembled model payload
            prediction_result = await self.telemetryMLService.predict_optimal_cornering(trackName, track_corner_payload)

            guidance_instructions = {
                "task": "follow the json_structure to generate racing guidance sentences for car operation techniques in JSON format",
                "json_structure": {
                    "throttle_guidance": [
                        "sentence 1 about throttle technique",
                        "sentence 2 about throttle technique", 
                        "sentence 3 about throttle technique",
                        "sentence 4 about throttle technique"
                    ],
                    "brake_guidance": [
                        "sentence 1 about brake technique",
                        "sentence 2 about brake technique",
                        "sentence 3 about brake technique", 
                        "sentence 4 about brake technique"
                    ],
                    "steering_guidance": [
                        "sentence 1 about steering technique",
                        "sentence 2 about steering technique",
                        "sentence 3 about steering technique",
                        "sentence 4 about steering technique"
                    ]
                },
                "sentence_requirements": [
                    "Generate exactly 12 sentences total (4 for each operation type)",
                    "Each sentence should describe a specific racing technique",
                    "Focus on HOW to operate the car effectively",
                    "Use descriptive technique words from the provided lists",
                    "Make sentences actionable and practical",
                    "No distance measurements or track position references",
                    "Return ONLY the JSON object",
                    "No markdown formatting (```json```)",
                    "No explanatory text",
                    "No code blocks",
                    "Start directly with { and end with }",
                    "Exactly 4 sentences per guidance type"
                ],
                "example_output": {
                    "throttle_guidance": [
                        "gradually apply throttle when exiting corners to avoid wheel spin",
                        "moderately increase throttle through corner exit",
                        "quickly apply throttle when straightening out of corners",
                        "rapidly apply throttle in dry conditions with good grip"
                    ],
                    "brake_guidance": [
                        "gradually apply brake pressure when entering corners",
                        "moderately increase brake pressure during cornering",
                        "quickly apply brake pressure when approaching tight corners",
                        "rapidly apply brake pressure in wet conditions to avoid locking up"
                    ],
                    "steering_guidance": [
                        "gradually apply steering input when entering corners",
                        "moderately increase steering input during cornering",
                        "quickly apply steering input when navigating tight corners", 
                        "rapidly to counter steering"
                    ]
                }
            }
        except Exception as e:
            raise Exception(f"Failed to enable racing guidance: {str(e)}")
        
        # Return both data for OpenAI and side products for external use
        return {
            # Data for OpenAI's second query
            'status': 'success',
            'message': guidance_instructions,
            
            # Side products (prefixed with _) for external use
            '_guidance_enabled': True,
            '_prediction_result': prediction_result.get("predictions", {}) if isinstance(prediction_result, dict) else {},
            '_skip_openai_processing': True,  # Special flag to skip second OpenAI query if needed
        }
