"""
Typed result shapes + JSON parsing helpers.

OUTSIDE the agent box — the box returns ``AgentResponse.raw_response`` as
free-form text. These helpers turn that text into typed results the UI
can render. Flow modules (``annotation.flows.*``) compose them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed results — what UIs / API responses consume
# ---------------------------------------------------------------------------


@dataclass
class AnnotationResult:
    """Result for the detailed (sub-segment discovery) flow."""

    final_labels: List[str]
    final_reasoning: str
    accepted: bool
    iterations: int
    messages: List[dict]
    graph_images: List[bytes] = field(default_factory=list)  # PNG bytes
    sub_start: Optional[int] = None
    sub_end: Optional[int] = None
    # Per-label proposals. Each entry:
    #   {label_id, start_index, end_index, reasoning}
    # The UI materialises one sub-segment per AI-discovered range.
    label_annotations: List[dict] = field(default_factory=list)


@dataclass
class LapAnnotationResult:
    """Result for the lap-section excerpter flow.

    ``label_ids`` is the flat list of parent labels the agent picked
    (circuit + circuit_section + segment_type + optional main). The UI
    persists this as a single annotated segment over
    ``[start_index, end_index]`` — potentially revised from the rough
    splitter boundary when ``revised`` is True.
    """

    section_id: str
    start_index: int
    end_index: int
    label_ids: List[str]
    reasoning: str
    revised: bool
    submitted: bool
    rough_start: int
    rough_end: int
    rejected_proposals: List[Dict[str, Any]] = field(default_factory=list)
    rendered_images: List[bytes] = field(default_factory=list)
    transcript: str = ""
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# JSON extraction — agent raw_response often arrives wrapped in code fences
# ---------------------------------------------------------------------------


def parse_json_response(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from a synthesizer response."""

    def _try_loads(s: str) -> Optional[dict]:
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            fixed = re.sub(
                r'"((?:[^"\\]|\\.)*)"',
                lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
                s,
            )
            return json.loads(fixed)
        except (json.JSONDecodeError, re.error):
            pass
        return None

    try:
        json_str = raw
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        result = _try_loads(json_str)
        if result is not None:
            return result
    except (IndexError, KeyError):
        pass

    brace_match = re.search(r'\{[\s\S]*\}', raw)
    if brace_match:
        result = _try_loads(brace_match.group())
        if result is not None:
            return result

    return None


