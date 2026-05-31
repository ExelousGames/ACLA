"""
Shared utilities for Segment Annotation App.
"""
import torch
torch.classes.__path__ = []
import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import re
import sys
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import importlib

# Ensure app module is on path
def _ensure_app_module_on_path() -> None:
    candidate = Path(__file__).resolve().parent
    for _ in range(4): # adjusting for depth in ui/segment_tabs/
        if (candidate / "app").exists():
            path_str = candidate.as_posix()
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
        candidate = candidate.parent

_ensure_app_module_on_path()

try:
    from app.storage import get_shared_telemetry_store
    from app.pipelines.training.config import TrainingPipelineConfig
    import app.domain.labels
    import app.domain.segment
    # Force reload to pick up model changes (e.g. new fields)
    importlib.reload(app.domain.labels)
    importlib.reload(app.domain.segment)
    from app.domain.labels import LABEL_MAPPING, LABEL_NAME_TO_ID, LABEL_CATEGORIES, MAIN_LABEL_GUIDELINES, LABEL_IMAGE_MAP
    from app.domain.segment import AnnotatedSegment, SegmentFeatureCatalog
    from app.pipelines.inference.segment_updater import SegmentUpdater

except ImportError:
    # Fallback or error handling if needed, though mostly we expect this to work if running from root or with pythonpath setup
    pass



@dataclass
class GraphConfig:
    """Configuration for a single telemetry graph analysis."""
    description: str
    features: List[str] = field(default_factory=list)
    reference_lines: List[Dict[str, Any]] = field(default_factory=list)

GRAPH_CONFIGS = [
    GraphConfig(
        description="Difference in speed between driver and expert (Expert - Driver). Positive values indicate the driver is slower than the expert, negative values indicate faster.",
        features=[
            "speed_difference"
        ]
    ),
    GraphConfig(
        description="Comparison of throttle input (0-1) between Driver (Physics_gas) and Expert (expert_optimal_throttle).",
        features=["expert_optimal_throttle", "Physics_gas"]
    ),
    GraphConfig(
        description="Comparison of brake input (0-1) between Driver (Physics_brake) and Expert (expert_optimal_brake).",
        features=["expert_optimal_brake", "Physics_brake"]
    )
]



def get_display_labels(labels):
    """Convert label IDs or strings to display strings."""
    if not isinstance(labels, list):
        labels = [labels]
    
    display_labels = []
    for l in labels:
        key = str(l)
        if key in LABEL_MAPPING:
            display_labels.append(LABEL_MAPPING[key])
        else:
            display_labels.append(key)
    return display_labels

def _run_async(func, *args, **kwargs):
    """Execute an async function from a synchronous context."""
    try:
        return asyncio.run(func(*args, **kwargs))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

@st.cache_resource
def get_store():
    return get_shared_telemetry_store()


def register_output_dir(cache_key: str, directory: Optional[str]) -> None:
    """Tell the shared store that ``cache_key`` lives in ``directory``.

    Custom output directories — picked via the first-time annotation
    popup — get registered here so every consumer that goes through
    ``get_shared_telemetry_store()`` (annotation pages, training-unit
    store, etc.) reads/writes the cache_key from the right path on
    disk. Pass ``directory=None`` to clear the override.
    """
    if not cache_key:
        return
    get_store().register_directory(cache_key, directory)


def get_available_sessions(cache_key: str) -> List[str]:
    """Get list of available session IDs from the store."""
    store = get_store()
    try:
        return store.list_chunk_ids(cache_key)
    except Exception:
        return []

def _is_segment_list(chunk) -> bool:
    """A chunk read from an annotation dataset is a list of segment dicts;
    a chunk read from a raw telemetry dataset is a list of row dicts.
    Segment dicts always carry ``start_index``/``end_index`` plus either
    ``labels`` or a nested ``telemetry_data`` list — telemetry rows don't.
    """
    if not isinstance(chunk, list) or not chunk:
        return False
    head = chunk[0]
    if not isinstance(head, dict):
        return False
    return (
        "start_index" in head and "end_index" in head
        and ("labels" in head or "telemetry_data" in head)
    )


def _segments_to_telemetry_df(segments: list) -> pd.DataFrame:
    """Concat each segment's ``telemetry_data`` into one flat telemetry df.
    Segments are ordered by ``start_index`` so the resulting iloc range
    follows the original sample order.
    """
    ordered = sorted(
        segments,
        key=lambda s: (
            s.get("start_index") if isinstance(s.get("start_index"), int) else 0
        ),
    )
    frames = []
    for seg in ordered:
        rows = seg.get("telemetry_data") or []
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(max_entries=1, show_spinner=False)
def load_session_data(cache_key: str, session_id: str) -> pd.DataFrame:
    """Load a specific session of data from the telemetry store.

    Handles both shapes a chunk can take:
      * raw telemetry — ``list[row_dict]``;
      * annotation output — ``list[segment_dict]`` with per-segment
        ``telemetry_data``. The latter is unrolled into a flat telemetry
        frame so views that expect telemetry rows (Detailed Annotation,
        rule-based, etc.) work uniformly regardless of whether they read
        from a telemetry dataset or from another annotation's output.
    """
    store = get_store()
    try:
        chunk = store.get_chunk(cache_key, session_id)
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
        return pd.DataFrame()

    if chunk is None:
        return pd.DataFrame()

    if isinstance(chunk, list):
        if _is_segment_list(chunk):
            return _segments_to_telemetry_df(chunk)
        return pd.DataFrame(chunk)
    if isinstance(chunk, dict):
        if "data" in chunk and isinstance(chunk["data"], list):
            data = chunk["data"]
            if _is_segment_list(data):
                return _segments_to_telemetry_df(data)
            return pd.DataFrame(data)
        try:
            return pd.DataFrame(chunk)
        except ValueError:
            return pd.DataFrame([chunk])
    return pd.DataFrame()

def load_annotations(session_id: str, annotation_key: str) -> List[AnnotatedSegment]:
    """Load annotations for a specific session."""
    store = get_store()
    chunk_data = store.get_chunk(annotation_key, session_id)
    
    raw_data = []
    if chunk_data:
        if isinstance(chunk_data, list):
            raw_data = chunk_data
        elif isinstance(chunk_data, dict) and "data" in chunk_data:
             raw_data = chunk_data["data"]
             
    segments = [AnnotatedSegment.from_dict(d) for d in raw_data]
    # Ensure chunk_index is set for loaded segments
    for s in segments:
        if s.chunk_index is None:
            s.chunk_index = session_id
    return segments

def build_segment(
    df,
    *,
    start: int,
    end: int,
    label_ids: List[str],
    notes: Optional[str] = None,
    parent_id: Optional[str] = None,
    chunk_index: Optional[Any] = None,
    id: Optional[str] = None,
) -> AnnotatedSegment:
    """Single construction point for ``AnnotatedSegment``.

    Slices ``df.iloc[start:end]`` into ``telemetry_data`` so every save
    path embeds the rows the segment refers to — downstream nodes that
    consume an annotation's output (children flow, training datasets)
    can recover the underlying telemetry without re-resolving the
    parent's input.

    ``end`` is exclusive (matches ``df.iloc`` semantics). The classifier
    paths use inclusive-end indexing and slice manually instead of going
    through this helper.
    """
    s = max(0, int(start))
    e = min(len(df), int(end)) if df is not None else int(end)
    rows: List[Dict[str, Any]] = []
    if df is not None and not getattr(df, "empty", False) and s < e:
        rows = df.iloc[s:e].to_dict(orient="records")
    return AnnotatedSegment(
        id=id or str(uuid.uuid4()),
        labels=list(label_ids),
        segment_length=len(rows) if rows else max(0, int(end) - int(start)),
        start_index=int(start),
        end_index=int(end),
        notes=notes,
        parent_id=parent_id,
        chunk_index=chunk_index,
        telemetry_data=rows,
    )


def save_annotations(session_id: str, annotations: List[AnnotatedSegment], annotation_key: str, silent: bool = False):
    """Save annotations to the telemetry store."""
    store = get_store()
    
    # Save to specific chunk index
    data_to_save = [a.to_dict() for a in annotations]
    
    if not data_to_save:
        # If empty, delete the chunk so it doesn't show up as an annotated session
        if hasattr(store, "delete_chunk"):
            store.delete_chunk(annotation_key, session_id)
            if not silent:
                st.success(f"All annotations deleted for session {session_id}.")
        else:
            # Fallback
            store.save_chunk(annotation_key, session_id, data_to_save)
            if not silent:
                st.success(f"Saved 0 annotations to {annotation_key} (session {session_id})")
    else:
        store.save_chunk(annotation_key, session_id, data_to_save)
        if not silent:
            st.success(f"Saved {len(annotations)} annotations to {annotation_key} (session {session_id})")


