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
import zarr
import sys
import os
import time
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
    from app.storage.zarr import get_shared_zarr_store
    from app.config.pipeline_config import PipelineConfig
    import app.domain.labels
    import app.domain.segment
    # Force reload to pick up model changes (e.g. new fields)
    importlib.reload(app.domain.labels)
    importlib.reload(app.domain.segment)
    from app.domain.labels import LABEL_MAPPING, LABEL_NAME_TO_ID, LABEL_CATEGORIES, MAIN_LABEL_GUIDELINES, LABEL_IMAGE_MAP
    from app.domain.segment import AnnotatedSegment, SegmentFeatureCatalog
    from app.services.segment_updater import SegmentUpdater

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
    return get_shared_zarr_store()

def get_available_sessions(cache_key: str) -> List[str]:
    """Get list of available session IDs from the store."""
    store = get_store()
    group_path = store._group_path(cache_key)
    if not group_path.exists():
        return []
    try:
        group = zarr.open_group(str(group_path), mode="r")
        # Filter out metadata chunk
        sessions = [k for k in group.array_keys() if k != "chunk_000000"]
        return sorted(sessions)
    except Exception:
        return []

@st.cache_data(max_entries=1, show_spinner=False)
def load_session_data(cache_key: str, session_id: str) -> pd.DataFrame:
    """Load a specific session of data from Zarr."""
    store = get_store()
    # Accessing internal method _group_path to avoid iterating all chunks
    group_path = store._group_path(cache_key)
    
    if not group_path.exists():
        return pd.DataFrame()
        
    try:
        # Open in read-only mode
        group = zarr.open_group(str(group_path), mode="r")
        chunk_name = session_id
        
        if chunk_name not in group:
            return pd.DataFrame()
            
        raw_bytes = bytes(group[chunk_name][:])
        chunk = json.loads(raw_bytes.decode("utf-8"))
        
        # Robust DataFrame creation
        if isinstance(chunk, list):
            df = pd.DataFrame(chunk)
        elif isinstance(chunk, dict):
            if "data" in chunk and isinstance(chunk["data"], list):
                df = pd.DataFrame(chunk["data"])
            else:
                try:
                    df = pd.DataFrame(chunk)
                except ValueError:
                    df = pd.DataFrame([chunk])
        else:
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
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

def save_annotations(session_id: str, annotations: List[AnnotatedSegment], annotation_key: str, silent: bool = False):
    """Save annotations to Zarr store."""
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


