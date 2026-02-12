"""
Shared utilities for Segment Annotation App.
"""
import torch
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
    from app.services.zarr_telemetry_store import get_shared_zarr_store
    from app.config.pipeline_config import PipelineConfig
    import app.models.segment_models
    # Force reload to pick up model changes (e.g. new fields)
    importlib.reload(app.models.segment_models)
    from app.models.segment_models import AnnotatedSegment, LABEL_MAPPING, LABEL_NAME_TO_ID, LABEL_CATEGORIES, SegmentFeatureCatalog
    from app.services.segment_updater import SegmentUpdater
    from app.services.local_vlm_service import LocalVLMService, LocalVLMConfig, VLMProcessManager
except ImportError:
    # Fallback or error handling if needed, though mostly we expect this to work if running from root or with pythonpath setup
    pass

LABEL_DESCRIPTIONS = {
    "Overtaking": "On the 2D Trajectory plot, the driver deviates from the expert line. Physics_brake occurs later or deeper than expert_optimal_brake. Physics_gas is applied more aggressively than expert_optimal_throttle. Ends when the driver realigns with the racing line.",
    "Missing data": "Plots show gaps or empty sections. Physics_gas and Physics_brake flatline or drop unexpectedly. speed_difference is discontinuous. Ends when valid telemetry signal returns.",
    "Expert Adherence": "Entire 'distance_to_expert_line' line must stays below the 'Distance Limit' dashed line, and entire 'speed_difference' line must stays between the 'Speed Min' and 'Speed Max' dashed lines, then it indicates the driver is closely following the expert.",
    "Recovery & Merge": "'speed_difference' line starts above the 'Speed Max' dashed lines and DECREASE towards 'Speed Max' dashed lines, or 'distance_to_expert_line' line starts above the 'Distance Limit' dashed line and DECREASES towards 'Distance Limit' dashed line. Represents the driver merging BACK to the expert trajectory.",
    "Superior Expert": "speed_difference is negative (faster than expert). Physics_brake starts later (deeper) than expert_optimal_brake. Physics_gas is applied earlier or smoother than expert_optimal_throttle. Ends when the performance advantage is lost or neutralizes.",
    "Unexpected driving behavior": "Erratic patterns. Physics_gas oscillates or Physics_brake is applied unexpectedly. speed_difference is unstable. Ends when stable driving resumes.",
    "Mistake": "Values rapidly increase and cross ABOVE the 'Speed Max' dashed line (if speed) or 'Distance Limit' dashed line (if distance). Ends when the upward trend stops."
} 

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
        ],
        reference_lines=[
            {"name": "Zero", "value": 0.0, "color": "gray"},
            {"name": "Speed Max", "value": 10.0, "color": "red"},
            {"name": "Speed Min", "value": -10.0, "color": "green"},
        ]
    ),
    GraphConfig(
        description="Comparison of throttle input (0-1) between Driver (Physics_gas) and Expert (expert_optimal_throttle).",
        features=["expert_optimal_throttle", "Physics_gas"],
        reference_lines=[
             {"name": "Full Throttle", "value": 1.0, "color": "gray"},
             {"name": "Idle", "value": 0.0, "color": "gray"}
        ]
    ),
    GraphConfig(
        description="Comparison of brake input (0-1) between Driver (Physics_brake) and Expert (expert_optimal_brake).",
        features=["expert_optimal_brake", "Physics_brake"],
        reference_lines=[
             {"name": "Max Brake", "value": 1.0, "color": "gray"},
             {"name": "No Brake", "value": 0.0, "color": "gray"}
        ]
    )
]

@st.cache_resource
def get_vlm_service():
    """Load the VLM service for inference."""
    try:
        # Explicitly set device to cuda or cpu to avoid accelerate's "auto" which can be unstable on some setups
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use 4-bit quantization for efficiency
        config = LocalVLMConfig(
            load_in_4bit=True, 
            load_in_8bit=False,
            device_map=device_map
        )
        service = VLMProcessManager(config)
        return service
    except Exception as e:
        print(f"Failed to load VLM service: {e}")
        return None

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

def save_annotations(session_id: str, annotations: List[AnnotatedSegment], annotation_key: str):
    """Save annotations to Zarr store."""
    store = get_store()
    
    # Save to specific chunk index
    data_to_save = [a.to_dict() for a in annotations]
    
    if not data_to_save:
        # If empty, delete the chunk so it doesn't show up as an annotated session
        if hasattr(store, "delete_chunk"):
            store.delete_chunk(annotation_key, session_id)
            st.success(f"All annotations deleted for session {session_id}.")
        else:
            # Fallback
            store.save_chunk(annotation_key, session_id, data_to_save)
            st.success(f"Saved 0 annotations to {annotation_key} (session {session_id})")
    else:
        store.save_chunk(annotation_key, session_id, data_to_save)
        st.success(f"Saved {len(annotations)} annotations to {annotation_key} (session {session_id})")

def extract_json_from_response(response_text: str) -> Optional[dict]:
    """
    Robustly extract JSON from VLM response text which might contain markdown or hallucinated text.
    Handles nested braces by counting balance.
    """
    clean_text = response_text.strip()
    
    # 1. Try to find markdown block with json tag
    match = re.search(r'```json\s*(.*?)\s*```', clean_text, re.DOTALL | re.IGNORECASE)
    if match:
        clean_text = match.group(1)
    else:
        # 2. Try finding just any code block
        match = re.search(r'```\s*(.*?)\s*```', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
            
    clean_text = clean_text.strip()
    
    # 3. Attempt direct parse
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # 4. Sliding window with brace balancing to find the FIRST valid JSON object with "found" key
    # or just the outermost object.
    
    start_indices = [m.start() for m in re.finditer(r'{', clean_text)]
    
    for start in start_indices:
        balance = 0
        for i in range(start, len(clean_text)):
            if clean_text[i] == '{':
                balance += 1
            elif clean_text[i] == '}':
                balance -= 1
                if balance == 0:
                    # Found a balanced block
                    candidate = clean_text[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        # We return the first valid JSON object we find that seems to encompass structure
                        # Or specifically check for our expected keys
                        if "found" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    # If this block was balanced but invalid JSON (e.g. unquoted keys), 
                    # we might continue inner loop? No, balance=0 means we reached end of this block representation.
                    # We continue the outer loop to try next '{'
                    break
                    
    # 5. Heuristic fallback for malformed JSON
    return _heuristic_json_extraction(clean_text)

def _extract_string_field(text: str, field_name: str) -> Optional[str]:
    """
    Helper to extract a string field value, handling unclosed quotes or missing comma separators.
    """
    # Regex to find the start of the field: "key":\s*"
    pattern = f'"{field_name}"\s*:\s*"'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start_idx = match.end()
    remainder = text[start_idx:]
    
    # 1. Try to find a valid closing quote first (standard case)
    in_escape = False
    valid_close_found = False
    end_idx = -1
    
    for i, char in enumerate(remainder):
        if in_escape:
            in_escape = False
        elif char == '\\':
            in_escape = True
        elif char == '"':
            # Check if this could be a closing quote
            # It's valid if followed by comma, or }, or whitespace then comma/}
            suffix = remainder[i+1:].lstrip()
            if not suffix or suffix.startswith(',') or suffix.startswith('}'):
                end_idx = i
                valid_close_found = True
                break
    
    if valid_close_found:
        return remainder[:end_idx].replace(r'\"', '"')

    # 2. Fallback: Not properly closed
    # Stop at the next likely JSON key or the end of the block
    
    known_keys = ["found", "label", "start_percentage", "end_percentage", "reasoning"]
    
    # Create pattern for next keys
    other_keys = [k for k in known_keys if k != field_name]
    if not other_keys:
        next_key_match = None
    else:
        next_key_pattern = r'"(?:' + '|'.join(other_keys) + r')"\s*:'
        next_key_match = re.search(next_key_pattern, remainder)

    limit_idx = len(remainder)
    
    if next_key_match:
        limit_idx = next_key_match.start()
    else:
        # If no next key found, maybe we hit the end of the object '}'
        close_brace_idx = remainder.rfind('}')
        if close_brace_idx != -1:
             limit_idx = close_brace_idx
             
    raw_content = remainder[:limit_idx]
    # Clean up trailing content (commas, newlines from the "cut" point)
    return raw_content.rstrip(', \t\n\r').replace(r'\"', '"')

def _heuristic_json_extraction(text: str) -> Optional[dict]:
    """
    Attempt to extract JSON fields using regex when standard parsing fails.
    Handles common LLM errors like missing quotes, extra strings, etc.
    """
    data = {}
    
    # Extract 'found'
    found_match = re.search(r'"found"\s*:\s*(true|false)', text, re.IGNORECASE)
    if found_match:
        data["found"] = found_match.group(1).lower() == "true"
        
    # Extract 'label'
    label_val = _extract_string_field(text, "label")
    if label_val is not None:
        data["label"] = label_val
        
    # Extract percentages
    for field in ["start_percentage", "end_percentage"]:
        # Try unquoted number first
        match = re.search(f'"{field}"\s*:\s*([0-9.]+)', text)
        if match:
            try:
                data[field] = float(match.group(1))
                continue
            except ValueError:
                pass
        
        # Try quoted (potentially unclosed) via string extraction
        val_str = _extract_string_field(text, field)
        if val_str:
             try:
                data[field] = float(val_str)
             except ValueError:
                pass

    # Extract 'reasoning'
    reasoning_val = _extract_string_field(text, "reasoning")
    if reasoning_val is not None:
        data["reasoning"] = reasoning_val

    # If we found at least one logical field, return it
    if data:
        return data
    return None
