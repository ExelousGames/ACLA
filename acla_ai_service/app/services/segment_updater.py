
import pandas as pd
import logging
from typing import List, Dict, Any
from app.models.segment_models import AnnotatedSegment
from app.services.zarr_telemetry_store import get_shared_zarr_store
import json
import zarr

logger = logging.getLogger(__name__)

class SegmentUpdater:
    """
    Service to update existing segments with new features from the source data.
    """

    def __init__(self):
        self.store = get_shared_zarr_store()

    def update_segments(self, source_session_key: str, segments: List[AnnotatedSegment]) -> List[AnnotatedSegment]:
        """
        Updates the telemetry_data of the provided segments by re-fetching data 
        from the source session store using the segment's start and end indices.

        Args:
            source_session_key: The cache key for the source sessions (e.g. 'racing_sessions_enriched_')
            segments: List of AnnotatedSegment objects to update.

        Returns:
            List of updated AnnotatedSegment objects.
        """
        if not segments:
            return []

        # Group segments by session ID (chunk_index)
        segments_by_session: Dict[str, List[AnnotatedSegment]] = {}
        for seg in segments:
            if seg.chunk_index is None:
                logger.warning(f"Segment missing chunk_index, skipping update: {seg}")
                continue
            
            if seg.chunk_index not in segments_by_session:
                segments_by_session[seg.chunk_index] = []
            segments_by_session[seg.chunk_index].append(seg)

        updated_segments = []
        
        # Process each session
        for session_id, session_segments in segments_by_session.items():
            try:
                df = self._load_session_data(source_session_key, session_id)
                
                if df.empty:
                    logger.warning(f"Could not load data for session {session_id}, skipping segments.")
                    updated_segments.extend(session_segments) # Keep original if load fails
                    continue

                for seg in session_segments:
                    if seg.start_index is not None and seg.end_index is not None:
                        # Ensure indices are within bounds
                        start = max(0, int(seg.start_index))
                        end = min(len(df), int(seg.end_index))
                        
                        if start < end:
                            # Slice and convert to dict
                            segment_slice = df.iloc[start:end]
                            seg.telemetry_data = segment_slice.to_dict(orient="records")
                            seg.segment_length = len(seg.telemetry_data)
                        else:
                            logger.warning(f"Invalid indices for segment in session {session_id}: {start}-{end}")
                    
                    updated_segments.append(seg)
                    
            except Exception as e:
                logger.error(f"Error updating segments for session {session_id}: {e}")
                updated_segments.extend(session_segments) # Keep original on error

        return updated_segments

    def _load_session_data(self, cache_key: str, session_id: str) -> pd.DataFrame:
        """Helper to load session data, similar to the UI app logic."""
        group_path = self.store._group_path(cache_key)
        
        if not group_path.exists():
            return pd.DataFrame()
            
        try:
            group = zarr.open_group(str(group_path), mode="r")
            if session_id not in group:
                return pd.DataFrame()
                
            raw_bytes = bytes(group[session_id][:])
            chunk = json.loads(raw_bytes.decode("utf-8"))
            
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
            logger.error(f"Error loading session {session_id}: {e}")
            return pd.DataFrame()
