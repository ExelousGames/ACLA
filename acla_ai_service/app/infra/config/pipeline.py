from dataclasses import dataclass
from typing import Optional

@dataclass
class PipelineConfig:
    """Configuration for cache cleanup operations with optional ID for grouping keys"""
    
    # Pipeline ID to group related cache keys together
    # When set, it will be appended to all cache key prefixes
    id: str = ""
    
    # Base prefixes for cache keys (without ID)
    _session_data_prefix: str = "racing_sessions_"
    _processed_session_data_prefix: str = "racing_sessions_processed_"
    _enriched_sessions_prefix: str = "racing_sessions_enriched_"
    _segments_prefix: str = "enriched_segments_"
    _top_laps_prefix: str = "top_laps_"
    _annotation_prefix: str = "manual_segment_annotations_FirstBatchOfSegmentAnnotation"
    _training_segments_prefix: str = "training_segments_"
    
    # Cleanup flags
    session_cleanup: bool = False # True will download new data and overwrite existing cache key
    processed_session_cleanup: bool = True
    segment_cleanup: bool = True
    top_laps_cleanup: bool = True
    
    @property
    def session_data_cache_key(self) -> str:
        """Get session data cache key with optional ID suffix"""
        return f"{self._session_data_prefix}{self.id}" if self.id else self._session_data_prefix
    
    @property
    def processed_session_data_cache_key(self) -> str:
        """Get processed session data cache key with optional ID suffix"""
        return f"{self._processed_session_data_prefix}{self.id}" if self.id else self._processed_session_data_prefix
    
    @property
    def enriched_sessions_cache_key(self) -> str:
        """Get enriched sessions cache key with optional ID suffix"""
        return f"{self._enriched_sessions_prefix}{self.id}" if self.id else self._enriched_sessions_prefix
    
    @property
    def segments_cache_key(self) -> str:
        """Get segments cache key with optional ID suffix"""
        return f"{self._segments_prefix}{self.id}" if self.id else self._segments_prefix
    
    @property
    def top_laps_cache_key(self) -> str:
        """Get top laps cache key with optional ID suffix"""
        return f"{self._top_laps_prefix}{self.id}" if self.id else self._top_laps_prefix
    
    @property
    def annotation_cache_key(self) -> str:
        """Get annotation cache key with optional ID suffix"""
        return f"{self._annotation_prefix}_{self.id}" if self.id else self._annotation_prefix
    
    @property
    def training_segments_cache_key(self) -> str:
        """Get training segments cache key with optional ID suffix"""
        return f"{self._training_segments_prefix}{self.id}" if self.id else self._training_segments_prefix
    ""
