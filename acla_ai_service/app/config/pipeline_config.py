from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """Configuration for cache cleanup operations"""
    session_data_cache_key: str = f"racing_sessions_"
    session_cleanup: bool = False
    processed_session_data_cache_key: str = f"racing_sessions_processed_"
    processed_session_cleanup: bool = True
    enriched_sessions_cache_key: str = f"racing_sessions_enriched_"
    segments_cache_key: str = f"enriched_segments_"
    segment_cleanup: bool = True
    top_laps_cache_key: str = f"top_laps_"
    annotation_cache_key: str = "manual_segment_annotations_FirstBatchOfSegmentAnnotation"
    ""
