"""
Racing session analysis endpoints for AI model training and analysis
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
from httpx import request
from pydantic import BaseModel
import asyncio
import pandas as pd
from app.pipelines.inference.preprocessing import (
    preprocess_inference_telemetry,
)
from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService
from app.racing_engineer.top_lap_reference_guidance import (
    generate_top_lap_reference_guidance,
)
from app.ml.model_hub import (
    get_opportunity_forecaster,
    get_segment_classifier,
    get_tire_grip_analysis,
    get_top_lap_reference_model,
)
from app.services.runtime_segment_splitter import (
    RuntimeSegmentSplitError,
    split_runtime_segments,
)
from app.top_laps.runtime import TopLapReferenceModelError
from app.services.user_session_analysis import analyze_user_sessions
from app.shared.expert_features import ExpertFeatureCatalog
from app.shared.label_hierarchy import build_track_area_segments
from app.shared.labels import (
    LABEL_CATEGORIES,
    LABEL_IMAGE_MAP,
    LABEL_MAPPING,
    LABEL_NAME_TO_ID,
)


router = APIRouter(prefix="/racing-session", tags=["racing-session"])

# Pydantic models for request/response validation
class TrainingRequest(BaseModel):
    session_id: str
    telemetry_data: List[Dict[str, Any]]
    target_variable: str = "lap_time"
    model_type: str = "lap_time_prediction"
    preferred_algorithm: Optional[str] = None
    user_id: Optional[str] = None
    existing_model_data: Optional[str] = None

class MultipleTrainingRequest(BaseModel):
    session_id: str
    telemetry_data: List[Dict[str, Any]]

    '''#example of models_config
        {
            "config_id": "rf_model",
            "target_variable": "lap_time", 
            "model_type": "lap_time_prediction",
            "preferred_algorithm": "random_forest",
            "existing_model_data": data
        }
    '''
    models_config: List[Dict[str, Any]]  # List of model configurations to train
    user_id: Optional[str] = None
    parallel_training: bool = True  # Whether to train models in parallel or sequentially

class PredictionRequest(BaseModel):
    telemetry_data: Dict[str, Any]
    model_data: str  # Base64 encoded model data from database
    model_type: Optional[str] = "lap_time_prediction"
    use_river: bool = True  # Whether to use River ML or legacy scikit-learn
    user_id: Optional[str] = None

class TopLapReferenceGuidanceRequest(BaseModel):
    current_telemetry: Dict[str, Any]
    human_request: Optional[str] = None
    delay_seconds: Optional[float] = 0.0
    track_name: str
    car_name: str   
    user_id: Optional[str] = None

class OpportunityForecastRequest(BaseModel):
    telemetry_data: List[Dict[str, Any]]
    horizon_seconds: Optional[float] = 10.0
    top_k: Optional[int] = 3

class TrackCornerKnowledgeRequest(BaseModel):
    track_name: str
    corner_name: str
    normalized_position: Optional[float] = None
    trigger_position: Optional[float] = None
    current_telemetry: Optional[Dict[str, Any]] = None

TRACK_CORNER_UNSUPPORTED_MESSAGE = "Track guide doesn't support the current track right now."

class AnalyzeUserSessionsRequest(BaseModel):
    user_id: str
    session_limit: Optional[int] = 10

class SegmentClassificationRequest(BaseModel):
    session_id: Optional[str] = None
    telemetry_data: List[Dict[str, Any]]
    track_name: Optional[str] = None
    car_name: Optional[str] = None

class LiveBaselineAnalysisRequest(BaseModel):
    track: Optional[str] = None
    car: Optional[str] = None
    baseline_lap: Optional[int] = None
    records: List[Dict[str, Any]]
    
# Initialize telemetry service
telemetryMLService = Full_dataset_TelemetryMLService()


def _classify_telemetry_segments(
    telemetry_data: List[Dict[str, Any]],
    track_name: Optional[str],
    include_empty_track_sections: bool = False,
    splitter_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    dataframe = pd.DataFrame(telemetry_data)
    splitter_result = splitter_result or split_runtime_segments(dataframe, track_name)
    predicted_segments = get_segment_classifier().classify_ranges(
        dataframe,
        splitter_result["segments"],
    )
    raw_segments = []

    for segment in predicted_segments:
        labels = list(dict.fromkeys([
            segment.label,
            *(child.label for child in segment.subsegments),
        ]))
        raw_segments.append({
            "id": segment.id,
            "labels": labels,
            "start_index": segment.start_index,
            "end_index": segment.end_index,
        })

    if splitter_result.get("opponent_session") and not raw_segments:
        return []

    return build_track_area_segments(
        raw_segments,
        telemetry_data,
        splitter_result["circuit_id"],
        include_empty_sections=include_empty_track_sections,
    )


def _project_expert_reference_data(
    enriched_rows: List[Dict[str, Any]],
    raw_indices: List[int],
) -> List[Dict[str, Any]]:
    expert_features = ExpertFeatureCatalog.ExpertFeatures
    reference_features = (
        expert_features.EXPERT_TIME_DIFFERENCE,
        expert_features.EXPERT_OPTIMAL_PLAYER_POS_X,
        expert_features.EXPERT_OPTIMAL_PLAYER_POS_Y,
        expert_features.EXPERT_OPTIMAL_PLAYER_POS_Z,
        expert_features.EXPERT_OPTIMAL_THROTTLE,
        expert_features.EXPERT_OPTIMAL_BRAKE,
        expert_features.EXPERT_OPTIMAL_GEAR,
    )

    return [
        {
            "raw_index": raw_index,
            **{
                feature.value: row[feature.value]
                for feature in reference_features
            },
            "Graphics_normalized_car_position": row[
                "Graphics_normalized_car_position"
            ],
        }
        for row, raw_index in zip(enriched_rows, raw_indices)
    ]


def _build_time_gap(
    expert_rows: List[Dict[str, Any]],
    start_index: Any,
    end_index: Any,
) -> Optional[Dict[str, float]]:
    if not expert_rows or start_index is None or end_index is None:
        return None

    try:
        start = max(0, int(start_index))
        end_exclusive = min(len(expert_rows), int(end_index))
    except (TypeError, ValueError):
        return None

    if start >= len(expert_rows) or end_exclusive <= start:
        return None

    time_difference = (
        ExpertFeatureCatalog.ExpertFeatures.EXPERT_TIME_DIFFERENCE.value
    )
    start_diff = expert_rows[start].get(time_difference)
    end_diff = expert_rows[end_exclusive - 1].get(time_difference)
    try:
        start_ms = float(start_diff)
        end_ms = float(end_diff)
    except (TypeError, ValueError):
        return None

    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "delta_ms": end_ms - start_ms,
    }


def _annotate_segments_with_time_gaps(
    segments: List[Dict[str, Any]],
    expert_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    annotated_segments: List[Dict[str, Any]] = []

    for segment in segments:
        annotated_segment = dict(segment)
        time_gap = _build_time_gap(
            expert_rows,
            segment.get("start_index"),
            segment.get("end_index"),
        )
        if time_gap is not None:
            annotated_segment["time_gap"] = time_gap

        annotated_segments.append(annotated_segment)

    return annotated_segments


def _translate_segment_ranges_to_raw_indices(
    segments: List[Dict[str, Any]],
    raw_indices: List[int],
) -> List[Dict[str, Any]]:
    translated_segments: List[Dict[str, Any]] = []

    for segment in segments:
        translated_segment = dict(segment)
        try:
            start = int(segment["start_index"])
            end_exclusive = int(segment["end_index"])
        except (KeyError, TypeError, ValueError):
            translated_segments.append(translated_segment)
            continue

        if (
            start < 0
            or end_exclusive <= start
            or start >= len(raw_indices)
        ):
            translated_segments.append(translated_segment)
            continue

        end_exclusive = min(end_exclusive, len(raw_indices))
        translated_segment["start_index"] = raw_indices[start]
        translated_segment["end_index"] = raw_indices[end_exclusive - 1] + 1
        translated_segments.append(translated_segment)

    return translated_segments


@router.get("/labels")
async def get_labels() -> Dict[str, Any]:
    return {
        "label_mapping": LABEL_MAPPING,
        "label_name_to_id": LABEL_NAME_TO_ID,
        "label_image_map": LABEL_IMAGE_MAP,
        "label_categories": LABEL_CATEGORIES,
    }


@router.post("/top-lap-reference-guidance")
async def get_top_lap_reference_guidance(
    request: TopLapReferenceGuidanceRequest,
) -> Dict[str, Any]:
    """
    Get driving guidance using the top-lap reference model.
    """
    try:
        try:
            result = await generate_top_lap_reference_guidance(
                telemetryMLService,
                telemetry_dict=request.current_telemetry,
                user_request=request.human_request,
                track_name=request.track_name,
                car_name=request.car_name,
            )

        except Exception as e:
            print(
                f"[ERROR] Exception in top-lap reference guidance service: "
                f"\n {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error in top-lap reference guidance service: {str(e)}",
            )
        
        return {
            "message": "Top-lap reference guidance generated successfully",
            "guidance_result": result,
            "timestamp": result.get("timestamp"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Top-lap reference guidance failed: {str(e)}",
        )


@router.post("/opportunity-forecast")
async def get_opportunity_forecast(request: OpportunityForecastRequest) -> Dict[str, Any]:
    try:
        if not request.telemetry_data:
            raise HTTPException(status_code=400, detail="telemetry_data is required")

        result = get_opportunity_forecaster().forecast(
            request.telemetry_data,
            horizon_seconds=request.horizon_seconds or 10.0,
            top_k=request.top_k or 3,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Opportunity forecast failed: {str(e)}")


@router.post("/track-corner-knowledge")
async def get_track_corner_knowledge(request: TrackCornerKnowledgeRequest) -> Dict[str, Any]:
    try:
        track_name = (request.track_name or "").strip()
        corner_name = (request.corner_name or "").strip()
        if not track_name:
            raise HTTPException(status_code=400, detail="track_name is required")
        if not corner_name:
            raise HTTPException(status_code=400, detail="corner_name is required")

        from app.external_knowledge_base import track_guide as track_guide_lookup

        track_key = track_name.lower().replace(" ", "_")
        result = track_guide_lookup(track_key, corner=corner_name)
        if result is None:
            return {
                "status": "unsupported",
                "message": TRACK_CORNER_UNSUPPORTED_MESSAGE,
                "reason": "track_not_in_corpus",
                "track_knowledge": None,
                "normalized_position": request.normalized_position,
                "trigger_position": request.trigger_position,
            }
        if result.get("error"):
            return {
                "status": "unsupported",
                "message": TRACK_CORNER_UNSUPPORTED_MESSAGE,
                "reason": "corner_not_in_corpus",
                "track_knowledge": result,
                "normalized_position": request.normalized_position,
                "trigger_position": request.trigger_position,
            }

        return {
            "status": "success",
            "track_knowledge": result,
            "normalized_position": request.normalized_position,
            "trigger_position": request.trigger_position,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Track corner knowledge failed: {str(e)}")


@router.post("/segment-classification")
async def classify_session_segments(request: SegmentClassificationRequest) -> Dict[str, Any]:
    try:
        if not request.telemetry_data:
            raise HTTPException(status_code=400, detail="telemetry_data is required")

        preprocessed = preprocess_inference_telemetry(request.telemetry_data)
        splitter_result = split_runtime_segments(
            pd.DataFrame(preprocessed.records),
            request.track_name,
        )
        enriched_rows = get_top_lap_reference_model().enrich(
            preprocessed.records,
            track=request.track_name or splitter_result["circuit_id"],
            car=request.car_name,
        )
        enriched_rows = await get_tire_grip_analysis().enrich(enriched_rows)
        expert_reference_data = _project_expert_reference_data(
            enriched_rows,
            preprocessed.raw_indices,
        )
        segments = _classify_telemetry_segments(
            enriched_rows,
            splitter_result["circuit_id"],
            splitter_result=splitter_result,
        )
        segments = _translate_segment_ranges_to_raw_indices(
            segments,
            preprocessed.raw_indices,
        )

        return {
            "status": "success",
            "session_id": request.session_id,
            "samples_analyzed": len(request.telemetry_data),
            "parent_segment_count": len(segments),
            "segments": segments,
            "expert_reference_data": expert_reference_data,
        }
    except HTTPException:
        raise
    except TopLapReferenceModelError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeSegmentSplitError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segment classification failed: {str(e)}")


@router.post("/live-baseline-analysis")
async def analyze_live_baseline(request: LiveBaselineAnalysisRequest) -> Dict[str, Any]:
    try:
        if not request.records:
            raise HTTPException(status_code=400, detail="records is required")

        preprocessed = preprocess_inference_telemetry(request.records)
        splitter_result = split_runtime_segments(
            pd.DataFrame(preprocessed.records),
            request.track,
        )
        enriched_rows = get_top_lap_reference_model().enrich(
            preprocessed.records,
            track=request.track or splitter_result["circuit_id"],
            car=request.car,
        )
        enriched_rows = await get_tire_grip_analysis().enrich(enriched_rows)
        expert_reference_data = _project_expert_reference_data(
            enriched_rows,
            preprocessed.raw_indices,
        )
        segments = _classify_telemetry_segments(
            enriched_rows,
            splitter_result["circuit_id"],
            include_empty_track_sections=True,
            splitter_result=splitter_result,
        )
        segments = _annotate_segments_with_time_gaps(segments, enriched_rows)
        segments = _translate_segment_ranges_to_raw_indices(
            segments,
            preprocessed.raw_indices,
        )

        return {
            "status": "success",
            "session_id": f"live-baseline-lap-{request.baseline_lap}"
                if request.baseline_lap is not None
                else "live-baseline",
            "samples_analyzed": len(request.records),
            "parent_segment_count": len(segments),
            "segments": segments,
            "expert_time_available": True,
            "expert_reference_data": expert_reference_data,
        }
    except HTTPException:
        raise
    except TopLapReferenceModelError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeSegmentSplitError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live baseline analysis failed: {str(e)}")


@router.post("/analyze-user-sessions")
async def analyze_all_user_sessions(request: AnalyzeUserSessionsRequest) -> Dict[str, Any]:
    try:
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        session_analysis = await analyze_user_sessions(request.user_id, request.session_limit or 10)
        return {
            "status": "success",
            "sessionAnalysis": session_analysis,
        }
    except HTTPException:
        raise
    except TopLapReferenceModelError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User session analysis failed: {str(e)}")
