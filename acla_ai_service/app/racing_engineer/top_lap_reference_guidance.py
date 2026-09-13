"""Top-lap reference guidance for the racing engineer.

Generates segment-purpose guidance using the LLM, classifying telemetry into
segment labels and asking the LLM to verbalize. The function operates on a
``Full_dataset_TelemetryMLService`` instance because the telemetry features and
LLM orchestrator live there; runtime enrichment comes from the model hub.
"""

import time
import pandas as pd
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.shared.telemetry import FeatureProcessor
from app.local_llm.local_llm import GenerationRequest
from app.ml.model_hub import (
    get_top_lap_reference_model,
    get_tire_grip_analysis,
)
from app.ml.prompts import generate_llm_prompt_from_labels

if TYPE_CHECKING:
    from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService


async def generate_top_lap_reference_guidance(
    service: "Full_dataset_TelemetryMLService",
    telemetry_dict: Dict[str, Any],
    *,
    sequence_length: int = 40,
    user_request: Optional[str] = None,
    track_name: Optional[str] = None,
    car_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate segment-purpose guidance using the LLM without requiring the transformer."""
    start_time = time.time()

    try:
        driver_request = (user_request or "").strip()

        telemetry_df = pd.DataFrame([telemetry_dict])
        processor = FeatureProcessor(telemetry_df)
        processed_df = processor.general_cleaning_for_analysis()

        processor.flip_y_z_features()
        features = (
            service._top_lap_reference_feature_names
            or service.telemetry_features.get_features_for_top_lap_reference()
        )

        filtered_df = processor.filter_features_by_list(processed_df, features)
        processed_telemetry_dict = (
            filtered_df.iloc[0].to_dict() if not filtered_df.empty else telemetry_dict
        )

        try:
            processed_telemetry_dict = get_top_lap_reference_model().enrich(
                [processed_telemetry_dict],
                track=track_name,
                car=car_name,
            )[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract top-lap reference features: {str(e)}"
            )

        tire_grip_service = get_tire_grip_analysis()
        chunk_grip_features = await tire_grip_service.extract_tire_grip_features(
            [processed_telemetry_dict]
        )

        if len(chunk_grip_features) > 0:
            processed_telemetry_dict.update(chunk_grip_features[0])

        future_payload: List[Dict[str, Any]] = []
        segment_metadata: Dict[str, Any] = {
            "sequence_length_hint": sequence_length,
        }
        if driver_request:
            segment_metadata["user_request"] = driver_request

        print("[DEBUG] Generating label-free LLM prompt for point-in-time guidance...")
        llm_model, llm_metadata = await service.llm_orchestrator.get_llm_for_inference()
        if llm_model is None:
            raise RuntimeError("LLM guidance model is not available")

        try:
            # This endpoint has one telemetry row. Temporal detection requires a
            # sequence, so guidance deliberately uses the existing generic prompt.
            user_prompt = generate_llm_prompt_from_labels([])
        except Exception as e:
            raise RuntimeError(f"Failed to generate LLM prompt from labels: {str(e)}")

        generation_request = GenerationRequest(
            user_prompt=user_prompt,
        )

        try:
            output_text = llm_model.generate(generation_request)
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")

        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": response_time_ms,
            "llm": {
                "user": user_prompt,
                "raw_output": output_text,
            }
        }

    except Exception as error:
        error_msg = f"Failed to generate top-lap reference guidance: {error}"
        print(f"[ERROR] {error_msg}")

        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)

        return {
            "status": "error",
            "error_message": error_msg,
            "error_type": type(error).__name__,
            "response_time_ms": response_time_ms,
        }
