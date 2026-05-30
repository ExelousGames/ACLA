"""Expert action prediction for the racing engineer.

Generates segment-purpose guidance using the LLM, classifying telemetry into
segment labels and asking the LLM to verbalize. The function operates on a
``Full_dataset_TelemetryMLService`` instance because the cache, backend client,
imitation/tire-grip services, and LLM orchestrator all live there.
"""

import time
import pandas as pd
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.domain.telemetry import FeatureProcessor
from app.local_llm.local_llm import GenerationRequest
from app.ml.prompts import generate_llm_prompt_from_labels
from app.ml.segment_classifier.service import segment_classifier

if TYPE_CHECKING:
    from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService


async def predict_expert_actions(
    service: "Full_dataset_TelemetryMLService",
    telemetry_dict: Dict[str, Any],
    *,
    sequence_length: int = 40,
    user_request: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate segment-purpose guidance using the LLM without requiring the transformer."""
    start_time = time.time()

    try:
        driver_request = (user_request or "").strip()

        telemetry_df = pd.DataFrame([telemetry_dict])
        processor = FeatureProcessor(telemetry_df)
        processed_df = processor.general_cleaning_for_analysis()

        processor.flip_y_z_features()
        features = service._imitate_expert_feature_names or service.telemetry_features.get_features_for_learning_expert()

        filtered_df = processor.filter_features_by_list(processed_df, features)
        processed_telemetry_dict = (
            filtered_df.iloc[0].to_dict() if not filtered_df.empty else telemetry_dict
        )

        chunk_imitation_features = []
        try:
            expert_service, _ = await service.model_cache_service.get_model_or_fetch(
                model_type="imitation_learning",
                model_subtype="imitation_model_data",
                service_instance=service._expert_service,
                deserializer_func=service._expert_service.deserialize_imitation_model,
                backend_fetcher=service.backend_service.getCompleteActiveModelData,
            )
            chunk_imitation_features = expert_service.extract_expert_state_for_telemetry(
                [processed_telemetry_dict]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract imitation features: {str(e)}")

        chunk_grip_features = []
        tire_grip_service, _ = await service.model_cache_service.get_model_or_fetch(
            model_type="tire_grip_analysis",
            model_subtype="tire_grip_model_data",
            service_instance=service._tire_grip_service,
            deserializer_func=service._tire_grip_service.deserialize_tire_grip_model,
            backend_fetcher=service.backend_service.getCompleteActiveModelData,
        )
        chunk_grip_features = await tire_grip_service.extract_tire_grip_features(
            [processed_telemetry_dict]
        )

        if len(chunk_imitation_features) > 0:
            processed_telemetry_dict.update(chunk_imitation_features[0])

        if len(chunk_grip_features) > 0:
            processed_telemetry_dict.update(chunk_grip_features[0])

        future_payload: List[Dict[str, Any]] = []
        segment_metadata: Dict[str, Any] = {
            "sequence_length_hint": sequence_length,
        }
        if driver_request:
            segment_metadata["user_request"] = driver_request

        try:
            predicted_labels = segment_classifier.predict_segment(pd.DataFrame([processed_telemetry_dict]))
        except ValueError as e:
            raise RuntimeError(f"Segment classifier prediction failed: {e}")

        print(f"[DEBUG] Generating LLM prompt from predicted labels...")
        llm_model, llm_metadata = await service.llm_orchestrator.get_llm_for_inference()
        if llm_model is None:
            raise RuntimeError("LLM guidance model is not available")

        try:
            user_prompt = generate_llm_prompt_from_labels(predicted_labels)
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
        error_msg = f"Failed to generate expert guidance: {error}"
        print(f"[ERROR] {error_msg}")

        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)

        return {
            "status": "error",
            "error_message": error_msg,
            "error_type": type(error).__name__,
            "response_time_ms": response_time_ms,
        }
