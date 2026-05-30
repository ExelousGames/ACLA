"""
Training stage of the pipeline.

Reads annotated segments from cache, filters them for transformer-relevant
labels, and trains / saves the coach transformer model.
"""

import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.integrations.backend.client import backend_service as default_backend_service
from app.pipelines.training.pipeline.cleaning import print_section_divider
from app.pipelines.training.transformer_trainer import (
    prepare_and_train_coach_transformer_model,
)


async def run_transformer_guidance_training(
    annotation_cache_key: str,
    *,
    telemetry_store,
    cache_config,
    backend_service=None,
    shuffle_dataset: bool = True,
) -> Dict[str, Any]:
    backend = backend_service or default_backend_service

    transformer_training: Optional[Dict[str, Any]] = None
    llm_training: Optional[Dict[str, Any]] = None
    generated_datasets: List[Any] = []

    try:
        print(f"[INFO] Filtering annotations from {annotation_cache_key}...")
        filtered_segments = []

        chunk_iterator = telemetry_store.get_cached_data_chunks(annotation_cache_key)

        total_segments_checked = 0

        for chunk in chunk_iterator:
            if isinstance(chunk, list):
                for seg_dict in chunk:
                    total_segments_checked += 1
                    labels = seg_dict.get("labels", [])
                    if any(l in ["EA", "RM"] for l in labels):
                        filtered_segments.append(seg_dict)
            elif isinstance(chunk, dict):
                total_segments_checked += 1
                labels = chunk.get("labels", [])
                if any(l in ["EA", "RM"] for l in labels):
                    filtered_segments.append(chunk)

        print(f"[INFO] Found {len(filtered_segments)} annotated segments with EA/RM labels (scanned {total_segments_checked}).")

        if not filtered_segments:
            raise RuntimeError(f"No segments found with EA or RM labels in {annotation_cache_key}")

        segments_cache_key = cache_config.training_segments_cache_key

        await telemetry_store.cache_chunks_streaming(segments_cache_key, [filtered_segments])
        print(f"[INFO] Cached filtered segments to {segments_cache_key}")

        transformer_training = await prepare_and_train_coach_transformer_model(
            data_cache=telemetry_store,
            segments_cache_key=segments_cache_key,
        )

        if not transformer_training.get("success"):
            raise RuntimeError(transformer_training.get("error") or "Transformer training failed")

        await backend.save_ai_model(
            model_type="transformer_expert_action",
            model_data=transformer_training["serialized_model"],
            metadata={
                "training_history": transformer_training["training_history"],
                "test_metrics": transformer_training["test_metrics"],
                "training_timestamp": datetime.now().isoformat(),
            },
            is_active=True,
        )
        print("[INFO] ✓ Transformer model trained and saved")

    except RuntimeError as runtime_error:
        print(f"[ERROR] Pipeline error: {runtime_error}")
        return {
            "success": False,
            "error": str(runtime_error),
        }
    except Exception as training_error:
        print(f"[ERROR] Unexpected error: {training_error}")
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Transformer training / LLM dataset generation failed: {training_error}",
        }

    result_payload = {
        "success": True,
        "datasets": generated_datasets,
        "transformer_training": transformer_training,
        "llm_dataset_generation": llm_training,
    }

    print_section_divider("DATASET GENERATION COMPLETED")
    return result_payload
