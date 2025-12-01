"""
Script to migrate annotation labels from strings to IDs.
"""
import sys
import os
from pathlib import Path
import zarr
import json

# Ensure app module is on path
def _ensure_app_module_on_path() -> None:
    candidate = Path(__file__).resolve().parent
    for _ in range(3):
        if (candidate / "app").exists():
            path_str = candidate.as_posix()
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
        candidate = candidate.parent

_ensure_app_module_on_path()

from app.services.zarr_telemetry_store import get_shared_zarr_store
from app.services.full_dataset_ml_service import PipelineConfig

LABEL_MAPPING = {
    1: "Overtaking",
    2: "Tire Strategy",
    3: "Expert Line Adherence",
    4: "Mistake - Brake too early",
    5: "Recovery",
    6: "Mistake - Brake too late",
    7: "Recovery - Brake too late",
    8: "Recovery - Brake too early",
}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_MAPPING.items()}

def migrate_labels():
    store = get_shared_zarr_store()
    pipeline_config = PipelineConfig()
    annotation_key = pipeline_config.annotation_cache_key
    
    print(f"Checking annotations in key: {annotation_key}")
    
    if annotation_key not in store.list_cache_keys():
        print("Annotation key not found.")
        return

    metadata = store.get_cache_metadata(annotation_key)
    if not metadata:
        print("No metadata found.")
        return
        
    print(f"Found {metadata.chunk_count} chunks.")
    
    total_migrated = 0
    
    for chunk_index in range(1, metadata.chunk_count + 1):
        chunk_data = store.get_chunk(annotation_key, chunk_index)
        
        annotations = []
        if isinstance(chunk_data, list):
            annotations = chunk_data
        elif isinstance(chunk_data, dict) and "data" in chunk_data:
             annotations = chunk_data["data"]
        
        if not annotations:
            continue
            
        modified = False
        new_annotations = []
        
        for ann in annotations:
            labels = ann.get("labels", [])
            new_labels = []
            ann_modified = False
            
            if not isinstance(labels, list):
                labels = [labels]
                
            for l in labels:
                if isinstance(l, str) and l in LABEL_NAME_TO_ID:
                    new_labels.append(LABEL_NAME_TO_ID[l])
                    ann_modified = True
                else:
                    new_labels.append(l)
            
            if ann_modified:
                ann["labels"] = new_labels
                modified = True
                total_migrated += 1
            
            new_annotations.append(ann)
            
        if modified:
            store.save_chunk(annotation_key, chunk_index, new_annotations)
            print(f"Migrated chunk {chunk_index}")
            
    print(f"Migration complete. Total annotations updated: {total_migrated}")

if __name__ == "__main__":
    migrate_labels()
