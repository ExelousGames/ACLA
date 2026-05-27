"""
Script to migrate legacy labels in the annotation dataset to new labels.
Can be run directly or imported by the UI app.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Configuration for legacy label mapping
# Add your mappings here: "OLD_LABEL": "NEW_LABEL"
LEGACY_LABEL_MAP: Dict[Any, str] = {
    # Original integer-coded main labels
    "3": "EA",
    "5": "RM",
    3: "EA",
    5: "RM",
    "1": "O",
    "2": "MD",
    "4": "PS",

    # Mistake split (Practice vs Racing) — parent + sub-label prefix rename.
    # Old MS (single bucket) → MSP (technical practice) + MSR (interaction).
    # MS53 / MS54 (the interaction-failure sub-labels) are reparented under MSR.
    "MS": "MSP",
    "MS1": "MSP1",
    "MS2": "MSP2",
    "MS3": "MSP3",
    "MS4": "MSP4",
    "MS5": "MSP5",
    "MS6": "MSP6",
    "MS7": "MSP7",
    "MS8": "MSP8",
    "MS9": "MSP9",
    "MS10": "MSP10",
    "MS11": "MSP11",
    "MS13": "MSP13",
    "MS14": "MSP14",
    "MS15": "MSP15",
    "MS16": "MSP16",
    "MS17": "MSP17",
    "MS18": "MSP18",
    "MS19": "MSP19",
    "MS20": "MSP20",
    "MS21": "MSP21",
    "MS22": "MSP22",
    "MS23": "MSP23",
    "MS24": "MSP24",
    "MS25": "MSP25",
    "MS26": "MSP26",
    "MS27": "MSP27",
    "MS28": "MSP28",
    "MS29": "MSP29",
    "MS30": "MSP30",
    "MS31": "MSP31",
    "MS32": "MSP32",
    "MS33": "MSP33",
    "MS34": "MSP34",
    "MS35": "MSP35",
    "MS36": "MSP36",
    "MS37": "MSP37",
    "MS38": "MSP38",
    "MS41": "MSP41",
    "MS42": "MSP42",
    "MS43": "MSP43",
    "MS44": "MSP44",
    "MS45": "MSP45",
    "MS46": "MSP46",
    "MS47": "MSP47",
    "MS48": "MSP48",
    "MS49": "MSP49",
    "MS50": "MSP50",
    "MS51": "MSP51",
    "MS52": "MSP52",
    "MS53": "MSR1",
    "MS54": "MSR2",

    # Overtaking split (offensive O vs defensive OD).
    # Old O2 / O6 (defensive sub-labels) move under the new OD parent.
    "O2": "OD1",
    "O6": "OD2",
}

def _ensure_paths():
    """Ensure app and ui modules are on path."""
    current_file = Path(__file__).resolve()
    # Assume script is in acla_ai_service/scripts/
    # root is acla_ai_service/
    root_dir = current_file.parent.parent
    
    if root_dir.exists():
        path_str = root_dir.as_posix()
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            
    # Add ui folder to path so we can import segment_tabs
    ui_dir = root_dir / "ui"
    if ui_dir.exists():
        path_str_ui = ui_dir.as_posix()
        if path_str_ui not in sys.path:
            sys.path.insert(0, path_str_ui)

_ensure_paths()

# Late imports to avoid path errors
try:
    from segment_tabs.shared import get_store, get_available_sessions, load_annotations, save_annotations, AnnotatedSegment
except ImportError:
    print("Could not import segment_tabs.shared. Ensure you are running from the correct environment.")
    sys.exit(1)

def update_legacy_labels(dataset_key: str, migration_map: Dict[Any, str] = LEGACY_LABEL_MAP, dry_run: bool = False) -> Dict[str, int]:
    """
    Iterate over all sessions in the dataset and update labels based on the map.
    Returns statistics of updates.
    """
    store = get_store()
    sessions = get_available_sessions(dataset_key)
    
    stats = {
        "sessions_processed": 0,
        "sessions_updated": 0,
        "segments_updated": 0,
        "labels_replaced": 0
    }
    
    print(f"Starting migration for dataset: {dataset_key}")
    print(f"Mapping: {migration_map}")
    
    for session_id in sessions:
        stats["sessions_processed"] += 1
        annotations = load_annotations(session_id, dataset_key)
        
        session_modified = False
        updates_in_session = 0
        
        for segment in annotations:
            original_labels = list(segment.labels)
            new_labels = []
            segment_modified = False
            
            for label in original_labels:
                if label in migration_map:
                    new_labels.append(migration_map[label])
                    segment_modified = True
                    stats["labels_replaced"] += 1
                else:
                    new_labels.append(label)
            
            if segment_modified:
                segment.labels = new_labels
                session_modified = True
                updates_in_session += 1
        
        if session_modified:
            stats["sessions_updated"] += 1
            stats["segments_updated"] += updates_in_session
            if not dry_run:
                save_annotations(session_id, annotations, dataset_key)
                # To avoid spamming logs/UI, we might want to be quieter or use a progress callback if integrated in UI
                # But here we just print
                pass
                
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate legacy labels in annotation dataset.")
    parser.add_argument("dataset_key", help="The cache key of the dataset to update (e.g. annotation_data_v1)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without saving changes.")
    
    args = parser.parse_args()
    
    results = update_legacy_labels(args.dataset_key, LEGACY_LABEL_MAP, dry_run=args.dry_run)
    
    print("\nMigration Complete.")
    print(f"Sessions Processed: {results['sessions_processed']}")
    print(f"Sessions Updated: {results['sessions_updated']}")
    print(f"Segments Updated: {results['segments_updated']}")
    print(f"Labels Replaced: {results['labels_replaced']}")
