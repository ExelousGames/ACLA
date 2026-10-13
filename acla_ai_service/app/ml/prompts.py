from typing import List, Any, Dict, Optional
from app.shared.label_hierarchy import build_track_area_segments
from app.shared.labels import LABEL_MAPPING
MAIN_EVENT_LABELS = [
    "1", "2", "EA", "4", "RM", "MSP", "MSR", "O", "OD"
]

PROMPT_TEMPLATES = {
    "1": [
        "The system detected the following event(s): {main_events_str} on the {track_locations_str} track at the {track_sections_str} section. The key features are: {sublabels_str}. Construct a descriptive sentence using this information.",
        "Focusing strictly on event '{main_events_str}': It occurred at {track_locations_str} ({track_sections_str}). Features include: {sublabels_str}. Describe this concisely."
    ],
    "2": [
        "The system detected the following event(s): {main_events_str} on the {track_locations_str} track at the {track_sections_str} section. The key features are: {sublabels_str}. Construct a descriptive sentence using this information.",
        "Detail the {main_events_str} event observed at {track_locations_str}, specifically near {track_sections_str}. Notable aspects: {sublabels_str}."
    ],
    "EA": [
        "The system detected the following event(s): {main_events_str} on the {track_locations_str} track at the {track_sections_str} section. The key features are: {sublabels_str}. Construct a descriptive sentence using this information.",
        "Explain the {main_events_str} event located at {track_locations_str} within {track_sections_str} considering these factors: {sublabels_str}."
    ],
    "4": [
        "The system detected the following event(s): {main_events_str} on the {track_locations_str} track at the {track_sections_str} section. The key features are: {sublabels_str}. Construct a descriptive sentence using this information."
    ],
    "RM": [
        "The driver is trying to recover on the {track_locations_str} track at the {track_sections_str} section. The key features are: {sublabels_str}. Construct a descriptive sentence using this information."
    ],
    "MSP": [
        "The driver made a practice mistake at the {track_sections_str} section. The key features are: {sublabels_str}. Construct just a descriptive sentence using this information."
    ],
    "MSR": [
        "The driver made a racing mistake interacting with another car at the {track_sections_str} section. The key features are: {sublabels_str}. Construct just a descriptive sentence using this information."
    ],
    "default": [
        "The system detected the following event(s): {main_events_str} on the {track_locations_str} track at the {track_sections_str} section. The key features are: {sublabels_str}. Construct a descriptive sentence using this information."
    ]
}

def get_available_prompts(labels: List[Any]) -> List[str]:
    """Returns the list of available prompt templates based on the first detected main event."""
    labels_str_list = [str(l) for l in labels]
    main_events = [l for l in labels_str_list if l in MAIN_EVENT_LABELS]
    
    if main_events and main_events[0] in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[main_events[0]]
    return PROMPT_TEMPLATES["default"]

def generate_llm_prompt_from_labels(
    labels: List[Any],
    template_index: int = 0,
    *,
    telemetry_data: Optional[List[Dict[str, Any]]] = None,
    track_name: Optional[str] = None,
) -> str:
    """Generate a prompt with classifier events and backend-map location context."""
    labels_str_list = [str(l) for l in labels]
    
    main_events = [l for l in labels_str_list if l in MAIN_EVENT_LABELS]
    sublabels = [l for l in labels_str_list if l not in MAIN_EVENT_LABELS]
    rows = telemetry_data or []
    track = track_name or next((row["Static_track"] for row in rows if row.get("Static_track")), None)
    sections = build_track_area_segments([], rows, track, include_empty_sections=True)
    section_names = list(dict.fromkeys(section["track_section_name"] for section in sections))

    # Convert to human readable strings
    main_events_str = ", ".join([LABEL_MAPPING.get(l, l) for l in main_events]) if main_events else "event"
    track_locations_str = track or "unknown track"
    track_sections_str = ", ".join(section_names) if section_names else "unknown section"
    sublabels_str = ", ".join([LABEL_MAPPING.get(l, l) for l in sublabels]) if sublabels else "none"

    templates = get_available_prompts(labels)
    if template_index < 0 or template_index >= len(templates):
        template_index = 0
    
    template = templates[template_index]
    
    return template.format(
        main_events_str=main_events_str,
        track_locations_str=track_locations_str,
        track_sections_str=track_sections_str,
        sublabels_str=sublabels_str
    )

def generate_training_prompt(
    labels: List[Any],
    template_index: int = 0,
    *,
    telemetry_data: Optional[List[Dict[str, Any]]] = None,
    track_name: Optional[str] = None,
) -> str:
    """Generate a training prompt with the same backend-map context as inference."""
    return generate_llm_prompt_from_labels(
        labels, template_index, telemetry_data=telemetry_data, track_name=track_name,
    )

def get_human_readable_labels(labels: List[Any]) -> List[str]:
    """Converts a list of label IDs to human readable labels."""
    return [LABEL_MAPPING.get(str(l), l) for l in labels]

