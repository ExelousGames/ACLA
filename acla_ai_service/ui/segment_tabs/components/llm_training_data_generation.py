from typing import List, Any
from app.models.segment_models import LABEL_MAPPING

MAIN_EVENT_LABELS = [
    "1", "2", "EA", "4", "RM", "MS"
]

TRACK_LOCATION_LABELS = [
    "brands_hatch", "silverstone"
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
    "MS": [
        "The driver made a mistake at the {track_sections_str} section. The key features are: {sublabels_str}. Construct just a descriptive sentence using this information."
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

def generate_training_prompt(labels: List[Any], template_index: int = 0) -> str:
    """Generates the training prompt based on segment labels using the selected template."""
    labels_str_list = [str(l) for l in labels]
    
    main_events = [l for l in labels_str_list if l in MAIN_EVENT_LABELS]
    track_locations = [l for l in labels_str_list if l in TRACK_LOCATION_LABELS]
    track_sections = [l for l in labels_str_list if any(l.startswith(tl) and l != tl for tl in TRACK_LOCATION_LABELS)]
    sublabels = [l for l in labels_str_list if l not in MAIN_EVENT_LABELS and l not in TRACK_LOCATION_LABELS and l not in track_sections]

    # Convert to human readable strings
    main_events_str = ", ".join([LABEL_MAPPING.get(l, l) for l in main_events]) if main_events else "event"
    track_locations_str = ", ".join([LABEL_MAPPING.get(l, l) for l in track_locations]) if track_locations else "unknown track"
    track_sections_str = ", ".join([LABEL_MAPPING.get(l, l) for l in track_sections]) if track_sections else "unknown section"
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

def get_human_readable_labels(labels: List[Any]) -> List[str]:
    """Converts a list of label IDs to human readable labels."""
    return [LABEL_MAPPING.get(str(l), l) for l in labels]

