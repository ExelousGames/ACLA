import sys
import types


backend_module = types.ModuleType("app.integrations.backend.client")
backend_module.backend_service = object()
classifier_module = types.ModuleType("app.ml.segment_classifier.service")
classifier_module.segment_classifier = object()
sys.modules["app.integrations.backend.client"] = backend_module
sys.modules["app.ml.segment_classifier.service"] = classifier_module

from app.services.user_session_analysis import _finalize_summary


def test_finalize_summary_adds_parent_segments_and_child_highlights():
    summary = {
        "tracks": {
            "brands_hatch": {
                "trackName": "Brands Hatch",
                "sessionsAnalyzed": 1,
                "sessionsSkipped": 0,
                "sessionsFailed": 0,
                "totalTelemetryRows": 120,
                "cars": {"BMW M4 GT3": 1},
                "sections": {
                    "brands_hatch2": {
                        "sectionName": "Paddock Hill Bend",
                        "expertLevelTurns": 3,
                        "mistakes": 2,
                        "practiceMistakes": 2,
                        "racingMistakes": 0,
                        "labelCounts": {
                            "EA": 3,
                            "MSP1": 2,
                        },
                    },
                },
            },
        },
    }

    finalized = _finalize_summary(summary)
    track = finalized["tracks"]["brands_hatch"]

    assert track["trackOverview"]["parentSegmentCount"] == 1
    assert track["parentSegments"][0]["parentSegmentId"] == "brands_hatch2"
    assert track["parentSegments"][0]["parentSegmentName"] == "Paddock Hill Bend"
    assert track["parentSegments"][0]["childSegments"][0]["childSegmentId"] == "EA"
    assert track["parentSegments"][0]["childSegments"][0]["kind"] == "strength"
    assert track["parentSegments"][0]["childSegments"][1]["childSegmentId"] == "MSP1"
    assert track["parentSegments"][0]["childSegments"][1]["kind"] == "needs_work"
    assert track["strengths"][0]["childSegmentName"] == "Expert Adherence (Training)"
    assert track["improvementAreas"][0]["childSegmentName"] == "Initiate brake too late"
