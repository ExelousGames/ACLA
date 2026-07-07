from app.local_annotation_agent.workflow.preflight import (
    _prune_exclusive_label_candidates,
)


def test_higher_scored_exclusive_candidate_wins_when_later():
    candidates = [
        {"id": "MSP14", "score": 0.62, "exclusive_with": ["MSP24"]},
        {"id": "MSP24", "score": 0.74, "exclusive_with": ["MSP14"]},
    ]

    pruned = _prune_exclusive_label_candidates(candidates)

    assert [c["id"] for c in pruned] == ["MSP24"]


def test_higher_scored_exclusive_candidate_wins_when_earlier():
    candidates = [
        {"id": "MSP14", "score": 0.74, "exclusive_with": ["MSP24"]},
        {"id": "MSP24", "score": 0.62, "exclusive_with": ["MSP14"]},
    ]

    pruned = _prune_exclusive_label_candidates(candidates)

    assert [c["id"] for c in pruned] == ["MSP14"]


def test_one_way_exclusive_candidate_conflict_is_pruned():
    candidates = [
        {"id": "MSP14", "score": 0.62, "exclusive_with": []},
        {"id": "MSP24", "score": 0.74, "exclusive_with": ["MSP14"]},
    ]

    pruned = _prune_exclusive_label_candidates(candidates)

    assert [c["id"] for c in pruned] == ["MSP24"]


def test_non_conflicting_candidates_keep_original_order():
    candidates = [
        {"id": "MSP14", "score": 0.62, "exclusive_with": ["MSP24"]},
        {"id": "MSP10", "score": 0.41, "exclusive_with": []},
        {"id": "MSP11", "score": 0.88, "exclusive_with": []},
    ]

    pruned = _prune_exclusive_label_candidates(candidates)

    assert [c["id"] for c in pruned] == ["MSP14", "MSP10", "MSP11"]


def test_equal_score_exclusive_conflict_keeps_earlier_candidate():
    candidates = [
        {"id": "MSP14", "score": 0.62, "exclusive_with": ["MSP24"]},
        {"id": "MSP24", "score": 0.62, "exclusive_with": ["MSP14"]},
    ]

    pruned = _prune_exclusive_label_candidates(candidates)

    assert [c["id"] for c in pruned] == ["MSP14"]
