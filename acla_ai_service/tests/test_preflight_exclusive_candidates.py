from app.local_annotation_agent.workflow.preflight import (
    _label_candidates,
    _prune_exclusive_label_candidates,
)
from app.shared.labels import LABEL_CATEGORIES


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


def test_label_candidates_prune_exclusives_before_parent_cap(monkeypatch):
    searched_docs = [
        _doc("MSP1", 0.99, exclusive_with=["MSP5"]),
        _doc("MSP5", 0.98, exclusive_with=["MSP1"]),
        _doc("MSP13", 0.97, exclusive_with=["MSP22"]),
        _doc("MSP22", 0.96, exclusive_with=["MSP13"]),
        _doc("MSP14", 0.95, exclusive_with=["MSP24"]),
        _doc("MSP24", 0.94, exclusive_with=["MSP14"]),
        _doc("MSP10", 0.93),
        _doc("MSP11", 0.92),
    ]
    search_top_ks = []

    def fake_search(_evidence, *, filters=None, top_k=10):
        assert filters == {"parent": "MSP"}
        search_top_ks.append(top_k)
        return searched_docs[:top_k]

    monkeypatch.setattr(
        "app.local_annotation_agent.workflow.preflight.search",
        fake_search,
    )
    monkeypatch.setattr(
        "app.local_annotation_agent.workflow.preflight.get_doc",
        lambda _label_id: None,
    )

    candidates = _label_candidates(
        "brake pressure evidence",
        candidate_label_ids=["MSP"],
    )
    candidate_ids = [candidate["id"] for candidate in candidates]

    assert search_top_ks == [len(LABEL_CATEGORIES["MSP"])]
    assert candidate_ids == ["MSP", "MSP1", "MSP13", "MSP14", "MSP10", "MSP11"]


def test_label_candidates_do_not_prune_exclusives_across_parent_groups(monkeypatch):
    def fake_search(_evidence, *, filters=None, top_k=10):
        if filters == {"parent": "MSP"}:
            return [_doc("MSP13", 0.97, exclusive_with=["MSR1"])][:top_k]
        if filters == {"parent": "MSR"}:
            return [_doc("MSR1", 0.96, parent="MSR")][:top_k]
        return []

    monkeypatch.setattr(
        "app.local_annotation_agent.workflow.preflight.search",
        fake_search,
    )
    monkeypatch.setattr(
        "app.local_annotation_agent.workflow.preflight.get_doc",
        lambda _label_id: None,
    )

    candidates = _label_candidates(
        "mixed practice and racing evidence",
        candidate_label_ids=["MSP", "MSR"],
    )

    assert [candidate["id"] for candidate in candidates] == [
        "MSP",
        "MSP13",
        "MSR",
        "MSR1",
    ]


def _doc(label_id, score, *, parent="MSP", exclusive_with=None):
    return {
        "id": label_id,
        "name": label_id,
        "type": "sub",
        "parent": parent,
        "score": score,
        "exclusive_with": list(exclusive_with or []),
    }
