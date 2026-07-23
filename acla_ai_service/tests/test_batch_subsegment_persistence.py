from types import SimpleNamespace

import pandas as pd

from ui.segment_tabs import batch_subsegment


def test_batch_children_inherit_main_labels_but_not_parent_segment_type(monkeypatch):
    saved_children = []

    def fake_build_segment(_df, **kwargs):
        child = SimpleNamespace(labels=kwargs["label_ids"])
        saved_children.append(child)
        return child

    monkeypatch.setattr(batch_subsegment, "build_segment", fake_build_segment)
    monkeypatch.setattr(batch_subsegment, "save_annotations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        batch_subsegment,
        "st",
        SimpleNamespace(session_state={"current_annotations": []}),
    )

    parent = SimpleNamespace(id="parent-1", labels=["RM", "ST2"])
    result = SimpleNamespace(label_annotations=[
        {
            "label_id": "RM7",
            "start_index": 1,
            "end_index": 5,
            "reasoning": "matched sub-label",
        },
        {
            "label_id": "ST14",
            "start_index": 1,
            "end_index": 5,
            "reasoning": "matched segment type",
        },
    ])

    count = batch_subsegment._persist_children_for_parent(
        parent,
        result,
        "session-1",
        "annotations",
        pd.DataFrame(index=range(6)),
    )

    assert count == 1
    assert saved_children[0].labels == ["RM", "RM7", "ST14"]
