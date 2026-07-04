from app.shared.labels import BEHAVIOR_LABELS, LABEL_CATEGORIES, LABEL_MAPPING, TRACK_LABELS


def test_track_labels_group_all_track_parent_labels():
    track_category_ids = [
        category_id
        for category_id in LABEL_CATEGORIES
        if category_id in LABEL_MAPPING
        and category_id not in BEHAVIOR_LABELS
    ]

    assert LABEL_CATEGORIES["Track"] == TRACK_LABELS
    assert track_category_ids == TRACK_LABELS
    assert all(label_id in LABEL_MAPPING for label_id in TRACK_LABELS)


def test_behavior_labels_group_all_behavior_parent_labels():
    behavior_category_ids = [
        category_id
        for category_id in LABEL_CATEGORIES
        if category_id in LABEL_MAPPING
        and category_id not in TRACK_LABELS
    ]

    assert LABEL_CATEGORIES["Behavior"] == BEHAVIOR_LABELS
    assert behavior_category_ids == BEHAVIOR_LABELS
    assert all(label_id in LABEL_MAPPING for label_id in BEHAVIOR_LABELS)
