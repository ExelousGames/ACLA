import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

from segment_tabs.batch_subsegment import _parent_main_label_ids
from segment_tabs.shared import LABEL_CATEGORIES, LABEL_MAPPING


def test_parent_main_label_ids_accepts_display_names():
    main_label_id = LABEL_CATEGORIES["Main Labels"][0]
    main_label_name = LABEL_MAPPING[main_label_id]

    assert _parent_main_label_ids([main_label_name]) == [main_label_id]


def test_parent_main_label_ids_accepts_ids_and_ignores_non_main_labels():
    main_label_id = LABEL_CATEGORIES["Main Labels"][0]

    assert _parent_main_label_ids([main_label_id, "not-a-main-label"]) == [
        main_label_id
    ]
