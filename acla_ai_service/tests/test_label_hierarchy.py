"""Tests for label parent/sub-label grouping helpers."""

from __future__ import annotations

from app.domain.label_hierarchy import normalize_grouped_label_ids


def test_normalize_grouped_label_ids_adds_missing_parent() -> None:
    cleaned, rejected, added = normalize_grouped_label_ids(["MSP1", "ST3"])

    assert cleaned == ["MSP", "MSP1", "ST3"]
    assert rejected == []
    assert added == ["MSP"]


def test_normalize_grouped_label_ids_keeps_existing_parent_order() -> None:
    cleaned, rejected, added = normalize_grouped_label_ids(["MSP", "ST3", "MSP1"])

    assert cleaned == ["MSP", "ST3", "MSP1"]
    assert rejected == []
    assert added == []
