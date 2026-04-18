"""
Label Catalog — loads and queries the YAML label knowledge base.

The catalog enriches the existing LABEL_MAPPING / LABEL_CATEGORIES dicts
in segment_models.py with rich descriptions, hierarchy rules, and
exclusive-with constraints.

Usage::

    from app.models.label_catalog import get_label_catalog

    catalog = get_label_catalog()
    info = catalog.get_label("MS1")
    subs = catalog.get_sublabels("MS")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.models.segment_models import LABEL_MAPPING, LABEL_CATEGORIES

LOGGER = logging.getLogger(__name__)

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "config" / "label_catalog.yaml"

# Module-level singleton
_catalog_instance: Optional["LabelCatalog"] = None


# ---------------------------------------------------------------------------
# Data container for a single label entry
# ---------------------------------------------------------------------------

class LabelEntry:
    """Metadata for a single label loaded from the YAML catalog."""

    __slots__ = (
        "id", "name", "type", "description",
        "parent", "children", "exclusive_with",
    )

    def __init__(self, label_id: str, raw: Dict[str, Any]) -> None:
        self.id: str = label_id
        self.name: str = raw.get("name", LABEL_MAPPING.get(label_id, label_id))
        self.type: str = raw.get("type", "unknown")
        self.description: str = (raw.get("description") or "").strip()
        self.parent: Optional[str] = raw.get("parent")
        self.children: List[str] = raw.get("children") or []
        self.exclusive_with: List[str] = raw.get("exclusive_with") or []


# ---------------------------------------------------------------------------
# LabelCatalog
# ---------------------------------------------------------------------------

class LabelCatalog:
    """Queryable catalog of all annotation labels."""

    def __init__(self, entries: Dict[str, LabelEntry]) -> None:
        self._entries = entries

        # Build reverse mapping: sub-label → parent
        self.parent_of: Dict[str, str] = {}
        for lid, entry in entries.items():
            if entry.parent:
                self.parent_of[lid] = entry.parent

    # -- single label -------------------------------------------------------

    def get_label(self, label_id: str) -> Optional[LabelEntry]:
        """Return the full metadata for *label_id*, or ``None``."""
        return self._entries.get(label_id)

    # -- bulk queries -------------------------------------------------------

    def get_main_labels(self) -> List[LabelEntry]:
        """Return entries for all main labels (from LABEL_CATEGORIES)."""
        main_ids = LABEL_CATEGORIES.get("Main Labels", [])
        return [self._entries[lid] for lid in main_ids if lid in self._entries]

    def get_segment_types(self) -> List[LabelEntry]:
        """Return entries for all Segment Type labels."""
        st_ids = LABEL_CATEGORIES.get("Segment Type", [])
        return [self._entries[lid] for lid in st_ids if lid in self._entries]

    def get_sublabels(self, parent_id: str) -> List[LabelEntry]:
        """Return sub-label entries for *parent_id*."""
        sub_ids = LABEL_CATEGORIES.get(parent_id, [])
        return [self._entries[sid] for sid in sub_ids if sid in self._entries]

    def get_hierarchy_rules(self, label_ids: List[str]) -> Dict[str, Any]:
        """Return hierarchy validation info for a set of label IDs.

        Returns
        -------
        dict with keys:
            missing_parents  – sub-labels whose parent is not in *label_ids*
            exclusive_conflicts – pairs of labels that are mutually exclusive
        """
        id_set = set(label_ids)
        missing_parents: List[Dict[str, str]] = []
        exclusive_conflicts: List[Dict[str, Any]] = []

        for lid in label_ids:
            entry = self._entries.get(lid)
            if not entry:
                continue
            # Check parent present
            if entry.parent and entry.parent not in id_set:
                missing_parents.append({"label": lid, "missing_parent": entry.parent})
            # Check exclusive_with
            for ex in entry.exclusive_with:
                if ex in id_set:
                    pair = tuple(sorted([lid, ex]))
                    conflict = {"labels": list(pair), "reason": f"{lid} and {ex} are mutually exclusive"}
                    if conflict not in exclusive_conflicts:
                        exclusive_conflicts.append(conflict)

        return {
            "missing_parents": missing_parents,
            "exclusive_conflicts": exclusive_conflicts,
        }

    @property
    def all_ids(self) -> List[str]:
        return list(self._entries.keys())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_label_catalog(path: Optional[Path] = None) -> LabelCatalog:
    """Load the YAML label catalog and return a :class:`LabelCatalog`.

    Parameters
    ----------
    path : Path, optional
        Override the default catalog file location.
    """
    catalog_path = path or _CATALOG_PATH
    with open(catalog_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    raw_labels: Dict[str, Any] = raw.get("labels", {})

    entries: Dict[str, LabelEntry] = {}
    for label_id, label_data in raw_labels.items():
        label_id_str = str(label_id)
        entries[label_id_str] = LabelEntry(label_id_str, label_data)

    # Warn about labels in LABEL_MAPPING that have no YAML entry
    for lid in LABEL_MAPPING:
        if lid not in entries:
            LOGGER.warning("Label '%s' (%s) exists in LABEL_MAPPING but has no YAML catalog entry.", lid, LABEL_MAPPING[lid])

    return LabelCatalog(entries)


def get_label_catalog() -> LabelCatalog:
    """Return the module-level singleton, loading on first call."""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = load_label_catalog()
    return _catalog_instance
