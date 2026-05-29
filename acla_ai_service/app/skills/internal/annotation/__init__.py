"""Skill manager — orchestrator code separated from skill data.

Layout::

    app/skills/internal/annotation/             (this package — code + json data)
      __init__.py, _registry.py, _query.py, _embedder.py, label_search.py
      <name>.json                      (data — drop a json in, restart)

Skills are single JSON files; no Python escape hatches. Filtering,
formatting, or merging skill data with non-skill state is done on the
caller side.

The ``skills`` singleton is a read-only document store with three verbs:

  * ``skills.get(path)``               — dotted-path lookup
  * ``skills.find(path, **filters)``   — Mongo-style filter over a collection
  * ``skills.iter(path)``              — yield every document in a collection

Semantic retrieval over the label catalog lives separately in
:mod:`app.skills.internal.annotation.label_search` (``search_labels``).
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.skills.internal.annotation._registry import (
    Skill,
    SkillRegistry,
    SkillSpec,
    get_registry,
)


class _SkillsProxy:
    """Lazy proxy — defers registry construction until first attribute access."""

    def __repr__(self) -> str:
        return f"<skills proxy: {get_registry().names()}>"

    # The three verbs

    def get(self, path: str, default: Any = None) -> Any:
        return get_registry().get(path, default)

    def find(self, collection_path: str, **filters: Any) -> List[Dict[str, Any]]:
        return get_registry().find(collection_path, **filters)

    def iter(self, collection_path: str) -> List[Dict[str, Any]]:
        return get_registry().iter(collection_path)

    # Introspection

    def names(self) -> List[str]:
        return get_registry().names()

    def skill(self, name: str) -> Skill:
        return get_registry().skill(name)


skills = _SkillsProxy()


__all__ = [
    "Skill",
    "SkillRegistry",
    "SkillSpec",
    "get_registry",
    "skills",
]
