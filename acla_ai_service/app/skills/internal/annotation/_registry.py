"""SkillRegistry — document-store interface over a folder of skill JSON files.

Layout::

    app/skills/internal/annotation/             (this package — code + json data)
      __init__.py, _registry.py, _query.py, _embedder.py
      <name>.json                      (skill definitions — drop a json in, restart)

The registry is a read-only document store. It exposes three verbs,
uniform across every skill:

  * ``get(path)``               — dotted-path lookup into any skill's document tree
  * ``find(path, **filters)``   — Mongo-style filter over a collection at *path*
  * ``iter(path)``              — yield every document in a collection

Semantic retrieval over the label catalog is a separate concern — see
:mod:`app.skills.internal.annotation.label_search`.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.skills.internal.annotation._query import (
    _split_path,
    find as _filter,
    get_path,
    iter_collection,
)

LOGGER = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SKILLS_ROOT = _PACKAGE_ROOT


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkillSpec:
    name: str
    description: str
    when_to_use: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class Skill:
    spec: SkillSpec
    raw_body: Dict[str, Any]
    source_path: Path

    @property
    def name(self) -> str:
        return self.spec.name


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Document-store registry of all skills."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root: Path = root or _SKILLS_ROOT
        self._skills: Dict[str, Skill] = {}
        self._load_skills()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_skills(self) -> None:
        for json_path in sorted(self._root.glob("*.json")):
            if json_path.name.startswith("_") or json_path.name.startswith("."):
                continue
            with open(json_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh) or {}
            header = raw.pop("_skill", None)
            if not header:
                LOGGER.warning(
                    "Skill at %s has no '_skill' header — skipping.", json_path,
                )
                continue
            spec = SkillSpec(
                name=str(header.get("name", json_path.stem)),
                description=str(header.get("description", "")).strip(),
                when_to_use=list(header.get("when_to_use") or []),
                tags=list(header.get("tags") or []),
            )

            self._skills[spec.name] = Skill(
                spec=spec,
                raw_body=raw,
                source_path=json_path,
            )
        LOGGER.info("SkillRegistry loaded %d skill(s): %s",
                    len(self._skills), sorted(self._skills.keys()))

    # ------------------------------------------------------------------
    # Skill access (lower level)
    # ------------------------------------------------------------------

    def skill(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Unknown skill '{name}'. Known: {sorted(self._skills)}")
        return self._skills[name]

    def all_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def names(self) -> List[str]:
        return sorted(self._skills.keys())

    # ------------------------------------------------------------------
    # Path resolution — the heart of the query API
    # ------------------------------------------------------------------

    def _resolve(self, path: str, default: Any = None) -> Any:
        """Resolve a dotted path of the form ``<skill>.<key>[.<sub>...]``."""
        segments = _split_path(path)
        if not segments:
            return default
        skill_name = segments[0]
        if skill_name not in self._skills:
            return default
        skill = self._skills[skill_name]
        if len(segments) == 1:
            return skill.raw_body
        second = segments[1]
        if second not in skill.raw_body:
            return default
        base = skill.raw_body[second]
        rest = segments[2:]
        if not rest:
            return base
        return get_path(base, rest, default)

    # ------------------------------------------------------------------
    # Public query verbs
    # ------------------------------------------------------------------

    def get(self, path: str, default: Any = None) -> Any:
        """Path lookup. ``skills.get("sub_label_annotation.labels.MS1.description")``."""
        return self._resolve(path, default)

    def find(self, collection_path: str, **filters: Any) -> List[Dict[str, Any]]:
        """Filter a collection by Mongo-style predicates."""
        coll = self._resolve(collection_path)
        return _filter(coll, filters)

    def iter(self, collection_path: str) -> List[Dict[str, Any]]:
        """Yield every document in a collection (with ``id`` injected from key)."""
        return iter_collection(self._resolve(collection_path))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[SkillRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is not None:
        return _registry
    with _registry_lock:
        if _registry is None:
            _registry = SkillRegistry()
    return _registry
