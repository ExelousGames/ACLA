"""All-in-one skill folder — document-store interface.

Layout: ``app/skills/<name>/{skill.yaml, data.py?, render.py?}``.

All callers use the ``skills`` singleton with five verbs:

  * ``skills.get(path)``               — dotted-path lookup
  * ``skills.find(path, **filters)``   — Mongo-style filter over a collection
  * ``skills.iter(path)``              — yield every document in a collection
  * ``skills.render(name, **params)``  — call the skill's renderer
  * ``skills.search(query, top_k)``    — embedding similarity over discovery headers

There is no skill-specific import surface (no ``LabelCatalog`` /
``LapAnnotationSkill`` / ``GraphAnalysisSkill``). The renderer for each
skill, if any, lives at ``app/skills/<name>/render.py`` and itself
uses the query verbs internally.

Sub-agents that need to embed text directly (e.g. label_verifier
scoring observations against label descriptions) import ``embed`` /
``cosine_sim`` from :mod:`app.skills._embedder` — same model singleton
as the registry's discovery index.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.skills._embedder import cosine_sim, embed
from app.skills._registry import (
    Skill,
    SkillMatch,
    SkillRegistry,
    SkillSpec,
    get_registry,
)


class _SkillsProxy:
    """Lazy proxy — defers registry construction (and embedding-model load)
    until the first attribute access."""

    def __repr__(self) -> str:
        return f"<skills proxy: {get_registry().names()}>"

    # The five verbs

    def get(self, path: str, default: Any = None) -> Any:
        return get_registry().get(path, default)

    def find(self, collection_path: str, **filters: Any) -> List[Dict[str, Any]]:
        return get_registry().find(collection_path, **filters)

    def iter(self, collection_path: str) -> List[Dict[str, Any]]:
        return get_registry().iter(collection_path)

    def render(self, skill_name: str, **params: Any) -> str:
        return get_registry().render(skill_name, **params)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SkillMatch]:
        return get_registry().search(query, top_k=top_k, min_score=min_score)

    # Introspection

    def names(self) -> List[str]:
        return get_registry().names()

    def skill(self, name: str) -> Skill:
        return get_registry().skill(name)


skills = _SkillsProxy()


__all__ = [
    "Skill",
    "SkillMatch",
    "SkillRegistry",
    "SkillSpec",
    "cosine_sim",
    "embed",
    "get_registry",
    "skills",
]
