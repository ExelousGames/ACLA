"""SkillRegistry — document-store interface over a folder of skill yamls.

Layout::

    app/skills/                        (this package — orchestrator code + yaml data)
      __init__.py, _registry.py, _query.py, _embedder.py    (code)
      <name>.yaml                      (skill definitions — drop a yaml in, restart)

A skill is a single yaml file. No Python escape hatches — filtering,
formatting, or merging with non-skill state belong to the caller.

The registry exposes four verbs, uniform across every skill:

  * ``get(path)``               — dotted-path lookup into any skill's document tree
  * ``find(path, **filters)``   — Mongo-style filter over a collection at *path*
  * ``iter(path)``              — yield every document in a collection
  * ``search(query, top_k)``    — embedding similarity over discovery headers

Discovery headers are indexed as cached embeddings for ``search``;
only headers whose YAML hash changed are re-embedded on startup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from app.skills.annotation._query import (
    _split_path,
    find as _filter,
    get_path,
    iter_collection,
)

LOGGER = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
# Annotation yamls live under app/skills/annotation/<name>.yaml. The racing-
# engineer corpus lives in the sibling app/skills/racing_engineer/ subpackage
# and is loaded by its own module — see app/skills/__init__.py.
_SKILLS_ROOT = _PACKAGE_ROOT
# Cache lives at <project root>/.cache/skills — three levels up from
# app/skills/annotation/_registry.py.
_CACHE_DIR = _PACKAGE_ROOT.parent.parent.parent / ".cache" / "skills"
_EMBEDDINGS_FILE = _CACHE_DIR / "embeddings.npz"
_MANIFEST_FILE = _CACHE_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkillSpec:
    name: str
    description: str
    when_to_use: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def discovery_text(self) -> str:
        parts = [self.name + ".", self.description.strip()]
        if self.when_to_use:
            parts.append("Used when: " + "; ".join(self.when_to_use) + ".")
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags) + ".")
        return " ".join(p for p in parts if p)


@dataclass
class Skill:
    spec: SkillSpec
    raw_body: Dict[str, Any]
    yaml_path: Path

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class SkillMatch:
    skill: Skill
    score: float


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Document-store registry of all skills."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root: Path = root or _SKILLS_ROOT
        self._skills: Dict[str, Skill] = {}
        self._names: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._load_skills()
        self._build_or_load_index()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_skills(self) -> None:
        for yaml_path in sorted(self._root.glob("*.yaml")):
            if yaml_path.name.startswith("_") or yaml_path.name.startswith("."):
                continue
            with open(yaml_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            header = raw.pop("_skill", None)
            if not header:
                LOGGER.warning(
                    "Skill at %s has no '_skill:' header — skipping.", yaml_path,
                )
                continue
            spec = SkillSpec(
                name=str(header.get("name", yaml_path.stem)),
                description=str(header.get("description", "")).strip(),
                when_to_use=list(header.get("when_to_use") or []),
                tags=list(header.get("tags") or []),
            )

            self._skills[spec.name] = Skill(
                spec=spec,
                raw_body=raw,
                yaml_path=yaml_path,
            )
        LOGGER.info("SkillRegistry loaded %d skill(s): %s",
                    len(self._skills), sorted(self._skills.keys()))

    # ------------------------------------------------------------------
    # Embedding index for search()
    # ------------------------------------------------------------------

    def _build_or_load_index(self) -> None:
        current_manifest: Dict[str, str] = {}
        discovery_texts: Dict[str, str] = {}
        for name, skill in self._skills.items():
            yaml_bytes = skill.yaml_path.read_bytes()
            current_manifest[name] = hashlib.sha256(yaml_bytes).hexdigest()
            discovery_texts[name] = skill.spec.discovery_text()

        cached_embeddings: Dict[str, np.ndarray] = {}
        cached_manifest: Dict[str, str] = {}
        if _EMBEDDINGS_FILE.exists() and _MANIFEST_FILE.exists():
            try:
                cached_manifest = json.loads(_MANIFEST_FILE.read_text())
                arrs = np.load(_EMBEDDINGS_FILE)
                for n in cached_manifest:
                    if n in arrs.files:
                        cached_embeddings[n] = arrs[n]
            except Exception as exc:
                LOGGER.warning("Skill index cache unreadable (%s) — rebuilding.", exc)
                cached_embeddings, cached_manifest = {}, {}

        embeddings_map: Dict[str, np.ndarray] = {}
        to_embed: List[str] = []
        for name, h in current_manifest.items():
            if cached_manifest.get(name) == h and name in cached_embeddings:
                embeddings_map[name] = cached_embeddings[name]
            else:
                to_embed.append(name)

        if to_embed:
            from app.skills.annotation._embedder import embed
            texts = [discovery_texts[n] for n in to_embed]
            vecs = embed(texts)
            if vecs.ndim == 1:
                vecs = vecs[np.newaxis, :]
            for n, v in zip(to_embed, vecs):
                embeddings_map[n] = v
            LOGGER.info(
                "Embedded %d skill header(s); reused %d from cache.",
                len(to_embed), len(current_manifest) - len(to_embed),
            )

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(_EMBEDDINGS_FILE, **embeddings_map)
        _MANIFEST_FILE.write_text(json.dumps(current_manifest, indent=2, sort_keys=True))

        self._names = sorted(embeddings_map.keys())
        if self._names:
            self._embeddings = np.stack([embeddings_map[n] for n in self._names])
        else:
            self._embeddings = None

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
        """Path lookup. ``skills.get("sub_label_catalog.labels.MS1.description")``."""
        return self._resolve(path, default)

    def find(self, collection_path: str, **filters: Any) -> List[Dict[str, Any]]:
        """Filter a collection by Mongo-style predicates.

        ``skills.find("sub_label_catalog.labels", type="sub", parent="MS")``
        """
        coll = self._resolve(collection_path)
        return _filter(coll, filters)

    def iter(self, collection_path: str) -> List[Dict[str, Any]]:
        """Yield every document in a collection (with ``id`` injected from key)."""
        return iter_collection(self._resolve(collection_path))

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SkillMatch]:
        """Embedding-similarity search over discovery headers."""
        if self._embeddings is None or not self._names:
            return []
        from app.skills.annotation._embedder import embed
        q = embed(query)
        scores = self._embeddings @ q
        order = np.argsort(-scores)[: max(0, top_k)]
        return [
            SkillMatch(self._skills[self._names[i]], float(scores[i]))
            for i in order
            if scores[i] >= min_score
        ]


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
