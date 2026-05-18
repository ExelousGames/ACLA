"""SkillRegistry — document-store interface over a folder of skill yamls.

Layout::

    app/skills/<name>/
      skill.yaml      _skill: discovery header + raw body
      data.py         (optional) COLLECTIONS = {name: callable(raw_body) -> dict|list}
      render.py       (optional) render(skills, **params) -> str

The registry exposes five verbs, uniform across every skill:

  * ``get(path)``               — dotted-path lookup into any skill's document tree
  * ``find(path, **filters)``   — Mongo-style filter over a collection at *path*
  * ``iter(path)``              — yield every document in a collection
  * ``render(skill, **params)`` — call the skill's renderer (prompt fragment)
  * ``search(query, top_k)``    — embedding similarity over discovery headers

Discovery headers are still indexed as cached embeddings for ``search``;
only headers whose YAML hash changed are re-embedded on startup.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np
import yaml

from app.skills._query import (
    OPERATORS,
    _split_path,
    find as _filter,
    get_path,
    iter_collection,
    matches,
)

LOGGER = logging.getLogger(__name__)

_SKILLS_ROOT = Path(__file__).resolve().parent
_CACHE_DIR = _SKILLS_ROOT.parent.parent / ".cache" / "skills"
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
    collection_providers: Dict[str, Callable[[Dict[str, Any]], Any]] = field(default_factory=dict)
    renderer: Optional[Callable[..., str]] = None
    _collections_cache: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    def collection(self, name: str) -> Any:
        """Materialise a collection by name; cache it.

        Resolution order: declared provider (data.py) → raw YAML body key.
        Returns ``None`` if neither has the name.
        """
        if name in self._collections_cache:
            return self._collections_cache[name]
        if name in self.collection_providers:
            result = self.collection_providers[name](self.raw_body)
        elif name in self.raw_body:
            result = self.raw_body[name]
        else:
            result = None
        self._collections_cache[name] = result
        return result


@dataclass
class SkillMatch:
    skill: Skill
    score: float


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

def _load_optional_module(folder_name: str, module_name: str):
    """Import ``app.skills.<folder>.<module>`` if it exists. Return module or None."""
    qualified = f"app.skills.{folder_name}.{module_name}"
    try:
        return importlib.import_module(qualified)
    except ImportError:
        return None


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
        for sub in sorted(self._root.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_") or sub.name.startswith("."):
                continue
            yaml_path = sub / "skill.yaml"
            if not yaml_path.exists():
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
                name=str(header.get("name", sub.name)),
                description=str(header.get("description", "")).strip(),
                when_to_use=list(header.get("when_to_use") or []),
                tags=list(header.get("tags") or []),
            )

            data_mod = _load_optional_module(sub.name, "data")
            collections: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
            if data_mod and hasattr(data_mod, "COLLECTIONS"):
                collections = dict(data_mod.COLLECTIONS)

            render_mod = _load_optional_module(sub.name, "render")
            renderer = getattr(render_mod, "render", None) if render_mod else None

            self._skills[spec.name] = Skill(
                spec=spec,
                raw_body=raw,
                yaml_path=yaml_path,
                collection_providers=collections,
                renderer=renderer,
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
            from app.skills._embedder import embed
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
        """Resolve a dotted path of the form ``<skill>.<collection_or_key>[.<sub>...]``."""
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
        rest = segments[2:]
        # Collections take priority over raw keys for the second segment
        if second in skill.collection_providers:
            base = skill.collection(second)
        elif second in skill.raw_body:
            base = skill.raw_body[second]
        else:
            return default
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

    def render(self, skill_name: str, **params: Any) -> str:
        """Call the skill's renderer. Empty string if no renderer is registered."""
        skill = self.skill(skill_name)
        if skill.renderer is None:
            LOGGER.warning("Skill '%s' has no renderer — returning empty string.", skill_name)
            return ""
        return skill.renderer(self, **params)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SkillMatch]:
        """Embedding-similarity search over discovery headers."""
        if self._embeddings is None or not self._names:
            return []
        from app.skills._embedder import embed
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
