"""SkillRegistry — document-store interface over a folder of skill JSON files.

Layout::

    app/skills/annotation/             (this package — code + json data)
      __init__.py, _registry.py, _query.py, _embedder.py
      <name>.json                      (skill definitions — drop a json in, restart)

The registry exposes four verbs, uniform across every skill:

  * ``get(path)``               — dotted-path lookup into any skill's document tree
  * ``find(path, **filters)``   — Mongo-style filter over a collection at *path*
  * ``iter(path)``              — yield every document in a collection
  * ``search(query, top_k)``    — hybrid (vector + BM25) discovery search

``search`` is backed by LlamaIndex: a ``VectorStoreIndex`` over the
skill discovery headers paired with a ``BM25Retriever``, fused by
``QueryFusionRetriever`` in ``relative_score`` mode. Cache lives under
``<repo>/acla_ai_service/.cache/skills_llama/``; rebuilt when any
source JSON or the embedding model changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.infra.config import settings
from app.skills.annotation._query import (
    _split_path,
    find as _filter,
    get_path,
    iter_collection,
)

LOGGER = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SKILLS_ROOT = _PACKAGE_ROOT
_CACHE_DIR = _PACKAGE_ROOT.parent.parent.parent / ".cache" / "skills_llama"
_PERSIST_DIR = _CACHE_DIR / "storage"
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
    source_path: Path

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
        self._retriever = None  # llama_index QueryFusionRetriever
        self._load_skills()
        self._build_or_load_index()

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
    # Hybrid index for search()
    # ------------------------------------------------------------------

    def _manifest(self) -> Dict[str, str]:
        m: Dict[str, str] = {"__model__": settings.annotation_skill_embedding_model}
        for name, skill in sorted(self._skills.items()):
            m[name] = hashlib.sha256(skill.source_path.read_bytes()).hexdigest()
        return m

    def _build_or_load_index(self) -> None:
        if not self._skills:
            self._retriever = None
            return

        from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
        from llama_index.core.schema import TextNode
        from llama_index.retrievers.bm25 import BM25Retriever

        from app.skills.annotation._embedder import get_llama_embedding

        embed_model = get_llama_embedding()
        current = self._manifest()
        index = None

        if _PERSIST_DIR.exists() and _MANIFEST_FILE.exists():
            try:
                cached = json.loads(_MANIFEST_FILE.read_text())
            except Exception as exc:
                LOGGER.warning("Skill index manifest unreadable (%s) — rebuilding.", exc)
                cached = None
            if cached == current:
                try:
                    storage = StorageContext.from_defaults(persist_dir=str(_PERSIST_DIR))
                    index = load_index_from_storage(storage, embed_model=embed_model)
                    LOGGER.info("Skill hybrid index loaded from cache.")
                except Exception as exc:
                    LOGGER.warning("Skill index reload failed (%s) — rebuilding.", exc)
                    index = None

        if index is None:
            nodes = [
                TextNode(
                    text=skill.spec.discovery_text(),
                    metadata={"skill_name": skill.name},
                    excluded_embed_metadata_keys=["skill_name"],
                    excluded_llm_metadata_keys=["skill_name"],
                )
                for skill in self._skills.values()
            ]
            index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=str(_PERSIST_DIR))
            _MANIFEST_FILE.write_text(json.dumps(current, indent=2, sort_keys=True))
            LOGGER.info("Skill hybrid index built (%d skill(s)).", len(self._skills))

        nodes = list(index.docstore.docs.values())
        pool = max(int(settings.hybrid_candidate_pool), len(self._skills))
        vec_retriever = VectorIndexRetriever(index=index, similarity_top_k=pool)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=pool)
        self._retriever = QueryFusionRetriever(
            [vec_retriever, bm25_retriever],
            similarity_top_k=pool,
            mode=settings.hybrid_fusion_mode,
            num_queries=1,
            use_async=False,
        )

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

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SkillMatch]:
        """Hybrid (vector + BM25) search over skill discovery headers."""
        if self._retriever is None or not self._skills:
            return []
        nodes = self._retriever.retrieve(query)
        out: List[SkillMatch] = []
        for n in nodes:
            score = float(n.score) if n.score is not None else 0.0
            if score < min_score:
                continue
            name = n.node.metadata.get("skill_name")
            skill = self._skills.get(name) if name else None
            if skill is None:
                continue
            out.append(SkillMatch(skill, score))
            if len(out) >= max(0, top_k):
                break
        return out


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
