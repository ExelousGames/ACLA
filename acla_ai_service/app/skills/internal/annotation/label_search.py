"""Hybrid (vector + BM25) retrieval over the skill knowledge base.

``search`` indexes **every record in every collection of every skill**
held by the ``skills`` document store. Drop a new JSON into the skill
folder and its records become searchable on the next run — no code
change, nothing to register.

Nothing is hardcoded: not the skill name, not the collection key, not
the field names. A record's searchable text is the concatenation of all
its string fields, so a doc with a totally different shape still indexes
on whatever prose it carries. Its scalar fields (``type``, ``parent``,
or anything else a doc happens to define) become metadata you can filter
on; a record that omits a field is simply not matched by a filter on
that field — it is never dropped from the index.

A "record" is an entry of any collection — a top-level key whose value
is a dict-of-dicts (e.g. ``sub_label_annotation.labels.MSP1``). The
index is built once and cached under
``<repo>/acla_ai_service/.cache/labels_llama/``; it rebuilds whenever
the embedding model or any indexed text changes (so adding/editing a
skill JSON invalidates it automatically).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.infra.config import settings
from app.skills.internal.annotation._registry import get_registry

LOGGER = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
# app/skills/internal/annotation/ → repo root (acla_ai_service) is 4 parents up.
_CACHE_DIR = _PACKAGE_ROOT.parents[3] / ".cache" / "labels_llama"
_PERSIST_DIR = _CACHE_DIR / "storage"
_MANIFEST_FILE = _CACHE_DIR / "manifest.json"

# Populated together on first build/load.
_retriever = None
_docs_by_id: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _corpus_docs() -> List[Dict[str, Any]]:
    """Every record across every skill, structure-agnostic.

    Walks each skill body for collections (a key whose value is a
    dict-of-dicts) and yields each inner entry with its ``id`` (the key)
    plus ``_skill`` / ``_collection`` provenance injected.
    """
    docs: List[Dict[str, Any]] = []
    for skill in get_registry().all_skills():
        body = skill.raw_body if isinstance(skill.raw_body, dict) else {}
        for coll_key, coll_val in body.items():
            if not isinstance(coll_val, dict) or not coll_val:
                continue
            if not all(isinstance(v, dict) for v in coll_val.values()):
                continue
            for rec_id, rec in coll_val.items():
                doc = dict(rec)
                doc["id"] = rec_id
                doc["_skill"] = skill.name
                doc["_collection"] = coll_key
                docs.append(doc)
    return docs


def _node_text(doc: Dict[str, Any]) -> str:
    """Searchable text = every string field the record carries."""
    parts = [
        v.strip()
        for k, v in doc.items()
        if not k.startswith("_") and k != "id" and isinstance(v, str) and v.strip()
    ]
    return " ".join(parts) or str(doc.get("id", ""))


def _metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Filterable metadata = id, provenance, and every scalar field."""
    md: Dict[str, Any] = {
        "record_id": doc["id"],
        "_skill": doc.get("_skill", ""),
        "_collection": doc.get("_collection", ""),
    }
    for k, v in doc.items():
        if k.startswith("_") or k == "id":
            continue
        if isinstance(v, (str, int, float, bool)):
            md[k] = v
    return md


def _manifest(docs: List[Dict[str, Any]]) -> Dict[str, str]:
    payload = json.dumps(
        [(d.get("id"), _node_text(d)) for d in docs], sort_keys=True
    ).encode("utf-8")
    return {
        "__model__": settings.annotation_skill_embedding_model,
        "__docs__": hashlib.sha256(payload).hexdigest(),
    }


def _build_or_load() -> None:
    """Build (or load from cache) the hybrid retriever + id→record map."""
    global _retriever, _docs_by_id

    from llama_index.core import (
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    from app.skills.internal.annotation._embedder import get_llama_embedding

    docs = [d for d in _corpus_docs() if d.get("id")]
    _docs_by_id = {d["id"]: d for d in docs}
    embed_model = get_llama_embedding()
    current = _manifest(docs)
    index = None

    if _PERSIST_DIR.exists() and _MANIFEST_FILE.exists():
        try:
            cached = json.loads(_MANIFEST_FILE.read_text())
        except Exception as exc:
            LOGGER.warning("Skill-doc index manifest unreadable (%s) — rebuilding.", exc)
            cached = None
        if cached == current:
            try:
                storage = StorageContext.from_defaults(persist_dir=str(_PERSIST_DIR))
                index = load_index_from_storage(storage, embed_model=embed_model)
                LOGGER.info("Skill-doc hybrid index loaded from cache.")
            except Exception as exc:
                LOGGER.warning("Skill-doc index reload failed (%s) — rebuilding.", exc)
                index = None

    if index is None:
        nodes = []
        for doc in docs:
            md = _metadata(doc)
            nodes.append(
                TextNode(
                    text=_node_text(doc),
                    metadata=md,
                    excluded_embed_metadata_keys=list(md.keys()),
                    excluded_llm_metadata_keys=list(md.keys()),
                )
            )
        index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(_PERSIST_DIR))
        _MANIFEST_FILE.write_text(json.dumps(current, indent=2, sort_keys=True))
        LOGGER.info("Skill-doc hybrid index built (%d record(s)).", len(docs))

    nodes = list(index.docstore.docs.values())
    pool = max(int(settings.hybrid_candidate_pool), len(nodes))
    vec_retriever = VectorIndexRetriever(index=index, similarity_top_k=pool)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=pool)
    _retriever = QueryFusionRetriever(
        [vec_retriever, bm25_retriever],
        similarity_top_k=pool,
        mode=settings.hybrid_fusion_mode,
        num_queries=1,
        use_async=False,
    )


def _ensure_built() -> None:
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _build_or_load()


def _match(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """A record matches when every requested field equals (or is one of)
    the filter value. A field the record lacks never matches."""
    for field, val in filters.items():
        mv = meta.get(field)
        if isinstance(val, (list, tuple, set)):
            if mv not in val:
                return False
        elif mv != val:
            return False
    return True


def search(
    query: str,
    *,
    top_k: int = 8,
    min_score: float = 0.0,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Hybrid-search the skill knowledge base for records matching *query*.

    *query* is plain text. *filters* optionally scopes the result to
    records whose metadata fields match — e.g. ``{"type": "main"}`` or
    ``{"parent": "MSP"}``; a value may be a list to mean "one of". The
    field names are whatever the indexed docs declare; unknown fields
    just match nothing. Returns up to ``top_k`` full records (their skill
    JSON shape) each augmented with a ``score`` float, best-first.
    """
    q = (query or "").strip()
    if not q:
        return []

    _ensure_built()

    active = {k: v for k, v in (filters or {}).items() if v not in (None, "")}

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for n in _retriever.retrieve(q):
        score = float(n.score) if n.score is not None else 0.0
        if score < min_score:
            continue
        meta = n.node.metadata
        if not _match(meta, active):
            continue
        rid = meta.get("record_id")
        if not rid or rid in seen:
            continue
        doc = _docs_by_id.get(rid)
        if doc is None:
            continue
        seen.add(rid)
        out.append({**doc, "score": score})
        if len(out) >= max(0, top_k):
            break
    return out


def get_doc(record_id: str) -> Optional[Dict[str, Any]]:
    """Return an indexed record by id (its skill JSON shape), or None."""
    _ensure_built()
    return _docs_by_id.get(record_id)


__all__ = ["search", "get_doc"]
