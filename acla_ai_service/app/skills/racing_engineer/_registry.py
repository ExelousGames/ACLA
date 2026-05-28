"""Racing-engineer knowledge-base RAG index.

Walks every ``.md`` under ``app/skills/racing_engineer/`` (recursively
across ``labels/``, ``main_labels/``, ``features/``, ``behaviors/``,
``tracks/``, ``knowledge/``), chunks each file by ``## Heading`` (or by
blank-line paragraphs if a file has no headings), and serves hybrid
(vector + BM25) search via LlamaIndex.

The folder a file lives in becomes its ``kind`` (``label`` / ``feature``
/ ``track`` / ``behavior`` / ``main_label`` / ``knowledge``). Frontmatter
``name`` is preserved as the human-readable subject. Internal routing
fields like ``id`` / ``family`` / ``common_co_labels`` are stripped —
they're never LLM-visible.

The hybrid retriever pairs a ``VectorStoreIndex`` (BGE embeddings)
with a ``BM25Retriever`` and fuses their candidates through
``QueryFusionRetriever`` in ``relative_score`` mode.

Persistence lives at ``<repo>/acla_ai_service/.cache/racing_knowledge_llama/``;
the manifest fingerprints the corpus + model name. Editing one file or
swapping the model triggers a full re-index on next start.

Public surface is a single function::

    from app.skills.racing_engineer._registry import get_registry
    hits = get_registry().search("wet setup at Spa", top_k=5)
    # hits: List[KnowledgeHit]
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from app.infra.config import settings

LOGGER = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_CORPUS_ROOT = _PACKAGE_ROOT
_CACHE_DIR = _PACKAGE_ROOT.parent.parent.parent / ".cache" / "racing_knowledge_llama"
_PERSIST_DIR = _CACHE_DIR / "storage"
_MANIFEST_FILE = _CACHE_DIR / "manifest.json"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class KnowledgeHit:
    """One retrieved chunk. Never carries internal ids — only LLM-safe fields."""

    kind: str     # folder name: label / main_label / feature / behavior / track / knowledge
    name: str     # human-readable subject from frontmatter ``name``, "" if absent
    section: str  # heading text, or "" for headingless paragraphs
    text: str
    score: float


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> Tuple[str, str]:
    """Return (name, body). Only ``name`` is kept from frontmatter; ids/families dropped."""
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        return "", text.strip()
    try:
        front = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        front = {}
    if not isinstance(front, dict):
        front = {}
    name = str(front.get("name") or "").strip()
    return name, m.group(2).strip()


def _prefix(name: str, section: str, body: str) -> str:
    parts = [p for p in (name, section) if p]
    return f"{'. '.join(parts)}. {body}" if parts else body


def _split_long(body: str, max_chars: int) -> List[str]:
    """Paragraph-aware split. Keep paragraphs intact; pack until cap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        return []
    out: List[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= max_chars:
            buf = f"{buf}\n\n{p}"
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _chunk_file(path: Path, max_chars: int, fallback_name: str = "") -> List[Dict[str, str]]:
    """Split one .md into ``[{name, section, text}, ...]`` chunks.

    Rules:
      * Frontmatter is parsed; only ``name`` is kept (ids/families dropped).
      * When the file has no frontmatter ``name`` (labels/, main_labels/
        store the human name in the filename itself), ``fallback_name``
        is used instead.
      * If the file has any ``## Heading`` lines, each heading owns the
        text from itself to the next heading.
      * Otherwise the whole body is split on blank lines.
      * Any single chunk longer than ``max_chars`` is paragraph-split.
      * Each emitted chunk's ``text`` is prefixed with ``name`` and the
        section heading so the embedder sees the subject, not just the
        body. A query like "oversteer at entry" then matches a chunk
        from labels/oversteering_at_entry.md via the name prefix.
    """
    raw = path.read_text(encoding="utf-8")
    name, body = _split_frontmatter(raw)
    if not name:
        name = fallback_name
    if not body:
        return []

    headings = list(_HEADING_RE.finditer(body))
    chunks: List[Dict[str, str]] = []

    if not headings:
        for piece in _split_long(body, max_chars):
            chunks.append({"name": name, "section": "", "text": _prefix(name, "", piece)})
        return chunks

    if headings[0].start() > 0:
        preamble = body[: headings[0].start()].strip()
        if preamble:
            for piece in _split_long(preamble, max_chars):
                chunks.append({"name": name, "section": "", "text": _prefix(name, "", piece)})

    for i, m in enumerate(headings):
        section = m.group(1).strip()
        body_start = m.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        section_body = body[body_start:body_end].strip()
        if not section_body:
            continue
        for piece in _split_long(section_body, max_chars):
            chunks.append({"name": name, "section": section, "text": _prefix(name, section, piece)})

    return chunks


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class KnowledgeRegistry:
    """Hybrid (vector + BM25) RAG index over the racing-engineer knowledge corpus."""

    def __init__(self) -> None:
        self._chunks: List[Dict[str, str]] = []
        self._retriever = None  # llama_index QueryFusionRetriever
        self._build_or_load_index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _collect_chunks(self) -> List[Dict[str, str]]:
        if not _CORPUS_ROOT.is_dir():
            return []
        out: List[Dict[str, str]] = []
        max_chars = int(settings.racing_kb_max_chunk_chars)
        for path in sorted(_CORPUS_ROOT.rglob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            if path.name.startswith("_") or path.name.startswith("."):
                continue
            rel_parts = path.relative_to(_CORPUS_ROOT).parts
            if len(rel_parts) < 2:
                # Top-level .md (only README, already skipped) — no kind folder.
                continue
            kind = rel_parts[0]
            rel = path.relative_to(_CORPUS_ROOT).as_posix()
            # Filename stem (slug) as fallback name — labels/ and main_labels/
            # have no frontmatter; their human name lives in the filename.
            fallback_name = path.stem.replace("_", " ").strip()
            try:
                file_chunks = _chunk_file(path, max_chars, fallback_name=fallback_name)
            except Exception:
                LOGGER.exception("racing_engineer KB: failed to chunk %s", path)
                continue
            for ch in file_chunks:
                out.append({
                    "_source": rel,        # internal — for manifest / debug only
                    "kind": kind,
                    "name": ch["name"],
                    "section": ch["section"],
                    "text": ch["text"],
                })
        return out

    def _manifest(self, chunks: List[Dict[str, str]]) -> Dict[str, str]:
        m: Dict[str, str] = {"__model__": settings.racing_kb_embedding_model}
        for i, ch in enumerate(chunks):
            key = f"{ch['_source']}#{i}"
            payload = (
                f"{ch.get('kind','')}\n{ch.get('name','')}\n"
                f"{ch['section']}\n{ch['text']}"
            ).encode("utf-8")
            m[key] = hashlib.sha256(payload).hexdigest()
        return m

    def _build_or_load_index(self) -> None:
        chunks = self._collect_chunks()
        self._chunks = chunks
        if not chunks:
            LOGGER.info(
                "racing_engineer KB: knowledge/ is empty — RAG search will return [].",
            )
            self._retriever = None
            return

        from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
        from llama_index.core.schema import TextNode
        from llama_index.retrievers.bm25 import BM25Retriever

        from app.skills.racing_engineer._embedder import get_llama_embedding

        embed_model = get_llama_embedding()
        current = self._manifest(chunks)
        index = None

        if _PERSIST_DIR.exists() and _MANIFEST_FILE.exists():
            try:
                cached = json.loads(_MANIFEST_FILE.read_text())
            except Exception as exc:
                LOGGER.warning("racing_engineer KB: manifest unreadable (%s) — rebuilding.", exc)
                cached = None
            if cached == current:
                try:
                    storage = StorageContext.from_defaults(persist_dir=str(_PERSIST_DIR))
                    index = load_index_from_storage(storage, embed_model=embed_model)
                    LOGGER.info(
                        "racing_engineer KB: hybrid index loaded from cache (%d chunk(s)).",
                        len(chunks),
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "racing_engineer KB: cache reload failed (%s) — rebuilding.", exc,
                    )
                    index = None

        if index is None:
            nodes = [
                TextNode(
                    text=ch["text"],
                    metadata={
                        "kind": ch["kind"],
                        "name": ch["name"],
                        "section": ch["section"],
                    },
                    excluded_embed_metadata_keys=["kind", "name", "section"],
                    excluded_llm_metadata_keys=["kind", "name", "section"],
                )
                for ch in chunks
            ]
            index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=str(_PERSIST_DIR))
            _MANIFEST_FILE.write_text(json.dumps(current, indent=2, sort_keys=True))
            LOGGER.info(
                "racing_engineer KB: hybrid index built (%d chunk(s)).", len(chunks),
            )

        nodes = list(index.docstore.docs.values())
        pool = max(int(settings.hybrid_candidate_pool), int(settings.racing_kb_default_top_k))
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
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[KnowledgeHit]:
        if self._retriever is None or not self._chunks:
            return []

        k = top_k if top_k and top_k > 0 else settings.racing_kb_default_top_k
        nodes = self._retriever.retrieve(query)
        out: List[KnowledgeHit] = []
        for n in nodes:
            score = float(n.score) if n.score is not None else 0.0
            if score < min_score:
                continue
            md = n.node.metadata or {}
            out.append(KnowledgeHit(
                kind=str(md.get("kind", "")),
                name=str(md.get("name", "")),
                section=str(md.get("section", "")),
                text=n.node.get_content(),
                score=score,
            ))
            if len(out) >= max(0, int(k)):
                break
        return out


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[KnowledgeRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> KnowledgeRegistry:
    global _registry
    if _registry is not None:
        return _registry
    with _registry_lock:
        if _registry is None:
            _registry = KnowledgeRegistry()
    return _registry


def reload() -> None:
    """Drop the cached registry singleton; next call re-walks knowledge/."""
    global _registry
    with _registry_lock:
        _registry = None


__all__ = ["KnowledgeHit", "KnowledgeRegistry", "get_registry", "reload"]
