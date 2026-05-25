"""Racing-engineer knowledge-base RAG index.

Walks ``app/skills/racing_engineer/knowledge/*.md`` (recursively),
chunks each file by ``## Heading`` (or by blank-line paragraphs if a
file has no headings), embeds the chunks with the racing-engineer
embedder, and serves cosine-similarity search.

Chunks are cached at ``<repo>/acla_ai_service/.cache/racing_knowledge/``
with a manifest keyed by per-file SHA + chunk index + embedding model
name. Editing one file re-embeds only its chunks on next start;
swapping the model invalidates the whole cache. Mirrors the pattern in
:mod:`app.skills.annotation._registry`.

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
from typing import Dict, List, Optional

import numpy as np

from app.infra.config import settings

LOGGER = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_KNOWLEDGE_DIR = _PACKAGE_ROOT / "knowledge"
# Cache lives at <project root>/acla_ai_service/.cache/racing_knowledge/
# — gitignored alongside the annotation cache.
_CACHE_DIR = _PACKAGE_ROOT.parent.parent.parent / ".cache" / "racing_knowledge"
_EMBEDDINGS_FILE = _CACHE_DIR / "embeddings.npz"
_MANIFEST_FILE = _CACHE_DIR / "manifest.json"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class KnowledgeHit:
    """One retrieved chunk."""

    source: str   # relative path from knowledge/, e.g. "sample_wet_setup.md"
    section: str  # heading text, or "" for headingless paragraphs
    text: str
    score: float


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _strip_frontmatter(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    return m.group(2).strip() if m else text.strip()


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


def _chunk_file(path: Path, max_chars: int) -> List[Dict[str, str]]:
    """Split one .md into ``[{section, text}, ...]`` chunks.

    Rules:
      * Frontmatter is stripped (it's metadata, not content).
      * If the file has any ``## Heading`` lines, each heading owns the
        text from itself to the next heading.
      * Otherwise the whole body is split on blank lines.
      * Any single chunk longer than ``max_chars`` is paragraph-split.
      * Each emitted chunk's ``text`` is **prefixed with the section
        heading** so the embedder sees both context and body — this
        materially improves retrieval of corner-specific snippets.
    """
    raw = path.read_text(encoding="utf-8")
    body = _strip_frontmatter(raw)
    if not body:
        return []

    headings = list(_HEADING_RE.finditer(body))
    chunks: List[Dict[str, str]] = []

    if not headings:
        for piece in _split_long(body, max_chars):
            chunks.append({"section": "", "text": piece})
        return chunks

    # Any preamble before the first heading
    if headings[0].start() > 0:
        preamble = body[: headings[0].start()].strip()
        if preamble:
            for piece in _split_long(preamble, max_chars):
                chunks.append({"section": "", "text": piece})

    for i, m in enumerate(headings):
        section = m.group(1).strip()
        body_start = m.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        section_body = body[body_start:body_end].strip()
        if not section_body:
            continue
        for piece in _split_long(section_body, max_chars):
            # Prefix the heading so the embedding captures both context and content.
            chunks.append({"section": section, "text": f"{section}. {piece}"})

    return chunks


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class KnowledgeRegistry:
    """RAG index over the racing-engineer knowledge corpus."""

    def __init__(self) -> None:
        self._chunks: List[Dict[str, str]] = []  # [{source, section, text}, ...]
        self._embeddings: Optional[np.ndarray] = None
        self._build_or_load_index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _collect_chunks(self) -> List[Dict[str, str]]:
        if not _KNOWLEDGE_DIR.is_dir():
            return []
        out: List[Dict[str, str]] = []
        max_chars = int(settings.racing_kb_max_chunk_chars)
        for path in sorted(_KNOWLEDGE_DIR.rglob("*.md")):
            # Skip the README — it's authoring docs, not knowledge content.
            if path.name.lower() == "readme.md":
                continue
            rel = path.relative_to(_KNOWLEDGE_DIR).as_posix()
            try:
                file_chunks = _chunk_file(path, max_chars)
            except Exception:
                LOGGER.exception("racing_engineer KB: failed to chunk %s", path)
                continue
            for ch in file_chunks:
                out.append({"source": rel, "section": ch["section"], "text": ch["text"]})
        return out

    def _build_or_load_index(self) -> None:
        from app.skills.racing_engineer._embedder import embed_documents, model_name

        chunks = self._collect_chunks()
        if not chunks:
            LOGGER.info(
                "racing_engineer KB: knowledge/ is empty — RAG search will return [].",
            )
            self._chunks = []
            self._embeddings = None
            return

        # Defer importing model_name() until after we know there's work; it
        # forces a model load. Once loaded, include the model name in the
        # manifest so a model swap invalidates the whole cache cleanly.
        active_model = model_name()

        # Per-chunk manifest keys: <source>#<chunk_index>:<sha256(text)>.
        current_manifest: Dict[str, str] = {}
        for i, ch in enumerate(chunks):
            key = f"{ch['source']}#{i}"
            current_manifest[key] = hashlib.sha256(ch["text"].encode("utf-8")).hexdigest()
        current_manifest["__model__"] = active_model

        cached_manifest: Dict[str, str] = {}
        cached_embeddings: Dict[str, np.ndarray] = {}
        cache_compatible = False
        if _EMBEDDINGS_FILE.exists() and _MANIFEST_FILE.exists():
            try:
                cached_manifest = json.loads(_MANIFEST_FILE.read_text())
                cache_compatible = cached_manifest.get("__model__") == active_model
                if cache_compatible:
                    arrs = np.load(_EMBEDDINGS_FILE)
                    for k in cached_manifest:
                        if k == "__model__":
                            continue
                        if k in arrs.files:
                            cached_embeddings[k] = arrs[k]
                else:
                    LOGGER.info(
                        "racing_engineer KB: cache was built with '%s', "
                        "active model is '%s' — full re-embed.",
                        cached_manifest.get("__model__"), active_model,
                    )
            except Exception as exc:
                LOGGER.warning(
                    "racing_engineer KB: cache unreadable (%s) — rebuilding.", exc,
                )
                cached_manifest, cached_embeddings, cache_compatible = {}, {}, False

        # Decide which chunks need (re-)embedding.
        to_embed_keys: List[str] = []
        to_embed_texts: List[str] = []
        embeddings_map: Dict[str, np.ndarray] = {}
        for i, ch in enumerate(chunks):
            key = f"{ch['source']}#{i}"
            h = current_manifest[key]
            if cache_compatible and cached_manifest.get(key) == h and key in cached_embeddings:
                embeddings_map[key] = cached_embeddings[key]
            else:
                to_embed_keys.append(key)
                to_embed_texts.append(ch["text"])

        if to_embed_texts:
            vecs = embed_documents(to_embed_texts)
            for k, v in zip(to_embed_keys, vecs):
                embeddings_map[k] = v
            LOGGER.info(
                "racing_engineer KB: embedded %d chunk(s); reused %d from cache.",
                len(to_embed_keys), len(chunks) - len(to_embed_keys),
            )
        else:
            LOGGER.info(
                "racing_engineer KB: %d chunk(s) all served from cache.", len(chunks),
            )

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(_EMBEDDINGS_FILE, **embeddings_map)
        _MANIFEST_FILE.write_text(json.dumps(current_manifest, indent=2, sort_keys=True))

        # Preserve corpus order so search results are stable per-build.
        ordered_keys = [f"{ch['source']}#{i}" for i, ch in enumerate(chunks)]
        self._chunks = chunks
        self._embeddings = np.stack([embeddings_map[k] for k in ordered_keys])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[KnowledgeHit]:
        if self._embeddings is None or not self._chunks:
            return []
        from app.skills.racing_engineer._embedder import embed_query

        k = top_k if top_k and top_k > 0 else settings.racing_kb_default_top_k
        q = embed_query(query)
        scores = self._embeddings @ q  # cosine, vectors are L2-normalised
        order = np.argsort(-scores)[: max(0, int(k))]
        out: List[KnowledgeHit] = []
        for i in order:
            score = float(scores[i])
            if score < min_score:
                continue
            ch = self._chunks[i]
            out.append(KnowledgeHit(
                source=ch["source"],
                section=ch["section"],
                text=ch["text"],
                score=score,
            ))
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
