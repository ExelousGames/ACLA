"""Racing-engineer knowledge corpus loader.

The racing engineer's brain (Qwen2.5-32B) reaches into this corpus via the
``explain_label`` / ``analyze_telemetry`` server-side tools. The corpus
itself is plain Markdown files with YAML frontmatter — one file per label,
per main-label family, or per telemetry feature.

Layout::

    app/skills/external/racing_engineer/
      __init__.py                  (this module — loader)
      README.md                    (format spec for human authors)
      labels/<slugged_name>.md     (one per sub-label, e.g. oversteering_at_entry.md)
      main_labels/<slugged_name>.md (one per parent label, e.g. mistake_practice.md)
      features/<NAME>.md           (one per telemetry channel, e.g. driver_push_to_limit.md)

Every file is addressed by its human name (slugged), never by an internal
label id. Core categories use direct keyed lookups via ``label(name)`` /
``feature(name)`` etc. Internal ids (``MSP44``, ``RM7``, …) live in
``LABEL_MAPPING`` in ``app/domain/labels.py``; convert id → name via
``LABEL_MAPPING[id]`` upstream before reaching this module.

Two newer surfaces sit alongside that:

  * ``track(track_id, corner=None)`` — keyed lookup over ``tracks/<id>.md``.
    The ``## <corner>`` sections become per-corner entries the LLM can
    address by name. See ``tracks/README.md`` for the file format.

  * ``search(query, top_k)`` — RAG retrieval over every ``.md`` in the
    racing-engineer corpus (``labels/``, ``main_labels/``, ``features/``,
    ``behaviors/``, ``tracks/``, ``knowledge/``) via the embedding index
    in :mod:`._registry`. Use when the right doc isn't obvious — works
    for prose (``knowledge/``) and for structured per-label / per-track
    docs alike.

The two surfaces are intentional siblings: keyed when internal code
already has an id (classifier output), retrieval when the LLM is
working from a free-text question.

A loaded entry is a flat ``dict`` mixing frontmatter fields and parsed
section bodies. Section headings (``## Some Heading``) become keys with
lowercase-snake names (``some_heading``); the body text below each
heading becomes the value. Frontmatter keys win over parsed section
bodies if a name collides.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent

# category name → {entry stem → loaded dict}. Populated lazily on first read.
# The stem is the slugged human name (labels/oversteering_at_entry.md →
# "oversteering_at_entry"). Internal label ids (MSP44, RM7, …) live only in
# app/domain/labels.py; they are never used to address files here.
_CACHE: Dict[str, Dict[str, dict]] = {}
_CACHE_LOCK = threading.Lock()

# Matches a file with YAML frontmatter delimited by --- lines, then a body.
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def _slug(name: str) -> str:
    """Slug a human name to match a filename stem: lowercase, non-alnum → _."""
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _parse_md(path: Path) -> dict:
    """Parse one Markdown file into a flat dict.

    - Optional frontmatter (between ``---`` lines at the top) is parsed
      as YAML and merged into the output.
    - Section headings (``## Heading``) are split out into keys
      (lowercased, spaces → underscores) whose values are the section
      bodies.
    - If a name appears in both frontmatter AND as a section heading,
      frontmatter wins. (Authors should pick one home per field.)
    - ``id`` defaults to the filename stem when frontmatter doesn't set it.
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        front: dict = {}
        body = text.strip()
    else:
        try:
            front = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            LOGGER.warning("racing_engineer: bad YAML frontmatter in %s: %s", path, exc)
            front = {}
        if not isinstance(front, dict):
            LOGGER.warning("racing_engineer: frontmatter in %s is not a mapping", path)
            front = {}
        body = match.group(2).strip()

    sections: Dict[str, object] = {}
    current_key: Optional[str] = None
    buf: List[str] = []

    def _flush() -> None:
        if current_key is None:
            return
        sections[current_key] = _shape_section(buf)

    for line in body.splitlines():
        if line.startswith("## "):
            _flush()
            current_key = line[3:].strip().lower().replace(" ", "_")
            buf = []
        else:
            buf.append(line)
    _flush()

    # Frontmatter wins over section bodies on name collisions.
    out: dict = {**sections, **front, "_raw_body": body}
    out.setdefault("id", path.stem)
    return out


def _shape_section(lines: List[str]) -> object:
    """Return ``list[str]`` for pure-bullet sections, ``str`` otherwise.

    A section whose every non-blank line starts with ``"- "`` is a bullet
    list — return the bullet contents as a list of strings (with line
    continuations folded). Anything else is prose; return the trimmed
    block as a single string.

    Tools that read structured fields like ``remedies`` / ``telemetry_signature``
    benefit from real lists; prose sections like ``definition`` /
    ``engineer_interpretation`` stay as strings. Keeps the corpus author-
    friendly (just write Markdown bullets) and the runtime payload clean.
    """
    text = "\n".join(lines).strip()
    if not text:
        return ""
    # A section is treated as a bullet list when the first non-blank line
    # starts with "- ". Subsequent indented (or unindented) continuation
    # lines are folded into the preceding bullet. Mixed prose + bullets
    # are rare; if a section starts with a bullet, treat the whole thing
    # as bullets — authors can split if they want.
    first_non_blank = next((ln for ln in text.splitlines() if ln.strip()), "")
    if not first_non_blank.lstrip().startswith("- "):
        return text
    items: List[str] = []
    current: List[str] = []
    for ln in text.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("- "):
            if current:
                items.append(" ".join(current).strip())
            current = [stripped[2:].rstrip()]
        elif stripped and current:
            # Continuation of the previous bullet (wrapped line).
            current.append(stripped)
    if current:
        items.append(" ".join(current).strip())
    return items


def _load_category(name: str) -> Dict[str, dict]:
    """Lazy-load every .md under app/skills/external/racing_engineer/<name>/.

    Thread-safe. Missing directories return an empty dict (no error) so
    Phase 2 can ship category-by-category without empty folders being
    fatal.
    """
    with _CACHE_LOCK:
        cached = _CACHE.get(name)
        if cached is not None:
            return cached
        out: Dict[str, dict] = {}
        category_dir = _DIR / name
        if category_dir.is_dir():
            for path in sorted(category_dir.glob("*.md")):
                # Skip authoring docs: files starting with "_" / "." (mirrors
                # the convention in app/skills/internal/annotation/_registry.py) and
                # any plain README.md, which is always documentation about
                # the folder's format, never indexable data.
                if path.name.startswith("_") or path.name.startswith("."):
                    continue
                if path.name.lower() == "readme.md":
                    continue
                try:
                    entry = _parse_md(path)
                except Exception:
                    LOGGER.exception("racing_engineer: failed to parse %s", path)
                    continue
                entry_id = str(entry.get("id") or path.stem)
                out[entry_id] = entry
        _CACHE[name] = out
        return out


def label(name: str) -> Optional[dict]:
    """Return the concept doc for one action label by *human name*.

    Pass the readable name (e.g. ``"Oversteering at entry"`` or
    ``"oversteering_at_entry"``) — the lookup slugs the input and
    matches the file stem (``labels/oversteering_at_entry.md``).

    Looks first in ``labels/`` (sub-labels), then in ``main_labels/``
    (parent families). Returns ``None`` if no file exists yet — callers
    should fall back gracefully (the ``explain_label`` tool already does).

    Internal label ids (``"MSP44"``) are not accepted here; resolve
    them upstream via ``LABEL_MAPPING`` before calling.
    """
    if not name:
        return None
    stem = _slug(name)
    if not stem:
        return None
    entry = _load_category("labels").get(stem)
    if entry is not None:
        return entry
    return _load_category("main_labels").get(stem)


def feature(name: str) -> Optional[dict]:
    """Return the concept doc for one telemetry channel (e.g. ``"push_limit"``)."""
    if not name:
        return None
    return _load_category("features").get(name)


def behavior(name: str) -> Optional[dict]:
    """Return the behavior spec for a named LLM behavior (e.g. ``"emotion"``).

    Behaviors live in ``behaviors/<name>.md`` and are loaded into the system
    prompt at pipeline build time — they define *how* the LLM should act, not
    what it knows. See ``behaviors/`` for the file format.
    """
    if not name:
        return None
    return _load_category("behaviors").get(name)


def track(track_id: str, corner: Optional[str] = None) -> Optional[dict]:
    """Return the keyed track entry, optionally filtered to one corner.

    Without ``corner``: returns the full track dict plus a ``corners``
    list (the names of every ``## <name>`` section in the file) so the
    LLM can pick one and call back.

    With ``corner``: returns a slimmed dict containing just the matching
    section under ``corner_detail`` (plus the track's ``id`` / ``name``).
    Corner names are matched case-insensitively after collapsing spaces
    and slashes to underscores, which matches how :func:`_parse_md`
    normalises section keys.
    """
    if not track_id:
        return None
    entry = _load_category("tracks").get(track_id.lower())
    if entry is None:
        return None

    overview = entry.get("_raw_body", "").split("\n##", 1)[0].strip()
    # Section keys (the lowercased/underscored ``## Heading`` names) are the
    # corners. Everything else is metadata fields from frontmatter or the
    # parser's bookkeeping.
    metadata_keys = {"id", "name", "_raw_body", "length_km", "country"}
    corner_keys = [k for k in entry.keys() if k not in metadata_keys and not k.startswith("_")]

    if corner is None:
        return {
            "id": entry.get("id", track_id),
            "name": entry.get("name", track_id),
            "overview": overview,
            "corners": corner_keys,
            # Surface remaining frontmatter (length_km, country, etc.) so the
            # LLM can mention them without a second call.
            **{k: v for k, v in entry.items() if k in metadata_keys - {"_raw_body", "id", "name"}},
        }

    # Section keys came out of _parse_md via lower() + " " → "_", so they
    # may still contain "/", "-", and other punctuation. Normalise both the
    # stored keys AND the incoming corner name through the same transform
    # for matching, but keep the original stored key for the returned dict.
    def _norm(s: str) -> str:
        out = "".join(c if c.isalnum() else "_" for c in s.lower())
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_")

    query_norm = _norm(corner)
    matched: Optional[str] = None
    for k in corner_keys:
        kn = _norm(k)
        if kn == query_norm or query_norm in kn or kn in query_norm:
            matched = k
            break
    if matched is None:
        return {
            "id": entry.get("id", track_id),
            "name": entry.get("name", track_id),
            "error": f"corner '{corner}' not found",
            "available_corners": corner_keys,
        }
    section = entry.get(matched)
    return {
        "id": entry.get("id", track_id),
        "name": entry.get("name", track_id),
        "corner": matched,
        "corner_detail": section,
    }


def search(query: str, top_k: Optional[int] = None) -> List[dict]:
    """RAG search over every ``.md`` in the racing-engineer corpus.

    Walks ``labels/``, ``main_labels/``, ``features/``, ``behaviors/``,
    ``tracks/``, and ``knowledge/`` — so a free-text question can find
    any doc the engineer might want, not just the prose in ``knowledge/``.

    Returns JSON-friendly dicts with only LLM-safe fields: ``kind``,
    ``name``, ``section``, ``text``, ``score``. Internal routing keys
    (file paths, ids, families) are never surfaced.
    """
    if not query or not query.strip():
        return []
    from app.external_knowledge_base._registry import get_registry
    hits = get_registry().search(query.strip(), top_k=top_k)
    return [
        {
            "kind": h.kind,
            "name": h.name,
            "section": h.section,
            "text": h.text,
            "score": round(h.score, 4),
        }
        for h in hits
    ]


def reload() -> None:
    """Drop in-memory caches so the next call re-reads from disk.

    Clears the keyed `.md` cache here and also resets the RAG registry
    singleton so a fresh ``search()`` re-scans ``knowledge/``. Intended
    for the authoring script; production startup just lazy-loads on
    first access.
    """
    with _CACHE_LOCK:
        _CACHE.clear()
    try:
        from app.external_knowledge_base._registry import reload as _kb_reload
        _kb_reload()
    except Exception:
        LOGGER.exception("racing_engineer: failed to reset KB registry")


__all__ = ["label", "feature", "behavior", "track", "search", "reload"]
