"""Racing-engineer knowledge corpus loader.

The racing engineer's brain (Qwen2.5-32B) reaches into this corpus via the
``explain_label`` / ``analyze_recent_segment`` server-side tools. The corpus
itself is plain Markdown files with YAML frontmatter — one file per label,
per main-label family, or per telemetry feature.

Layout::

    app/skills/racing_engineer/
      __init__.py           (this module — loader)
      README.md             (format spec for human authors)
      labels/<ID>.md        (one per sub-label, e.g. MS44.md)
      main_labels/<ID>.md   (one per parent label, e.g. MS.md)
      features/<NAME>.md    (one per telemetry channel, e.g. push_limit.md)

This module is intentionally **simple**: direct ``label(id)`` /
``feature(name)`` lookups, no query DSL, no embedding index, no Mongo-style
filters. Phase 2b can layer a ``_registry.py`` / ``_embedder.py`` here if
free-text search across the corpus becomes useful (see plan: "The racing
engineer subfolder is fully independent and can grow its own parallel
infrastructure as needs grow").

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

# category name → {entry id → loaded dict}. Populated lazily on first read.
_CACHE: Dict[str, Dict[str, dict]] = {}
_CACHE_LOCK = threading.Lock()

# Matches a file with YAML frontmatter delimited by --- lines, then a body.
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def _parse_md(path: Path) -> dict:
    """Parse one Markdown-with-frontmatter file into a flat dict.

    - Frontmatter (between ``---`` lines) is parsed as YAML.
    - Section headings (``## Heading``) are split out into keys
      (lowercased, spaces → underscores) whose values are the section
      bodies.
    - If a name appears in both frontmatter AND as a section heading,
      frontmatter wins. (Authors should pick one home per field.)
    - Files without frontmatter still load — the whole body lands at
      key ``"body"`` and ``"id"`` defaults to the filename stem.
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {"id": path.stem, "body": text.strip()}

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
    """Lazy-load every .md under app/skills/racing_engineer/<name>/.

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
                try:
                    entry = _parse_md(path)
                except Exception:
                    LOGGER.exception("racing_engineer: failed to parse %s", path)
                    continue
                entry_id = str(entry.get("id") or path.stem)
                out[entry_id] = entry
        _CACHE[name] = out
        return out


def label(label_id: str) -> Optional[dict]:
    """Return the concept doc for one action label by id (e.g. ``"MS44"``).

    Looks first in ``labels/`` (sub-labels), then in ``main_labels/``
    (parent families). Returns ``None`` if no file exists yet — callers
    should fall back gracefully (the ``explain_label`` tool already does).
    """
    if not label_id:
        return None
    entry = _load_category("labels").get(label_id)
    if entry is not None:
        return entry
    return _load_category("main_labels").get(label_id)


def feature(name: str) -> Optional[dict]:
    """Return the concept doc for one telemetry channel (e.g. ``"push_limit"``)."""
    if not name:
        return None
    return _load_category("features").get(name)


def reload() -> None:
    """Drop the in-memory cache so the next ``label`` / ``feature`` call
    re-reads from disk. Intended for the authoring script after writing
    new files in a long-running process; production startup just lazy-
    loads on first access."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = ["label", "feature", "reload"]
