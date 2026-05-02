"""
Vocabulary — pipeline-wide phrase -> definition lookup.

Loaded once from ``acla_ai_service/app/skills/vocabulary.yaml``.  Every
prompt assembled in the annotation pipeline (step_describer, planner,
proposal_synthesizer, ...) is scanned against this lookup; phrases that
appear in the prompt body get a Glossary block prepended that lists each
matched phrase and its definition.

Usage::

    from app.models.vocabulary import get_vocabulary

    vocab = get_vocabulary()
    glossary = vocab.build_glossary_block(prompt)
    if glossary:
        prompt = glossary + "\\n" + prompt
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

LOGGER = logging.getLogger(__name__)

_VOCAB_PATH = Path(__file__).resolve().parent.parent / "skills" / "vocabulary.yaml"

# Module-level singleton
_instance: Optional["Vocabulary"] = None


class Vocabulary:
    """Phrase -> definition lookup with prompt scanning."""

    __slots__ = ("_entries", "_phrase_rx")

    def __init__(self, entries: Dict[str, str]) -> None:
        self._entries: Dict[str, str] = {
            k.strip().lower(): v.strip()
            for k, v in entries.items()
            if isinstance(k, str) and v
        }
        # Sort phrases by length DESC so the regex matches the longest
        # phrase first.  Without this, a short phrase like "rear-slip
        # dominant" would shadow longer ones like
        # "rear-slip dominant at corner entry".
        sorted_phrases = sorted(self._entries.keys(), key=len, reverse=True)
        if sorted_phrases:
            pattern = "|".join(re.escape(p) for p in sorted_phrases)
            self._phrase_rx: Optional[re.Pattern[str]] = re.compile(
                pattern, flags=re.IGNORECASE
            )
        else:
            self._phrase_rx = None

    # -- single lookup ------------------------------------------------------

    def get(self, phrase: str) -> Optional[str]:
        """Return the definition for *phrase*, or ``None``."""
        return self._entries.get(phrase.strip().lower())

    # -- prompt scanning ----------------------------------------------------

    def phrases_used_in(self, text: str) -> List[Tuple[str, str]]:
        """Return ``[(phrase, definition), ...]`` for every vocab phrase in *text*.

        The result preserves first-occurrence order in *text*.  Duplicates are
        removed (a phrase that appears twice is reported once).
        """
        if not text or self._phrase_rx is None:
            return []
        seen: set[str] = set()
        results: List[Tuple[str, str]] = []
        for m in self._phrase_rx.finditer(text):
            key = m.group(0).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            defn = self._entries.get(key)
            if defn:
                results.append((m.group(0), defn))
        return results

    def build_glossary_block(self, text: str) -> str:
        """Return a glossary block ready to prepend to *text*, or ``""`` if no phrase matched."""
        used = self.phrases_used_in(text)
        if not used:
            return ""
        lines = ["=== Vocabulary used in this prompt ==="]
        for phrase, defn in used:
            lines.append(f'- "{phrase}": {defn}')
        lines.append("")
        return "\n".join(lines)

    # -- bulk introspection -------------------------------------------------

    @property
    def all_phrases(self) -> List[str]:
        return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_vocabulary(path: Optional[Path] = None) -> Vocabulary:
    """Load *vocabulary.yaml* and return a :class:`Vocabulary`."""
    p = path or _VOCAB_PATH
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    entries: Dict[str, str] = raw.get("vocabulary", {}) or {}
    LOGGER.info("Loaded vocabulary with %d phrase entries.", len(entries))
    return Vocabulary(entries)


def get_vocabulary() -> Vocabulary:
    """Return the module-level singleton, loading on first call."""
    global _instance
    if _instance is None:
        _instance = load_vocabulary()
    return _instance
