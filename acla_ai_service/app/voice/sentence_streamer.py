"""Buffer LLM tokens, emit complete sentences for downstream TTS.

Used by /naturallanguagequery/stream (Phase 2.5) to pace Kokoro TTS calls.
Without this, we'd either synth one word at a time (choppy) or wait for
the full answer (slow first-audio).

Design:
    - feed(text): add a chunk of streamed tokens
    - drain_sentences(): yield any complete sentences ready to synthesize
    - flush(): yield whatever remains at end-of-stream

A "complete sentence" is text ending with [.!?] followed by a space/newline
or end-of-buffer, AND meeting `min_words` (default 6). The word-count gate
prevents the synthesizer firing on tiny phrases like "Sure." which sound
choppy when chained.

Edge cases:
    - Decimals ("1.2s") are NOT treated as sentence ends — the splitter
      requires a space after the punctuation.
    - Common abbreviations (e.g. "F.I.A.", "i.e.") have at most a few chars
      between dots; the min_words gate handles these in practice.
    - On flush(), the min_words gate is bypassed so partial trailing text
      is still synthesized.
"""

from __future__ import annotations

import re
from typing import Iterator, List


# A sentence end is .!? followed by whitespace OR end-of-string.
# We use lookahead to keep the punctuation attached to the sentence.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Buffer ends with a sentence terminator preceded by a letter — i.e. the
# buffer's final sentence is complete even though no trailing whitespace
# arrived. The letter check avoids splitting on decimals ("1.2") or
# ellipses ("Wait..."). Acronyms like "F.I.A." can split prematurely, but
# they're vanishingly rare in engineer-voice answers; the trade favors
# faster time-to-first-audio for the common single-sentence case.
_TAIL_TERMINATOR_RE = re.compile(r"[A-Za-z][.!?]\s*$")


class SentenceStreamer:
    """Stateful buffer that yields complete sentences from token deltas."""

    def __init__(self, min_words: int = 6) -> None:
        self._buffer: str = ""
        self._min_words = max(1, min_words)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, text: str) -> None:
        """Append streamed tokens to the buffer."""
        if not text:
            return
        self._buffer += text

    def drain_sentences(self) -> Iterator[str]:
        """Yield each complete sentence that's now ready.

        Removes yielded sentences from the buffer. A sentence is "ready"
        when it has a terminal [.!?] followed by whitespace AND meets
        min_words. If a candidate sentence falls below min_words it stays
        in the buffer to merge with the next one.
        """
        if not self._buffer:
            return

        parts = _SENTENCE_END_RE.split(self._buffer)
        # parts[-1] is everything after the LAST sentence terminator — usually
        # incomplete, BUT if it ends in [letter][.!?] it's actually a complete
        # sentence that just lacks trailing whitespace (single-sentence LLM
        # responses always hit this — see _TAIL_TERMINATOR_RE comment).
        tail_is_complete = bool(parts) and bool(_TAIL_TERMINATOR_RE.search(parts[-1]))

        if len(parts) <= 1 and not tail_is_complete:
            return

        if tail_is_complete:
            complete_candidates = parts
            tail = ""
        else:
            complete_candidates = parts[:-1]
            tail = parts[-1]

        # Re-join short sentences with the next one until min_words is met.
        emitted: List[str] = []
        accumulator = ""
        for piece in complete_candidates:
            piece = piece.strip()
            if not piece:
                continue
            combined = (accumulator + " " + piece) if accumulator else piece
            if _word_count(combined) >= self._min_words:
                emitted.append(combined)
                accumulator = ""
            else:
                accumulator = combined

        # Anything still in accumulator wasn't long enough — push it back
        # onto the buffer so it can merge with future tokens.
        # CRITICAL: keep a trailing space separator. Without it, the next
        # feed() chunk concatenates directly onto the accumulator and the
        # next sentence ends up glued onto this one (e.g. "racer.Brake").
        if accumulator and tail:
            self._buffer = accumulator + " " + tail
        elif accumulator:
            self._buffer = accumulator + " "
        else:
            self._buffer = tail

        for sentence in emitted:
            yield sentence

    def flush(self) -> Iterator[str]:
        """At end-of-stream, yield whatever text remains regardless of length."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            yield remaining

    @property
    def buffered(self) -> str:
        """Inspect what hasn't been yielded yet (mostly for debugging)."""
        return self._buffer


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _word_count(text: str) -> int:
    # Split on whitespace; non-empty pieces count as words. Good enough
    # for an "is this big enough to synth?" gate.
    return sum(1 for piece in text.split() if piece)
