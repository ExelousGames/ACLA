#!/usr/bin/env python3
"""Synthesize racing-engineer corpus entries from anchor examples.

Generates Markdown-with-frontmatter `.md` files for the labels in
``app/domain/labels.py:LABEL_MAPPING`` that don't have a file yet,
using the 6 hand-authored anchor labels (MSP1, MSP22, MSP44, MSP47,
MSP17, MSP9) as worked examples for the LLM.

The script talks to the **local llama-server** by default — the same
endpoint the racing engineer runtime uses
(``settings.llama_server_url``). OpenAI-compatible HTTP, no API key
needed. Swap in a Claude or OpenAI client by passing ``--backend
claude`` / ``--backend openai`` and the corresponding env vars
(``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``).

Usage::

    # Generate every missing MSP*/MSR*/RM* sub-label label .md file.
    python -m scripts.synthesize_racing_engineer_corpus

    # Generate only specific labels (useful for re-running on failures).
    python -m scripts.synthesize_racing_engineer_corpus --only MSP2 MSP3 RM1

    # Overwrite existing files.
    python -m scripts.synthesize_racing_engineer_corpus --force

    # Use Claude instead of the local llama-server.
    python -m scripts.synthesize_racing_engineer_corpus --backend claude --model claude-opus-4-7

Outputs to ``app/skills/racing_engineer/labels/<ID>.md``. **No
post-processing or validation** — per the project's "trust VLM output"
policy. Review before merging.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.domain.labels import LABEL_MAPPING  # noqa: E402

_CORPUS_DIR = _PROJECT_ROOT / "app" / "skills" / "racing_engineer"
_LABELS_DIR = _CORPUS_DIR / "labels"

# Labels to generate (sub-labels only — main_labels are hand-authored).
_TARGET_FAMILIES = ("MSP", "MSR", "RM")

# Anchor examples that set the tone. The script reads these from disk
# at runtime so any hand-edits propagate to future syntheses.
_ANCHORS = ["MSP1", "MSP22", "MSP44", "MSP47", "MSP17", "MSP9"]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

async def _llama_generate(prompt: str, *, model: Optional[str] = None) -> str:
    """Call the local llama-server's OpenAI-compatible /chat/completions."""
    from openai import AsyncOpenAI
    from app.infra.config import settings

    client = AsyncOpenAI(
        base_url=settings.llama_server_url,
        api_key="not-needed",
    )
    resp = await client.chat.completions.create(
        model=model or settings.llama_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1500,
    )
    return (resp.choices[0].message.content or "").strip()


async def _claude_generate(prompt: str, *, model: Optional[str] = None) -> str:
    """Call Claude via the anthropic SDK. Requires ANTHROPIC_API_KEY."""
    from anthropic import AsyncAnthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = AsyncAnthropic(api_key=api_key)
    msg = await client.messages.create(
        model=model or "claude-opus-4-7",
        max_tokens=1500,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    # anthropic SDK returns a list of content blocks
    parts: List[str] = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


async def _openai_generate(prompt: str, *, model: Optional[str] = None) -> str:
    """Call OpenAI via the openai SDK. Requires OPENAI_API_KEY."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = AsyncOpenAI(api_key=api_key)
    resp = await client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1500,
    )
    return (resp.choices[0].message.content or "").strip()


_BACKENDS = {
    "llama": _llama_generate,
    "claude": _claude_generate,
    "openai": _openai_generate,
}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_PROMPT_HEADER = """You are authoring one entry in a racing-engineer
knowledge corpus. The corpus teaches a local LLM (Qwen 2.5-32B) to talk
to drivers like a real race engineer about specific driving actions /
mistakes the segment classifier detected.

Each entry is a single Markdown file with YAML frontmatter, in EXACTLY
this format:

```markdown
---
id: <LABEL_ID>
name: <human-readable name>
family: <MSP|MSR|RM|EA|...>
common_co_labels: [<other label ids>]
causes_to_check: [<other label ids>]
---

## Definition
<one-paragraph statement of what the action is>

## Physics
<why it happens, vehicle-dynamics view>

## Telemetry signature
- <bullet referencing telemetry channels>
- <bullet>
- <bullet>

## Engineer interpretation
<2-4 sentences in the voice of a real race engineer talking to a driver>

## Remedies
- <concrete next-time action>
- <concrete next-time action>
- <concrete next-time action>
```

Rules:
- Match the tone of the worked examples below precisely. They are the
  benchmark — same length, same plain-spoken engineer voice, same
  level of physics detail.
- Never refer to the label by its code (e.g. "MSP44") in the prose;
  use the natural name.
- Use telemetry channels from the worked examples where applicable
  (`Physics_brake`, `Physics_gas`, `expert_optimal_brake`,
  `expert_optimal_throttle`, `speed_difference`, `driver_push_to_limit`,
  slip angles, yaw rate, etc.). Don't invent channels.
- Keep `common_co_labels` and `causes_to_check` to label ids the
  classifier actually emits — use the worked examples as a guide.
- If the action's vehicle-dynamics meaning is genuinely unclear from
  the name alone, write a best-effort interpretation rather than
  refusing — the human reviewer will adjust.
- Output ONLY the Markdown file content, starting with `---` and
  ending with the last bullet of the Remedies section. No preamble,
  no commentary, no triple-backticks around the file.

WORKED EXAMPLES (study tone + structure carefully):
"""


def _load_anchor_examples() -> str:
    """Read every hand-authored anchor file verbatim into the prompt."""
    parts: List[str] = []
    for label_id in _ANCHORS:
        path = _LABELS_DIR / f"{label_id}.md"
        if not path.exists():
            print(f"[warn] anchor {label_id} not found at {path} — skipping")
            continue
        text = path.read_text(encoding="utf-8")
        parts.append(f"### EXAMPLE — {label_id} ({LABEL_MAPPING.get(label_id, label_id)})\n\n{text}")
    return "\n\n".join(parts)


def _build_prompt(label_id: str, label_name: str, anchors: str) -> str:
    return (
        _PROMPT_HEADER
        + anchors
        + f"\n\n---\n\nNow author the entry for:\n\n"
        + f"- id: {label_id}\n"
        + f"- name: {label_name}\n\n"
        + "Output the Markdown file:\n"
    )


# ---------------------------------------------------------------------------
# Target enumeration
# ---------------------------------------------------------------------------

def _targets(only: Optional[Iterable[str]] = None) -> List[str]:
    """Every label id in the target families that isn't a hand-authored anchor."""
    if only:
        return list(only)
    out: List[str] = []
    for label_id in LABEL_MAPPING:
        if not any(label_id.startswith(f) for f in _TARGET_FAMILIES):
            continue
        if not label_id[len(label_id.rstrip("0123456789")):].isdigit():
            # Skip family-bare entries like "MSP" itself — those live in main_labels/.
            continue
        if label_id in _ANCHORS:
            continue
        out.append(label_id)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=sorted(_BACKENDS), default="llama",
        help="Which LLM backend to use (default: llama via local llama-server).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model id override (otherwise backend default).",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Only synthesize the listed label ids (e.g. --only MSP2 MSP3).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files (default: skip).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max concurrent LLM calls (default: 4).",
    )
    args = parser.parse_args()

    _LABELS_DIR.mkdir(parents=True, exist_ok=True)
    generate = _BACKENDS[args.backend]
    anchors_block = _load_anchor_examples()
    if not anchors_block:
        print("[fatal] no anchor examples found — author MSP1/MSP22/MSP44/MSP47/MSP17/MSP9 first.")
        return 2

    targets = _targets(args.only)
    if not targets:
        print("[ok] nothing to synthesize.")
        return 0

    semaphore = asyncio.Semaphore(args.concurrency)
    written = 0
    skipped = 0
    failed = 0

    async def _one(label_id: str) -> None:
        nonlocal written, skipped, failed
        path = _LABELS_DIR / f"{label_id}.md"
        if path.exists() and not args.force:
            skipped += 1
            print(f"[skip] {label_id} (exists)")
            return
        name = LABEL_MAPPING.get(label_id, label_id)
        prompt = _build_prompt(label_id, name, anchors_block)
        async with semaphore:
            try:
                content = await generate(prompt, model=args.model)
            except Exception as exc:
                failed += 1
                print(f"[fail] {label_id}: {exc}")
                return
        # Strip wrapping ```markdown fences if the model added them.
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
        text = text.strip() + "\n"
        path.write_text(text, encoding="utf-8")
        written += 1
        print(f"[ok]   {label_id} → {path.relative_to(_PROJECT_ROOT)}")

    await asyncio.gather(*[_one(lid) for lid in targets])

    print(f"\n=== {written} written, {skipped} skipped, {failed} failed ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
