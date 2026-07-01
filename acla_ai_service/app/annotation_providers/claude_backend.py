"""
Claude SDK backend — wraps ``claude-agent-sdk``'s ``query()``.

Reuses the user's existing Claude Code OAuth session (Claude Max
subscription); no ``ANTHROPIC_API_KEY`` required. Provides a ``generate()``
that mirrors the local VLM service's contract so the runner swaps
backends via one config flag.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional

LOGGER = logging.getLogger(__name__)

CLAUDE_VLM_MODELS: dict[str, dict] = {
    "claude-sonnet-4-6": {
        "label": "Claude Sonnet 4.6 (recommended)",
    },
    "claude-opus-4-7": {
        "label": "Claude Opus 4.7",
    },
}


class ClaudeSDKBackend:
    """VLM backend that calls Claude via ``claude-agent-sdk``.

    Stateless across calls — each ``generate()`` spawns a fresh ``claude``
    subprocess via the SDK.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        use_thinking: bool = False,
    ) -> None:
        self._model = model
        self._use_thinking = use_thinking

    def generate(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream_callback: Optional[Callable[[str], None]] = None,
        reasoning_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Send prompt (+ optional images) to Claude.

        ``max_tokens`` and ``temperature`` are accepted for interface
        compatibility but ignored — the Claude CLI manages these internally.
        """
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "claude-agent-sdk is not installed. Install with: "
                "`pip install claude-agent-sdk` and ensure the `claude` CLI "
                "is installed and logged in (Claude Max subscription)."
            ) from exc

        with tempfile.TemporaryDirectory(prefix="acla_claude_imgs_") as tmp:
            full_prompt = self._build_prompt(prompt, images, Path(tmp))
            return asyncio.run(
                self._run_query(
                    full_prompt,
                    stream_callback=stream_callback,
                    reasoning_callback=reasoning_callback,
                )
            )

    def is_running(self) -> bool:
        return True

    def _build_prompt(
        self,
        prompt: str,
        images: Optional[List[bytes]],
        tmp_dir: Path,
    ) -> str:
        thinking_prefix = ""
        if self._use_thinking:
            thinking_prefix = (
                "Think carefully and step-by-step about the evidence before "
                "answering. Surface your reasoning, then produce the final "
                "answer in the requested format.\n\n"
            )

        if not images:
            return thinking_prefix + prompt

        paths: list[str] = []
        for i, img in enumerate(images):
            p = tmp_dir / f"graph_{i:02d}.png"
            p.write_bytes(img)
            paths.append(str(p))

        image_block = "\n".join(f"- {p}" for p in paths)
        return (
            f"{thinking_prefix}"
            f"Telemetry graph images to analyse (PNG, on disk):\n"
            f"{image_block}\n\n"
            f"{prompt}"
        )

    async def _run_query(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str], None]],
        reasoning_callback: Optional[Callable[[str], None]],
    ) -> str:
        from claude_agent_sdk import query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            model=self._model,
            allowed_tools=[],
        )

        answer_parts: list[str] = []
        async for message in query(prompt=prompt, options=options):
            text, thinking = self._extract(message)
            if thinking:
                if reasoning_callback is not None:
                    reasoning_callback(thinking)
                elif stream_callback is not None:
                    stream_callback(thinking)
            if text:
                answer_parts.append(text)
                if stream_callback is not None:
                    stream_callback(text)

        return "".join(answer_parts).strip()

    @staticmethod
    def _extract(message: Any) -> tuple[str, str]:
        try:
            from claude_agent_sdk.types import AssistantMessage, TextBlock
        except ImportError:
            AssistantMessage = None
            TextBlock = None

        text = ""
        thinking = ""

        if AssistantMessage is not None and isinstance(message, AssistantMessage):
            for block in getattr(message, "content", None) or []:
                if TextBlock is not None and isinstance(block, TextBlock):
                    text += getattr(block, "text", "") or ""
                    continue
                btype = getattr(block, "type", None)
                if btype == "thinking":
                    thinking += (
                        getattr(block, "thinking", None)
                        or getattr(block, "text", "")
                        or ""
                    )
                elif btype == "text":
                    text += getattr(block, "text", "") or ""
            return text, thinking

        if isinstance(message, dict):
            if message.get("type") == "assistant":
                for block in message.get("content", []) or []:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text += block.get("text", "") or ""
                    elif btype == "thinking":
                        thinking += (
                            block.get("thinking") or block.get("text", "") or ""
                        )

        return text, thinking


# Backwards-compatible alias.
ClaudeAgentBackend = ClaudeSDKBackend


def get_or_start_claude_backend(
    model: str = "claude-sonnet-4-6",
    use_thinking: bool = False,
) -> ClaudeSDKBackend:
    """Return a fresh Claude backend (stateless — no caching)."""
    return ClaudeSDKBackend(model=model, use_thinking=use_thinking)
