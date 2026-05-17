"""
Claude-backed VLM service for the annotation agent pipeline.

Drives ``claude-agent-sdk``'s ``query()`` so the user's existing Claude Code
OAuth session (Claude Max subscription) is reused — no ``ANTHROPIC_API_KEY``
required. Provides a ``generate()`` method that mirrors the contract of
:class:`AnnotationAgentLLMService` so the pipeline can swap backends with a
single config flag.

Typical usage::

    from app.services.llm.claude_agent_backend import (
        get_or_start_claude_backend,
        CLAUDE_VLM_MODELS,
    )

    backend = get_or_start_claude_backend(model="claude-sonnet-4-6")
    answer = backend.generate("Describe the telemetry graph.", images=[png_bytes])
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional

LOGGER = logging.getLogger(__name__)

# Models exposed in the UI dropdown for the Claude backend.
CLAUDE_VLM_MODELS: dict[str, dict] = {
    "claude-sonnet-4-6": {
        "label": "Claude Sonnet 4.6 (recommended)",
    },
    "claude-opus-4-7": {
        "label": "Claude Opus 4.7",
    },
}


class ClaudeAgentBackend:
    """VLM backend that calls Claude via ``claude-agent-sdk``.

    Stateless across calls — each ``generate()`` spawns a fresh ``claude``
    subprocess via the SDK, mirroring the no-cross-call-memory semantics of
    the llama-server backend.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        use_thinking: bool = False,
    ) -> None:
        self._model = model
        self._use_thinking = use_thinking

    # -- API compatible with AnnotationAgentLLMService.generate() ----------

    def generate(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream_callback: Optional[Callable[[str], None]] = None,
        reasoning_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Send prompt (+ optional images) to Claude via claude-agent-sdk.

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

        # Images are passed as on-disk PNG paths because claude-agent-sdk's
        # `prompt=str` form is the documented, stable shape. Claude reads
        # referenced files natively with its vision capability.
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
        # Stateless — no subprocess to keep alive between calls.
        return True

    # -- internals ---------------------------------------------------------

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

        # No permission_mode: with allowed_tools=[] the CLI has no tools to
        # prompt for, so bypassPermissions is unnecessary — and the
        # `--dangerously-skip-permissions` flag it maps to is rejected when
        # the process runs as root (e.g. inside our dev container).
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
        """Best-effort extraction of (text, thinking) from an SDK message.

        Handles both the typed ``AssistantMessage`` shape and the raw dict
        shape; ignores ``ResultMessage``, ``SystemMessage``, etc.
        """
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


# ---------------------------------------------------------------------------
# Factory (mirrors get_or_start_service in annotation_agent_llm_service.py)
# ---------------------------------------------------------------------------

def get_or_start_claude_backend(
    model: str = "claude-sonnet-4-6",
    use_thinking: bool = False,
) -> ClaudeAgentBackend:
    """Return a Claude backend configured for the requested model.

    The backend is cheap to construct (no subprocess held), so no caching is
    needed — every call returns a fresh instance.
    """
    return ClaudeAgentBackend(model=model, use_thinking=use_thinking)
