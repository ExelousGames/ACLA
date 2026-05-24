"""Resolve the chat-sidecar GGUF model — download from HF on first boot.

Ported from scripts/start_llama_server.sh (the bash one this replaces).
Shard-pattern handling is preserved: if the configured filename matches
``<base>-NNNNN-of-NNNNN.gguf``, every sibling shard is fetched in one
``snapshot_download`` call so llama-server can auto-load them.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$")


def ensure_chat_gguf(
    *,
    model_dir: str,
    model_repo: str,
    model_file: str,
    hf_token: str | None = None,
) -> Path:
    """Return the local path to the chat GGUF, downloading it if missing."""
    target_dir = Path(model_dir)
    target_path = target_dir / model_file

    if target_path.is_file():
        return target_path

    target_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token or os.environ.get("HF_TOKEN")

    shard_match = _SHARD_RE.match(model_file)
    if shard_match:
        from huggingface_hub import snapshot_download
        pattern = f"{shard_match.group(1)}-*.gguf"
        LOGGER.info(
            "Chat GGUF missing — fetching sharded %s from %s into %s",
            pattern, model_repo, target_dir,
        )
        snapshot_download(
            repo_id=model_repo,
            allow_patterns=[pattern],
            local_dir=str(target_dir),
            token=token,
        )
    else:
        from huggingface_hub import hf_hub_download
        LOGGER.info(
            "Chat GGUF missing — fetching %s from %s into %s",
            model_file, model_repo, target_dir,
        )
        hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            local_dir=str(target_dir),
            token=token,
        )

    if not target_path.is_file():
        raise FileNotFoundError(
            f"HF download completed but {target_path} still missing. "
            f"Check that model_file matches what {model_repo} actually publishes."
        )
    return target_path


__all__ = ["ensure_chat_gguf"]
