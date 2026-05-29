"""Llama-server startup — resolve the chat GGUF and launch the sidecar.

Split out of the old ``app/main.py`` so the FastAPI startup (``startup/app.py``)
and the llama-server boot live in separate modules.
"""

from __future__ import annotations

from app.infra.config import settings
from app.llama.chat_model import ensure_chat_gguf, ensure_draft_gguf
from app.llama.process import LlamaServerConfig, LlamaServerProcess


def start_chat_sidecar() -> LlamaServerProcess:
    """Resolve the chat GGUF (downloading on first boot) and launch llama-server."""
    gguf_path = ensure_chat_gguf(
        model_dir=settings.llama_model_dir,
        model_repo=settings.llama_model_repo,
        model_file=settings.llama_model_file,
        hf_token=settings.hf_token,
    )

    draft_path = None
    if settings.llama_speculative_enabled:
        try:
            draft_path = ensure_draft_gguf(
                model_dir=settings.llama_model_dir,
                model_repo=settings.llama_draft_model_repo,
                model_file=settings.llama_draft_model_file,
                hf_token=settings.hf_token,
            )
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            print(f"⚠️  Draft GGUF resolve failed ({exc}); starting without speculative decoding.")

    sidecar = LlamaServerProcess(
        LlamaServerConfig(
            model_path=gguf_path,
            host=settings.llama_host,
            port=settings.llama_port,
            n_ctx=settings.llama_n_ctx,
            n_gpu_layers=settings.llama_n_gpu_layers,
            jinja=True,
            startup_timeout_seconds=settings.llama_startup_timeout_seconds,
            draft_model_path=draft_path,
            draft_n_gpu_layers=settings.llama_draft_n_gpu_layers,
            draft_max=settings.llama_draft_max,
            draft_min=settings.llama_draft_min,
        )
    )
    sidecar.start_or_attach()
    return sidecar
