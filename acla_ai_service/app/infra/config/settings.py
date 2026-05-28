"""
Configuration settings for ACLA AI Service
"""

import os
from typing import Optional, List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "ACLA AI Service"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API Configuration
    backend_server_ip: Optional[str] = None
    backend_proxy_port: Optional[str] = None
    

    # AI Service Authentication (for backend login)
    ai_service_username: Optional[str] = None
    ai_service_password: Optional[str] = None
    
    # Local LLM (llama-server / llama-cpp-python) Configuration
    # llama-server runs as a sidecar inside the ai_service container and exposes
    # an OpenAI-compatible HTTP API at this URL. The chat code calls it as if it
    # were OpenAI, just with a different base_url.
    llama_server_url: str = "http://127.0.0.1:8080/v1"
    # Default targets the racing-engineer's brain: Qwen2.5-32B-Instruct. Override
    # via LLAMA_MODEL_NAME / LLAMA_MODEL_REPO / LLAMA_MODEL_FILE env vars for
    # local iteration on a smaller GGUF.
    llama_model_name: str = "qwen2.5-32b-instruct"
    llama_model_repo: str = "Qwen/Qwen2.5-32B-Instruct-GGUF"
    llama_model_file: str = "qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf"
    llama_model_dir: str = "/app/models/llama_server"
    llama_host: str = "127.0.0.1"
    llama_port: int = 8080
    llama_n_ctx: int = 8192
    llama_n_gpu_layers: int = 99  # 0 disables GPU offload; 99 = offload all layers
    llama_health_timeout_seconds: float = 2.0
    # Seconds to wait for the chat sidecar to come up on first boot (model
    # download can take many minutes). Matches the LLAMA_WAIT_SECONDS default
    # used by the previous bash bootstrap.
    llama_startup_timeout_seconds: int = 300

    # Speculative decoding — uses a tiny draft model to predict the next
    # tokens, then the main model verifies them in parallel. Same output
    # distribution; reported 1.5-2.5x throughput gain when accept-rate is
    # good. Draft must share a tokenizer with the main model — Qwen2.5-0.5B
    # is the right pairing for Qwen2.5-32B.
    llama_speculative_enabled: bool = True
    llama_draft_model_repo: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    llama_draft_model_file: str = "qwen2.5-0.5b-instruct-q5_k_m.gguf"
    llama_draft_n_gpu_layers: int = 99
    llama_draft_max: int = 16
    llama_draft_min: int = 0

    # Hosted LLM (OpenAI-compatible third-party endpoint). When
    # HOSTED_LLM_BASE_URL is set, the chat + voice pipelines use it instead
    # of the local llama-server sidecar. Works with Groq, Cerebras, Together,
    # Fireworks, OpenRouter, etc., by just changing the base_url. If
    # HOSTED_LLM_BASE_URL is set, HOSTED_LLM_API_KEY and HOSTED_LLM_MODEL
    # are required — missing values fail loudly at startup.
    hosted_llm_base_url: Optional[str] = None   # e.g. https://api.groq.com/openai/v1
    hosted_llm_api_key: Optional[str] = None
    hosted_llm_model: Optional[str] = None      # e.g. qwen/qwen3-32b

    # Kokoro TTS Configuration (Phase 2)
    # Neural TTS that replaces window.speechSynthesis in the frontend.
    # Apache-2.0 ONNX model — downloaded on first run, persisted in a volume.
    # URLs match kokoro-onnx upstream's documented setup (examples/save.py):
    # https://github.com/thewh1teagle/kokoro-onnx#getting-started
    kokoro_model_dir: str = "/app/models/kokoro"
    kokoro_model_url: str = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/kokoro-v1.0.onnx"
    )
    kokoro_voices_url: str = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/voices-v1.0.bin"
    )
    kokoro_default_voice: str = "af_bella"
    kokoro_sample_rate: int = 24000

    # Racing-engineer knowledge base (RAG over knowledge/ + keyed tracks/).
    # Default = BAAI/bge-large-en-v1.5 — production-grade English retrieval,
    # 335M params / 1024-dim / ~1.3GB. Strong MTEB recall on prose, runs
    # comfortably on GPU. Swap down to bge-base-en-v1.5 (~400MB, 768-dim)
    # if RAM is tight, or up to a 7B-class embedder for marginal gains.
    racing_kb_embedding_model: str = "BAAI/bge-large-en-v1.5"
    # bge-en-v1.5 was trained with this query-side instruction; documents go
    # in unprefixed. Empty string disables the prefix (use for non-bge models).
    racing_kb_query_prefix: str = "Represent this sentence for searching relevant passages: "
    # Default top_k for search_racing_knowledge when the LLM doesn't specify.
    racing_kb_default_top_k: int = 5
    # Soft cap on chunk character length when splitting a long section.
    # 2000 chars ≈ 500 tokens for English prose — well under bge-base's
    # 512-token max.
    racing_kb_max_chunk_chars: int = 2000

    # Annotation skill registry (hybrid index over discovery headers).
    # Same bge-large default as the racing KB — the index is tiny (one
    # vector per skill) so the heavier model is paid for once at startup.
    annotation_skill_embedding_model: str = "BAAI/bge-large-en-v1.5"
    annotation_skill_query_prefix: str = "Represent this sentence for searching relevant passages: "
    # Hybrid retrieval — vector and BM25 sub-retrievers per registry,
    # combined by QueryFusionRetriever in `relative_score` mode (the
    # user-selected fusion strategy).
    hybrid_fusion_mode: str = "relative_score"
    # How many candidates each sub-retriever pulls before fusion. Wider
    # than the final top_k so the fusion has overlap to work with.
    hybrid_candidate_pool: int = 20

    # Hugging Face Configuration
    hf_token: Optional[str] = None
    hf_username: Optional[str] = None
    
    # CORS Configuration
    allowed_origins: List[str] = ["*"]
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
