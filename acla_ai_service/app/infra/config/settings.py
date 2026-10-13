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
    
    # Chat, voice, and guidance LLM selector. Format: '<provider>:<model>',
    # where provider is 'openai' or 'hosted'. All LLM inference is remote.
    chat_llm_model: str = "openai:gpt-5.5"
    chat_openai_api_key_env: str = "OPENAI_API_KEY"

    # Hosted LLM (OpenAI-compatible third-party endpoint). When CHAT_LLM_MODEL
    # starts with 'hosted:', all LLM pipelines use this endpoint.
    # Works with Groq, Cerebras, Together, Fireworks, OpenRouter, etc., by just
    # changing the base_url. HOSTED_LLM_API_KEY is then required.
    hosted_llm_base_url: Optional[str] = None   # e.g. https://api.groq.com/openai/v1
    hosted_llm_api_key: Optional[str] = None

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

    # Hybrid retrieval settings used by the external racing knowledge base.
    hybrid_fusion_mode: str = "relative_score"
    # How many candidates each sub-retriever pulls before fusion. Wider
    # than the final top_k so the fusion has overlap to work with.
    hybrid_candidate_pool: int = 20
    # AI annotation providers. This is intentionally separate from the
    # hosted Groq/chatbot settings above; annotation provider selection is
    # per-run in the Streamlit annotation UI.
    annotation_enabled_providers: Optional[str] = None
    annotation_openai_api_key_env: str = "OPENAI_API_KEY"
    annotation_openai_models: str = "gpt-4o,gpt-4.1"
    annotation_openai_default_model: str = "gpt-4o"
    annotation_openai_compatible_base_url: Optional[str] = None
    annotation_openai_compatible_api_key_env: str = "ANNOTATION_OPENAI_COMPATIBLE_API_KEY"
    annotation_openai_compatible_models: Optional[str] = None
    annotation_openai_compatible_default_model: Optional[str] = None

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
        extra = "ignore"


# Global settings instance
settings = Settings()
