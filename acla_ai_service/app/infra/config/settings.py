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
    
    # OpenAI Configuration (legacy — being phased out in favor of local llama-server)
    openai_api_key: Optional[str] = None

    # Active chat backend selector. "llama" uses the local llama-server sidecar
    # (canonical, Phase 1+). "openai" reverts to the legacy gpt-4o path — only
    # useful as a rollback during Phase 1 rollout and requires OPENAI_API_KEY.
    llm_provider: str = "llama"

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
