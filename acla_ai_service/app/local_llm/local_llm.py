"""Local LLM inference utilities for telemetry guidance."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.infra.config.settings import settings

import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError as exc:  # pragma: no cover - guarding runtime deps
    raise ImportError(
        "transformers is required for LocalTelemetryLLM. Please install `transformers`."
    ) from exc

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - guarding runtime deps
    raise ImportError("peft is required to load LoRA adapters. Install `peft`.") from exc

LOGGER = logging.getLogger(__name__)

# Determine persistent model cache directory.
# __file__ = app/llm/local_llm.py → parents[2] is the project root (was
# parents[3] when the file lived at app/services/llm/local_llm_service.py,
# fixed in refactor/hexagonal-v2 Step 10).
_SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE = str(_SERVICE_ROOT / "models" / "huggingface_cache")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    base_model: str = "mistralai/Ministral-3-14B-Base-2512"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = DEFAULT_HF_CACHE
    gguf_file: Optional[str] = None

    #specify the name or directory path of a specific fine-tuned LoRA
    adapter: Optional[str] = None

    # transformers: The default local inference provider.
    # llama_cpp: Optimized inference for CPU-only or limited hardware.
    provider: str = "transformers"  

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    device_map: Union[str, Dict[str, Union[int, str]]] = "auto"
    max_memory: Optional[Dict[str, Union[int, str]]] = None
    offload_folder: Optional[str] = None
    offload_state_dict: bool = False
    low_cpu_mem_usage: bool = True
    bf16: bool = field(default_factory=lambda: torch.cuda.is_available())
    fp16: bool = False

@dataclass
class GenerationConfig:
    max_input_tokens: int = 2353642
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    do_sample: bool = True

@dataclass
class LocalLLMConfig:
    """Configuration options for local LLM inference."""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class GenerationRequest:
    """Payload for inference requests."""

    user_prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    model_id: Optional[str] = None
    api_token: Optional[str] = None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class LocalTelemetryLLM:
    """High-level wrapper for local LLM inference."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LocalTelemetryLLM, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[LocalLLMConfig] = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self.config = config or LocalLLMConfig()
        self.tokenizer = None
        self.processor = None
        self.model = None
        self._initialized = True

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _raise_missing_local_resource(self, resource_name: str, cause: Exception) -> None:
        """Raise a helpful error when required local files are missing or download fails."""
        
        hint = (
            f"Failed to load or automatic download failed for '{resource_name}'. "
            f"Original error: {cause}"
        )
        # We don't want to enforce manual downloads, transformers can do it
        # Try adjusting your model ID, HF_TOKEN or connection
        raise RuntimeError(hint) from cause

    def _resolve_llama_cpp_model_path(self, adapter_path: Optional[Path] = None) -> Path:
        """Resolve the GGUF file to load for llama.cpp inference."""

        configured_gguf = self.config.model.gguf_file
        if configured_gguf:
            configured_path = Path(configured_gguf)
            if configured_path.exists() and configured_path.is_file():
                resolved_path = configured_path.resolve()
                self.config.model.gguf_file = str(resolved_path)
                return resolved_path

            raise FileNotFoundError(
                "Configured GGUF file for llama_cpp inference does not exist: "
                f"{configured_path}"
            )

        candidate_paths: List[Path] = []

        if adapter_path is not None:
            adapter_path = Path(adapter_path)
            if adapter_path.is_file():
                candidate_paths.append(adapter_path)
            elif adapter_path.is_dir():
                matching_files = sorted(adapter_path.glob("*.gguf"))
                candidate_paths.extend(matching_files)

                adapter_named_gguf = adapter_path / f"{adapter_path.name}.gguf"
                if adapter_named_gguf.exists():
                    candidate_paths.insert(0, adapter_named_gguf)

        for candidate_path in candidate_paths:
            if candidate_path.exists() and candidate_path.is_file():
                resolved_path = candidate_path.resolve()
                self.config.model.gguf_file = str(resolved_path)
                return resolved_path

        searched_locations = [str(path) for path in candidate_paths] or ["<none>"]
        raise FileNotFoundError(
            "Unable to locate a GGUF file for llama_cpp inference. "
            f"Searched: {searched_locations}"
        )

    def _ensure_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return

        tokenizer_name = self.config.model.tokenizer_name or self.config.model.base_model
        LOGGER.info("Loading tokenizer %s", tokenizer_name)

        tokenizer_kwargs = {
            "cache_dir": self.config.model.cache_dir,
            "token": settings.hf_token,
            "trust_remote_code": self.config.model.trust_remote_code,
        }
        if self.config.model.gguf_file:
            tokenizer_kwargs["gguf_file"] = self.config.model.gguf_file

        # Try loading AutoProcessor first (recommended for multimodal models like Mistral 3)
        try:
            self.processor = AutoProcessor.from_pretrained(
                tokenizer_name,
                **tokenizer_kwargs,
            )
            if hasattr(self.processor, "tokenizer"):
                self.tokenizer = self.processor.tokenizer
                LOGGER.info("Loaded AutoProcessor and extracted tokenizer")
        except Exception as e:
            LOGGER.debug("AutoProcessor load failed or not applicable: %s", e)
            self.processor = None

        if self.tokenizer is not None:
             # Ensure padding settings are correct even if loaded via processor
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.padding_side != "right":
                self.tokenizer.padding_side = "right"
            return
        
        # Try loading fast tokenizer first, fall back to slow tokenizer if it fails
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
                **tokenizer_kwargs,
            )
        except OSError as os_error:
            self._raise_missing_local_resource(tokenizer_name, os_error)
        except Exception as e:
            msg = str(e)
            if "ModelWrapper" in msg:
                LOGGER.warning(
                    "Tokenization schema error detected. Your `transformers` version may be too old for this model (e.g. requires Tekken support)."
                )

            LOGGER.warning(
                "Failed to load fast tokenizer for %s: %s. Attempting slow tokenizer fallback.",
                tokenizer_name,
                msg
            )
            
            fallback_kwargs = {
                "use_fast": False,
                **tokenizer_kwargs,
            }

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    **fallback_kwargs,
                )
            except OSError as fallback_os_error:
                self._raise_missing_local_resource(tokenizer_name, fallback_os_error)
            except Exception as e2:
                LOGGER.error("Failed to load slow tokenizer as well: %s", str(e2))
                raise
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.padding_side != "right":
            self.tokenizer.padding_side = "right"

    def _load_model(self, adapter_path: Optional[Path] = None) -> Union[torch.nn.Module, Any]:
        if self.config.model.provider == "llama_cpp":
            gguf_path = self._resolve_llama_cpp_model_path(adapter_path)

            from app.llama.process import LlamaServerConfig, LlamaServerProcess

            n_gpu_layers = -1 if torch.cuda.is_available() else 0
            proc = LlamaServerProcess(
                LlamaServerConfig(
                    model_path=gguf_path,
                    n_ctx=self.config.generation.max_input_tokens,
                    n_gpu_layers=n_gpu_layers,
                    startup_timeout_seconds=60,
                )
            )
            proc.start_or_attach()

            # Stored for generate() — keep .llama_url/.llama_process names so
            # any existing callers keep working.
            self.llama_url = proc.base_url
            self.llama_process = proc
            return proc

        self._ensure_tokenizer()

        torch_dtype = torch.bfloat16 if self.config.model.bf16 else (
            torch.float16 if self.config.model.fp16 else torch.float32
        )

        LOGGER.info("Loading base model %s", self.config.model.base_model)
        load_kwargs: Dict[str, Any] = {
            "cache_dir": self.config.model.cache_dir,
            "dtype": None if (self.config.model.load_in_8bit or self.config.model.load_in_4bit) else torch_dtype,
            "device_map": self.config.model.device_map,
            "token": settings.hf_token,
            "trust_remote_code": self.config.model.trust_remote_code,
        }
        if self.config.model.gguf_file:
            load_kwargs["gguf_file"] = self.config.model.gguf_file

        if self.config.model.load_in_8bit or self.config.model.load_in_4bit:
            LOGGER.info("Configuring quantization: 8bit=%s, 4bit=%s", self.config.model.load_in_8bit, self.config.model.load_in_4bit)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.config.model.load_in_8bit,
                load_in_4bit=self.config.model.load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype if self.config.model.load_in_4bit else None
            )
            load_kwargs["quantization_config"] = quantization_config

        if self.config.model.max_memory:
            load_kwargs["max_memory"] = self.config.model.max_memory

        if self.config.model.offload_folder:
            offload_path = Path(self.config.model.offload_folder)
            offload_path.mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = str(offload_path)

        if self.config.model.offload_state_dict:
            load_kwargs["offload_state_dict"] = True

        if self.config.model.low_cpu_mem_usage is not None:
            load_kwargs["low_cpu_mem_usage"] = self.config.model.low_cpu_mem_usage

        try:
            LOGGER.info("Attempting to load model with AutoModelForCausalLM")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model,
                **load_kwargs,
            )
        except (OSError, ValueError, RuntimeError) as e:
            LOGGER.info("AutoModelForCausalLM failed (%s), falling back to AutoModelForImageTextToText", str(e))
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.config.model.base_model,
                    **load_kwargs,
                )
            except OSError as load_error:
                self._raise_missing_local_resource(self.config.model.base_model, load_error)

        if adapter_path:
            adapter_path = Path(adapter_path)
            LOGGER.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            return model

        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load_for_inference(self, adapter_path: Optional[Path] = None) -> None:
        """Load base model (and optional adapter) for inference."""
        if self.model is not None:
            # If adapter_path is provided, we might need to load it. 
            # For now, we assume if model is loaded, it's sufficient, 
            # or the user accepts the current state due to Singleton constraint.
            # In a full implementation, we would check if the adapter is attached.
            LOGGER.info("Model already loaded. Skipping reload.")
            if hasattr(self.model, "eval") and self.config.model.provider != "llama_cpp":
                self.model.eval()
            return

        self.model = self._load_model(adapter_path=adapter_path)
        if hasattr(self.model, "eval") and self.config.model.provider != "llama_cpp":
            self.model.eval()

    def generate(self, request: GenerationRequest) -> str:
        """Generate telemetry narrative using the loaded model."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_for_inference() first.")

        prompt = self._format_generation_prompt(
            user_prompt=request.user_prompt,
        )

        if self.config.model.provider == "llama_cpp":
            import requests
            generation_kwargs = {
                "prompt": prompt,
                "n_predict": request.max_new_tokens or self.config.generation.max_new_tokens,
                "temperature": request.temperature or self.config.generation.temperature,
                "top_p": request.top_p or self.config.generation.top_p,
            }
            
            try:
                response = requests.post(
                    f"{self.llama_url}/completion",
                    json=generation_kwargs,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                return data.get("content", "").strip()
            except requests.RequestException as e:
                raise RuntimeError(f"Native llama-server generation failed: {e}")

        self._ensure_tokenizer()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.generation.max_input_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": request.max_new_tokens or self.config.generation.max_new_tokens,
            "max_length": None,
            "temperature": request.temperature or self.config.generation.temperature,
            "top_p": request.top_p or self.config.generation.top_p,
            "do_sample": request.do_sample if request.do_sample is not None else self.config.generation.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    @staticmethod
    def _format_generation_prompt(user_prompt: str) -> str:
        user_block = f"[USER]\n{user_prompt}\n[/USER]\n\n"
        assistant_prefix = "[ASSISTANT]\n"
        return f"{user_block}{assistant_prefix}"


__all__ = [
    "LocalLLMConfig",
    "ModelConfig",
    "GenerationConfig",
    "GenerationRequest",
    "LocalTelemetryLLM",
]
