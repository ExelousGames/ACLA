"""
Local VLM backend — VLM-specific HTTP wrapper around a llama-server.

Uses llama.cpp's OpenAI-compatible ``/v1/chat/completions`` endpoint for
multimodal (text + image) inference. Mirrors the ``generate()`` contract
of the Claude backend so the runner can swap them with one config flag.

Process lifecycle (spawn/health/stop) is delegated to
``app.llama.process.LlamaServerProcess`` — the single owner of any
``llama-server`` process in the service.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import requests

from app.llama.process import LlamaServerConfig, LlamaServerProcess

LOGGER = logging.getLogger(__name__)

# Default model directory — climb back to project root.
# __file__ = app/local_annotation_agent/backend.py → parents[2] is the project root.
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONVERT_SCRIPT = Path("/opt/llama.cpp/convert_hf_to_gguf.py")
_QUANTIZE_BIN = Path("/opt/llama.cpp/build/bin/llama-quantize")

QWEN25_VL_MODELS: dict[str, dict] = {
    "Qwen/Qwen2.5-VL-72B-Instruct": {
        "label": "Qwen2.5-VL-72B",
        "max_context": 131072,
        "max_new_tokens": 8192,
    },
}


@dataclass
class LocalVLMConfig:
    """Configuration for the llama-server backed VLM service."""

    gguf_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    context_size: int = 32768
    n_gpu_layers: int = -1
    host: str = "127.0.0.1"
    port: Optional[int] = None
    flash_attention: bool = True
    startup_timeout: int = 180
    hf_repo: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    quantization_type: str = "Q4_K_M"
    hf_token: Optional[str] = None
    auto_download: bool = True


# Backwards-compatible alias — the old name in case anything else still
# references it during the transition. Remove once the migration is done.
AnnotationAgentLLMConfig = LocalVLMConfig


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class LocalVLMService:
    """VLM-specific HTTP wrapper around a llama-server process.

    Process lifecycle is owned by ``LlamaServerProcess``; this class adds
    OpenAI-format chat-completion, tool-call loop, and streaming helpers
    on top.
    """

    def __init__(self, config: LocalVLMConfig) -> None:
        self._config = config
        self._proc: Optional[LlamaServerProcess] = None

    @property
    def _base_url(self) -> Optional[str]:
        return self._proc.base_url if self._proc is not None and self._proc.is_running() else None

    def start(self) -> None:
        if self.is_running():
            LOGGER.info("llama-server already running at %s", self._base_url)
            return

        gguf_path = self._resolve_gguf()
        mmproj_path = self._resolve_mmproj()

        self._proc = LlamaServerProcess(
            LlamaServerConfig(
                model_path=gguf_path,
                mmproj_path=mmproj_path,
                host=self._config.host,
                port=self._config.port,
                n_ctx=self._config.context_size,
                n_gpu_layers=self._config.n_gpu_layers,
                flash_attention=self._config.flash_attention,
                startup_timeout_seconds=self._config.startup_timeout,
            )
        )
        self._proc.start_or_attach()

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.stop()
        self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.is_running()

    def __enter__(self) -> "LocalVLMService":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def generate(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream_callback: Optional[Callable[[str], None]] = None,
        reasoning_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Send a chat-completion request to the running llama-server."""
        if not self.is_running():
            raise RuntimeError("llama-server is not running — call start() first.")

        content: list[dict] = []
        if images:
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        content.append({"type": "text", "text": prompt})

        payload = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            if stream_callback is not None:
                return self._generate_streaming(
                    payload, stream_callback, reasoning_callback,
                )

            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            answer = (message.get("content") or "").strip()
            reasoning = (message.get("reasoning_content") or "").strip()
            if not answer and reasoning:
                LOGGER.warning(
                    "llama-server returned only reasoning_content (%d chars) "
                    "with empty content — the model likely hit max_tokens "
                    "during its thinking phase.  Increase max_new_tokens.",
                    len(reasoning),
                )
            return answer
        except requests.RequestException as exc:
            raise RuntimeError(f"llama-server chat completion failed: {exc}") from exc

    def chat_with_tools(
        self,
        prompt: str,
        tools: List[dict],
        tool_handler: Callable[[str, dict], str],
        images: Optional[List[bytes]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        max_rounds: int = 3,
        stream_callback: Optional[Callable[[str], None]] = None,
        reasoning_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Run a chat-completion loop that lets the VLM emit tool calls.

        ``tools`` follows the OpenAI tool schema
        (``[{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}]``).
        ``tool_handler`` receives ``(name, parsed_args_dict)`` and returns
        a string the loop sends back as the tool result. Loop terminates
        when the model returns no tool calls OR after ``max_rounds``
        tool-call turns — final assistant text content is returned.

        Streaming is disabled inside this loop; llama-server's tool-call
        deltas don't surface a stable streaming format. The
        ``stream_callback`` fires on the final assistant content only.
        """
        if not self.is_running():
            raise RuntimeError("llama-server is not running — call start() first.")

        user_content: list[dict] = []
        if images:
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode("ascii")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        user_content.append({"type": "text", "text": prompt})

        messages: list[dict] = [{"role": "user", "content": user_content}]

        for round_idx in range(max_rounds + 1):
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "tools": tools,
            }
            try:
                resp = requests.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"llama-server chat completion failed: {exc}"
                ) from exc

            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            content = (message.get("content") or "").strip()
            reasoning = (message.get("reasoning_content") or "").strip()
            if reasoning and reasoning_callback is not None:
                reasoning_callback(reasoning)

            if not tool_calls:
                if content and stream_callback is not None:
                    stream_callback(content)
                if not content and reasoning:
                    LOGGER.warning(
                        "llama-server tool-call loop: empty content with "
                        "reasoning (%d chars) — likely hit max_tokens.",
                        len(reasoning),
                    )
                return content

            if round_idx >= max_rounds:
                LOGGER.warning(
                    "llama-server tool-call loop: reached max_rounds=%d "
                    "with tool calls still pending — returning content '%s'",
                    max_rounds, content[:120],
                )
                if content and stream_callback is not None:
                    stream_callback(content)
                return content

            # Echo the assistant turn (with tool_calls) so the next round
            # can carry the matching tool-result messages.
            messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                fn = call.get("function") or {}
                name = str(fn.get("name") or "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                try:
                    result = tool_handler(name, args)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("tool handler '%s' raised", name)
                    result = json.dumps({"error": str(exc)})
                if not isinstance(result, str):
                    result = json.dumps(result, default=str)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "content": result,
                })

        return ""

    def _generate_streaming(
        self,
        payload: dict,
        stream_callback: Callable[[str], None],
        reasoning_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        payload = {**payload, "stream": True}
        answer_text = ""
        reasoning_text = ""
        with requests.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    answer_delta = delta.get("content") or ""
                    reasoning_delta = delta.get("reasoning_content") or ""
                    if answer_delta:
                        answer_text += answer_delta
                        stream_callback(answer_delta)
                    if reasoning_delta:
                        reasoning_text += reasoning_delta
                        if reasoning_callback is not None:
                            reasoning_callback(reasoning_delta)
                        else:
                            stream_callback(reasoning_delta)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        if not answer_text and reasoning_text:
            LOGGER.warning(
                "llama-server stream returned only reasoning_content (%d "
                "chars) with empty content — the model likely hit "
                "max_tokens during its thinking phase.  Increase "
                "max_new_tokens.",
                len(reasoning_text),
            )
        return answer_text.strip()

    # -- model file resolution / HF download ------------------------------

    def _resolve_gguf(self) -> Path:
        if self._config.gguf_path:
            p = Path(self._config.gguf_path)
            if p.is_file():
                return p
            raise FileNotFoundError(f"GGUF model file not found: {p}")

        if self._config.hf_repo:
            model_name = self._config.hf_repo.split("/")[-1]
            quant = self._config.quantization_type
            specific = _MODELS_DIR / f"{model_name}-{quant}.gguf"
            if specific.is_file():
                LOGGER.info("Found GGUF for %s: %s", self._config.hf_repo, specific)
                return specific
            if self._config.auto_download:
                gguf, _ = self._convert_from_hf()
                return gguf
            raise FileNotFoundError(
                f"GGUF for {self._config.hf_repo} not found at {specific}.  "
                "Enable auto_download or convert manually."
            )

        candidates = sorted(_MODELS_DIR.glob("*.gguf"))
        for c in candidates:
            name_lower = c.name.lower()
            if "vl" in name_lower and "mmproj" not in name_lower:
                LOGGER.info("Auto-detected GGUF model: %s", c)
                return c

        raise FileNotFoundError(
            f"No VL GGUF model file found in {_MODELS_DIR}.  Set hf_repo "
            "to a Qwen2.5-VL repo and enable auto_download, or place a "
            "pre-converted GGUF file there manually."
        )

    def _resolve_mmproj(self) -> Optional[Path]:
        if self._config.mmproj_path:
            p = Path(self._config.mmproj_path)
            if p.is_file():
                return p
            raise FileNotFoundError(f"mmproj file not found: {p}")

        if self._config.hf_repo:
            model_name = self._config.hf_repo.split("/")[-1]
            specific = _MODELS_DIR / f"mmproj-{model_name}-f16.gguf"
            if specific.is_file():
                LOGGER.info("Found mmproj for %s: %s", self._config.hf_repo, specific)
                return specific
            if self._config.auto_download:
                _, mmproj = self._convert_from_hf()
                return mmproj
            raise FileNotFoundError(
                f"mmproj for {self._config.hf_repo} not found at {specific}.  "
                "Enable auto_download or convert manually."
            )

        candidates = sorted(_MODELS_DIR.glob("*mmproj*.gguf"))
        if candidates:
            LOGGER.info("Auto-detected mmproj: %s", candidates[0])
            return candidates[0]

        raise FileNotFoundError(
            f"No mmproj file found in {_MODELS_DIR}.  Set hf_repo "
            "to a Qwen2.5-VL repo and enable auto_download, or place a "
            "pre-converted mmproj GGUF file there manually."
        )

    def _convert_from_hf(self) -> tuple[Path, Path]:
        import os
        import shutil
        from huggingface_hub import snapshot_download

        repo = self._config.hf_repo
        model_name = repo.split("/")[-1]
        quant = self._config.quantization_type
        token = self._config.hf_token or os.environ.get("HF_TOKEN")

        final_gguf = _MODELS_DIR / f"{model_name}-{quant}.gguf"
        mmproj_gguf = _MODELS_DIR / f"mmproj-{model_name}-f16.gguf"

        if final_gguf.is_file() and mmproj_gguf.is_file():
            LOGGER.info("Using cached GGUF files: %s, %s", final_gguf, mmproj_gguf)
            return final_gguf, mmproj_gguf

        if not _CONVERT_SCRIPT.is_file():
            raise FileNotFoundError(
                f"llama.cpp convert script not found at {_CONVERT_SCRIPT}.  "
                "Rebuild the Docker image or install llama.cpp manually."
            )
        quantize_bin = _QUANTIZE_BIN
        if not quantize_bin.is_file():
            import shutil as _sh
            found = _sh.which("llama-quantize")
            if found:
                quantize_bin = Path(found)
            else:
                raise FileNotFoundError(
                    "llama-quantize binary not found.  Rebuild the Docker "
                    "image with the updated Dockerfile."
                )

        hf_cache = _MODELS_DIR / "huggingface_cache"
        local_dir = hf_cache / repo.replace("/", "--")
        LOGGER.info("Downloading %s from HuggingFace Hub …", repo)
        model_dir = snapshot_download(
            repo_id=repo,
            local_dir=str(local_dir),
            token=token,
        )
        model_dir = Path(model_dir)
        LOGGER.info("HF model downloaded to %s", model_dir)

        f16_gguf = _MODELS_DIR / f"{model_name}-f16.gguf"
        if not final_gguf.is_file():
            if not f16_gguf.is_file():
                LOGGER.info("Converting %s → f16 GGUF …", model_name)
                subprocess.run(
                    [
                        sys.executable, str(_CONVERT_SCRIPT),
                        str(model_dir),
                        "--outfile", str(f16_gguf),
                        "--outtype", "f16",
                    ],
                    check=True,
                )
                LOGGER.info("f16 GGUF created: %s", f16_gguf)

            LOGGER.info("Quantizing %s → %s …", f16_gguf.name, quant)
            subprocess.run(
                [str(quantize_bin), str(f16_gguf), str(final_gguf), quant],
                check=True,
            )
            LOGGER.info("Quantized GGUF created: %s", final_gguf)

            f16_gguf.unlink(missing_ok=True)

        if not mmproj_gguf.is_file():
            LOGGER.info("Extracting mmproj from %s …", model_name)
            subprocess.run(
                [
                    sys.executable, str(_CONVERT_SCRIPT),
                    str(model_dir),
                    "--mmproj",
                    "--outfile", str(mmproj_gguf),
                    "--outtype", "f16",
                ],
                check=True,
            )
            LOGGER.info("mmproj GGUF created: %s", mmproj_gguf)

        if local_dir.is_dir():
            LOGGER.info("Cleaning up HF download cache at %s …", local_dir)
            shutil.rmtree(local_dir, ignore_errors=True)

        return final_gguf, mmproj_gguf


# Backwards-compatible alias.
AnnotationAgentLLMService = LocalVLMService


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service_instance: Optional[LocalVLMService] = None
_service_config: Optional[LocalVLMConfig] = None
_service_lock = threading.Lock()


def _configs_match(a: LocalVLMConfig, b: LocalVLMConfig) -> bool:
    return (
        a.hf_repo == b.hf_repo
        and a.quantization_type == b.quantization_type
        and a.gguf_path == b.gguf_path
        and a.mmproj_path == b.mmproj_path
        and a.context_size == b.context_size
        and a.n_gpu_layers == b.n_gpu_layers
    )


def get_or_start_service(
    config: Optional[LocalVLMConfig] = None,
) -> LocalVLMService:
    """Return the running service, starting/restarting it as needed."""
    global _service_instance, _service_config
    with _service_lock:
        cfg = config or LocalVLMConfig()

        if _service_instance is not None and _service_instance.is_running():
            if _service_config is not None and _configs_match(_service_config, cfg):
                return _service_instance
            LOGGER.info(
                "Model config changed (was %s, now %s) — restarting llama-server.",
                _service_config.hf_repo if _service_config else "unknown",
                cfg.hf_repo,
            )
            _service_instance.stop()
            _service_instance = None
            _service_config = None

        svc = LocalVLMService(cfg)
        svc.start()
        _service_instance = svc
        _service_config = cfg
        return svc


def stop_service() -> None:
    global _service_instance
    with _service_lock:
        if _service_instance is not None:
            _service_instance.stop()
            _service_instance = None
