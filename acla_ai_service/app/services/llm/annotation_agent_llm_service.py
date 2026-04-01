"""
Dedicated llama-cpp VLM service for the annotation agent pipeline.

Manages its own ``llama-server`` subprocess using the OpenAI-compatible
``/v1/chat/completions`` endpoint for multimodal (text + image) inference.
Completely independent from :class:`LocalTelemetryLLM` — no bitsandbytes,
no HuggingFace transformers.

Typical usage::

    from app.services.llm.annotation_agent_llm_service import (
        get_or_start_service,
        AnnotationAgentLLMConfig,
    )

    config = AnnotationAgentLLMConfig(
        gguf_path="/app/models/Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf",
        mmproj_path="/app/models/mmproj-Qwen2.5-VL-72B-Instruct-f16.gguf",
    )
    service = get_or_start_service(config)
    answer = service.generate("Describe the telemetry graph.", images=[png_bytes])
"""

from __future__ import annotations

import base64
import logging
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

LOGGER = logging.getLogger(__name__)

# Default model directory (inside the container / dev workspace)
_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths to llama.cpp tools (built in the Docker image)
_CONVERT_SCRIPT = Path("/opt/llama.cpp/convert_hf_to_gguf.py")
_QUANTIZE_BIN = Path("/opt/llama.cpp/build/bin/llama-quantize")

# Well-known Qwen2.5-VL sizes (repo → short name)
QWEN25_VL_MODELS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
    "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
}


@dataclass
class AnnotationAgentLLMConfig:
    """Configuration for the llama-server backed VLM service."""

    gguf_path: Optional[str] = None
    """Path to the main GGUF model file.  ``None`` = auto-detect in models/."""

    mmproj_path: Optional[str] = None
    """Path to the mmproj (clip vision projector) GGUF.  ``None`` = auto-detect."""

    context_size: int = 16384
    """Context window size.  Annotation prompts (guidelines + stats + graphs)
    are long — 16 k tokens is a safe default."""

    n_gpu_layers: int = -1
    """Number of layers to offload to GPU.  ``-1`` = all layers (recommended
    for AMD AI Max 395+ with 128 GB unified memory)."""

    host: str = "127.0.0.1"
    port: Optional[int] = None
    """TCP port for llama-server.  ``None`` = pick an ephemeral port."""

    flash_attention: bool = True
    """Enable flash-attention (``-fa`` flag) for better throughput on ROCm."""

    startup_timeout: int = 180
    """Seconds to wait for the llama-server ``/health`` endpoint after spawn."""

    # --- HuggingFace source + conversion -----------------------------------
    hf_repo: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    """HuggingFace Hub repo ID.  The model will be downloaded and converted
    to GGUF locally using llama.cpp tools."""

    quantization_type: str = "Q4_K_M"
    """GGUF quantization type applied after f16 conversion.
    Common choices: Q4_K_M, Q5_K_M, Q6_K, Q8_0."""

    hf_token: Optional[str] = None
    """HuggingFace API token.  ``None`` = read from ``HF_TOKEN`` env var
    or the token saved by ``huggingface-cli login``."""

    auto_download: bool = True
    """If ``True``, download and convert missing model files automatically."""


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class AnnotationAgentLLMService:
    """Manages a single ``llama-server`` process for VLM inference."""

    def __init__(self, config: AnnotationAgentLLMConfig) -> None:
        self._config = config
        self._process: Optional[subprocess.Popen] = None
        self._base_url: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn ``llama-server`` and block until the health-check passes."""
        if self.is_running():
            LOGGER.info("llama-server already running at %s", self._base_url)
            return

        gguf_path = self._resolve_gguf()
        mmproj_path = self._resolve_mmproj()
        port = self._config.port or _pick_free_port()

        cmd: list[str] = [
            "llama-server",
            "-m", str(gguf_path),
            "-c", str(self._config.context_size),
            "--port", str(port),
            "-ngl", str(self._config.n_gpu_layers),
            "--host", self._config.host,
        ]
        if mmproj_path:
            cmd += ["--mmproj", str(mmproj_path)]
        if self._config.flash_attention:
            cmd += ["-fa", "on"]

        LOGGER.info("Starting llama-server: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=sys.stderr,
        )
        self._base_url = f"http://{self._config.host}:{port}"

        # Poll /health until ready
        if not self._wait_for_health():
            self.stop()
            raise RuntimeError(
                f"llama-server failed to become healthy within "
                f"{self._config.startup_timeout}s.  Check logs above."
            )
        LOGGER.info("llama-server ready at %s", self._base_url)

    def stop(self) -> None:
        """Terminate the llama-server process gracefully."""
        if self._process is None:
            return
        LOGGER.info("Stopping llama-server (pid %d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            LOGGER.warning("llama-server did not exit, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None
        self._base_url = None

    def is_running(self) -> bool:
        """Return ``True`` if the subprocess is alive and /health returns 200."""
        if self._process is None or self._process.poll() is not None:
            return False
        try:
            r = requests.get(f"{self._base_url}/health", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "AnnotationAgentLLMService":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- inference ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat-completion request to the running llama-server.

        Parameters
        ----------
        prompt:
            The user-message text.
        images:
            Optional list of raw PNG image bytes.  Each image is sent as a
            base64-encoded ``image_url`` content part (OpenAI multimodal
            format).
        max_tokens:
            Maximum tokens to generate.
        temperature:
            Sampling temperature.

        Returns
        -------
        str
            The assistant's reply text.
        """
        if not self.is_running():
            raise RuntimeError("llama-server is not running — call start() first.")

        # Build multimodal content list
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
            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            raise RuntimeError(f"llama-server chat completion failed: {exc}") from exc

    # -- internal helpers ---------------------------------------------------

    def _resolve_gguf(self) -> Path:
        """Return an absolute path to the main GGUF file."""
        if self._config.gguf_path:
            p = Path(self._config.gguf_path)
            if p.is_file():
                return p
            raise FileNotFoundError(f"GGUF model file not found: {p}")

        # Auto-detect: look for a VL-related GGUF in the models directory
        candidates = sorted(_MODELS_DIR.glob("*.gguf"))
        for c in candidates:
            name_lower = c.name.lower()
            if "vl" in name_lower and "mmproj" not in name_lower:
                LOGGER.info("Auto-detected GGUF model: %s", c)
                return c

        # Convert from HuggingFace if enabled
        if self._config.auto_download:
            gguf, _ = self._convert_from_hf()
            return gguf

        raise FileNotFoundError(
            f"No VL GGUF model file found in {_MODELS_DIR}.  Set hf_repo "
            "to a Qwen2.5-VL repo and enable auto_download, or place a "
            "pre-converted GGUF file there manually."
        )

    def _resolve_mmproj(self) -> Optional[Path]:
        """Return an absolute path to the mmproj file, or ``None``."""
        if self._config.mmproj_path:
            p = Path(self._config.mmproj_path)
            if p.is_file():
                return p
            raise FileNotFoundError(f"mmproj file not found: {p}")

        # Auto-detect
        candidates = sorted(_MODELS_DIR.glob("*mmproj*.gguf"))
        if candidates:
            LOGGER.info("Auto-detected mmproj: %s", candidates[0])
            return candidates[0]

        # Convert from HuggingFace if enabled
        if self._config.auto_download:
            _, mmproj = self._convert_from_hf()
            return mmproj

        raise FileNotFoundError(
            f"No mmproj file found in {_MODELS_DIR}.  Set hf_repo "
            "to a Qwen2.5-VL repo and enable auto_download, or place a "
            "pre-converted mmproj GGUF file there manually."
        )

    # -- HuggingFace download & GGUF conversion -----------------------------

    def _convert_from_hf(self) -> tuple[Path, Path]:
        """Download a HuggingFace VL model and convert to GGUF + mmproj.

        Returns (main_gguf_path, mmproj_path).  Results are cached — if the
        target files already exist they are returned immediately.
        """
        import os
        import shutil
        from huggingface_hub import snapshot_download

        repo = self._config.hf_repo
        model_name = repo.split("/")[-1]
        quant = self._config.quantization_type
        token = self._config.hf_token or os.environ.get("HF_TOKEN")

        final_gguf = _MODELS_DIR / f"{model_name}-{quant}.gguf"
        mmproj_gguf = _MODELS_DIR / f"mmproj-{model_name}-f16.gguf"

        # Return cached files if both already exist
        if final_gguf.is_file() and mmproj_gguf.is_file():
            LOGGER.info("Using cached GGUF files: %s, %s", final_gguf, mmproj_gguf)
            return final_gguf, mmproj_gguf

        # -- Step 0: Validate tools -----------------------------------------
        if not _CONVERT_SCRIPT.is_file():
            raise FileNotFoundError(
                f"llama.cpp convert script not found at {_CONVERT_SCRIPT}.  "
                "Rebuild the Docker image or install llama.cpp manually."
            )
        quantize_bin = _QUANTIZE_BIN
        if not quantize_bin.is_file():
            # Try PATH fallback
            import shutil as _sh
            found = _sh.which("llama-quantize")
            if found:
                quantize_bin = Path(found)
            else:
                raise FileNotFoundError(
                    "llama-quantize binary not found.  Rebuild the Docker "
                    "image with the updated Dockerfile."
                )

        # -- Step 1: Download HF model (safetensors) -----------------------
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

        # -- Step 2: Convert main model to f16 GGUF ------------------------
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

            # -- Step 3: Quantize f16 → target quant -----------------------
            LOGGER.info("Quantizing %s → %s …", f16_gguf.name, quant)
            subprocess.run(
                [str(quantize_bin), str(f16_gguf), str(final_gguf), quant],
                check=True,
            )
            LOGGER.info("Quantized GGUF created: %s", final_gguf)

            # Remove f16 intermediate to reclaim disk space
            f16_gguf.unlink(missing_ok=True)

        # -- Step 4: Extract mmproj (vision projector) ---------------------
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

        # -- Step 5: Clean up HF download cache ----------------------------
        if local_dir.is_dir():
            LOGGER.info("Cleaning up HF download cache at %s …", local_dir)
            shutil.rmtree(local_dir, ignore_errors=True)

        return final_gguf, mmproj_gguf

    def _wait_for_health(self) -> bool:
        """Poll ``/health`` until 200 or timeout."""
        deadline = time.monotonic() + self._config.startup_timeout
        url = f"{self._base_url}/health"
        while time.monotonic() < deadline:
            # Check that the process hasn't exited unexpectedly
            if self._process and self._process.poll() is not None:
                LOGGER.error(
                    "llama-server exited with code %d during startup",
                    self._process.returncode,
                )
                return False
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        return False


# ---------------------------------------------------------------------------
# Module-level persistent holder
# ---------------------------------------------------------------------------

_service_instance: Optional[AnnotationAgentLLMService] = None
_service_lock = threading.Lock()


def get_or_start_service(
    config: Optional[AnnotationAgentLLMConfig] = None,
) -> AnnotationAgentLLMService:
    """Return the running service instance, starting it if necessary.

    The server stays alive across multiple ``run_annotation_pipeline()``
    calls so the (expensive) model load happens only once.
    """
    global _service_instance
    with _service_lock:
        if _service_instance is not None and _service_instance.is_running():
            return _service_instance

        cfg = config or AnnotationAgentLLMConfig()
        svc = AnnotationAgentLLMService(cfg)
        svc.start()
        _service_instance = svc
        return svc


def stop_service() -> None:
    """Stop the held llama-server instance and release resources."""
    global _service_instance
    with _service_lock:
        if _service_instance is not None:
            _service_instance.stop()
            _service_instance = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_free_port() -> int:
    """Bind to port 0, read the assigned port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
