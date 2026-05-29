"""Single owner of `llama-server` subprocess lifecycle.

All three llama-server users (chat sidecar, annotation VLM, telemetry-LoRA
inference) go through this class so spawning, health-waiting, port allocation,
and shutdown live in exactly one place.

It does NOT speak HTTP to the server — each caller keeps whatever client it
already uses (openai-async, requests, httpx). It just owns the process.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class LlamaServerConfig:
    """Knobs for one llama-server invocation."""

    model_path: Path
    mmproj_path: Optional[Path] = None
    host: str = "127.0.0.1"
    # None means "pick a free port". A fixed int is used as-is.
    port: Optional[int] = None
    n_ctx: int = 8192
    n_gpu_layers: int = 99
    flash_attention: bool = False
    jinja: bool = False
    startup_timeout_seconds: int = 300
    # Speculative decoding — draft_model_path None means disabled.
    draft_model_path: Optional[Path] = None
    draft_n_gpu_layers: int = 99
    draft_max: int = 16
    draft_min: int = 0
    extra_args: List[str] = field(default_factory=list)


class LlamaServerProcess:
    """Owns one llama-server process — start, attach-if-running, stop."""

    def __init__(self, config: LlamaServerConfig) -> None:
        self._config = config
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._owned: bool = False  # False when we attached to a pre-existing server

    # -- lifecycle --------------------------------------------------------

    def start_or_attach(self) -> None:
        """Spawn llama-server, OR attach silently if one is already healthy.

        Attach mode matters for two scenarios:
        1. uvicorn --reload kills the worker; the orphan llama-server keeps
           running. The new worker calls start_or_attach() and reuses it
           instead of trying (and failing) to bind the same port.
        2. Tests/scripts that launch their own llama-server externally.
        """
        target_port = self._config.port
        if target_port is not None and self._is_port_healthy(target_port):
            self._port = target_port
            self._owned = False
            LOGGER.info(
                "llama-server already healthy at %s:%d — attaching (not spawned)",
                self._config.host, target_port,
            )
            return

        self._spawn()

    def _spawn(self) -> None:
        port = self._config.port or _pick_free_port()
        cmd: List[str] = [
            "llama-server",
            "--model", str(self._config.model_path),
            "--host", self._config.host,
            "--port", str(port),
            "--n-gpu-layers", str(self._config.n_gpu_layers),
            "--ctx-size", str(self._config.n_ctx),
        ]
        if self._config.mmproj_path is not None:
            cmd += ["--mmproj", str(self._config.mmproj_path)]
        if self._config.flash_attention:
            cmd += ["-fa", "on"]
        if self._config.jinja:
            cmd += ["--jinja"]
        if self._config.draft_model_path is not None:
            # Speculative decoding flag names verified against llama.cpp
            # common/arg.cpp: -md / --model-draft, -ngld /
            # --n-gpu-layers-draft, --draft-max, --draft-min.
            cmd += [
                "-md", str(self._config.draft_model_path),
                "-ngld", str(self._config.draft_n_gpu_layers),
                "--draft-max", str(self._config.draft_max),
                "--draft-min", str(self._config.draft_min),
            ]
        cmd += list(self._config.extra_args)

        LOGGER.info("Starting llama-server: %s", " ".join(cmd))
        # start_new_session so the child survives parent reload (uvicorn).
        # On container shutdown the docker init kills the whole tree anyway.
        self._process = subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=sys.stderr,
            start_new_session=True,
        )
        self._port = port
        self._owned = True

        if not self._wait_for_health():
            self.stop()
            raise RuntimeError(
                f"llama-server failed to become healthy on {self._config.host}:{port} "
                f"within {self._config.startup_timeout_seconds}s. Check logs above."
            )
        LOGGER.info("llama-server ready at %s", self.base_url)

    def stop(self) -> None:
        if not self._owned or self._process is None:
            self._process = None
            self._port = None
            self._owned = False
            return
        LOGGER.info("Stopping llama-server (pid %d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            LOGGER.warning("llama-server did not exit on SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None
        self._port = None
        self._owned = False

    def is_running(self) -> bool:
        if self._port is None:
            return False
        if self._owned and self._process is not None and self._process.poll() is not None:
            return False
        return self._is_port_healthy(self._port)

    # -- accessors --------------------------------------------------------

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("llama-server has not been started")
        return self._port

    @property
    def base_url(self) -> str:
        """Root URL, e.g. http://127.0.0.1:8080 — no /v1 suffix."""
        return f"http://{self._config.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """OpenAI-style base URL, e.g. http://127.0.0.1:8080/v1."""
        return f"{self.base_url}/v1"

    # -- internals --------------------------------------------------------

    def _wait_for_health(self) -> bool:
        deadline = time.monotonic() + self._config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                LOGGER.error(
                    "llama-server exited with code %d during startup",
                    self._process.returncode,
                )
                return False
            if self._is_port_healthy(self._port or 0):
                return True
            time.sleep(2)
        return False

    def _is_port_healthy(self, port: int) -> bool:
        if port <= 0:
            return False
        try:
            r = requests.get(
                f"http://{self._config.host}:{port}/health", timeout=2,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


__all__ = ["LlamaServerConfig", "LlamaServerProcess"]
