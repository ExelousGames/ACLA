"""Subprocess + log-tail infrastructure shared by every card on the Training tab.

Each "job" (`classifier`, `transformer`, `llm`, `runall`) gets its own marker
JSON under ``models/.training/<job>.json`` and its own per-run log file. Marker
files survive browser refreshes and process restarts, so the UI can recover the
status of an in-flight training even after a reload.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import streamlit as st


_AI_SERVICE_DIR = Path(__file__).resolve().parents[2]  # acla_ai_service/
_TRAINING_STATE_DIR = _AI_SERVICE_DIR / "models" / ".training"


def _marker_path(job: str) -> Path:
    return _TRAINING_STATE_DIR / f"{job}.json"


def read_marker(job: str) -> Optional[dict]:
    p = _marker_path(job)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_marker(job: str, payload: dict) -> None:
    _TRAINING_STATE_DIR.mkdir(parents=True, exist_ok=True)
    _marker_path(job).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_marker(job: str) -> None:
    try:
        _marker_path(job).unlink()
    except FileNotFoundError:
        pass


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tail_log(log_path: Path, max_lines: int = 80) -> str:
    if not log_path.exists():
        return "(log file not created yet)"
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 16_384))  # ~last 16 KB
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"(failed to read log: {exc})"
    return "\n".join(tail.splitlines()[-max_lines:])


def spawn(job: str, cmd: List[str], *, extra_info: Optional[dict] = None) -> dict:
    """Launch ``cmd`` in a new process group, redirect stdout+stderr to a log
    file, persist a marker for UI recovery. Returns the marker."""
    _TRAINING_STATE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _TRAINING_STATE_DIR / f"{job}_{timestamp}.log"

    log_fh = log_path.open("w", buffering=1, encoding="utf-8")
    log_fh.write(f"$ {' '.join(cmd)}\n")
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(_AI_SERVICE_DIR),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    marker = {
        "pid": proc.pid,
        "log_path": str(log_path),
        "started_at": datetime.now().isoformat(),
        "cmd": cmd,
        **(extra_info or {}),
    }
    write_marker(job, marker)
    return marker


def stop(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def render_card(
    job: str,
    *,
    title: str,
    description: str,
    render_start_form: Callable[[], None],
) -> None:
    """Renders one training card: either the start form (no marker) or the
    status/log tail (marker present). ``render_start_form`` must call
    :func:`spawn` and ``st.rerun()`` when the user submits."""
    st.subheader(title)
    if description:
        st.caption(description)

    marker = read_marker(job)
    if marker is None:
        render_start_form()
        return

    pid = int(marker["pid"])
    running = pid_alive(pid)
    log_path = Path(marker["log_path"])

    info_col, meta_col = st.columns([3, 2])
    if running:
        info_col.info(
            f"**Running** · PID `{pid}` · started {marker['started_at']}"
        )
    else:
        info_col.success(
            f"**Finished** · started {marker['started_at']} · last PID `{pid}`"
        )
    meta_col.code(" ".join(marker.get("cmd", [])), language="bash")

    bc = st.columns(3)
    if bc[0].button("🔄 Refresh", key=f"{job}_refresh", use_container_width=True):
        st.rerun()
    if running:
        if bc[1].button("⏹ Stop", key=f"{job}_stop", use_container_width=True):
            stop(pid)
            st.warning("SIGTERM sent. Click Refresh in a few seconds.")
    else:
        if bc[1].button(
            "🧹 Clear status", key=f"{job}_clear", use_container_width=True,
        ):
            clear_marker(job)
            st.rerun()
    if bc[2].button(
        "🆕 New run", key=f"{job}_new", use_container_width=True, disabled=running,
        help="Clear status and show the start form again.",
    ):
        clear_marker(job)
        st.rerun()

    st.markdown(f"**Log tail** — `{log_path}`")
    st.code(tail_log(log_path), language="text")
