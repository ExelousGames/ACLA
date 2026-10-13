"""Run Labelme on a virtual desktop exposed through noVNC in the training container."""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


NOVNC_DIR = Path("/usr/share/novnc")


def _wait_for_port(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{process.args[0]} exited during startup; check the output above.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"{process.args[0]} did not start listening on port {port}.")


def _stop(process: subprocess.Popen) -> None:
    # Websockify forks connection handlers; stop its whole session as well.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


def run_browser(command: list[str]) -> int:
    missing = [name for name in ("Xtigervnc", "openbox") if shutil.which(name) is None]
    if importlib.util.find_spec("websockify") is None:
        missing.append("websockify")
    if not (NOVNC_DIR / "vnc.html").is_file():
        missing.append("noVNC")
    if missing:
        raise RuntimeError(
            f"Missing browser annotation dependencies: {', '.join(missing)}. "
            "Rebuild the ai_training Docker image."
        )

    env = {**os.environ, "DISPLAY": ":99", "QT_QPA_PLATFORM": "xcb"}
    processes = []
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        signal.signal(signal.SIGTERM, _terminate)
        signal.signal(signal.SIGHUP, _terminate)
        display = subprocess.Popen([
            "Xtigervnc", ":99", "-geometry", "1600x900", "-depth", "24",
            "-rfbport", "5900", "-localhost", "yes", "-SecurityTypes", "None",
            "-nolisten", "tcp", "-desktop", "Labelme",
        ], env=env, start_new_session=True)
        processes.append(display)
        _wait_for_port(display, 5900)
        processes.append(subprocess.Popen(["openbox"], env=env, start_new_session=True))
        proxy = subprocess.Popen([
            sys.executable, "-m", "websockify", "--web", str(NOVNC_DIR),
            "0.0.0.0:6080", "127.0.0.1:5900",
        ], env=env, start_new_session=True)
        processes.append(proxy)
        _wait_for_port(proxy, 6080)
        editor = subprocess.Popen(command, env=env, start_new_session=True)
        processes.append(editor)
        print(
            "Open http://localhost:6080/vnc.html?autoconnect=true&resize=remote\n"
            "Choose Open Dir in Labelme to browse /app/storage. Press Ctrl+C to stop.",
            flush=True,
        )
        while True:
            for process in processes[:-1]:
                if process.poll() is not None:
                    raise RuntimeError(f"{process.args[0]} stopped; check the output above.")
            try:
                return editor.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        return 130
    finally:
        # A second Ctrl+C or terminal hangup must not strand the remaining desktop.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            for process in reversed(processes):
                _stop(process)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
