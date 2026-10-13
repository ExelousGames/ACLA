"""Stop existing Labelme editors before replacing their desktop or browser session."""

from pathlib import Path

import psutil


def _is_editor(command: list[str]) -> bool:
    if not command:
        return False
    executable = Path(command[0]).name.lower()
    if executable in ("labelme", "labelme.exe"):
        return True
    return executable.startswith("python") and (
        command[1:3] == ["-m", "labelme"]
        or (len(command) > 1 and Path(command[1]).name in ("labelme", "labelme_editor.py"))
    )


def _is_launcher(command: list[str]) -> bool:
    return len(command) > 1 and (
        Path(command[1]).name == "open_labelme.py"
        or command[1:4] == ["-m", "training.image_segmentation", "annotate"]
    )


def stop_existing_labelme() -> None:
    current = psutil.Process()
    excluded = {current.pid, *(parent.pid for parent in current.parents())}
    username = current.username()
    editors = []
    launchers = {}
    for process in psutil.process_iter(["pid", "username", "cmdline"]):
        if process.pid in excluded or process.info["username"] != username:
            continue
        if not _is_editor(process.info["cmdline"] or []):
            continue
        editors.append(process)
        try:
            parent = process.parent()
            if parent and parent.pid not in excluded and _is_launcher(parent.cmdline()):
                launchers[parent.pid] = parent
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    for process in editors:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(editors, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    if alive:
        raise RuntimeError("The previous Labelme editor did not stop.")

    # The browser launcher releases its VNC display and ports after the editor exits.
    _, alive = psutil.wait_procs(list(launchers.values()), timeout=20)
    if alive:
        raise RuntimeError("The previous Labelme launcher did not finish cleaning up.")
