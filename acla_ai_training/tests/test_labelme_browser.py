from __future__ import annotations

import signal
import subprocess
from unittest.mock import MagicMock

import pytest

from training.image_segmentation import browser


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    (tmp_path / "vnc.html").touch()
    monkeypatch.setattr(browser, "NOVNC_DIR", tmp_path)
    monkeypatch.setattr(browser.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(browser.importlib.util, "find_spec", lambda name: object())
    processes = [
        MagicMock(pid=index + 100, args=[name])
        for index, name in enumerate(["Xtigervnc", "openbox", "websockify", "labelme"])
    ]
    for process in processes:
        process.poll.return_value = None
        process.wait.return_value = 0
    launch = MagicMock(side_effect=processes)
    monkeypatch.setattr(browser.subprocess, "Popen", launch)
    monkeypatch.setattr(browser, "_wait_for_port", MagicMock())
    monkeypatch.setattr(browser.os, "killpg", MagicMock())
    return processes, launch


def test_browser_uses_virtual_display_and_reaps_children_on_editor_exit(desktop, monkeypatch):
    processes, launch = desktop
    monkeypatch.setenv("DISPLAY", ":0")
    processes[-1].wait.return_value = 7
    previous_handler = signal.getsignal(signal.SIGTERM)

    assert browser.run_browser(["python", "-m", "labelme"]) == 7

    assert all(call.kwargs["env"]["DISPLAY"] == ":99" for call in launch.call_args_list)
    assert all(call.kwargs["start_new_session"] for call in launch.call_args_list)
    assert browser.os.environ["DISPLAY"] == ":0"
    assert [call.args for call in browser.os.killpg.call_args_list] == [
        (process.pid, signal.SIGTERM) for process in reversed(processes)
    ]
    assert all(process.wait.called for process in processes)
    assert signal.getsignal(signal.SIGTERM) == previous_handler


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit(143)])
def test_interruption_cleans_up_the_desktop(desktop, interruption):
    processes, _ = desktop
    processes[-1].wait.side_effect = [interruption, 0]

    if isinstance(interruption, SystemExit):
        with pytest.raises(SystemExit) as exc:
            browser.run_browser(["labelme"])
        assert exc.value.code == 143
    else:
        assert browser.run_browser(["labelme"]) == 130

    assert browser.os.killpg.call_count == 4
    assert all(process.wait.called for process in processes)


@pytest.mark.parametrize("shutdown_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_further_signals_cannot_interrupt_cleanup(desktop, shutdown_signal):
    processes, _ = desktop
    processes[-1].wait.side_effect = [KeyboardInterrupt, 0]
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    def interrupt_cleanup(pid, signum):
        handler = signal.getsignal(shutdown_signal)
        if handler != signal.SIG_IGN:
            assert callable(handler), "Shutdown would terminate the launcher without cleanup"
            handler(shutdown_signal, None)

    browser.os.killpg.side_effect = interrupt_cleanup
    try:
        result = browser.run_browser(["labelme"])
        assert all(signal.getsignal(signum) == handler for signum, handler in previous_handlers.items())
    except (KeyboardInterrupt, SystemExit):
        pytest.fail("A further signal interrupted desktop cleanup")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    assert result == 130
    assert browser.os.killpg.call_count == 4
    assert all(process.wait.called for process in processes)


def test_terminal_hangup_cleans_up_the_desktop(desktop):
    processes, _ = desktop
    previous_handler = signal.getsignal(signal.SIGHUP)

    def hangup(*args, **kwargs):
        processes[-1].wait.side_effect = None
        handler = signal.getsignal(signal.SIGHUP)
        assert callable(handler), "Terminal hangup would bypass desktop cleanup"
        handler(signal.SIGHUP, None)

    processes[-1].wait.side_effect = hangup
    with pytest.raises(SystemExit) as exc:
        browser.run_browser(["labelme"])

    assert exc.value.code == 128 + signal.SIGHUP
    assert browser.os.killpg.call_count == 4
    assert all(process.wait.called for process in processes)
    assert signal.getsignal(signal.SIGHUP) == previous_handler


def test_partial_startup_failure_stops_children_already_started(desktop):
    processes, launch = desktop
    browser._wait_for_port.side_effect = [None, RuntimeError("port unavailable")]

    with pytest.raises(RuntimeError, match="port unavailable"):
        browser.run_browser(["labelme"])

    assert launch.call_count == 3
    assert browser.os.killpg.call_count == 3
    assert all(process.wait.called for process in processes[:3])


def test_lost_browser_service_stops_the_editor(desktop):
    processes, _ = desktop
    processes[2].poll.return_value = 1

    with pytest.raises(RuntimeError, match="websockify stopped"):
        browser.run_browser(["labelme"])

    assert browser.os.killpg.call_count == 4


def test_missing_dependencies_fail_before_starting_processes(desktop, monkeypatch):
    _, launch = desktop
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="Xtigervnc, openbox.*Rebuild"):
        browser.run_browser(["labelme"])

    launch.assert_not_called()


def test_stubborn_process_is_killed_and_reaped(monkeypatch):
    kill = MagicMock()
    monkeypatch.setattr(browser.os, "killpg", kill)
    process = MagicMock(pid=100)
    process.wait.side_effect = [subprocess.TimeoutExpired("labelme", 5), 0]

    browser._stop(process)

    assert [call.args for call in kill.call_args_list] == [
        (100, signal.SIGTERM), (100, signal.SIGKILL),
    ]
    assert process.wait.call_count == 2


def test_port_wait_reports_an_early_process_exit():
    process = MagicMock(args=["Xtigervnc"])
    process.poll.return_value = 1

    with pytest.raises(RuntimeError, match="Xtigervnc exited during startup"):
        browser._wait_for_port(process, 5900)
