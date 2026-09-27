import unittest
from unittest.mock import MagicMock, call, patch

import psutil

from training.image_segmentation import processes


class LabelmeProcessTests(unittest.TestCase):
    def setUp(self):
        self.current = MagicMock(pid=100)
        self.current.username.return_value = "driver"
        self.current.parents.return_value = [MagicMock(pid=99)]
        self.patchers = [
            patch.object(processes.psutil, "Process", return_value=self.current),
            patch.object(processes.psutil, "process_iter", return_value=[]),
            patch.object(processes.psutil, "wait_procs", return_value=([], [])),
        ]
        _, self.scan, self.wait = [patcher.start() for patcher in self.patchers]
        for patcher in self.patchers:
            self.addCleanup(patcher.stop)

    def editor(self, pid=200, command=None, username="driver"):
        process = MagicMock(pid=pid)
        process.info = {
            "pid": pid, "username": username,
            "cmdline": command or ["python3", "/app/training/image_segmentation/labelme_editor.py"],
        }
        process.parent.return_value = None
        return process

    def test_no_existing_editor(self):
        processes.stop_existing_labelme()
        self.wait.assert_has_calls([call([], timeout=5), call([], timeout=5), call([], timeout=20)])

    def test_only_labelme_editors_owned_by_current_user_are_stopped(self):
        editors = [self.editor(pid=200 + index, command=command) for index, command in enumerate([
            ["python3", "/app/training/image_segmentation/labelme_editor.py"],
            ["/usr/bin/python3.11", "-m", "labelme"],
            ["/usr/bin/python", "/usr/bin/labelme"],
            ["/usr/bin/labelme"],
            ["labelme.exe"],
        ])]
        untouched = [
            self.editor(pid=100), self.editor(pid=99), self.editor(username="someone_else"),
            self.editor(command=["python3", "train_labelme.py"]),
            self.editor(command=["python3", "open_labelme.py"]),
            self.editor(command=["python3", "worker.py", "labelme_editor.py"]),
            self.editor(command=["sh", "-c", "python -m labelme"]),
        ]
        unknown = self.editor()
        unknown.info["cmdline"] = None
        untouched.append(unknown)
        self.scan.return_value = editors + untouched

        processes.stop_existing_labelme()

        for process in editors:
            process.terminate.assert_called_once_with()
            process.kill.assert_not_called()
        for process in untouched:
            process.terminate.assert_not_called()

    def test_waits_for_browser_launcher_cleanup_after_editor_exit(self):
        for command in (["python", "/app/scripts/open_labelme.py"],
                        ["python", "-m", "training.image_segmentation", "annotate"]):
            with self.subTest(command=command):
                editor = self.editor()
                launcher = MagicMock(pid=201)
                launcher.cmdline.return_value = command
                editor.parent.return_value = launcher
                self.scan.return_value = [editor]
                events = MagicMock()
                events.attach_mock(editor.terminate, "terminate")
                events.attach_mock(self.wait, "wait")

                processes.stop_existing_labelme()

                self.assertEqual(events.mock_calls, [
                    call.terminate(), call.wait([editor], timeout=5),
                    call.wait([], timeout=5), call.wait([launcher], timeout=20),
                ])
                launcher.terminate.assert_not_called()

    def test_does_not_wait_for_unrelated_parent(self):
        editor = self.editor()
        editor.parent.return_value = MagicMock(pid=201)
        editor.parent.return_value.cmdline.return_value = ["bash"]
        self.scan.return_value = [editor]

        processes.stop_existing_labelme()

        self.wait.assert_called_with([], timeout=20)

    def test_stubborn_editor_is_killed_and_waited_for(self):
        editor = self.editor()
        self.scan.return_value = [editor]
        self.wait.side_effect = [([], [editor]), ([editor], []), ([], [])]

        processes.stop_existing_labelme()

        editor.kill.assert_called_once_with()
        self.assertEqual(self.wait.call_args_list[:2], [
            call([editor], timeout=5), call([editor], timeout=5),
        ])

    def test_editor_exiting_during_discovery_is_harmless(self):
        editor = self.editor()
        editor.parent.side_effect = psutil.NoSuchProcess(editor.pid)
        editor.terminate.side_effect = psutil.NoSuchProcess(editor.pid)
        self.scan.return_value = [editor]

        processes.stop_existing_labelme()

        editor.kill.assert_not_called()

    def test_cleanup_timeout_prevents_replacement(self):
        editor = self.editor()
        self.scan.return_value = [editor]
        self.wait.side_effect = [([editor], []), ([], []), ([], [MagicMock(pid=201)])]

        with self.assertRaisesRegex(RuntimeError, "launcher did not finish cleaning up"):
            processes.stop_existing_labelme()

    def test_editor_that_survives_kill_prevents_replacement(self):
        editor = self.editor()
        self.scan.return_value = [editor]
        self.wait.return_value = ([], [editor])

        with self.assertRaisesRegex(RuntimeError, "editor did not stop"):
            processes.stop_existing_labelme()
