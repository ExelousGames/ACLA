import json
import os
import sys
import unittest
from dataclasses import dataclass
from enum import Enum
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ACCCheckAvailableSession import ACCSessionChecker
from util.streaming import StreamingEvent


class Status(Enum):
    OFF = 0
    LIVE = 2


@dataclass
class NativeGraphics:
    status: Status
    completed_lap: int = 1


@dataclass
class NativeSnapshot:
    Graphics: NativeGraphics


class ACCSessionCheckerTests(unittest.TestCase):
    def setUp(self):
        self.checker = ACCSessionChecker()
        self.checker.asm = Mock()
        self.checker.asm.read_shared_memory.return_value = NativeSnapshot(NativeGraphics(Status.LIVE))
        self.checker.emit_update = Mock()

    def test_poll_emits_canonical_status_without_native_objects_or_control_fields(self):
        self.checker.poll()
        self.checker.emit_update.assert_called_once_with({'Graphics_status': 2})

    def test_requested_update_emits_canonical_status_and_preserves_request_id(self):
        self.checker.handle_event(StreamingEvent('request_update', {}, 'request-1'))
        self.checker.emit_update.assert_called_once_with({'Graphics_status': 2}, request_id='request-1')

    def test_status_changes_emit_updates_but_unrelated_telemetry_changes_do_not(self):
        self.checker.poll()
        self.checker.asm.read_shared_memory.return_value.Graphics.completed_lap = 2
        self.checker.poll()
        self.checker.emit_update.assert_called_once_with({'Graphics_status': 2})

        self.checker.asm.read_shared_memory.return_value.Graphics.status = Status.OFF
        self.checker.poll()
        self.checker.emit_update.assert_called_with({'Graphics_status': 0})
        self.assertEqual(self.checker.emit_update.call_count, 2)

    def test_offline_status_is_preserved_as_zero(self):
        result = self.checker._serialize_snapshot(NativeSnapshot(NativeGraphics(Status.OFF)))
        self.assertEqual(json.loads(result), {'Graphics_status': 0})

    def test_absent_shared_memory_remains_a_separate_control_update(self):
        self.checker.asm.read_shared_memory.return_value = None
        self.checker.poll()
        self.checker.emit_update.assert_called_once_with({
            'available': False, 'checking': True, 'message': 'Checking for live ACC session',
        }, request_id=None)


if __name__ == '__main__':
    unittest.main()
