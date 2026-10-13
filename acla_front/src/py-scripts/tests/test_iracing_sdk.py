import ctypes
from ctypes import wintypes
import mmap
import os
import struct
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from iracing_sdk import BUFFER, HEADER, VARIABLE, SnapshotReader, WindowsMemory, header_layout


def sdk_fixture():
    memory = bytearray(4096)
    variables = [
        (0, 0, 8, False, b'Name'),
        (1, 8, 1, False, b'IsOnTrack'),
        (2, 12, 1, False, b'Gear'),
        (3, 16, 1, False, b'SessionFlags'),
        (4, 20, 1, False, b'Speed'),
        (5, 24, 1, False, b'SessionTime'),
        (4, 32, 3, True, b'RPM'),
        (2, 44, 3, False, b'CarIdxTrackSurface'),
        (4, 56, 1, False, b'Invalid'),
    ]
    session = b'WeekendInfo:\n TrackName: caf\xe9\n\0'
    HEADER.pack_into(memory, 0, 2, 1, 60, 1, len(session), 2048, len(variables), 112, 2, 64)
    BUFFER.pack_into(memory, 48, 9, 3072)
    BUFFER.pack_into(memory, 64, 10, 3200)
    for index, (kind, offset, count, as_time, name) in enumerate(variables):
        VARIABLE.pack_into(memory, 112 + index * 144, kind, offset, count, as_time, name, b'description', b'unit')
    memory[2048:2048 + len(session)] = session
    memory[3200:3208] = b'car\0\0\0\0\0'
    struct.pack_into('<?3xiIfd3f3if', memory, 3208, True, -1, 0x80000000, 50.5, 12.125, 100, 200, 300, 3, -1, 1, float('nan'))
    return memory, [name.decode() for _, _, _, _, name in variables]


class FixtureMemory:
    def __init__(self, content):
        self.content = content

    def read(self, offset, length):
        if offset < 0 or offset + length > len(self.content):
            raise ValueError('Fixture read out of bounds')
        return bytes(self.content[offset:offset + length])


class SnapshotTests(unittest.TestCase):
    def test_all_sdk_types_newest_buffer_arrays_and_nonfinite_values(self):
        content, names = sdk_fixture()
        reader = SnapshotReader(FixtureMemory(content), names)
        packet = reader.snapshot()
        self.assertEqual(packet['tick'], 10)
        self.assertEqual(packet['values'], {
            'Name': 'car', 'IsOnTrack': True, 'Gear': -1, 'SessionFlags': 0x80000000,
            'Speed': 50.5, 'SessionTime': 12.125, 'RPM': 300,
            'CarIdxTrackSurface': [3, -1, 1],
        })
        self.assertIn('café', packet['sessionInfo'])
        self.assertIsNone(reader.snapshot())
        BUFFER.pack_into(content, 64, 11, 3200)
        self.assertNotIn('sessionInfo', reader.snapshot())
        struct.pack_into('<i', content, 12, 2)
        BUFFER.pack_into(content, 64, 12, 3200)
        self.assertIn('sessionInfo', reader.snapshot())

    def test_only_requested_variables_are_decoded(self):
        content, _ = sdk_fixture()
        self.assertEqual(SnapshotReader(FixtureMemory(content), ['Speed']).snapshot()['values'], {'Speed': 50.5})

    def test_rejects_torn_rows_and_retries_without_consuming_tick(self):
        content, names = sdk_fixture()

        class TornMemory(FixtureMemory):
            def read(self, offset, length):
                result = super().read(offset, length)
                if offset == 3200:
                    BUFFER.pack_into(self.content, 64, 11, 3200)
                return result

        reader = SnapshotReader(TornMemory(content), names)
        self.assertIsNone(reader.snapshot())
        self.assertIsNone(reader.last_tick)
        self.assertEqual(reader.snapshot()['tick'], 11)

    def test_rejects_metadata_updated_during_snapshot(self):
        content, names = sdk_fixture()

        class ChangingSession(FixtureMemory):
            def read(self, offset, length):
                result = super().read(offset, length)
                if offset == 2048:
                    struct.pack_into('<i', self.content, 12, 2)
                return result

        reader = SnapshotReader(ChangingSession(content), names)
        self.assertIsNone(reader.snapshot())
        self.assertEqual(reader.snapshot()['tick'], 10)

    def test_disconnect_and_tick_restart_do_not_reuse_schema(self):
        content, names = sdk_fixture()
        reader = SnapshotReader(FixtureMemory(content), names)
        reader.snapshot()
        BUFFER.pack_into(content, 48, 1, 3072)
        BUFFER.pack_into(content, 64, 2, 3200)
        self.assertEqual(reader.snapshot(), {'type': 'disconnected'})
        struct.pack_into('<i', content, 4, 0)
        self.assertEqual(reader.snapshot(), {'type': 'disconnected'})

    def test_rejects_invalid_header_and_variable_ranges(self):
        for offset, value in [(0, 3), (24, 5000), (32, 5), (36, -1), (20, -1)]:
            with self.subTest(offset=offset):
                content, _ = sdk_fixture()
                struct.pack_into('<i', content, offset, value)
                with self.assertRaises(ValueError):
                    header_layout(content)
        content, names = sdk_fixture()
        struct.pack_into('<i', content, 112 + 4, 4096)
        with self.assertRaises(ValueError):
            SnapshotReader(FixtureMemory(content), names).snapshot()

    @unittest.skipUnless(sys.platform == 'win32', 'Windows SDK integration')
    def test_real_windows_read_only_mapping_event_and_cleanup(self):
        content, names = sdk_fixture()
        unique = uuid.uuid4().hex
        map_name, event_name = 'Local\\ACLA-test-map-' + unique, 'Local\\ACLA-test-event-' + unique
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateEventW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        event = kernel.CreateEventW(None, False, True, event_name)
        self.assertTrue(event)
        capture = WindowsMemory(map_name, event_name)
        try:
            with mmap.mmap(-1, len(content), tagname=map_name) as producer:
                producer[:] = content
                self.assertTrue(capture.open())
                capture.wait()
                self.assertEqual(SnapshotReader(capture, names).snapshot()['values']['Speed'], 50.5)
                capture.close()
            self.assertFalse(capture.open())
        finally:
            capture.close()
            kernel.CloseHandle(event)


if __name__ == '__main__':
    unittest.main()
