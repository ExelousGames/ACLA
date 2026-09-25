import contextlib
import io
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acc_broadcasting import ACCBroadcasting, Packet, broadcasting_config_path
from ACCMemoryExtractor import ACCRecording


def string(value):
    data = value.encode('utf-8')
    return struct.pack('<H', len(data)) + data


def registration(success=1):
    return struct.pack('<BiBB', 1, 7, success, 0) + string('')


def car_update(car_id=1052, position=0.25, location=1):
    # v4 includes driverCount between driverIndex and gear. Nonzero filler
    # values ensure a one-byte offset error cannot accidentally pass the test.
    car = struct.pack('<BHHBBfffBHHHHfHi', 3, car_id, 2, 3, 4,
                      100.0, -20.0, 1.5, location, 150, 5, 2, 4, position, 7, -120)
    lap = struct.pack('<iHHBiiiBBBB', 90000, car_id, 2, 3, 30000, 31000, 29000, 0, 1, 0, 0)
    return car + lap * 3


def session_update(session=1, elapsed=10000.0, replay=False):
    data = struct.pack('<BHHBBffi', 2, 0, session, 10, 5, elapsed, 3600000.0, 1052)
    data += string('Drivable') + string('Dash') + string('Basic HUD') + bytes([replay])
    if replay:
        data += struct.pack('<ff', elapsed, 1000.0)
    return data + struct.pack('<fBBBBBiHHBBBBB', 43200.0, 20, 25, 0, 0, 0, 90000, 1052, 2, 0, 0, 1, 0, 0)


def track_update(name='spa', track_id=1):
    return struct.pack('<Bi', 5, 7) + string(name) + struct.pack('<iiBB', track_id, 7004, 0, 0)


class FakeSocket:
    def __init__(self):
        self.queue = []
        self.sent = []
        self.closed = False

    def setblocking(self, value):
        self.blocking = value

    def connect(self, address):
        self.address = address

    def send(self, data):
        self.sent.append(data)

    def recv(self, size):
        if not self.queue:
            raise BlockingIOError()
        value = self.queue.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


class ACCBroadcastingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / 'broadcasting.json'
        self.config.write_text(json.dumps({'updListenerPort': 9000, 'connectionPassword': 'päss'}), encoding='utf-8-sig')
        self.now = 0.0
        self.sock = FakeSocket()
        self.factory = Mock(return_value=self.sock)
        self.client = ACCBroadcasting(self.config, socket_factory=self.factory, clock=lambda: self.now)
        self.addCleanup(self.client.close)
        self.stderr = io.StringIO()

    def poll(self, *packets):
        self.sock.queue.extend(packets)
        with contextlib.redirect_stderr(self.stderr):
            return self.client.poll()

    def test_registration_uses_v4_utf8_password_and_nonblocking_local_socket(self):
        self.assertIsNone(self.poll())
        packet = Packet(self.sock.sent[0])
        self.assertEqual(packet.read('BB'), (1, 4))
        self.assertEqual(packet.string(), 'Kestrel Motorsport Analyst')
        self.assertEqual(packet.string(), 'päss')
        self.assertEqual(packet.read('i'), (100,))
        self.assertEqual(packet.string(), '')
        self.assertEqual(packet.offset, len(packet.data))
        self.assertFalse(self.sock.blocking)
        self.assertEqual(self.sock.address, ('127.0.0.1', 9000))
        self.assertEqual(self.poll(registration()), {})
        self.assertEqual(self.sock.sent[1:], [struct.pack('<Bi', code, 7) for code in (10, 11)])

    def test_real_car_packets_preserve_ids_endpoints_and_pit_cars(self):
        result = self.poll(registration(), session_update(), track_update(),
                           car_update(0, 0), car_update(1052, 0.25), car_update(65535, 1, 2))
        self.assertEqual(result, {'0': 0.0, '1052': 0.25, '65535': 1.0})
        self.assertEqual(self.poll(car_update(1052, 0.5))['1052'], 0.5)

    def test_invalid_or_missing_cars_are_removed_without_placeholder_zero(self):
        self.poll(registration())
        for value, location in [(-1, 1), (1.1, 1), (float('nan'), 1), (float('inf'), 1), (0.5, 0)]:
            with self.subTest(value=value, location=location):
                self.poll(car_update())
                self.assertEqual(self.poll(car_update(position=value, location=location)), {})

    def test_stale_values_expire_even_when_other_cars_still_update(self):
        self.poll(registration(), car_update())
        self.now = 2.1
        self.assertEqual(self.poll(car_update(2, 0.5)), {'2': 0.5})

    def test_session_track_and_replay_changes_clear_cached_cars(self):
        self.poll(registration(), session_update(), track_update(), car_update())
        self.assertEqual(self.poll(session_update(session=2)), {})
        self.poll(car_update())
        self.assertEqual(self.poll(track_update('monza', 3)), {})
        self.poll(car_update())
        self.assertEqual(self.poll(session_update(session=2, elapsed=1000)), {})
        self.poll(car_update())
        self.assertEqual(self.poll(session_update(replay=True), car_update()), {})
        self.assertEqual(self.poll(session_update(), car_update()), {'1052': 0.25})

    def test_entry_list_removes_departed_cars(self):
        self.poll(registration(), car_update(1), car_update(1052))
        self.assertEqual(self.poll(struct.pack('<BiHH', 4, 7, 1, 1052)), {'1052': 0.25})

    def test_truncated_datagrams_do_not_mutate_state_or_stop_capture(self):
        self.poll(registration(), car_update())
        data = car_update(position=0.75)
        for length in range(len(data)):
            self.assertEqual(self.poll(data[:length]), {'1052': 0.25})
        self.assertEqual(self.poll(data), {'1052': 0.75})

    def test_rejected_registration_and_socket_errors_retry_without_old_positions(self):
        self.assertIsNone(self.poll(registration(0)))
        self.assertTrue(self.sock.closed)
        self.now = 1
        self.poll()
        self.assertEqual(self.factory.call_count, 1)
        self.now = 5
        self.assertEqual(self.poll(registration(), car_update()), {'1052': 0.25})
        self.assertIsNone(self.poll(ConnectionResetError()))
        self.now = 10
        self.assertEqual(self.poll(registration()), {})

    def test_silent_server_times_out_and_reregisters(self):
        self.poll(registration(), car_update())
        self.now = 5.1
        self.assertIsNone(self.poll())
        self.assertTrue(self.sock.closed)
        self.now = 10.1
        self.assertEqual(self.poll(registration()), {})
        self.assertEqual(self.factory.call_count, 2)

    def test_missing_or_disabled_config_is_optional_and_does_not_leak_passwords(self):
        for content in ['', 'null', '[]', '{"updListenerPort":0}', '{"updListenerPort":true}']:
            with self.subTest(content=content):
                self.config.write_text(content, encoding='utf-8')
                self.assertIsNone(self.poll())
                self.now += 5
        self.factory.assert_not_called()
        self.assertEqual(len(self.stderr.getvalue().splitlines()), 1)
        self.assertNotIn('päss', self.stderr.getvalue())
        self.config.unlink()
        self.assertIsNone(self.poll())

    def test_accepts_acc_utf16_config_and_explicit_path_override(self):
        self.config.write_text('{"updListenerPort":9000}', encoding='utf-16')
        self.assertEqual(self.poll(registration()), {})
        with patch.dict('os.environ', {'ACLA_ACC_BROADCAST_CONFIG': str(self.config)}):
            self.assertEqual(broadcasting_config_path(), self.config)

    def test_acc_config_without_a_byte_order_marker_registers_and_delivers_positions(self):
        self.config.write_text(json.dumps({'updListenerPort': 9000, 'connectionPassword': 'päss'},
                                         ensure_ascii=False), encoding='utf-16-le')
        self.assertFalse(self.config.read_bytes().startswith(b'\xff\xfe'))
        self.assertEqual(self.poll(registration(), car_update()), {'1052': 0.25})
        packet = Packet(self.sock.sent[0])
        packet.read('BB')
        packet.string()
        self.assertEqual(packet.string(), 'päss')
        self.assertEqual(self.stderr.getvalue(), '')

    def test_close_unregisters_and_releases_socket(self):
        self.poll(registration(), car_update())
        self.client.close()
        self.assertTrue(self.sock.closed)
        self.assertEqual(self.sock.sent[-1], struct.pack('<Bi', 9, 7))
        self.assertEqual(self.client.positions, {})
        self.client.close()

    def test_loopback_udp_registration_and_car_delivery(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.bind(('127.0.0.1', 0))
            server.settimeout(1)
            self.config.write_text(json.dumps({'updListenerPort': server.getsockname()[1]}), encoding='utf-8')
            client = ACCBroadcasting(self.config)
            try:
                self.assertIsNone(client.poll())
                request, address = server.recvfrom(65535)
                self.assertEqual(request[:2], bytes([1, 4]))
                server.sendto(registration(), address)
                server.sendto(car_update(), address)
                # Wait for queued loopback packets without sleeping in production.
                import select
                self.assertTrue(select.select([client.sock], [], [], 1)[0])
                self.assertEqual(client.poll(), {'1052': 0.25})
            finally:
                client.close()


class ACCExtractorBroadcastingTests(unittest.TestCase):
    def setUp(self):
        with patch('ACCMemoryExtractor.accSharedMemory'), patch('ACCMemoryExtractor.ACCBroadcasting'):
            self.recorder = ACCRecording()
        self.scheduler = Mock()

    def record(self, status=2):
        self.recorder.asm.read_shared_memory.return_value = SimpleNamespace(
            Graphics=SimpleNamespace(status=status, normalized_car_position=0.4))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.recorder.recordOnce(self.scheduler, None)
        return json.loads(output.getvalue())

    def test_emits_broadcast_positions_alongside_shared_memory_on_one_json_line(self):
        self.recorder.broadcasting.poll.return_value = {'7': 0.25, '1052': 0.5}
        self.assertEqual(self.record(), {'Graphics_status': 2, 'Graphics_normalized_car_position': 0.4,
                                        'Graphics_normalized_positions': {'7': 0.25, '1052': 0.5}})

    def test_unavailable_broadcasting_preserves_shared_memory_capture(self):
        self.recorder.broadcasting.poll.return_value = None
        self.assertEqual(self.record(), {'Graphics_status': 2, 'Graphics_normalized_car_position': 0.4})

    def test_nonlive_and_missing_shared_memory_release_udp_state(self):
        for status in (0, 1, 3):
            self.assertNotIn('Graphics_normalized_positions', self.record(status))
        self.recorder.broadcasting.poll.assert_not_called()
        self.assertEqual(self.recorder.broadcasting.close.call_count, 3)
        self.recorder.asm.read_shared_memory.return_value = None
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.recorder.recordOnce(self.scheduler, None)
        self.assertEqual(json.loads(output.getvalue()), {'available': False})
        self.assertEqual(self.recorder.broadcasting.close.call_count, 4)


if __name__ == '__main__':
    unittest.main()
