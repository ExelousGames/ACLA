"""Nonblocking ACC Broadcasting v4 supplement to the shared-memory extractor.

Wire reference: Kunos ksBroadcastingNetwork, also implemented at
https://github.com/EmperorCookie/accapi/tree/main/src/accapi
Only registration, entry-list/track requests and unregistration are sent.
"""

import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import time


def broadcasting_config_path():
    override = os.environ.get('ACLA_ACC_BROADCAST_CONFIG')
    if override:
        return Path(override)
    documents = Path.home() / 'Documents'
    if sys.platform == 'win32':
        # Respect redirected Documents folders, including OneDrive.
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
                documents = Path(os.path.expandvars(winreg.QueryValueEx(key, 'Personal')[0]))
        except OSError:
            pass
    return documents / 'Assetto Corsa Competizione' / 'Config' / 'broadcasting.json'


def encode_string(value):
    encoded = value.encode('utf-8')
    return struct.pack('<H', len(encoded)) + encoded


class Packet:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def read(self, fmt):
        values = struct.unpack_from('<' + fmt, self.data, self.offset)
        self.offset += struct.calcsize('<' + fmt)
        return values

    def string(self):
        length, = self.read('H')
        return self.read(f'{length}s')[0].decode('utf-8')

    def lap(self):
        _, _, _, splits = self.read('iHHB')
        self.read('i' * splits + 'BBBB')


class ACCBroadcasting:
    UPDATE_INTERVAL_MS = 100
    STALE_SECONDS = 2.0
    RETRY_SECONDS = 5.0

    def __init__(self, config_path=None, *, socket_factory=socket.socket, clock=time.monotonic):
        self.config_path = Path(config_path) if config_path is not None else broadcasting_config_path()
        self.socket_factory = socket_factory
        self.clock = clock
        self.sock = None
        self.connection_id = None
        self.positions = {}
        self.session = None
        self.session_time = None
        self.track = None
        self.replay = False
        self.last_received = 0.0
        self.next_retry = 0.0
        self.last_warning = None

    def _warn(self, message):
        if message != self.last_warning:
            print(f'ACC broadcasting: {message}', file=sys.stderr, flush=True)
            self.last_warning = message

    def _connect(self, now):
        self.next_retry = now + self.RETRY_SECONDS
        try:
            # json.loads detects UTF-8/16/32, including ACC's BOM-less UTF-16 LE.
            config = json.loads(self.config_path.read_bytes())
            port = config.get('updListenerPort')  # ACC's config intentionally spells this "upd".
            password = config.get('connectionPassword', '')
            if type(port) is not int or not 1 <= port <= 65535 or not isinstance(password, str):
                raise ValueError('Invalid broadcasting settings')
            registration = (bytes([1, 4]) + encode_string('Kestrel Motorsport Analyst')
                            + encode_string(password) + struct.pack('<i', self.UPDATE_INTERVAL_MS)
                            + encode_string(''))
            self.sock = self.socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.sock.connect(('127.0.0.1', port))
            self.sock.send(registration)
            self.last_received = now
        except (OSError, ValueError, AttributeError, struct.error):
            self.close()
            self._warn('Unavailable; enable updListenerPort in broadcasting.json. Shared-memory capture continues.')

    def _handle(self, data, now):
        packet = Packet(data)
        message, = packet.read('B')
        if message == 1:
            connection_id, success, _ = packet.read('iBB')
            packet.string()  # Do not expose server messages or credentials in diagnostics.
            if not success:
                self.close()
                self._warn('Registration rejected; check connectionPassword in broadcasting.json.')
                return
            self.positions.clear()
            self.connection_id = connection_id
            self.last_warning = None
            for request in (10, 11):
                self.sock.send(struct.pack('<Bi', request, connection_id))
        elif self.connection_id is None:
            return
        elif message == 2:
            event, session, session_type, _, session_time, _, _ = packet.read('HHBBffi')
            for _ in range(3):
                packet.string()
            replay, = packet.read('B')
            identity = (event, session, session_type)
            if (identity != self.session or bool(replay) != self.replay
                    or (self.session_time is not None and session_time < self.session_time - 1000)):
                self.positions.clear()
            self.session = identity
            self.session_time = session_time
            self.replay = bool(replay)
        elif message == 3:
            car = packet.read('HHBBfffBHHHHfHi')
            for _ in range(3):
                packet.lap()
            car_id, location, position = car[0], car[7], car[12]
            if not self.replay and location in (1, 2, 3, 4) and math.isfinite(position) and 0 <= position <= 1:
                self.positions[car_id] = (position, now)
            else:
                self.positions.pop(car_id, None)
        elif message == 4:
            connection_id, count = packet.read('iH')
            car_ids = set(packet.read('H' * count))
            if connection_id == self.connection_id:
                self.positions = {key: value for key, value in self.positions.items() if key in car_ids}
        elif message == 5:
            connection_id, = packet.read('i')
            name = packet.string()
            track_id, meters = packet.read('ii')
            identity = (name, track_id, meters)
            if connection_id == self.connection_id:
                if identity != self.track:
                    self.positions.clear()
                self.track = identity
        self.last_received = now

    def poll(self):
        """Return fresh car-ID -> lap-fraction values, or None if disconnected."""
        now = self.clock()
        if self.sock is None:
            if now < self.next_retry:
                return None
            self._connect(now)
        if self.sock is None:
            return None
        try:
            # Bound each poll so a UDP burst cannot starve the shared-memory reader.
            for _ in range(256):
                try:
                    data = self.sock.recv(65535)
                except BlockingIOError:
                    break
                try:
                    self._handle(data, now)
                except (struct.error, UnicodeError):
                    continue  # A malformed datagram never terminates recording.
                if self.sock is None:
                    break
        except OSError:
            self.close()
            self.next_retry = now + self.RETRY_SECONDS
        if self.sock is not None and now - self.last_received >= self.RETRY_SECONDS:
            self.close()
            self.next_retry = now + self.RETRY_SECONDS
            self._warn('No UDP updates; reconnecting. Shared-memory capture continues.')
        if self.connection_id is None:
            return None
        self.positions = {key: value for key, value in self.positions.items()
                          if now - value[1] <= self.STALE_SECONDS}
        return {str(key): value[0] for key, value in self.positions.items()}

    def close(self):
        if self.sock is not None:
            try:
                if self.connection_id is not None:
                    self.sock.send(struct.pack('<Bi', 9, self.connection_id))
            except OSError:
                pass
            finally:
                self.sock.close()
                self.sock = None
        self.connection_id = None
        self.positions.clear()
        self.session = None
        self.session_time = None
        self.track = None
        self.replay = False
