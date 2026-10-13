"""Read-only IRSDK capture. Protocol reference: iRacing's irsdk_defines.h /
irsdk_utils.cpp (https://forums.iracing.com/discussion/62/iracing-sdk).

Only this subprocess touches shared memory. A pull request permits one snapshot;
adaptation, YAML parsing, disk writes and rendering happen in other processes.
"""

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import struct
import sys
import time


HEADER = struct.Struct('<10i8x')
BUFFER = struct.Struct('<ii8x')
VARIABLE = struct.Struct('<iii?3x32s64s32s')
FORMATS = ('c', '?', 'i', 'I', 'f', 'd')
MAX_BYTES = 64 * 1024 * 1024


class MappingChanged(Exception):
    pass


def header_layout(data):
    values = HEADER.unpack_from(data)
    version, status, rate, update, session_len, session_offset, count, offset, buffers, row_len = values
    if version != 2 or not 1 <= buffers <= 4 or not 0 < rate <= 360:
        raise ValueError('Unsupported IRSDK header version, buffer count or tick rate')
    if not 0 < count <= 4096 or not 0 < row_len <= MAX_BYTES:
        raise ValueError('Invalid IRSDK variable table or row length')
    slots = [BUFFER.unpack_from(data, HEADER.size + i * BUFFER.size) for i in range(buffers)]
    ranges = [(session_offset, session_len), (offset, count * VARIABLE.size)]
    ranges.extend((position, row_len) for _, position in slots)
    if any(position < 0 or length < 0 or position + length > MAX_BYTES for position, length in ranges):
        raise ValueError('Invalid IRSDK memory range')
    size = max([112] + [position + length for position, length in ranges])
    return values, slots, size


class SnapshotReader:
    """Bounds-checked decoder, also usable with an in-memory SDK fixture."""

    def __init__(self, memory, names):
        self.memory = memory
        self.names = set(names)
        self.last_tick = None
        self.schema = None
        self.variables = []
        self.session_update = None

    def snapshot(self):
        raw_header = self.memory.read(0, 112)
        # A disconnected map can contain a partially initialized header.
        if not struct.unpack_from('<i', raw_header, 4)[0] & 1:
            return {'type': 'disconnected'}
        header, slots, _ = header_layout(raw_header)
        version, _, _, update, session_len, session_offset, count, offset, _, row_len = header
        tick, position = max(slots, key=lambda slot: slot[0])
        if self.last_tick is not None and tick < self.last_tick:
            return {'type': 'disconnected'}
        if tick == self.last_tick:
            return None
        schema = (version, count, offset, row_len)
        if schema != self.schema:
            table = self.memory.read(offset, count * VARIABLE.size)
            variables = []
            for index in range(count):
                kind, start, length, as_time, name, _, _ = VARIABLE.unpack_from(table, index * VARIABLE.size)
                name = name.split(b'\0', 1)[0].decode('ascii')
                if not 0 <= kind < len(FORMATS) or length < 1:
                    raise ValueError('Invalid IRSDK variable type or count')
                decoder = struct.Struct('<' + str(length) + FORMATS[kind])
                if start < 0 or start + decoder.size > row_len:
                    raise ValueError('IRSDK variable exceeds row bounds')
                if name in self.names:
                    variables.append((name, kind, start, length, as_time, decoder))
            self.variables = variables
            self.schema = schema

        # The SDK copies the newest row and checks its tick again to reject tears.
        row = self.memory.read(position, row_len)
        session = None
        if self.session_update != update:
            session = self.memory.read(session_offset, session_len).split(b'\0', 1)[0].decode('cp1252', errors='replace')
        after = self.memory.read(0, 112)
        if raw_header[:40] != after[:40]:
            return None
        slot_index = slots.index((tick, position))
        if BUFFER.unpack_from(after, HEADER.size + slot_index * BUFFER.size) != (tick, position):
            return None

        values = {}
        for name, kind, start, length, as_time, decoder in self.variables:
            entries = decoder.unpack_from(row, start)
            if kind == 0:
                value = b''.join(entries).split(b'\0', 1)[0].decode('cp1252', errors='replace')
            elif length == 1 or as_time:
                value = entries[-1]
            else:
                value = list(entries)
            # JSON has no NaN/Infinity; unavailable channels must not reach the adapter.
            items = value if isinstance(value, list) else [value]
            if all(not isinstance(item, float) or math.isfinite(item) for item in items):
                values[name] = value
        self.last_tick = tick
        self.session_update = update
        result = {'type': 'sample', 'tick': tick, 'values': values}
        if session is not None:
            result['sessionInfo'] = session
        return result


class WindowsMemory:
    def __init__(self, map_name=r'Local\IRSDKMemMapFileName', event_name=r'Local\IRSDKDataValidEvent'):
        if sys.platform != 'win32':
            raise OSError('iRacing live telemetry requires Windows')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'OpenFileMappingW': ([wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
            'MapViewOfFile': ([wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t], ctypes.c_void_p),
            'UnmapViewOfFile': ([ctypes.c_void_p], wintypes.BOOL),
            'OpenEventW': ([wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
            'WaitForSingleObject': ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes = arguments
            function.restype = result
        self.mapping = self.view = self.event = None
        self.size = 0
        self.map_name = map_name
        self.event_name = event_name

    def open(self):
        self.close()
        self.mapping = self.kernel.OpenFileMappingW(4, False, self.map_name)
        if not self.mapping:
            if ctypes.get_last_error() == 2:
                return False
            raise ctypes.WinError(ctypes.get_last_error())
        self.view = self.kernel.MapViewOfFile(self.mapping, 4, 0, 0, 112)
        if not self.view:
            raise ctypes.WinError(ctypes.get_last_error())
        self.size = 112
        header = self.read(0, 112)
        if not struct.unpack_from('<i', header, 4)[0] & 1:
            self.close()
            return False
        _, _, size = header_layout(header)
        self.kernel.UnmapViewOfFile(self.view)
        self.view = self.kernel.MapViewOfFile(self.mapping, 4, 0, 0, size)
        if not self.view:
            raise ctypes.WinError(ctypes.get_last_error())
        self.size = size
        self.event = self.kernel.OpenEventW(0x100000, False, self.event_name)
        if not self.event:
            error = ctypes.get_last_error()
            self.close()
            if error == 2:
                return False
            raise ctypes.WinError(error)
        return True

    def read(self, offset, length):
        if offset < 0 or length < 0 or offset + length > self.size:
            raise MappingChanged('IRSDK layout grew beyond mapped memory')
        return ctypes.string_at(self.view + offset, length)

    def wait(self):
        if self.kernel.WaitForSingleObject(self.event, 100) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.event:
            self.kernel.CloseHandle(self.event)
        if self.view:
            self.kernel.UnmapViewOfFile(self.view)
        if self.mapping:
            self.kernel.CloseHandle(self.mapping)
        self.mapping = self.view = self.event = None
        self.size = 0


def emit(message):
    print(json.dumps(message, allow_nan=False, separators=(',', ':')), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variables', required=True)
    args = parser.parse_args()
    memory = WindowsMemory()
    reader = None
    last_data = time.monotonic()
    emit({'type': 'ready'})
    try:
        # Pipe EOF also cleans up if the parent process exits unexpectedly.
        for command in sys.stdin:
            if command.strip() != 'next':
                raise ValueError('Invalid iRacing capture request')
            if reader is None:
                if memory.open():
                    reader = SnapshotReader(memory, args.variables.split(','))
                    last_data = time.monotonic()
                else:
                    time.sleep(0.1)
            try:
                result = reader.snapshot() if reader else None
                if reader and result is None:
                    memory.wait()
                    result = reader.snapshot()
            except MappingChanged:
                result = {'type': 'disconnected'}
            if result and result['type'] == 'sample':
                last_data = time.monotonic()
            elif (result and result['type'] == 'disconnected') or time.monotonic() - last_data > 30:
                memory.close()
                reader = None
                result = {'type': 'disconnected'}
            emit(result or {'type': 'idle'})
    finally:
        memory.close()


if __name__ == '__main__':
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception as error:
        print(str(error), file=sys.stderr, flush=True)
        sys.exit(1)
