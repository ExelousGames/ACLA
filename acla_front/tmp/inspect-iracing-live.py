"""Read-only, bounded live inventory using the project's existing IRSDK reader."""
import datetime
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'py-scripts'))
from iracing_sdk import WindowsMemory, SnapshotReader, VARIABLE, header_layout

memory = WindowsMemory()
try:
    if not memory.open():
        raise SystemExit('No connected live iRacing shared-memory mapping.')
    header, _, _ = header_layout(memory.read(0, 112))
    table = memory.read(header[7], header[6] * VARIABLE.size)
    channels = {}
    decode = lambda value: value.split(b'\0', 1)[0].decode('cp1252', errors='replace')
    for index in range(header[6]):
        kind, offset, count, as_time, name, description, unit = VARIABLE.unpack_from(table, index * VARIABLE.size)
        channels[decode(name)] = {
            'type': ('char', 'bool', 'int', 'bitfield', 'float', 'double')[kind],
            'count': count, 'countAsTime': as_time,
            'description': decode(description), 'unit': decode(unit),
            'observations': 0, 'changes': 0,
        }
    reader = SnapshotReader(memory, channels)
    started = datetime.datetime.now().astimezone()
    start = time.monotonic()
    samples = []
    disconnected = False
    while time.monotonic() - start < 12:
        sample = reader.snapshot()
        if sample and sample['type'] == 'disconnected':
            disconnected = True
            break
        if sample and sample['type'] == 'sample':
            samples.append(sample)
            for name, value in sample['values'].items():
                channel = channels[name]
                if channel['observations']:
                    channel['changes'] += int(channel['latest'] != value)
                else:
                    channel['first'] = value
                channel['latest'] = value
                channel['observations'] += 1
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    channel['min'] = min(channel.get('min', value), value)
                    channel['max'] = max(channel.get('max', value), value)
        memory.wait()
    elapsed = time.monotonic() - start
    result = {
        'capturedAt': started.isoformat(), 'elapsedSeconds': elapsed,
        'connectedAtStart': True, 'disconnectedDuringCapture': disconnected,
        'headerVersion': header[0], 'advertisedHz': header[2],
        'nativeChannelCount': header[6], 'sampleCount': len(samples),
        'firstTick': samples[0]['tick'] if samples else None,
        'lastTick': samples[-1]['tick'] if samples else None,
        'channels': channels, 'samples': samples,
    }
    output = ROOT / 'tmp' / ('iracing-live-' + started.strftime('%Y%m%dT%H%M%S') + '.json')
    output.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    summary = {key: value for key, value in result.items() if key not in ('channels', 'samples')}
    summary['output'] = str(output)
    summary['changingChannels'] = [name for name, channel in channels.items() if channel['changes']]
    summary['selected'] = {name: channels[name] for name in (
        'IsOnTrack', 'IsOnTrackCar', 'IsInGarage', 'IsReplayPlaying', 'SessionTime',
        'Speed', 'RPM', 'Gear', 'Throttle', 'Brake', 'BrakeRaw', 'Clutch',
        'SteeringWheelAngle', 'FuelLevel', 'Lap', 'LapCompleted', 'LapDistPct',
        'OnPitRoad', 'PlayerTrackSurface', 'TrackTempCrew', 'AirTemp',
        'Precipitation', 'TrackWetness', 'LFpressure', 'LFcoldPressure',
        'LFtempCL', 'LFwearL', 'LFbrakeLinePress', 'LFshockDefl',
    ) if name in channels}
    print(json.dumps(summary, allow_nan=False, indent=2))
finally:
    memory.close()
