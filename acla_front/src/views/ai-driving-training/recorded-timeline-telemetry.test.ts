import type { RecordedDrivingRow, RecordedDrivingSession } from './recorded-driving-session';
import { buildRecordedTimelinePaths, recordedTimelineValues } from './recorded-timeline-telemetry';

function recording(samples: Array<[number, RecordedDrivingRow['sample'], number?]>): RecordedDrivingSession {
    const rows = samples.map(([timeSeconds, sample, segment = 0], index) => ({ timeSeconds, sample, segment, index }));
    return { id: 'timeline', game: 'iracing', rows, startTimeSeconds: 100,
        durationSeconds: rows[rows.length - 1].timeSeconds, timestampField: 'Graphics_analyst_session_time' };
}

it('aligns pedals and discrete gear changes to video time and a zoomed window', () => {
    const session = recording([
        [0, { Physics_gas: 0, Physics_brake: 1, Physics_gear: 0 }],
        [0.25, { Physics_gas: 1, Physics_brake: 0.5, Physics_gear: 1 }],
        [0.5, { Physics_gas: 0.5, Physics_brake: 0, Physics_gear: 3 }],
    ]);
    const full = buildRecordedTimelinePaths(session, 99.75, 0, 1);
    expect(full.throttle).toBe('M250.00 70.00 L500.00 8.00 L750.00 39.00');
    expect(full.brake).toBe('M250.00 8.00 L500.00 39.00 L750.00 70.00');
    expect(full.gear).toBe('M250.00 116.00 H500.00V104.67 H750.00V82.00');
    expect(full.maxGear).toBe(2);

    const zoomed = buildRecordedTimelinePaths(session, 99.75, 0.375, 0.625);
    expect(zoomed.throttle).toContain('L500.00 8.00');
    expect(zoomed.maxGear).toBe(2);
    expect(buildRecordedTimelinePaths(session, 100.5, 0.25, 0.5).gear).toBe('');
});

it('breaks each trace at missing values, recorded gaps, and session changes', () => {
    const session = recording([
        [0, { Physics_gas: 0.4, Physics_brake: 0.1 }],
        [0.25, { Physics_gas: 0.5 }],
        [0.5, { Physics_gas: 0.6, Physics_brake: 0.2 }],
        [1.25, { Physics_gas: 0.7, Physics_brake: 0.3 }],
        [1.5, { Physics_gas: 0.8, Physics_brake: 0.4 }, 1],
    ]);
    const { throttle, brake, gear } = buildRecordedTimelinePaths(session, 100, 0, 2);
    expect(throttle.match(/M/g)).toHaveLength(3);
    expect(brake.match(/M/g)).toHaveLength(4);
    expect(gear).toBe('');
    expect(buildRecordedTimelinePaths(session, 100, 0.6, 1.2)).toMatchObject({ throttle: '', brake: '', gear: '' });
});

it('bounds path size and preserves a brief pedal spike in a dense chart', () => {
    const sampleCount = 10_001;
    const session = recording(Array.from({ length: sampleCount }, (_, index) => [
        index / 1000, { Physics_brake: index === 5553 ? 1 : 0 },
    ]));
    const { brake } = buildRecordedTimelinePaths(session, 100, 0, 10);
    expect(brake).toContain('L555.30 8.00');
    expect(brake.match(/[ML]/g)!.length).toBeLessThanOrEqual(4004);
});

it('reads the last aligned sample with available channels, within the recorded time and freshness limits', () => {
    const session = recording([
        [0, { Physics_gas: 0.75, Physics_gear: 0 }],
        [0.25, { Physics_brake: 0.2, Physics_gear: 1 }],
        [0.25, { Physics_brake: 0.33, Physics_gear: 2 }],
        [1.25, { Physics_gas: 0.5, Physics_gear: 4 }],
    ]);
    const empty = { throttle: null, brake: null, gear: null };
    expect(recordedTimelineValues(session, 99.5, 0.4)).toEqual(empty);
    expect(recordedTimelineValues(session, 99.5, 0.6)).toEqual({ throttle: 0.75, brake: null, gear: -1 });
    expect(recordedTimelineValues(session, 99.5, 0.75)).toEqual({ throttle: null, brake: 0.33, gear: 1 });
    expect(recordedTimelineValues(session, 99.5, 1.5)).toEqual(empty);
    expect(recordedTimelineValues(session, 99.5, 1.75)).toEqual({ throttle: 0.5, brake: null, gear: 3 });
    expect(recordedTimelineValues(session, 99.5, 2)).toEqual(empty);
});
