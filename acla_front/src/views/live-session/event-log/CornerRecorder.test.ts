import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { CornerRecorder, DriverCornerRecord, formatCornerTime, parseSessionTimeMs } from './CornerRecorder';

const map: CircuitMapDto = {
    id: 'track', game: 'acc', circuit_name: 'Test circuit', resolution: 1000, samples: {},
    centerline_segments: [{ id: 'turn-1', tags: ['corner'], start_position: 0.2, end_position: 0.3 }],
};
const BASE_TIME = 2027758;
const positions = [0.1, 0.13, 0.16, 0.185, 0.205, 0.22, 0.23, 0.24, 0.255, 0.275, 0.3];

describe('CornerRecorder', () => {
    let recorder: CornerRecorder;
    beforeEach(() => { recorder = new CornerRecorder(); recorder.setMap(map); });
    const tick = (time: number, cars: Record<string, number>, sampleIndex = time) => recorder.tick({
        Graphics_game_time: (BASE_TIME + time) / 1000,
        Graphics_normalized_positions: cars,
        Graphics_player_car_id: 7,
        // Deliberately unrelated to opponent motion or the requested telemetry clock.
        Physics_speed_kmh: 900, Physics_brake: 1, Physics_gas: 0, Graphics_current_time: 1,
    }, sampleIndex);

    it('converts elapsed session seconds to milliseconds and formats times without wrapping at an hour', () => {
        expect(parseSessionTimeMs(2027.758)).toBe(BASE_TIME);
        expect(parseSessionTimeMs(62.003)).toBe(62003);
        expect(parseSessionTimeMs(0)).toBe(0);
        expect(parseSessionTimeMs(1 / 60)).toBe(17);
        expect(formatCornerTime(3600001)).toBe('60:00:001');
        expect(formatCornerTime(758)).toBe('00:00:758');
        [undefined, null, '', '2027.758', NaN, Infinity, -1, Number.MAX_VALUE]
            .forEach((value) => expect(parseSessionTimeMs(value)).toBeNull());
    });

    it('creates a record only at the end, backtracks braking and acceleration, and interpolates entry', () => {
        const records: DriverCornerRecord[] = [];
        positions.forEach((position, index) => {
            const completed = tick(index * 1000, { '7': position }, index);
            if (index < positions.length - 1) expect(completed).toEqual([]);
            records.push(...completed);
        });
        expect(records).toHaveLength(1);
        expect(records[0].exitSpeed).toBeCloseTo(0.025);
        expect(records[0].mapId).toBe('track');
        expect(records[0]).toMatchObject({
            carId: '7', isPlayer: true, cornerId: 'turn-1',
            decelerationStart: { position: 0.16, timeMs: BASE_TIME + 2000 },
            accelerationStart: { position: 0.24, timeMs: BASE_TIME + 7000 },
            entry: { position: 0.2, timeMs: BASE_TIME + 3750 },
            exit: { position: 0.3, timeMs: BASE_TIME + 10000 },
            cornerTimeMs: 6250, decelerationToExitMs: 8000,
        });
        expect(tick(11000, { '7': 0.32 })).toEqual([]);
    });

    it('tracks multiple cars independently and does not copy the player physics or timing', () => {
        const records: DriverCornerRecord[] = [];
        positions.forEach((position, index) => {
            records.push(...tick(index * 1000, { '7': 0.1 + index * 0.01, '63': position, '1052': 0.1 + index * 0.025 }));
        });
        expect(records.map((record) => record.carId)).toEqual(['1052', '63']);
        expect(records[0]).toMatchObject({ isPlayer: false, decelerationStart: null, accelerationStart: null, cornerTimeMs: 4000 });
        expect(records[1]).toMatchObject({ isPlayer: false, decelerationStart: { position: 0.16 }, cornerTimeMs: 6250 });
    });

    it('measures each car at the exit crossing even without an entry or a full rate window', () => {
        tick(0, { '7': 0.29, '63': 0.295 });
        const records = tick(100, { '7': 0.31, '63': 0.305 });
        expect(records).toHaveLength(2);
        expect(records[0].exit.timeMs).toBe(BASE_TIME + 50);
        expect(records[0].exitSpeed).toBeCloseTo(0.2);
        expect(records[1].exitSpeed).toBeCloseTo(0.1);
        expect(records.every((record) => record.cornerTimeMs === null)).toBe(true);
    });

    it('ignores repeated broadcast positions and duplicate clock ticks', () => {
        const records: DriverCornerRecord[] = [];
        positions.forEach((position, index) => {
            records.push(...tick(index * 1000, { '7': position }));
            expect(tick(index * 1000, { '7': position + 0.01 })).toEqual([]);
            expect(tick(index * 1000 + 100, { '7': position })).toEqual([]);
        });
        expect(records[0].decelerationStart?.position).toBe(0.16);
        expect(records[0].cornerTimeMs).toBe(6250);
        expect(records[0].exitSpeed).toBeCloseTo(0.025);
    });

    it('interpolates corners across start/finish and records every passage once', () => {
        recorder.setMap({ ...map, centerline_segments: [{ id: 'wrap', tags: ['slow corner'], start_position: 0.95, end_position: 0.02 }] });
        const records: DriverCornerRecord[] = [];
        [0.93, 0.96, 0.99, 0.01, 0.04].forEach((position, index) => records.push(...tick(index * 1000, { '7': position })));
        expect(records).toHaveLength(1);
        expect(records[0]).toMatchObject({ entry: { timeMs: BASE_TIME + 667 }, exit: { timeMs: BASE_TIME + 3333 }, cornerTimeMs: 2666 });
        expect(records[0].exitSpeed).toBeCloseTo(0.03);
        for (let index = 5; index <= 23; index += 1) tick(index * 1000, { '7': 0.04 + (index - 4) * 0.05 });
        expect(tick(24000, { '7': 0.03 })).toHaveLength(1);
    });

    it('keeps player and opponent corner timing across player lap clock resets', () => {
        recorder.setMap({ ...map, centerline_segments: [
            ...map.centerline_segments!,
            { id: 'wrap', tags: ['corner'], start_position: 0.95, end_position: 0.02 },
        ] });
        const records: DriverCornerRecord[] = [];
        [0.93, 0.96, 0.99, 0.01, 0.04].forEach((position, index) => {
            records.push(...recorder.tick({
                Graphics_game_time: (BASE_TIME + index * 1000) / 1000,
                Graphics_current_time_str: formatCornerTime(position * 25000),
                Graphics_current_time: position * 25000,
                Graphics_player_car_id: 7,
                Graphics_normalized_car_position: position,
                Graphics_normalized_positions: { '63': (18 + index * 3) / 100 },
            }, index));
        });
        expect(records).toHaveLength(2);
        expect(records).toEqual(expect.arrayContaining([
            expect.objectContaining({ carId: '7', isPlayer: true, cornerId: 'wrap', cornerTimeMs: 2666 }),
            expect.objectContaining({ carId: '63', isPlayer: false, cornerId: 'turn-1', cornerTimeMs: 3333 }),
        ]));
        expect(records.every(({ exitSpeed }) => Math.abs(exitSpeed! - 0.03) < 1e-9)).toBe(true);
    });

    it('measures an exit crossing over the start/finish line', () => {
        recorder.setMap({ ...map, centerline_segments: [{ id: 'wrap', tags: ['slow corner'], start_position: 0.95, end_position: 0 }] });
        tick(0, { '7': 0.99 });
        expect(tick(1000, { '7': 0.01 })[0].exitSpeed).toBeCloseTo(0.02);
    });

    it('records missing entry or motion as not observed when capture starts inside a corner', () => {
        tick(0, { '7': 0.28 });
        expect(tick(1000, { '7': 0.31 })[0]).toMatchObject({ entry: null, decelerationStart: null, accelerationStart: null, cornerTimeMs: null });
    });

    it('does not join history across missing cars, backwards motion, jumps, gaps or clock resets', () => {
        tick(0, { '7': 0.28 });
        tick(500, {});
        expect(tick(1000, { '7': 0.31 })).toEqual([]);
        expect(tick(1500, { '7': 0.28 })).toEqual([]);
        expect(tick(2000, { '7': 0.8 })).toEqual([]);
        tick(2500, { '7': 0.28 });
        expect(tick(5000, { '7': 0.31 })).toEqual([]);
        tick(6000, { '7': 0.28 });
        expect(tick(0, { '7': 0.31 })).toEqual([]);
    });

    it('discards missing or invalid car timing even when the clock timestamp repeats', () => {
        for (const cars of [{}, { '7': NaN }] as Array<Record<string, number>>) {
            recorder.reset();
            tick(0, { '7': 0.19 });
            tick(1000, { '7': 0.24 });
            tick(1000, cars);
            expect(tick(2000, { '7': 0.31 })).toEqual([]);
        }
    });

    it('accepts a forward sample exactly 0.1 lap away', () => {
        tick(0, { '7': 0.15 });
        tick(1000, { '7': 0.25 });
        expect(tick(2000, { '7': 0.35 })[0]).toMatchObject({ cornerTimeMs: 1000 });
    });

    it('requires the requested clock and live data, and resets when maps or streams change', () => {
        for (const reset of [
            () => recorder.reset(),
            () => recorder.setMap(map),
            () => recorder.tick({ Graphics_current_time: BASE_TIME + 500 }, 1),
            () => recorder.tick({ Graphics_game_time: (BASE_TIME + 500) / 1000, Graphics_status: 1 }, 1),
        ]) {
            recorder.reset();
            tick(0, { '7': 0.28 });
            reset();
            expect(tick(1000, { '7': 0.31 })).toEqual([]);
        }
        recorder.setMap(null);
        tick(0, { '7': 0.28 });
        expect(tick(1000, { '7': 0.31 })).toEqual([]);
    });

    it('uses legacy tags, deduplicates ranges, excludes sequences, and merges player identity once', () => {
        expect(recorder.setMap({ ...map, centerline_segments: undefined, centerline_tags: [
            { id: 'legacy', label: 'corner', start_position: 0.2, end_position: 0.3 },
            { id: 'slow', label: 'slow corner', start_position: 0.2, end_position: 0.3 },
            { id: 'sequence', label: 'consecutive corners', start_position: 0.1, end_position: 0.4 },
        ] })).toBe(1);
        tick(0, { '7': 0.28 });
        const records = recorder.tick({ Graphics_game_time: (BASE_TIME + 1000) / 1000,
            Graphics_player_car_id: 7, Graphics_normalized_car_position: 0.31, Graphics_normalized_positions: { '7': 0.28 } }, 1);
        expect(records).toHaveLength(1);
        expect(records[0]).toMatchObject({ carId: '7', isPlayer: true, cornerId: 'legacy' });
    });
});
