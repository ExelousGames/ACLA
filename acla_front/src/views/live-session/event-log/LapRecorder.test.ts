import type { StandardTelemetrySample } from '../live-session-types';
import { formatCornerTime } from './CornerRecorder';
import { LapRecorder, type DriverLapRecord } from './LapRecorder';

const BASE_TIME = 2027758;
const sample = (time: number, positions: Record<string, number>): StandardTelemetrySample => ({
    Graphics_game_time: (BASE_TIME + time) / 1000,
    Graphics_player_car_id: 7, Graphics_normalized_positions: positions,
    // These fields must never supply lap timing.
    Graphics_current_time: 1, Graphics_clock: 99999, Graphics_completed_lap: 400,
});

describe('LapRecorder', () => {
    let recorder: LapRecorder;
    const tick = (time: number, position: number, index = time) => recorder.tick(sample(time, { '7': position }), index);
    beforeEach(() => { recorder = new LapRecorder(); });

    it('discards the initial partial lap and interpolates consecutive finish-line crossings', () => {
        expect(tick(0, 0.97, 0)).toEqual([]);
        expect(tick(1000, 0.02, 1)).toEqual([]); // First crossing at +600 ms.
        for (let i = 1; i <= 19; i += 1) expect(tick(1000 + i * 1000, (2 + i * 5) / 100, i + 1)).toEqual([]);
        expect(tick(21500, 0.03, 21)).toEqual([expect.objectContaining({
            carId: '7', isPlayer: true, lapTimeMs: 20150,
            start: { timeMs: BASE_TIME + 600, position: 0, sampleIndex: 1 },
            end: { timeMs: BASE_TIME + 20750, position: 0, sampleIndex: 21 },
        })]);
    });

    it('records each car independently using the player position override and no map', () => {
        const records: DriverLapRecord[] = [];
        for (let i = 0; i <= 41; i += 1) records.push(...recorder.tick({
            ...sample(i * 1000, { '7': 0.5, '63': ((97 + i * 5) % 100) / 100 }),
            Graphics_normalized_car_position: ((97 + i * 4) % 100) / 100,
        }, i));
        expect(records.map(({ carId, lapTimeMs, isPlayer }) => ({ carId, lapTimeMs, isPlayer }))).toEqual([
            { carId: '63', lapTimeMs: 20000, isPlayer: false },
            { carId: '7', lapTimeMs: 25000, isPlayer: true },
            { carId: '63', lapTimeMs: 20000, isPlayer: false },
        ]);
    });

    it('does not advance on repeated timestamps, including changed positions', () => {
        tick(0, 0.97);
        expect(tick(0, 0.02)).toEqual([]);
        expect(tick(1000, 0.02)).toEqual([]);
        for (let i = 1; i <= 19; i += 1) {
            expect(tick(1000 + i * 1000, (2 + i * 5) / 100)).toEqual([]);
            expect(tick(1000 + i * 1000, 0.5)).toEqual([]);
        }
        expect(tick(21000, 0.02)[0].lapTimeMs).toBe(20000);
    });

    it('counts stationary time when telemetry remains continuous', () => {
        tick(0, 0.97);
        tick(1000, 0.02);
        for (let i = 2; i <= 5; i += 1) tick(i * 1000, 0.02);
        for (let i = 1; i <= 19; i += 1) tick(5000 + i * 1000, (2 + i * 5) / 100);
        expect(tick(25000, 0.02)[0].lapTimeMs).toBe(24000);
    });

    it.each([
        ['missing session clock', { ...sample(2000, { '7': 0.07 }), Graphics_game_time: undefined,
            Graphics_current_time_str: formatCornerTime(BASE_TIME + 2000), Graphics_current_time: BASE_TIME + 2000 }],
        ['invalid session clock', { ...sample(2000, { '7': 0.07 }), Graphics_game_time: NaN }],
        ['negative session clock', { ...sample(2000, { '7': 0.07 }), Graphics_game_time: -1 }],
        ['missing cars', sample(2000, {})],
        ['invalid position', sample(2000, { '7': NaN })],
        ['out of range position', sample(2000, { '7': 1.2 })],
        ['missing cars at a repeated timestamp', sample(1000, {})],
        ['paused telemetry', { ...sample(2000, { '7': 0.07 }), Graphics_status: 1 }],
        ['clock rollback', sample(500, { '7': 0.07 })],
        ['gap over two seconds', sample(3001, { '7': 0.07 })],
        ['reverse movement', sample(2000, { '7': 0.01 })],
        ['forward jump', sample(2000, { '7': 0.13 })],
    ] as Array<[string, StandardTelemetrySample]>)('discards the incomplete lap on %s and recovers', (_reason, interruption) => {
        tick(0, 0.97);
        tick(1000, 0.02);
        expect(recorder.tick(interruption, 2)).toEqual([]);
        const records: DriverLapRecord[] = [];
        for (let i = 0; i <= 40; i += 1) {
            records.push(...tick(4000 + i * 1000, ((7 + i * 5) % 100) / 100));
            if (i <= 19) expect(records).toEqual([]);
        }
        expect(records).toHaveLength(1);
        expect(records[0].lapTimeMs).toBe(20000);
    });

    it('discards unfinished laps on reset and reverse finish-line crossings', () => {
        for (const interrupt of [() => recorder.reset(), () => tick(1500, 0.99)]) {
            recorder.reset();
            tick(0, 0.97);
            tick(1000, 0.02);
            interrupt();
            for (let i = 0; i < 20; i += 1) expect(tick(2000 + i * 1000, ((2 + i * 5) % 100) / 100)).toEqual([]);
        }
    });

    it('accepts exactly two seconds and 0.1 lap and treats 1 and 0 as the same finish line', () => {
        const records: DriverLapRecord[] = [];
        for (let i = 0; i <= 20; i += 1) {
            const position = ((9 + i) % 10) / 10;
            records.push(...tick(i * 2000, position === 0 ? 1 : position));
            if (position === 0) expect(tick(i * 2000 + 1, 0)).toEqual([]);
        }
        expect(records).toHaveLength(1);
        expect(records[0].lapTimeMs).toBe(20000);
    });

    it('invalidates only the missing car while keeping the other car timing', () => {
        const records: DriverLapRecord[] = [];
        for (let i = 0; i <= 21; i += 1) {
            const position = ((97 + i * 5) % 100) / 100;
            records.push(...recorder.tick(sample(i * 1000, i === 5 ? { '63': position } : { '7': position, '63': position }), i));
        }
        expect(records).toHaveLength(1);
        expect(records[0]).toMatchObject({ carId: '63', lapTimeMs: 20000 });
    });

    it('records player and opponent laps using session time while the player lap clock resets', () => {
        const records: DriverLapRecord[] = [];
        for (let i = 0; i <= 41; i += 1) {
            const playerPosition = ((97 + i * 4) % 100) / 100;
            records.push(...recorder.tick({
                ...sample(i * 1000, { '63': ((97 + i * 5) % 100) / 100 }),
                Graphics_game_time: (BASE_TIME + i * 1000) / 1000,
                Graphics_normalized_car_position: playerPosition,
                Graphics_current_time_str: formatCornerTime(playerPosition * 25000),
                Graphics_current_time: playerPosition * 25000,
            }, i));
        }
        expect(records.map(({ carId, lapTimeMs, isPlayer }) => ({ carId, lapTimeMs, isPlayer }))).toEqual([
            { carId: '63', lapTimeMs: 20000, isPlayer: false },
            { carId: '7', lapTimeMs: 25000, isPlayer: true },
            { carId: '63', lapTimeMs: 20000, isPlayer: false },
        ]);
    });
});
