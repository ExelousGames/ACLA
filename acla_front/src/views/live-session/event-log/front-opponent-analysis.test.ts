import type { StandardTelemetrySample } from '../live-session-types';
import type { DriverCornerRecord } from './CornerRecorder';
import type { DriverLapRecord } from './LapRecorder';
import { analyzeFrontOpponent } from './front-opponent-analysis';

const sample: StandardTelemetrySample = {
    Graphics_player_car_id: 7, Graphics_normalized_car_position: 0.95,
    Graphics_normalized_positions: { '7': 0.96, '63': 0.02, '1052': 0.08, '20': 0.94 },
};
const point = { timeMs: 0, position: 0, sampleIndex: 0 };
const cornerRecords = (carId: string, cornerId: string, times: Array<number | null>, cornerName = cornerId): DriverCornerRecord[] => times.map((cornerTimeMs, i) => ({
    id: `${carId}:${cornerId}:${i}`, carId, isPlayer: carId === '7', cornerId, cornerName,
    entry: cornerTimeMs === null ? null : point, exit: point, cornerTimeMs,
    decelerationStart: null, accelerationStart: null, decelerationToExitMs: null,
    exitSpeed: null, mapId: 'track',
}));
const lapRecords = (carId: string, times: number[]): DriverLapRecord[] => times.map((lapTimeMs, i) => ({
    id: `${carId}:${i}`, carId, isPlayer: carId === '7', start: point, end: point, lapTimeMs,
}));
const three = (time: number) => [time - 100, time, time + 100];

describe('front opponent analysis', () => {
    it('selects the nearest car across the finish line, excluding the player regardless of lap or race order', () => {
        expect(analyzeFrontOpponent({ ...sample, Graphics_completed_lap: 100 }, [], [])).toMatchObject({
            status: 'insufficient_data', player_car_id: '7', opponent_car_id: '63',
            comparison_methods: { minimum_samples_per_car: 3, lap_average: expect.stringContaining('finish_line_crossings') },
        });
        expect(analyzeFrontOpponent({ ...sample, Graphics_normalized_car_position: 0.01 }, [], []).opponent_car_id).toBe('63');
    });

    it.each([
        [{}, /player ID/],
        [{ ...sample, Graphics_player_car_id: -1 }, /player ID/],
        [{ ...sample, Graphics_player_car_id: 1.5 }, /player ID/],
        [{ ...sample, Graphics_normalized_car_position: NaN }, /normalized position/],
        [{ ...sample, Graphics_status: 1 }, /Live player/],
        [{ ...sample, Graphics_normalized_positions: undefined }, /telemetry is unavailable/],
        [{ ...sample, Graphics_normalized_car_position: 0.25, Graphics_normalized_positions: {
            '7': 0.3, '63': 0.25, '1052': 0.75, '20': 0.24, '21': NaN, '22': 1.1,
        } }, /No qualifying opponent/],
    ] as Array<[StandardTelemetrySample, RegExp]>)('explains unavailable opponent selection: %p', (telemetry, reason) => {
        const result = analyzeFrontOpponent(telemetry, [], []);
        expect(result.status).toBe('unavailable');
        expect(result.opponent_car_id).toBeNull();
        expect(result.reasons[0]).toMatch(reason);
        expect(result.overall_lap_comparison).toMatchObject({ status: 'unavailable', delta_ms: null, reason: expect.stringMatching(reason) });
    });

    it('uses trimmed full-lap averages, matches corners by ID, and sorts both signs by magnitude', () => {
        const corners = [
            ...cornerRecords('7', 'fast-small', three(900), 'Same name'),
            ...cornerRecords('7', 'slow-small', three(1100)),
            ...cornerRecords('7', 'equal', three(1000)),
            ...cornerRecords('7', 'fast-large', three(700)),
            ...cornerRecords('7', 'slow-large', three(1500)),
            ...['slow-large', 'fast-large', 'equal', 'slow-small', 'fast-small'].flatMap((id) => cornerRecords('63', id, three(1000), 'Same name')),
            ...cornerRecords('1052', 'unrelated', three(500)),
        ];
        const laps = [...lapRecords('7', [1000, 60000, 61000, 62000, 63000, 64000, 65000, 900000]),
            ...lapRecords('63', three(63000))];
        const result = analyzeFrontOpponent(sample, corners, laps);
        expect(result.status).toBe('complete');
        expect(result.reasons).toEqual([]);
        expect(result.overall_lap_comparison).toMatchObject({
            status: 'available', player_average_time_ms: 62500, opponent_average_time_ms: 63000,
            player_sample_count: 8, opponent_sample_count: 3, player_included_count: 6, opponent_included_count: 1, delta_ms: -500,
        });
        expect(result.faster_corners.map(({ corner_id, delta_ms }) => [corner_id, delta_ms])).toEqual([['fast-large', -300], ['fast-small', -100]]);
        expect(result.slower_corners.map(({ corner_id, delta_ms }) => [corner_id, delta_ms])).toEqual([['slow-large', 500], ['slow-small', 100]]);
        expect(result.equal_corners).toEqual([expect.objectContaining({ corner_id: 'equal', delta_ms: 0, player_sample_count: 3, opponent_sample_count: 3 })]);
    });

    it('rounds sub-millisecond deltas before classifying and does not return negative zero', () => {
        const result = analyzeFrontOpponent(sample, [
            ...cornerRecords('7', 'rounded-equal', [900, 1000, 1001, 1000, 1100]),
            ...cornerRecords('63', 'rounded-equal', three(1000)),
            ...cornerRecords('7', 'negative-equal', three(1000)),
            ...cornerRecords('63', 'negative-equal', [900, 1000, 1001, 1000, 1100]),
        ], []);
        expect(result.equal_corners).toHaveLength(2);
        expect(result.equal_corners.map(({ delta_ms }) => delta_ms)).toEqual([0, 0]);
        expect(result.faster_corners).toEqual([]);
        expect(result.slower_corners).toEqual([]);
    });

    it('returns corner results immediately with explicit insufficient corners and unavailable laps', () => {
        const result = analyzeFrontOpponent(sample, [
            ...cornerRecords('7', 'ready', three(900)), ...cornerRecords('63', 'ready', three(1000)),
            ...cornerRecords('7', 'short', [null, NaN, 0, -1, 1000, 1100]), ...cornerRecords('63', 'short', three(1000)),
            ...cornerRecords('63', 'opponent-only', three(1000)),
        ], [...lapRecords('7', [60000, 61000]), ...lapRecords('63', three(62000))]);
        expect(result.status).toBe('partial');
        expect(result.faster_corners).toHaveLength(1);
        expect(result.overall_lap_comparison).toMatchObject({
            status: 'unavailable', delta_ms: null, player_average_time_ms: null, opponent_average_time_ms: 62000,
            player_sample_count: 2, reason: expect.stringContaining('uninterrupted Graphics_current_time_str'),
        });
        expect(result.insufficient_corners).toEqual([
            expect.objectContaining({ corner_id: 'short', player_sample_count: 2, opponent_sample_count: 3, reason: expect.stringContaining('three complete corner passes') }),
            expect.objectContaining({ corner_id: 'opponent-only', player_sample_count: 0, opponent_sample_count: 3 }),
        ]);
    });

    it('returns lap-only comparisons without fabricating corner data and reports player slower', () => {
        const result = analyzeFrontOpponent(sample, [], [...lapRecords('7', three(61000)), ...lapRecords('63', three(60000))]);
        expect(result.status).toBe('partial');
        expect(result.overall_lap_comparison).toMatchObject({ status: 'available', delta_ms: 1000 });
        expect(result.reasons).toEqual([expect.stringContaining('No corner timing records')]);
    });

    it.each([0, 1, 2])('requires three complete laps per car, ignoring invalid laps (%i valid)', (count) => {
        const result = analyzeFrontOpponent(sample, [], [
            ...lapRecords('7', [NaN, Infinity, 0, -1, ...three(60000).slice(0, count)]), ...lapRecords('63', three(60000)),
        ]);
        expect(result.status).toBe('insufficient_data');
        expect(result.overall_lap_comparison).toMatchObject({ status: 'unavailable', player_sample_count: count, player_average_time_ms: null, delta_ms: null });
    });
});
