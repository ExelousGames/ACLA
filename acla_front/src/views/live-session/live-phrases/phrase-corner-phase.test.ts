import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import { copyPhraseReference } from './phrase-driving-reference';
import { readReferenceGuidance } from './phrase-reference-guidance';

type Knot = readonly [number, number];
const shape: readonly Knot[] = [[0, 0.05], [0.04, 0.3], [0.08, 0.3], [0.12, 0.2], [0.16, 0.1], [0.2, 0]];
const interpolate = (position: number, knots: readonly Knot[]) => {
    if (position < 0 || position > 0.2) return 0;
    const index = Math.max(1, knots.findIndex(([at]) => at >= position));
    const [from, a] = knots[index - 1], [to, b] = knots[index];
    return a + (b - a) * (position - from) / (to - from);
};
function reference(game: 'acc' | 'iracing' = 'acc', sign: -1 | 1 = 1, start = 0.1,
    knots: readonly Knot[] = shape): TopLapReferenceEntry {
    return {
        game, track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_steering', 'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_speed'],
        x: Array.from({ length: 200 }, (_, i) => i / 200),
        y: Array.from({ length: 200 }, (_, i) => {
            const angle = i / 200 * Math.PI * 2, ground = Math.sin(angle) * 200 * sign;
            const progress = ((i / 200 - start) % 1 + 1) % 1;
            return [Math.cos(angle) * 200, game === 'acc' ? ground : 0, game === 'iracing' ? ground : 0,
                interpolate(progress, knots) * sign, 0, 0.5, 90];
        }),
    };
}
const sample = (position: number): StandardTelemetrySample => ({
    Static_track: 'test', Static_car_model: 'car', Graphics_track_grip_status: 2,
    Graphics_normalized_car_position: position, Physics_speed_kmh: 90, Physics_brake: 0, Physics_gas: 0.5,
});
const read = (lap: TopLapReferenceEntry, position: number, sign: -1 | 1 = 1, start = 0.1,
    patch: StandardTelemetrySample = {}) => readReferenceGuidance({ ...sample(position), ...patch }, null, lap, lap.game, 1000,
    lap, undefined, undefined, sign, { start, end: (start + 0.2) % 1 });

it.each(['acc', 'iracing'] as const)('follows loaded steering and final unwind in both directions in %s without vision', (game) => {
    for (const sign of [-1, 1] as const) {
        const lap = reference(game, sign);
        expect(read(lap, 0.12, sign).cornerPhase).toBeUndefined();
        expect(read(lap, 0.16, sign)).toMatchObject({ cornerPhase: 'hold', steeringPhase: 'turn' });
        expect(read(lap, 0.23, sign)).toMatchObject({ cornerPhase: 'unwind', cornerExit: 2500 });
        expect(read(lap, 0.31, sign).cornerPhase).toBeUndefined();
    }
});

it.each([false, true])('keeps phases across the lap seam (duplicated endpoint: %s)', (duplicate) => {
    const lap = reference('acc', 1, 0.9);
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    expect(read(lap, 0.96, 1, 0.9).cornerPhase).toBe('hold');
    expect(read(lap, 0.03, 1, 0.9)).toMatchObject({ cornerPhase: 'unwind', cornerExit: 2500 });
});

it.each([0.25, 0.3, 0.4])('waits through a temporary release before a later steering peak of %s', (laterPeak) => {
    const lap = reference('acc', 1, 0.1, [[0, 0.05], [0.04, 0.3], [0.08, 0.1],
        [0.12, laterPeak], [0.14, laterPeak], [0.2, 0]]);
    expect(read(lap, 0.165).cornerExit).toBeUndefined();
    expect(read(lap, 0.265)).toMatchObject({ cornerPhase: 'unwind', cornerExit: 2500 });
});

it('does not treat acceleration with constant loaded steering as an unwind', () => {
    const lap = reference('acc', 1, 0.1, [[0, 0.3], [0.2, 0.3]]);
    lap.y.forEach((row, i) => { row[6] = 80 + i; });
    const guidance = read(lap, 0.23, 1, 0.1, { Physics_speed_kmh: 126 });
    expect(guidance.acceleration).toBeGreaterThan(0);
    expect(guidance.cornerPhase).toBe('hold');
    expect(guidance.cornerExit).toBeUndefined();
});

it.each(['absent', 'invalid', 'zero', 'opposing'] as const)('withholds phases when steering is %s', (failure) => {
    const lap = reference();
    if (failure === 'absent') lap.target_features[3] = 'missing';
    if (failure === 'invalid') lap.y[30][3] = NaN;
    if (failure === 'zero') lap.y.forEach((row) => { row[3] = 0; });
    if (failure === 'opposing') lap.y[30][3] = -0.3;
    expect(read(lap, 0.16).cornerPhase).toBeUndefined();
    expect(read(lap, 0.23).cornerExit).toBeUndefined();
});

it.each([0, 50, 100])('keeps steering phases independent of driving aggression %s', (aggression) => {
    const lap = copyPhraseReference(reference(), aggression, [{ start: 0.1, end: 0.3 }])!;
    expect(read(lap, 0.16).cornerPhase).toBe('hold');
    expect(read(lap, 0.23).cornerExit).toBe(2500);
});

it.each([{ Physics_brake: 0.1 }, { Physics_brake: undefined }, { Physics_speed_kmh: 19 }])(
    'withholds unwind speech without adequate live speed and a released brake: %j', (patch) => {
        const guidance = read(reference(), 0.23, 1, 0.1, patch);
        expect(guidance.cornerPhase).toBe('unwind');
        expect(guidance.cornerExit).toBeUndefined();
    });

it('keeps required braking ahead of unwind speech', () => {
    const lap = reference();
    lap.y.forEach((row) => { row[4] = 0.3; });
    expect(read(lap, 0.23)).toMatchObject({ cornerPhase: 'unwind', braking: true });
    expect(read(lap, 0.23).cornerExit).toBeUndefined();
    expect(read(reference(), 0.23, 1, 0.1, { Physics_speed_kmh: 110 }).cornerExit).toBeUndefined();
});

it('requires a matching mapped corner direction and bounds', () => {
    const lap = reference();
    expect(read(lap, 0.23, -1).cornerPhase).toBeUndefined();
    expect(readReferenceGuidance(sample(0.23), null, lap, lap.game, 1000).cornerPhase).toBeUndefined();
});
