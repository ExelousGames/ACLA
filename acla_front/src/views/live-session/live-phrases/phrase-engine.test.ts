import { createLiveTelemetryStore, LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import { COOLDOWN_MS, PHRASE_RULES, PhraseEngine, TELEMETRY_MAX_AGE_MS, describeConditions } from './phrase-engine';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';
import type { CornerPosition } from '../track-vision/track-vision-types';
import { vision } from './test-fixtures';

const frame = (sample: StandardTelemetrySample, status = 2): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: status,
    committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedSequence: 0, committedCount: 0 },
});
const driving = { Physics_speed_kmh: 100 };
const defaultRule = 'corner-left-player-inside-opponent-outside';
const ids = (engine: PhraseEngine, now: number) => engine.evaluate(now).events.map((event) => event.ruleId);
const update = (engine: PhraseEngine, now: number, sample: StandardTelemetrySample = driving) => {
    engine.receiveVision(vision(now), now);
    return engine.receiveTelemetry(frame(sample), now);
};

describe('visual corner position phrase rules', () => {
    it('uses published analysis without accessing raw screen detections', () => {
        const engine = new PhraseEngine();
        const result = vision(0, { corner: 'right', player: 'outside', opponent: 'inside' });
        Object.defineProperty(result, 'detections', { get: () => { throw new Error('Phrase rules must not read screen detections'); } });
        engine.receiveVision(result, 0);
        engine.receiveTelemetry(frame(driving), 0);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual(['corner-right-player-outside-opponent-inside']);
    });

    it('describes both cars in every rule without passing advice or G-force conditions', () => {
        expect(PHRASE_RULES).toHaveLength(18);
        expect(PHRASE_RULES.every((rule) => rule.category === 'Corner position'
            && ['cornerDirection', 'playerPosition', 'opponentPosition'].every((input) => rule.conditions.some((condition) => condition.input === input)))).toBe(true);
        PHRASE_RULES.forEach((rule) => {
            expect(rule.sentence).toMatch(/you are .+; the opponent ahead is/);
            expect(rule.sentence).not.toMatch(/\b(?:pass|opening|exit|brake)\b/i);
            expect(describeConditions(rule)).not.toMatch(/force|acceleration|brake/i);
        });
    });

    it('emits once after a sustained match, only on a fresh telemetry frame', () => {
        const engine = new PhraseEngine();
        update(engine, 0);
        update(engine, 799);
        engine.receiveVision(vision(800), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 800);
        update(engine, 1000);
        expect(ids(engine, 1000)).toEqual([defaultRule]);
    });

    const positions: CornerPosition[] = ['inside', 'middle', 'outside'];
    it.each((['left', 'right'] as const).flatMap((corner) => positions.flatMap((player) => positions.map((opponent) => (
        { corner, player, opponent }
    )))))('emits one accurate phrase for $corner / $player / $opponent', ({ corner, player, opponent }) => {
        const engine = new PhraseEngine();
        engine.receiveVision(vision(0, { corner, player, opponent }), 0);
        engine.receiveTelemetry(frame(driving), 0);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([`corner-${corner}-player-${player}-opponent-${opponent}`]);
    });

    it.each([
        driving,
        { ...driving, Physics_g_force_x: -20, Physics_g_force_y: 50, Physics_g_force_z: 100 },
        { ...driving, Physics_g_force_x: NaN, Physics_g_force_y: Infinity },
        { ...driving, Physics_brake: 1 },
    ])('uses the same visual positions regardless of G-forces or brake data: %s', (sample) => {
        const engine = new PhraseEngine();
        update(engine, 0, sample);
        update(engine, 800, sample);
        expect(ids(engine, 800)).toEqual([defaultRule]);
    });

    it('requires a clear period and cooldown before repeating a suggestion', () => {
        const engine = new PhraseEngine();
        update(engine, 0);
        update(engine, 800);
        update(engine, 900, { ...driving, Physics_speed_kmh: 0 });
        update(engine, 1000);
        update(engine, 1800);
        expect(ids(engine, 1800)).toEqual([defaultRule]);
        update(engine, 1900, { ...driving, Physics_speed_kmh: 0 });
        update(engine, 2500);
        const snapshot = update(engine, 3300);
        expect(snapshot.rules.find((rule) => rule.id === defaultRule)?.status).toBe('Cooldown');
        for (let now = 3800; now <= 8800; now += 500) update(engine, now);
        expect(ids(engine, 8800)).toEqual([defaultRule, defaultRule]);
    });

    it.each([
        {},
        { Physics_speed_kmh: undefined },
        { Physics_speed_kmh: NaN },
        { Physics_speed_kmh: -1 },
        { ...driving, Physics_speed_kmh: 29 },
    ])('does not fill absent or invalid driving inputs with zero: %s', (sample) => {
        const engine = new PhraseEngine();
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(sample), 0);
        engine.receiveTelemetry(frame(sample), 1000);
        expect(ids(engine, 1000)).toEqual([]);
    });

    it.each([
        ['missing vision', null],
        ['missing published analysis', { ...vision(0), analysis: null }],
        ['car pack without an individual position', { ...vision(0), analysis: { cornerDirection: 'left' as const, playerPosition: 'inside' as const, carAhead: 1 as const } }],
        ['missing camera alignment', vision(0, { playerCenterX: null })],
        ['no car ahead', vision(0, { carAhead: false })],
        ['straight road', vision(0, { corner: 'straight' })],
        ['future timestamp', vision(5000)],
        ['depth only', { capturedAt: 0, analysis: null, width: 100, height: 100, detections: { depth: {
            task: 'depth' as const, width: 1, height: 1, values: new Float32Array([10]), inferenceMs: 1, classNames: [],
        } } }],
    ])('withholds every suggestion with %s', (_label, detection) => {
        const engine = new PhraseEngine();
        engine.receiveVision(detection, 0);
        engine.receiveTelemetry(frame(driving), 0);
        engine.receiveTelemetry(frame(driving), 1000);
        expect(ids(engine, 1000)).toEqual([]);
    });

    it('expires vision while telemetry remains live', () => {
        const engine = new PhraseEngine();
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(driving), 1300);
        const snapshot = engine.receiveTelemetry(frame(driving), VISION_MAX_AGE_MS + 1);
        expect(snapshot.telemetryReady).toBe(true);
        expect(snapshot.visionReady).toBe(false);
        expect(snapshot.events).toEqual([]);
    });

    it('restarts the hold when vision stops or capture resumes after a gap', () => {
        const engine = new PhraseEngine();
        update(engine, 0);
        engine.receiveVision(null, 500);
        update(engine, 700);
        update(engine, 1499);
        expect(ids(engine, 1499)).toEqual([]);
        update(engine, 1500);
        expect(ids(engine, 1500)).toEqual([defaultRule]);

        const gap = new PhraseEngine();
        gap.receiveVision(vision(0), 0);
        gap.receiveTelemetry(frame(driving), 1400);
        update(gap, 2100);
        expect(ids(gap, 2100)).toEqual([]);
        update(gap, 2900);
        expect(ids(gap, 2900)).toEqual([defaultRule]);
    });

    it('restarts the hold when visual positions change', () => {
        const engine = new PhraseEngine();
        update(engine, 0);
        engine.receiveVision(vision(500, { player: 'outside', opponent: 'inside' }), 500);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 1300);
        expect(ids(engine, 1300)).toEqual(['corner-left-player-outside-opponent-inside']);
    });

    it('restarts the hold when the camera alignment changes even within the same position band', () => {
        const engine = new PhraseEngine();
        update(engine, 0);
        engine.receiveVision(vision(500, { playerCenterX: 0.51 }), 500);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 1300);
        expect(ids(engine, 1300)).toEqual([defaultRule]);
    });

    it('uses all velocity components as a speed fallback without position data', () => {
        const sample = { Physics_velocity_x: 6, Physics_velocity_y: 0, Physics_velocity_z: 8 };
        const engine = new PhraseEngine();
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(sample), 0);
        engine.receiveTelemetry(frame(sample), 800);
        expect(ids(engine, 800)).toEqual([defaultRule]);
        const other = new PhraseEngine();
        other.receiveVision(vision(0), 0);
        other.receiveTelemetry(frame({ ...sample, Physics_velocity_y: undefined }), 0);
        other.receiveTelemetry(frame({ ...sample, Physics_velocity_y: undefined }), 800);
        expect(ids(other, 800)).toEqual([]);
    });

    it.each([0, 1, 3])('suppresses output for simulator status %s', (status) => {
        const engine = new PhraseEngine();
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(driving, status), 0);
        engine.receiveTelemetry(frame(driving, status), 1000);
        expect(ids(engine, 1000)).toEqual([]);
        expect(engine.evaluate(1000).telemetryReady).toBe(false);
    });

    it('expires telemetry and does not count a silent gap toward the hold', () => {
        const engine = new PhraseEngine();
        update(engine, 0);
        update(engine, TELEMETRY_MAX_AGE_MS + 1);
        expect(ids(engine, TELEMETRY_MAX_AGE_MS + 1)).toEqual([]);
        const expired = engine.evaluate(TELEMETRY_MAX_AGE_MS * 2 + 2);
        expect(expired.telemetryReady).toBe(false);
        expect(expired.events).toEqual([]);
    });

    it.each(['session-reset', 'stream-reset'] as const)('clears history, holds and old vision on %s', (type) => {
        const engine = new PhraseEngine();
        update(engine, 0);
        update(engine, 800);
        engine.receiveTelemetry({ type, snapshot: createLiveTelemetryStore().getSnapshot() }, 900);
        engine.receiveVision(vision(800), 900);
        engine.receiveTelemetry(frame(driving), 1000);
        engine.receiveTelemetry(frame(driving), 1800);
        expect(ids(engine, 1800)).toEqual([]);
        update(engine, 1900);
        update(engine, 2700);
        expect(ids(engine, 2700)).toEqual([defaultRule]);
    });

    it('bounds session history', () => {
        const engine = new PhraseEngine();
        for (let index = 0; index < 60; index++) {
            const now = index * (COOLDOWN_MS + 1000);
            update(engine, now);
            update(engine, now + 800);
            engine.receiveVision(null, now + 900);
        }
        expect(engine.evaluate(60 * (COOLDOWN_MS + 1000)).events).toHaveLength(50);
    });
});
