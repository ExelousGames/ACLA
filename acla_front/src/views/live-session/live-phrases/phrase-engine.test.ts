import { createLiveTelemetryStore, LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import { COOLDOWN_MS, PHRASE_RULES, PhraseCondition, PhraseEngine, TELEMETRY_MAX_AGE_MS } from './phrase-engine';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';
import { circuitMap, shapedCornerMap, vision } from './test-fixtures';

const frame = (sample: StandardTelemetrySample, status = 2): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: status,
    committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedSequence: 0, committedCount: 0 },
});
const driving = { Physics_speed_kmh: 100, Graphics_normalized_car_position: 0.11 };
const map = circuitMap();
const defaultRule = 'inside-outbraking';
const ids = (engine: PhraseEngine, now: number) => engine.evaluate(now).events.map((event) => event.ruleId);
const update = (engine: PhraseEngine, now: number, sample: StandardTelemetrySample = driving) => {
    engine.receiveVision(vision(now), now);
    return engine.receiveTelemetry(frame(sample), now);
};

describe('vision and Live Map overtaking guides', () => {
    it('reports individual fits even when another condition fails or has missing input', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const belowThreshold = update(engine, 0, { ...driving, Physics_speed_kmh: 29 });
        const conditions = belowThreshold.rules.find((rule) => rule.id === defaultRule)!.conditions;
        expect(conditions.find((condition) => condition.input === 'speed')).toMatchObject({ conditionFit: false, inputMissing: false });
        expect(conditions.filter((condition) => condition.input !== 'speed').every((condition) => condition.conditionFit)).toBe(true);

        engine.receiveVision(null, 100);
        const missingVision = engine.receiveTelemetry(frame({ ...driving, Physics_speed_kmh: 30 }), 100);
        const current = missingVision.rules.find((rule) => rule.id === defaultRule)!;
        expect(current.status).toBe('Missing input');
        expect(current.conditions.find((condition) => condition.input === 'speed')).toMatchObject({ conditionFit: true, inputMissing: false });
        expect(current.conditions.find((condition) => condition.input === 'carAhead')).toMatchObject({ conditionFit: false, inputMissing: true });
        expect(current.conditions.find((condition) => condition.input === 'phase')).toMatchObject({ conditionFit: true, inputMissing: false });
    });

    it('evaluates all conditions in guides suppressed by a higher-priority match', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(circuitMap('slow', true), 0);
        engine.receiveVision(vision(0, { player: 'outside', opponent: 'inside' }), 0);
        const snapshot = engine.receiveTelemetry(frame(driving), 0);
        expect(snapshot.rules.find((rule) => rule.id === 'next-corner')?.status).toBe('Confirming');
        const switchback = snapshot.rules.find((rule) => rule.id === 'switchback')!;
        expect(switchback.status).toBe('Not matched');
        expect(switchback.conditions.every((condition) => condition.conditionFit)).toBe(true);
    });

    it('keeps evaluated condition instances independent of past snapshots, other engines and the catalog', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const before = update(engine, 0);
        const after = update(engine, 100, { ...driving, Physics_speed_kmh: 0 });
        const other = new PhraseEngine().evaluate(100);
        expect(before.rules.find((rule) => rule.id === defaultRule)!.conditions.every((condition) => condition.conditionFit)).toBe(true);
        expect(after.rules.find((rule) => rule.id === defaultRule)!.conditions.find((condition) => condition.input === 'speed')?.conditionFit).toBe(false);
        expect(other.rules.every((rule) => rule.conditions.every((condition) => !condition.conditionFit))).toBe(true);
        for (const rule of [...PHRASE_RULES, ...before.rules, ...after.rules]) {
            rule.conditions.forEach((condition) => expect(condition).toBeInstanceOf(PhraseCondition));
        }
        expect(PHRASE_RULES.every((rule) => rule.conditions.every((condition) => !condition.conditionFit))).toBe(true);
    });

    it('uses published analysis without accessing raw screen detections', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const result = vision(0, { corner: 'right' });
        Object.defineProperty(result, 'detections', { get: () => { throw new Error('Phrase rules must not read screen detections'); } });
        engine.receiveVision(result, 0);
        engine.receiveTelemetry(frame(driving), 0);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([defaultRule]);
    });

    it('lists shape-specific guides before general overtaking guides', () => {
        expect(PHRASE_RULES.map((rule) => rule.id)).toEqual([
            'next-corner', 'same-direction', 'sequence-exit', 's-bend',
            'tightening-corner', 'opening-corner', 'hairpin-exit',
            'inside-outbraking', 'around-outside', 'switchback',
            'better-exit', 'slipstream', 'pressure-feint',
        ]);
        PHRASE_RULES.forEach((rule) => {
            expect(rule.sentence).not.toMatch(/you are .+; the opponent ahead is/);
            expect(rule.conditions.find((condition) => condition.input === 'phase')?.description).toContain('Live Map section');
        });
    });

    it.each([
        { shape: 's-bend', position: 0.12, tactic: 's-bend' },
        { shape: 'tightening', position: 0.12, tactic: 'tightening-corner' },
        { shape: 'opening', position: 0.2, tactic: 'opening-corner' },
        { shape: 'hairpin', position: 0.2, tactic: 'hairpin-exit' },
    ] as const)('selects $tactic from centerline geometry', ({ shape, position, tactic }) => {
        const engine = new PhraseEngine();
        engine.receiveMap(shapedCornerMap(shape), 0);
        const sample = { ...driving, Graphics_normalized_car_position: position };
        update(engine, 0, sample);
        update(engine, 800, sample);
        expect(ids(engine, 800)).toEqual([tactic]);
    });

    it.each(['opposite', 'same'] as const)('uses the %s direction sequence guide and reserves exit guidance for the last corner', (direction) => {
        const sequence = circuitMap('slow', true);
        if (direction === 'same') sequence.samples.middle_line!.find((point) => point.normalized_position === 0.31)!.x = 1100;
        sequence.centerline_tags!.push({ id: 'sequence', label: 'consecutive corners', start_position: 0.1, end_position: 0.31 });
        for (const [position, tactic] of [
            [0.11, direction === 'opposite' ? 'next-corner' : 'same-direction'],
            [0.18, 'sequence-exit'],
            [0.29, direction === 'same' ? 'hairpin-exit' : 'better-exit'],
        ] as const) {
            const engine = new PhraseEngine();
            engine.receiveMap(sequence, 0);
            const sample = { ...driving, Graphics_normalized_car_position: position };
            engine.receiveVision(vision(0, { player: 'outside', opponent: 'inside' }), 0);
            engine.receiveTelemetry(frame(sample), 0);
            engine.receiveTelemetry(frame(sample), 800);
            expect(ids(engine, 800)).toEqual([tactic]);
        }
    });

    it('does not inherit a partly confirmed phrase when moving to another corner in the same area', () => {
        const sequence = shapedCornerMap('hairpin');
        sequence.centerline_segments = [
            { id: 'area', tags: ['consecutive corners'], start_position: 0.1, end_position: 0.3 },
            { id: 'first', tags: ['corner', 'slow'], start_position: 0.1, end_position: 0.2 },
            { id: 'second', tags: ['corner', 'slow'], start_position: 0.2, end_position: 0.3 },
        ];
        const engine = new PhraseEngine();
        engine.receiveMap(sequence, 0);
        const first = { ...driving, Graphics_normalized_car_position: 0.11 };
        const second = { ...driving, Graphics_normalized_car_position: 0.21 };
        update(engine, 0, first);
        update(engine, 700, second);
        update(engine, 800, second);
        expect(ids(engine, 800)).toEqual([]);
        update(engine, 1500, second);
        expect(ids(engine, 1500)).toEqual(['inside-outbraking']);
    });

    it.each([
        { tactic: 'inside-outbraking', speed: 'slow', position: 0.11, player: 'inside', opponent: 'outside' },
        { tactic: 'around-outside', speed: 'fast', position: 0.11, player: 'outside', opponent: 'inside' },
        { tactic: 'switchback', speed: 'slow', position: 0.15, player: 'outside', opponent: 'inside' },
        { tactic: 'better-exit', speed: 'slow', position: 0.18, player: 'middle', opponent: 'inside' },
        { tactic: 'better-exit', speed: 'fast', position: 0.18, player: 'middle', opponent: 'inside' },
        { tactic: 'next-corner', speed: 'fast', position: 0.11, player: 'outside', opponent: 'inside', linked: true },
        { tactic: 'next-corner', speed: 'slow', position: 0.15, player: 'outside', opponent: 'inside', linked: true },
        { tactic: 'pressure-feint', speed: 'slow', position: 0.11, player: 'middle', opponent: 'outside' },
        { tactic: 'slipstream', speed: 'slow', position: 0.5, player: 'middle', opponent: 'middle', straight: true },
    ] as const)('selects only $tactic for $speed at $position', (scenario) => {
        const engine = new PhraseEngine();
        engine.receiveMap(circuitMap(scenario.speed, 'linked' in scenario), 0);
        const sample = { ...driving, Graphics_normalized_car_position: scenario.position };
        engine.receiveVision(vision(0, { corner: 'straight' in scenario ? 'straight' : 'left', player: scenario.player, opponent: scenario.opponent }), 0);
        engine.receiveTelemetry(frame(sample), 0);
        engine.receiveTelemetry(frame(sample), 800);
        expect(ids(engine, 800)).toEqual([scenario.tactic]);
    });

    it.each([null, { ...map, centerline_tags: [] }, circuitMap('fast')])('withholds inside outbraking without a mapped slow corner', (input) => {
        const engine = new PhraseEngine();
        engine.receiveMap(input, 0);
        update(engine, 0);
        update(engine, 800);
        expect(ids(engine, 800)).toEqual([]);
    });

    it.each([undefined, NaN, -0.1, 1.1])('withholds guidance for invalid lap position %s', (position) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const sample = { ...driving, Graphics_normalized_car_position: position };
        update(engine, 0, sample);
        update(engine, 800, sample);
        expect(ids(engine, 800)).toEqual([]);
    });

    it('restarts confirmation when the map is removed or replaced', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        engine.receiveMap(null, 500);
        update(engine, 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveMap(circuitMap(), 900);
        update(engine, 1000);
        update(engine, 1700);
        expect(ids(engine, 1700)).toEqual([defaultRule]);
    });

    it('emits once after a sustained match, only on a fresh telemetry frame', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, 799);
        engine.receiveVision(vision(800), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 800);
        update(engine, 1000);
        expect(ids(engine, 1000)).toEqual([defaultRule]);
    });

    it.each(['other-track', 'other-game'])('withholds an old map after switching to %s', (change) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        const event = frame({ ...driving, Static_track: change === 'other-track' ? 'spa' : 'test' });
        if (change === 'other-game') event.update.game = 'iracing';
        engine.receiveTelemetry(event, 800);
        const result = engine.receiveTelemetry(event, 1600);
        expect(result.mapReady).toBe(false);
        expect(result.events).toEqual([]);
    });

    it('normalizes ACC display names when matching the current map', () => {
        const engine = new PhraseEngine();
        engine.receiveMap({ ...map, source_track_key: 'brands_hatch' }, 0);
        update(engine, 0, { ...driving, Static_track: 'Brands Hatch Circuit' });
        update(engine, 800, { ...driving, Static_track: 'Brands Hatch Circuit' });
        expect(ids(engine, 800)).toEqual([defaultRule]);
    });

    it('preserves the cooldown across map reloads', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, 800);
        engine.receiveMap(null, 900);
        engine.receiveMap(circuitMap(), 1500);
        update(engine, 1600);
        update(engine, 2400);
        expect(ids(engine, 2400)).toEqual([defaultRule]);
        expect(engine.evaluate(2400).rules.find((rule) => rule.id === defaultRule)?.status).toBe('Cooldown');
    });

    it('restarts the hold when moving straight into another tagged section', () => {
        const engine = new PhraseEngine();
        const split = circuitMap();
        split.centerline_tags = [
            { id: 'first', label: 'straight', start_position: 0.4, end_position: 0.5 },
            { id: 'second', label: 'straight', start_position: 0.5, end_position: 0.8 },
        ];
        engine.receiveMap(split, 0);
        update(engine, 0, { ...driving, Graphics_normalized_car_position: 0.49 });
        update(engine, 700, { ...driving, Graphics_normalized_car_position: 0.51 });
        update(engine, 800, { ...driving, Graphics_normalized_car_position: 0.52 });
        expect(ids(engine, 800)).toEqual([]);
        update(engine, 1500, { ...driving, Graphics_normalized_car_position: 0.53 });
        expect(ids(engine, 1500)).toEqual(['slipstream']);
    });

    it.each([
        driving,
        { ...driving, Physics_g_force_x: -20, Physics_g_force_y: 50, Physics_g_force_z: 100 },
        { ...driving, Physics_g_force_x: NaN, Physics_g_force_y: Infinity },
        { ...driving, Physics_brake: 1 },
    ])('uses the same visual positions regardless of G-forces or brake data: %s', (sample) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0, sample);
        update(engine, 800, sample);
        expect(ids(engine, 800)).toEqual([defaultRule]);
    });

    it('requires a clear period and cooldown before repeating a suggestion', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
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
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(sample), 0);
        engine.receiveTelemetry(frame(sample), 1000);
        expect(ids(engine, 1000)).toEqual([]);
    });

    it.each([
        ['missing vision', null],
        ['missing published analysis', { ...vision(0), analysis: null }],
        ['car pack without an individual position', { ...vision(0), analysis: { ...vision(0).analysis, opponents: [] } }],
        ['missing camera calibration', vision(0, { cameraOffset: null })],
        ['no car ahead', vision(0, { carAhead: false })],
        ['straight road', vision(0, { corner: 'straight' })],
        ['future timestamp', vision(5000)],
        ['depth only', { capturedAt: 0, analysis: null, width: 100, height: 100, detections: { depth: {
            task: 'depth' as const, width: 1, height: 1, values: new Float32Array([10]), inferenceMs: 1, classNames: [],
        } } }],
    ])('withholds every suggestion with %s', (_label, detection) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(detection, 0);
        engine.receiveTelemetry(frame(driving), 0);
        engine.receiveTelemetry(frame(driving), 1000);
        expect(ids(engine, 1000)).toEqual([]);
    });

    it('expires vision while telemetry remains live', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(driving), 1300);
        const snapshot = engine.receiveTelemetry(frame(driving), VISION_MAX_AGE_MS + 1);
        expect(snapshot.telemetryReady).toBe(true);
        expect(snapshot.visionReady).toBe(false);
        expect(snapshot.events).toEqual([]);
        const conditions = snapshot.rules.find((rule) => rule.id === defaultRule)!.conditions;
        expect(conditions.find((condition) => condition.input === 'speed')).toMatchObject({ conditionFit: true, inputMissing: false });
        expect(conditions.find((condition) => condition.input === 'carAhead')).toMatchObject({ conditionFit: false, inputMissing: true });
    });

    it('restarts the hold when vision stops or capture resumes after a gap', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        engine.receiveVision(null, 500);
        update(engine, 700);
        update(engine, 1499);
        expect(ids(engine, 1499)).toEqual([]);
        update(engine, 1500);
        expect(ids(engine, 1500)).toEqual([defaultRule]);

        const gap = new PhraseEngine();
        gap.receiveMap(map, 0);
        gap.receiveVision(vision(0), 0);
        gap.receiveTelemetry(frame(driving), 1400);
        update(gap, 2100);
        expect(ids(gap, 2100)).toEqual([]);
        update(gap, 2900);
        expect(ids(gap, 2900)).toEqual([defaultRule]);
    });

    it('restarts the hold when visual positions change', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        engine.receiveVision(vision(500, { player: 'middle', opponent: 'outside' }), 500);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 1300);
        expect(ids(engine, 1300)).toEqual(['pressure-feint']);
    });

    it('restarts the hold when the camera alignment changes even within the same position band', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        engine.receiveVision(vision(500, { cameraOffset: -0.35 }), 500);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 1300);
        expect(ids(engine, 1300)).toEqual([defaultRule]);
    });

    it('uses all velocity components as a speed fallback without position data', () => {
        const sample = { Graphics_normalized_car_position: 0.11, Physics_velocity_x: 6, Physics_velocity_y: 0, Physics_velocity_z: 8 };
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(sample), 0);
        engine.receiveTelemetry(frame(sample), 800);
        expect(ids(engine, 800)).toEqual([defaultRule]);
        const other = new PhraseEngine();
        other.receiveMap(map, 0);
        other.receiveVision(vision(0), 0);
        other.receiveTelemetry(frame({ ...sample, Physics_velocity_y: undefined }), 0);
        other.receiveTelemetry(frame({ ...sample, Physics_velocity_y: undefined }), 800);
        expect(ids(other, 800)).toEqual([]);
    });

    it.each([0, 1, 3])('suppresses output for simulator status %s', (status) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(driving, status), 0);
        engine.receiveTelemetry(frame(driving, status), 1000);
        expect(ids(engine, 1000)).toEqual([]);
        expect(engine.evaluate(1000).telemetryReady).toBe(false);
    });

    it('expires telemetry and does not count a silent gap toward the hold', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, TELEMETRY_MAX_AGE_MS + 1);
        expect(ids(engine, TELEMETRY_MAX_AGE_MS + 1)).toEqual([]);
        const expired = engine.evaluate(TELEMETRY_MAX_AGE_MS * 2 + 2);
        expect(expired.telemetryReady).toBe(false);
        expect(expired.events).toEqual([]);
        const conditions = expired.rules.find((rule) => rule.id === defaultRule)!.conditions;
        expect(conditions.find((condition) => condition.input === 'speed')).toMatchObject({ conditionFit: false, inputMissing: true });
        expect(conditions.find((condition) => condition.input === 'phase')).toMatchObject({ conditionFit: false, inputMissing: true });
        expect(conditions.find((condition) => condition.input === 'carAhead')).toMatchObject({ conditionFit: true, inputMissing: false });
    });

    it.each(['session-reset', 'stream-reset'] as const)('clears history, holds and old vision on %s', (type) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, 800);
        const reset = engine.receiveTelemetry({ type, snapshot: createLiveTelemetryStore().getSnapshot() }, 900);
        expect(reset.rules.every((rule) => rule.conditions.every((condition) => !condition.conditionFit && condition.inputMissing))).toBe(true);
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
        engine.receiveMap(map, 0);
        for (let index = 0; index < 60; index++) {
            const now = index * (COOLDOWN_MS + 1000);
            update(engine, now);
            update(engine, now + 800);
            engine.receiveVision(null, now + 900);
        }
        expect(engine.evaluate(60 * (COOLDOWN_MS + 1000)).events).toHaveLength(50);
    });
});
