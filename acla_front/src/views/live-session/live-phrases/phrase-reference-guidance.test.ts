import { conditionTestEngine } from './test-fixtures';
import { PhraseInstance } from './phrase';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { createLiveTelemetryStore, type LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { readReferenceGuidance, referenceRangeSegments } from './phrase-reference-guidance';
import { copyPhraseReference } from './phrase-driving-reference';
import { brakeReleaseSettings, SLOWLY_RELEASE_BRAKE } from './phrase-brake-release';
import { getPhraseSpeechInstances } from './phrase-speech';
import { createPhraseRoot, PHRASE_DEFINITIONS } from './phrase-engine';
import { Closure } from './closure';
import { circuitMap, variableBrakingReference, vision as trafficVision } from './test-fixtures';

beforeEach(() => jest.spyOn(Math, 'random').mockReturnValue(0));
afterEach(() => jest.restoreAllMocks());

// A 1000 m square lap, sampled every 10 m. Independent, known control onsets.
function referenceLap(): TopLapReferenceEntry {
    return { game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_steering', 'expert_optimal_brake', 'expert_optimal_throttle'],
        x: Array.from({ length: 100 }, (_, i) => i / 100),
        y: Array.from({ length: 100 }, (_, i) => {
            const d = i * 10;
            const xy = d < 250 ? [d, 0] : d < 500 ? [250, d - 250] : d < 750 ? [750 - d, 250] : [0, 1000 - d];
            return [...xy, 0, i >= 41 && i <= 49 ? 0.4 : 0, i >= 21 && i <= 29 ? 0.8 : 0, i >= 61 && i <= 90 ? 0.8 : 0];
        }),
    };
}
function addSpeed(lap: TopLapReferenceEntry) {
    lap.target_features.push('expert_optimal_speed');
    lap.y.forEach((row, i) => row.push(i < 20 || i >= 70 ? 120
        : i < 30 ? 120 - (i - 20) * 4 : i <= 60 ? 80 : 80 + (i - 60) * 4));
}
const driving = (position: number): StandardTelemetrySample => ({ Static_track: 'test', Static_car_model: 'car',
    Graphics_normalized_car_position: position, Physics_speed_kmh: 36, Physics_steer_angle: 0, Physics_brake: 0, Physics_gas: 0 });
// These fixtures have 10 m per 0.01 progress (the circular fixture supplies its chord length).
// Position-based rule tests specify the decision point, so place measured telemetry 500 ms before it.
const beforeDecision = (sample: StandardTelemetrySample, metresPerLap = 1000): StandardTelemetrySample => {
    const position = sample.Graphics_normalized_car_position, speed = sample.Physics_speed_kmh;
    if (position === undefined || speed === undefined || !Number.isFinite(speed) || speed < 0) return sample;
    const before = position - speed / 3.6 * 0.5 / metresPerLap;
    return { ...sample, Graphics_normalized_car_position: before < 0 ? before + 1 : before };
};
const localMotion = (vx: number, vy: number, heading = 0): StandardTelemetrySample => ({
    Physics_heading: heading, Physics_pitch: 0, Physics_roll: 0,
    Physics_local_velocity_x: vx * Math.cos(heading) + vy * Math.sin(heading),
    Physics_local_velocity_y: -vx * Math.sin(heading) + vy * Math.cos(heading), Physics_local_velocity_z: 0,
});
function detection(lap: TopLapReferenceEntry, sample: StandardTelemetrySample, now = 1000): TrackVisionDetection {
    return { capturedAt: now, width: 100, height: 100, detections: {}, reconstruction: null, geometry: null,
        telemetry: { ...createLiveTelemetryStore().getSnapshot(), game: 'acc', receivedAt: now, currentTelemetry: sample,
            telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0 },
        position: { coordinateFrame: 'car_local', inferenceMs: 10, referenceDelta: [0, 0, 0],
            normalizedPosition: sample.Graphics_normalized_car_position!, referenceLap: lap, references: { reference: { x: 0, y: 0, z: 0 } } } };
}
const frame = (sample: StandardTelemetrySample): LiveTelemetryFrameEvent => ({ type: 'frame', sample, sampleIndex: 0,
    telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0, committedSampleCount: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 } });

const steeringGuidance = (sample: StandardTelemetrySample, lap: TopLapReferenceEntry, cornerTurnSign: -1 | 1 = 1,
    cornerBounds = { start: 0.25, end: 0.55 }) =>
    readReferenceGuidance(sample, null, lap, lap.game, 1000, lap, undefined, undefined, cornerTurnSign, cornerBounds);

// Keep control channels independent from geometry; each segment is exactly 10 raw coordinate units.
function setTrajectory(lap: TopLapReferenceEntry, heading: (index: number) => number) {
    let x = 0, y = 0;
    lap.y.forEach((row, i) => {
        row[0] = x;
        row[lap.game === 'acc' ? 1 : 2] = y;
        row[lap.game === 'acc' ? 2 : 1] = 0;
        const angle = heading(i) * Math.PI / 180;
        x += 10 * Math.cos(angle);
        y += 10 * Math.sin(angle);
    });
}
function trajectoryLap(degrees = 12, sign = 1, game = 'acc') {
    const lap = referenceLap();
    lap.game = game;
    addSpeed(lap);
    lap.y.forEach((row) => { row[3] = 0; row[6] = 36; });
    // Start the first changing segment at 250; a centered tangent must not move onset back to 240.
    setTrajectory(lap, (i) => sign * degrees * (Math.max(0, Math.min(15, i - 25)) + (i === 25 ? 0.5 : 0)));
    return lap;
}

describe('Brake-zone direction-change exit', () => {
    it.each([0.22, 0.25, 0.27])('releases at or below turn-in speed independently of local deceleration at %s', (position) => {
        const lap = trajectoryLap(2);
        lap.y.forEach((row, i) => { row[6] = i < 25 ? 60 : i === 25 ? 36 : 20; });
        for (const speed of [35.99, 36, 36.01, 50]) {
            const sample = { ...driving(position), Physics_speed_kmh: 50, Physics_brake: 0.8 };
            const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000, lap,
                undefined, undefined, 1, { start: 0.25, end: 0.55 }, false, position,
                { ...sample, Physics_speed_kmh: speed });
            expect(guidance.releaseBrake).toBe(speed <= 36 ? 2500 : undefined);
        }
    });

    it.each(['missing speed', 'invalid speed', 'missing corner', 'released pedal', 'invalid pedal'])(
        'withholds release with %s', (kind) => {
            const lap = trajectoryLap(2);
            if (kind === 'missing speed') lap.target_features[6] = 'missing';
            if (kind === 'invalid speed') lap.y[25][6] = NaN;
            const sample = { ...driving(0.26), Physics_speed_kmh: 35,
                Physics_brake: kind === 'released pedal' ? 0 : kind === 'invalid pedal' ? NaN : 0.8 };
            const guidance = kind === 'missing corner' ? readReferenceGuidance(sample, null, lap, 'acc', 1000)
                : steeringGuidance(sample, lap);
            expect(guidance.releaseBrake).toBeUndefined();
        });

    it.each([0.3, 0.31])('releases after the reference brake interval at %s while keeping speed and pressure limits', (position) => {
        const lap = trajectoryLap(2);
        for (const speed of [35.99, 36, 36.01]) {
            for (const brake of [0, 0.099, 0.1, 0.8]) {
                const guidance = steeringGuidance({ ...driving(position), Physics_speed_kmh: speed, Physics_brake: brake }, lap);
                expect(guidance.pedalZone).toBe('none');
                expect(guidance.releaseBrake).toBe(speed <= 36 && brake >= 0.1 ? 2500 : undefined);
                expect(guidance.holdBrake).toBeUndefined();
            }
        }
    });

    it('uses predicted speed to release before the measured speed reaches turn-in speed', () => {
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, trajectoryLap(2));
        const at = (position: number, speed: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
        }), now);
        expect(at(0.22, 40, 1000).events).toEqual([]);
        const released = at(0.2211, 39, 1100);
        expect(released.state.path).toEqual(['root', 'Brake zone', 'Release the brake', 'say phrase']);
        expect(released.events.map(({ sentence }) => sentence)).toEqual(['Off brake.']);
        expect(engine.evaluate(1101).state.path).toEqual(['root', 'Brake zone']);
        expect(at(0.222, 36, 1200).state.path).toEqual(['root', 'Brake zone']);
        expect(at(0.223, 35.99, 1300).state.path).toEqual(['root']);
    });

    it('releases when the speed prediction reaches the target beyond the reference brake interval', () => {
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, trajectoryLap(2));
        const at = (position: number, speed: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
        }), now);
        expect(at(0.294, 42, 1000).events).toEqual([]);
        const released = at(0.2951, 41, 1100);
        expect(released.state.path).toEqual(['root', 'Brake zone', 'Release the brake', 'say phrase']);
        expect(released.events.map(({ sentence }) => sentence)).toEqual(['Off brake.']);
    });

    it('does not speak Release outside an entered Brake zone', () => {
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, trajectoryLap(2));
        const entered = engine.receiveTelemetry(frame({ ...driving(0.31), Physics_speed_kmh: 35, Physics_brake: 0.8 }), 1000);
        expect(entered.state.path).toEqual(['root', 'Steering']);
        expect(entered.events).toEqual([]);
    });

    it.each([false, true])('finishes release speech before the speed exit (playback started: %s)', (started) => {
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, trajectoryLap(2));
        engine.setSpeechDurations(new Map(getPhraseSpeechInstances().map((instance) => [instance.cacheKey, 500])));
        const at = (position: number, speed: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
        }), now);
        expect(at(0.21, 42, 1000).state.path).toContain('Brake zone');
        const released = at(0.23, 35, 1100);
        expect(released.state.path).toEqual(['root', 'Brake zone', 'Release the brake', 'say phrase']);
        const event = released.events[0];
        expect(event.sentence).toBe('Off brake.');
        if (started) engine.speechStarted(event.id, 1150);
        expect(at(0.24, 35, 1200).state.path).toContain('Release the brake');
        expect(engine.speechFinished(event.id, 1250).state.path).toEqual(['root']);
        expect(at(0.26, 35, 1300).events.map(({ sentence }) => sentence)).toEqual(['Off brake.']);
    });

    it('waits for release on the next telemetry frame when Brake playback finishes below target', () => {
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, trajectoryLap(2));
        engine.setSpeechDurations(new Map(getPhraseSpeechInstances().map((instance) => [instance.cacheKey, 500])));
        const at = (position: number, speed: number, brake: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed, Physics_brake: brake,
        }), now);
        const braking = at(0.21, 42, 0, 1000);
        expect(braking.events[0].sentence).toBe('Brake.');
        expect(at(0.23, 35, 0.8, 1100).state.path).toContain('Brake');
        expect(engine.speechFinished(braking.events[0].id, 1150).state.path).toContain('Brake zone');
        const released = at(0.24, 35, 0.8, 1200);
        expect(released.events.map(({ sentence }) => sentence)).toEqual(['Brake.', 'Off brake.']);
        expect(released.state.path).toContain('Release the brake');
    });

    it.each([0.22, 0.25, 0.27])('exits strictly below the direction-start target at position %s', (position) => {
        const lap = trajectoryLap(2);
        lap.y.forEach((row, i) => { row[6] = i < 25 ? 60 : i === 25 ? 36 : 20; });
        for (const speed of [35.99, 36, 36.01, 50]) {
            const guidance = steeringGuidance({ ...driving(position), Physics_speed_kmh: speed }, lap);
            expect(guidance.brakeZoneExit).toBe(speed < 36);
            expect(guidance.steeringPhase).toBe(position < 0.25 ? undefined : 'prepare');
        }
    });

    it('exits before turn-in and keeps Steering waiting for its direction range', () => {
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, trajectoryLap(2));
        const at = (position: number, speed: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
        }), now);
        expect(at(0.21, 42, 1000).state.path).toContain('Brake zone');
        expect(at(0.23, 35, 1100).state.path).toEqual(['root', 'Brake zone', 'Release the brake', 'say phrase']);
        expect(at(0.24, 35, 1200).state.path).toEqual(['root']);
        expect(at(0.26, 35, 1300).state.path).toContain('Steering');
    });

    it('keeps a quiet brake zone after turn-in until speed falls below the start target', () => {
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, trajectoryLap(2));
        const at = (position: number, speed: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
        }), now);
        expect(at(0.21, 42, 1000).state.path).toContain('Brake zone');
        expect(at(0.25, 42, 1100).state.path).toEqual(['root', 'Brake zone']);
        expect(at(0.26, 36, 1200).state.path).toContain('Brake zone');
        expect(at(0.27, 35.99, 1300).state.path).toContain('Steering');
    });

    it.each(['missing', 'invalid'])('withholds the speed exit with %s reference speed', (kind) => {
        const lap = trajectoryLap(2);
        if (kind === 'missing') lap.target_features[6] = 'missing speed';
        else lap.y[25][6] = NaN;
        const guidance = steeringGuidance({ ...driving(0.26), Physics_speed_kmh: 10 }, lap);
        expect(guidance.brakeZoneExit).toBeUndefined();
        expect(guidance.steeringPhase).toBe('prepare');
    });

    it('uses the interpolated target at the clipped direction start from the active reference', () => {
        const source = trajectoryLap(2), active = trajectoryLap(2);
        active.y[27][6] = 40;
        active.y[28][6] = 20;
        const target = Math.sqrt((40 ** 2 + 20 ** 2) / 2);
        for (const speed of [target - 0.01, target, target + 0.01, 35]) {
            const sample = { ...driving(0.26), Physics_speed_kmh: speed, Physics_brake: 0.8 };
            const guidance = readReferenceGuidance(sample, null, active, 'acc', 1000, source,
                undefined, undefined, 1, { start: 0.275, end: 0.35 });
            expect(guidance.brakeZoneExit).toBe(speed < target);
            expect(guidance.releaseBrake).toBe(speed <= target ? 2500 : undefined);
            expect(guidance.steeringRange).toBeNull();
        }
    });

    it.each([false, true])('uses the same start target before and across the lap seam (duplicate: %s)', (duplicate) => {
        const lap = trajectoryLap(2);
        lap.y.forEach((row, i) => { row[6] = i < 25 ? 60 : i === 25 ? 36 : 20; });
        lap.y = [...lap.y.slice(30), ...lap.y.slice(0, 30)];
        if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
        for (const position of [0.92, 0.95, 0.99, 0, 1, 0.01]) {
            for (const speed of [35.99, 36, 36.01]) {
                const guidance = steeringGuidance({ ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8 }, lap, 1,
                    { start: 0.95, end: 0.05 });
                expect(guidance.brakeZoneExit).toBe(speed < 36);
                expect(guidance.releaseBrake).toBe(speed <= 36 ? 2500 : undefined);
            }
        }
    });

    it.each([0.22, 0.31])('uses the mapped turn-in speed after reference braking ends from position %s', (position) => {
        const lap = trajectoryLap(2);
        setTrajectory(lap, (i) => 2 * (Math.max(0, Math.min(15, i - 35)) + (i === 35 ? 0.5 : 0)));
        lap.y.forEach((row, i) => { row[6] = i < 35 ? 60 : i === 35 ? 36 : 20; });
        for (const speed of [0, 35.99, 36, 36.01]) {
            const guidance = steeringGuidance({ ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8 }, lap);
            expect(guidance.brakeZoneExit).toBe(speed < 36);
            expect(guidance.releaseBrake).toBe(speed <= 36 ? 2500 : undefined);
        }
    });

    it.each([0.22, 0.31])('releases at a complete stop in an active Brake zone before later mapped turn-in from %s', (position) => {
        const lap = trajectoryLap(2);
        setTrajectory(lap, (i) => 2 * (Math.max(0, Math.min(15, i - 35)) + (i === 35 ? 0.5 : 0)));
        const map: CircuitMapDto = { ...circuitMap(),
            samples: { middle_line: lap.y.map((row, i) => ({ bin: i, normalized_position: lap.x[i],
                x: row[0], y: row[1], z: row[2], sample_count: 1, updated_at: '' })) },
            centerline_segments: [{ id: 'late-turn', tags: ['corner'], start_position: 0.25, end_position: 0.55 }],
        };
        const engine = conditionTestEngine();
        engine.receiveMap(map, 0, lap);
        const at = (position: number, speed: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
        }), now);
        expect(at(0.22, 42, 1000).state.path).toContain('Brake zone');
        expect(at(position, 42, 1100).state.path).toEqual(['root', 'Brake zone']);
        const released = at(position, 0, 1200);
        expect(released.state.path).toEqual(['root', 'Brake zone', 'Release the brake', 'say phrase']);
        expect(released.events.map(({ sentence }) => sentence)).toEqual(['Off brake.']);
        const exited = at(position, 0, 1300);
        expect(exited.state.path).toEqual(['root']);
        expect(exited.events).toEqual(released.events);
    });

    it('uses the first upcoming direction change in the mapped corner', () => {
        const lap = trajectoryLap(2);
        setTrajectory(lap, (i) => 2 * Math.max(0, Math.min(3, i - 25))
            + 2 * Math.max(0, Math.min(10, i - 35)));
        lap.y.forEach((row, i) => { row[4] = i >= 21 && i < 40 ? 0.8 : 0; row[6] = i < 35 ? 36 : 60; });
        for (const speed of [35, 40]) {
            expect(steeringGuidance({ ...driving(0.22), Physics_speed_kmh: speed }, lap).brakeZoneExit).toBe(speed < 36);
        }
    });

    it.each([false, true])('uses mapped turn-in beyond reference braking across the lap seam (duplicate: %s)', (duplicate) => {
        const lap = trajectoryLap(2);
        setTrajectory(lap, (i) => 2 * (Math.max(0, Math.min(15, i - 35)) + (i === 35 ? 0.5 : 0)));
        lap.y = [...lap.y.slice(33), ...lap.y.slice(0, 33)];
        if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
        for (const position of [0.89, 0.98, 0, 1, 0.01]) {
            const guidance = steeringGuidance({ ...driving(position), Physics_speed_kmh: 0, Physics_brake: 0.8 }, lap, 1,
                { start: 0.92, end: 0.22 });
            expect(guidance.brakeZoneExit).toBe(true);
            expect(guidance.releaseBrake).toBe(2500);
        }
    });

    it('does not reuse a passed direction change or borrow one outside the mapped corner', () => {
        const lap = trajectoryLap(2);
        setTrajectory(lap, (i) => 2 * Math.max(0, Math.min(3, i - 25))
            + 2 * Math.max(0, Math.min(10, i - 60)));
        const guidance = steeringGuidance({ ...driving(0.32), Physics_speed_kmh: 0, Physics_brake: 0.8 }, lap);
        expect(guidance.brakeZoneExit).toBe(false);
        expect(guidance.releaseBrake).toBeUndefined();
    });

    it.each([0, 10, 19.99])('stays in Brake zone past the pedal boundary without reference speed at %s km/h', (speed) => {
        const lap = trajectoryLap(2);
        lap.target_features[6] = 'missing speed';
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, lap);
        const at = (position: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed,
        }), now);
        const braking = at(0.24, 1000);
        expect(braking.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
        expect(braking.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
        expect(at(0.25, 1100).state.path).toContain('Brake zone');
        const afterRelease = at(0.30, 1200);
        expect(afterRelease.state.path).toEqual(['root', 'Brake zone']);
        expect(afterRelease.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    });

    it.each(['acc', 'iracing'] as const)('gives way to Steering below target at a gentle range start in %s', (game) => {
        for (const sign of [-1, 1] as const) {
            const lap = trajectoryLap(2, sign, game);
            const engine = conditionTestEngine();
            engine.receiveMap(steeringMap(sign, game), 0, lap);
            const at = (position: number, now: number) => {
                const event = frame({ ...driving(position), Physics_speed_kmh: position < 0.2451 ? 36 : 35.99,
                    Physics_brake: 0.8, Physics_steer_angle: -sign });
                event.update.game = game;
                return engine.receiveTelemetry(event, now);
            };
            const approaching = at(0.2441, 1000);
            expect(approaching.state.path).toEqual(['root', 'Brake zone', 'Release the brake', 'say phrase']);
            const exited = at(0.2451, 1100);
            expect(exited.state.path).toEqual(['root', 'Steering']);
            expect(exited.referenceSteeringPhase).toBe('prepare');
            expect(exited.events).toEqual(approaching.events);
            expect(at(0.251, 1200).state.path).toEqual(['root', 'Steering']);
            expect(at(0.26, 1300).state.path).toEqual(['root', 'Steering']);
        }
    });

    it.each(['constant', 'falling', 'opposite', 'missing', 'invalid'])('ignores %s reference steering for brake exit', (kind) => {
        const lap = trajectoryLap(2);
        lap.y.forEach((row, i) => { row[3] = kind === 'falling' ? 1 - i / 100 : kind === 'opposite' ? -0.5 : 0; });
        if (kind === 'missing') lap.target_features[3] = 'missing';
        if (kind === 'invalid') lap.y[25][3] = NaN;
        expect(steeringGuidance({ ...driving(0.25), Physics_speed_kmh: 35, Physics_steer_angle: undefined }, lap))
            .toMatchObject({ brakeZoneExit: true, pedalZone: 'brake' });
    });

    it.each([0, 50])(
        'waits until below the direction-start speed with %s%% caution', (caution) => {
            const lap = trajectoryLap(2);
            const engine = conditionTestEngine();
            engine.receiveMap(steeringMap(), 0, lap);
            engine.setDrivingAggression(caution, 0);
            const adjusted = copyPhraseReference(lap, caution)!;
            const at = (position: number, excess: number, now: number) => {
                const after = adjusted.x.findIndex((point) => point > 0.25), before = after - 1;
                const fraction = (0.25 - adjusted.x[before]) / (adjusted.x[after] - adjusted.x[before]);
                const a = adjusted.y[before][6], b = adjusted.y[after][6];
                const speed = b < a ? Math.sqrt(a ** 2 + (b ** 2 - a ** 2) * fraction) : a + (b - a) * fraction;
                return engine.receiveTelemetry(frame({
                    ...driving(position), Physics_speed_kmh: speed + excess, Physics_brake: 0.8,
                }), now);
            };
            const matched = at(0.244, 0, 900);
            expect(matched.state.path).toContain('Brake zone');
            expect(matched.events.map(({ sentence }) => sentence)).toEqual(['Off brake.']);
            const started = at(0.25, 1, 901);
            expect(started.referenceSteeringPhase).toBe('prepare');
            expect(started.state.path).toContain('Brake zone');
            expect(at(0.265, 0.000001, 902).state.path).toContain('Brake zone');
            expect(at(0.265, 0, 903).state.path).toContain('Brake zone');
            expect(at(0.266, -1, 904).state.path).toEqual(['root', 'Steering']);
        });

    it.each(['opposite turn', 'no overlap', 'flat trajectory'])(
        'keeps the brake zone with %s', (kind) => {
            const lap = trajectoryLap(2);
            if (kind === 'flat trajectory') setTrajectory(lap, () => 0);
            const map = steeringMap(kind === 'opposite turn' ? -1 : 1);
            if (kind === 'no overlap') setTrajectory(lap, (i) => 2 * Math.max(0, Math.min(2, i - 20)));
            const engine = conditionTestEngine();
            engine.receiveMap(map, 0, lap);
            const position = kind === 'no overlap' ? 0.21 : 0.25;
            const result = engine.receiveTelemetry(frame({ ...driving(position), Physics_brake: 0.8 }), 1000);
            expect(result.state.path.slice(0, 2)).toEqual(['root', 'Brake zone']);
        });

    it.each([['brake release', 36], ['brake release', 35.99], ['throttle onset', 36], ['throttle onset', 35.99]] as const)(
        'finishes Release before leaving Brake zone after reference %s at %s km/h', (boundary, speed) => {
            const lap = trajectoryLap(2);
            lap.y.forEach((row, i) => {
                if (boundary === 'brake release' && i >= 25) row[4] = 0;
                row[5] = boundary === 'throttle onset' && i >= 26 ? 0.8 : 0;
            });
            const engine = conditionTestEngine();
            engine.receiveMap(steeringMap(), 0, lap);
            engine.setSpeechDurations(new Map(getPhraseSpeechInstances().map((instance) => [instance.cacheKey, 500])));
            const at = (position: number, speed: number, now: number) => engine.receiveTelemetry(frame({
                ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
            }), now);
            expect(at(0.244, 42, 1000).state.path).toContain('Brake zone');
            const entered = at(0.250001, 42, 1100);
            expect(entered.referenceSteeringPhase).toBe('prepare');
            expect(entered.state.path).toEqual(['root', 'Brake zone']);
            const released = at(0.265, speed, 1200);
            expect(released.state.path).toEqual(['root', 'Brake zone', 'Release the brake', 'say phrase']);
            expect(released.events.map(({ sentence }) => sentence)).toEqual(['Off brake.']);
            const event = released.events[0];
            engine.speechStarted(event.id, 1210);
            expect(at(0.266, 35.99, 1250).state.path).toContain('Release the brake');
            expect(engine.speechFinished(event.id, 1300).state.path).toEqual(['root']);
            const turned = at(0.267, 35.99, 1400);
            expect(turned.state.path).toEqual(['root', 'Steering']);
            expect(turned.events.map(({ sentence }) => sentence)).toEqual(['Off brake.']);
        });

    it.each([0, 10, 19.99, 36, 35, 42])('checks brake exit before the next Steering range at %s km/h', (speed) => {
        const lap = trajectoryLap(2);
        setTrajectory(lap, (i) => 2 * Math.max(0, Math.min(5, i - 25))
            + 2 * Math.max(0, Math.min(10, i - 35)));
        lap.y.forEach((row, i) => {
            row[3] = i < 31 ? 0.2 : 0;
            row[4] = i >= 32 && i <= 38 ? 0.8 : 0;
            row[5] = 0;
        });
        const map: CircuitMapDto = { ...circuitMap(),
            samples: { middle_line: lap.y.map((row, i) => ({ bin: i, normalized_position: lap.x[i],
                x: row[0], y: row[1], z: row[2], sample_count: 1, updated_at: '' })) },
            centerline_segments: [
                { id: 'turn-1', tags: ['corner'], start_position: 0.25, end_position: 0.31 },
                { id: 'turn-2', tags: ['corner'], start_position: 0.35, end_position: 0.46 },
            ],
        };
        const engine = conditionTestEngine();
        engine.receiveMap(map, 0, lap);
        const at = (position: number, speed: number, now: number) => engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: speed, Physics_brake: 0.8,
        }), now);
        expect(at(0.26, 36, 1000).events).toEqual([]);
        expect(at(0.31, 42, 1100).state.path).toEqual(['root']);
        expect(at(0.32, 42, 1200).state.path).toContain('Brake zone');
        const entered = at(0.36, speed, 1300);
        expect(entered.referenceSteeringPhase).toBe('prepare');
        expect(entered.state.path).toContain('Brake zone');
        expect(entered.events.some(({ sentence }) => sentence === 'Off brake.')).toBe(speed <= 36);
        expect(at(0.361, speed, 1500).state.path).toContain(speed < 36 ? 'Steering' : 'Brake zone');
        expect(at(0.37, 35, 1600).state.path).toContain(speed > 36 ? 'Release the brake' : 'Steering');
        expect(at(0.371, 35, 1700).state.path).toContain('Steering');
    });

    it.each([false, true])('releases after a brake phrase before exiting with fresh telemetry (expired first: %s)', (expired) => {
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, trajectoryLap(2));
        const braking = engine.receiveTelemetry(frame({ ...driving(0.24), Physics_speed_kmh: 42 }), 1000);
        expect(braking.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
        if (expired) expect(engine.evaluate(2601).state.path).toContain('Brake zone');
        const now = expired ? 2700 : 1100;
        const released = engine.receiveTelemetry(frame({ ...driving(0.25), Physics_speed_kmh: 35, Physics_brake: 0.8 }), now);
        expect(released.state.path).toEqual(['root', 'Brake zone', 'Release the brake', 'say phrase']);
        expect(released.events).toEqual([...braking.events, expect.objectContaining({ sentence: 'Off brake.' })]);
        const exited = engine.receiveTelemetry(frame({ ...driving(0.26), Physics_speed_kmh: 35, Physics_brake: 0.8 }), now + 100);
        expect(exited.state.path).toEqual(['root', 'Steering']);
        expect(exited.events).toEqual(released.events);
    });

    it('exits without waiting for a usable 500 ms heading endpoint', () => {
        const lap = trajectoryLap(2);
        lap.y[26][6] = 0;
        const guidance = steeringGuidance({ ...driving(0.25), Physics_speed_kmh: 35 }, lap);
        expect(guidance.brakeZoneExit).toBe(true);
        expect(guidance.steeringPhase).toBe('prepare');
    });

    it.each([false, true])('checks a qualifying direction-change range across start/finish (duplicate: %s)', (duplicate) => {
        const lap = trajectoryLap(2);
        lap.y.forEach((row, i) => {
            row[0] = 300 * Math.cos(i * 2 * Math.PI / 100);
            row[1] = 300 * Math.sin(i * 2 * Math.PI / 100);
        });
        if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
        for (const position of [0.998, 0.999, 0, 1, 0.01]) {
            const guidance = steeringGuidance({ ...driving(position), Physics_speed_kmh: 35 }, lap, 1, { start: 0.998, end: 0.02 });
            expect(guidance.brakeZoneExit).toBe(true);
            expect(guidance.steeringPhase).toBe('prepare');
        }
        for (const position of [0.99, 0.02, 0.03]) {
            expect(steeringGuidance(driving(position), lap, 1, { start: 0.998, end: 0.02 }))
                .toMatchObject({ brakeZoneExit: false, steeringRange: null });
        }
    });
});

it.each([-1, 1] as const)('speaks Turn without orientation and exits when trajectory flattens in direction %s', (sign) => {
    const lap = trajectoryLap(12, sign);
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.setDrivingAggression(100, 0);
    engine.receiveMap(steeringMap(sign), 0, lap);
    const at = (position: number, now: number) => engine.receiveTelemetry(frame({
        ...driving(position), Physics_steer_angle: sign,
    }), now);
    expect(at(0.26, 1000).state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(at(0.28, 1100).state.path).toEqual(['root', 'Steering', 'say phrase']);
    const exited = at(0.41, 1200);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events).toEqual([expect.objectContaining({ sentence: 'Turn in', timestamp: 1000 })]);
});

it.each([20, 36, 180])('uses the curve through Steering end for correction at %s km/h', (speed) => {
    const lap = trajectoryLap();
    // At 300 the reference tangent is 54 degrees; Steering ends at 410, leaving 110 coordinate units.
    const angle = 54 * Math.PI / 180;
    const sample = { ...driving(0.30), Physics_speed_kmh: speed,
        ...localMotion((Math.cos(angle) - 4 / 110) * speed, Math.sin(angle) * speed) };
    const view = detection(lap, sample);
    view.position!.referenceDelta = [4, 5, 0];
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000, lap, undefined, undefined, 1).steeringDirection).toBe('aligned');
});

it.each([-1, 1] as const)('follows the curved reference tangent instead of aiming across a hairpin in direction %s', (sign) => {
    const lap = trajectoryLap(12, sign);
    const angle = sign * 6 * Math.PI / 180;
    const sample = { ...driving(0.26), ...localMotion(10 * Math.cos(angle), 10 * Math.sin(angle)) };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, undefined, undefined, sign)
        .steeringDirection).toBe('aligned');
});

it('changes the correction with Steering end while the current reference geometry stays the same', () => {
    const angle = 6 * Math.PI / 180;
    const sample = { ...driving(0.26), ...localMotion(10 * Math.cos(angle), 10 * Math.sin(angle), angle - Math.PI / 2) };
    for (const [turnSegments, expected] of [[2, 'right'], [15, 'aligned']] as const) {
        const lap = trajectoryLap();
        setTrajectory(lap, (i) => 12 * Math.max(0, Math.min(turnSegments, i - 25)));
        const view = detection(lap, sample);
        view.position!.referenceDelta = [12, 0, 0];
        // The shorter turn leaves 20 units to rejoin; the longer turn leaves 150.
        expect(readReferenceGuidance(sample, view, lap, 'acc', 1000, lap, undefined, undefined, 1).steeringDirection).toBe(expected);
    }
});

it('uses the centerline corner end for direction correction and excludes entry positioning', () => {
    const lap = trajectoryLap();
    const angle = 6 * Math.PI / 180;
    const sample = { ...driving(0.26), ...localMotion(10 * Math.cos(angle), 10 * Math.sin(angle), angle - Math.PI / 2) };
    const view = detection(lap, sample);
    view.position!.referenceDelta = [12, 0, 0];
    const read = (bounds: { start: number; end: number }) =>
        readReferenceGuidance(sample, view, lap, 'acc', 1000, lap, undefined, undefined, 1, bounds);
    expect(read({ start: 0.25, end: 0.28 }).steeringDirection).toBe('right');
    expect(read({ start: 0.25, end: 0.41 }).steeringDirection).toBe('aligned');
    expect(read({ start: 0.27, end: 0.41 }).steeringDirection).toBeUndefined();
    expect(read({ start: 0.25, end: 0.26 }).steeringDirection).toBeUndefined();
});

it.each([false, true])('uses the remaining Steering range after a bend at start/finish with duplicated endpoint=%s', (duplicate) => {
    const lap = referenceLap();
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    // The actual bend begins at 0 and the filtered range ends at 10. At 5, 5 raw coordinate units remain.
    const angle = -Math.PI / 8;
    const sample = { ...driving(0.005), ...localMotion(10 * (Math.cos(angle) - 3 / 5), 10 * Math.sin(angle)) };
    const view = detection(lap, sample);
    view.position!.referenceDelta = [3, 0, 0];
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000, lap, undefined, undefined, 1).steeringDirection).toBe('aligned');
    const end = { ...sample, Graphics_normalized_car_position: 0.01 };
    expect(readReferenceGuidance(end, detection(lap, end), lap, 'acc', 1000, lap, undefined, undefined, 1).steeringDirection).toBeUndefined();
});

it('uses capture-paired motion and progress, and withholds direction when the reference error expires', () => {
    const lap = referenceLap();
    const captured = { ...driving(0.25), ...localMotion(10 / Math.sqrt(2) - 3, 10 / Math.sqrt(2)) };
    const view = detection(lap, captured);
    view.position!.referenceDelta = [3, 0, 0];
    const live = { ...captured, Graphics_normalized_car_position: 0.251, ...localMotion(-10, 0, Math.PI) };
    expect(readReferenceGuidance(live, view, lap, 'acc', 1100, lap, undefined, undefined, 1).steeringDirection).toBe('aligned');
    expect(readReferenceGuidance(live, view, lap, 'acc', 1501, lap, undefined, undefined, 1).steeringDirection).toBeUndefined();
    view.position!.referenceDelta[0] = NaN;
    expect(readReferenceGuidance(live, view, lap, 'acc', 1100, lap, undefined, undefined, 1).steeringDirection).toBeUndefined();
});

it.each(['acc', 'iracing'])('starts qualifying direction-change ranges at the same absolute threshold in %s', (game) => {
    for (const sign of [-1, 1] as const) {
        for (const degrees of [2, 9.998, 10, 10.002]) {
            const lap = trajectoryLap(degrees, sign, game);
            expect(steeringGuidance(driving(0.2499), lap, sign).steering).toBeUndefined();
            for (const position of [0.25, 0.2501, 0.26]) {
                const guidance = steeringGuidance(driving(position), lap, sign);
                expect(guidance.steeringPhase).toBe('prepare');
                expect(guidance.steeringRange).toEqual(expect.objectContaining({ turnSign: sign }));
            }
            expect(steeringGuidance(driving(0.41), lap, sign).steering).toBeUndefined();
        }
    }
});

it.each([20, 36, 180])('matches the reference direction-change range independently of live speed %s and steering input', (speed) => {
    const lap = trajectoryLap();
    for (const steer of [-1, 0, 1, undefined]) {
        expect(steeringGuidance({ ...driving(0.26), Physics_speed_kmh: speed, Physics_steer_angle: steer }, lap))
            .toMatchObject({ steering: 2500, steeringPhase: 'prepare', steeringRange: expect.any(Object) });
    }
});

it.each(['constant', 'falling', 'opposite', 'missing', 'invalid'])('enters and exits with %s reference steering', (kind) => {
    const lap = trajectoryLap();
    lap.y.forEach((row, i) => { row[3] = kind === 'falling' ? 1 - i / 100 : kind === 'opposite' ? -0.5 : 0; });
    if (kind === 'missing') lap.target_features[3] = 'missing';
    if (kind === 'invalid') lap.y[25][3] = NaN;
    expect(steeringGuidance(driving(0.26), lap).steering).toBe(2500);
    expect(steeringGuidance(driving(0.41), lap).steeringRange).toBeNull();
});

it('retains the local 0.08 reference steering threshold only for Turn', () => {
    const lap = trajectoryLap(2);
    lap.y[26][3] = 0.079;
    lap.y[27][3] = 0.08;
    lap.y[28][3] = 0.08;
    expect(steeringGuidance(driving(0.26), lap).steeringPhase).toBe('prepare');
    expect(steeringGuidance(driving(0.27), lap).steeringPhase).toBe('turn');
    expect(steeringGuidance(driving(0.28), lap).steeringPhase).toBe('turn');
});

it.each([
    [0.25, 0.45, true], // Entire remaining turn inside the corner.
    [0.35, 0.45, false], // Entry-position adjustment before the corner.
    [0.28, 0.32, true], // Current position is inside the corner.
    [0.314, 0.316, false], // The corner has not started yet.
    [0.25, 0.29, false], // Earlier overlap is behind the current point.
    [0.25, 0.30, false], [0.41, 0.45, false], // Boundary contact alone.
] as const)('checks direction only inside corner %s to %s', (start, end, overlaps) => {
    const lap = trajectoryLap();
    expect(steeringGuidance(driving(0.30), lap, 1, { start, end }).steeringPhase)
        .toBe(overlaps ? 'prepare' : undefined);
});

it.each(['acc', 'iracing'])('clips direction checking to exact centerline corner bounds in %s', (game) => {
    const lap = trajectoryLap(12, 1, game);
    const bounds = { start: 0.275, end: 0.355 };
    for (const position of [0.26, 0.274999, 0.355, 0.36]) {
        expect(steeringGuidance(driving(position), lap, 1, bounds)).toMatchObject({
            steeringRange: null, brakeZoneExit: false,
        });
        expect(steeringGuidance(driving(position), lap, 1, bounds).steering).toBeUndefined();
    }
    for (const position of [0.275, 0.3, 0.354999]) {
        const guidance = steeringGuidance(driving(position), lap, 1, bounds);
        expect(guidance).toMatchObject({ steeringPhase: 'prepare', brakeZoneExit: false });
        expect(guidance.steeringRange!.start).toBeCloseTo(275);
        expect(guidance.steeringRange!.end).toBeCloseTo(355);
    }
});

it.each(['straight', 'reversal'])('ends the direction change at a %s before a later corner', (kind) => {
    const lap = trajectoryLap();
    setTrajectory(lap, (i) => i <= 27 ? Math.max(0, i - 25) * 12
        : i <= 30 ? (kind === 'straight' ? 24 : 12) : 24 + (i - 30) * 12);
    expect(steeringGuidance(driving(0.26), lap, 1, { start: 0.30, end: 0.35 }).steering).toBeUndefined();
    expect(steeringGuidance(driving(0.31), lap, 1, { start: 0.30, end: 0.35 }).steeringPhase).toBe('prepare');
});

it('requires mapped corner bounds and a matching trajectory direction for entry, but not exit', () => {
    const lap = trajectoryLap();
    expect(readReferenceGuidance(driving(0.26), null, lap, 'acc', 1000).steering).toBeUndefined();
    expect(readReferenceGuidance(driving(0.26), null, lap, 'acc', 1000, lap, undefined, undefined, 1).steering).toBeUndefined();
    expect(steeringGuidance(driving(0.26), lap, -1).steering).toBeUndefined();
    expect(readReferenceGuidance(driving(0.41), null, lap, 'acc', 1000).steeringRange).toBeNull();
});

it.each(['missing speed', 'invalid speed', 'zero speed'])(
    'matches the direction-change range independently of %s', (kind) => {
        const lap = trajectoryLap();
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[27][6] = NaN;
        if (kind === 'zero speed') lap.y[27][6] = 0;
        const guidance = steeringGuidance(driving(0.26), lap);
        expect(guidance.steeringPhase).toBe('prepare');
        expect(guidance.steeringRange).toEqual(expect.objectContaining({ turnSign: 1 }));
    });

it.each(['invalid position', 'no horizontal direction'])(
    'withholds the direction-change range with %s', (kind) => {
        const lap = trajectoryLap();
        if (kind === 'invalid position') lap.y[27][0] = NaN;
        if (kind === 'no horizontal direction') lap.y.forEach((row, i) => { row[0] = row[1] = 0; row[2] = i; });
        const guidance = steeringGuidance(driving(0.26), lap);
        expect(guidance.steering).toBeUndefined();
        expect(guidance.steeringRange).toBeUndefined();
    });

it.each(['acceleration', 'braking'])('keeps the same direction-change match during reference %s', (kind) => {
    const lap = trajectoryLap(2);
    lap.y.forEach((row) => { row[0] *= 0.2; row[1] *= 0.2; });
    // Changes in future reference travel time do not affect range membership.
    lap.y.forEach((row) => { row[6] = kind === 'acceleration' ? 32.4 : 39.6; });
    for (let i = 31; i < 40; i++) lap.y[i][6] = kind === 'acceleration' ? 108 : 3.6;
    const guidance = steeringGuidance(driving(0.30), lap);
    expect(guidance.steeringPhase).toBe('prepare');
    expect(guidance.steeringRange).toEqual(expect.objectContaining({ turnSign: 1 }));
});

it('keeps a turn continuous and advances through an inserted fractional-distance sample', () => {
    const lap = trajectoryLap();
    const fraction = 0.86952861952862;
    const row = lap.y[26].map((value, axis) => value + (lap.y[27][axis] - value) * fraction);
    lap.x.splice(27, 0, 0.26 + 0.01 * fraction);
    lap.y.splice(27, 0, row);
    expect(steeringGuidance(driving(0.265), lap)).toMatchObject({ steeringPhase: 'prepare', steeringRange: expect.any(Object) });
});

it.each(['acc', 'iracing'] as const)('ends Steering at the centerline corner end during a continuing turn in %s', (game) => {
    const lap = trajectoryLap(12, 1, game);
    setTrajectory(lap, (i) => i < 30 ? 12 * Math.max(0, i - 25)
        : i < 35 ? 60 + 10 * (i - 30) : 110 + 8 * Math.min(5, i - 35));
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(steeringMap(1, game), 0, lap);
    const at = (position: number, now: number) => {
        const event = frame(driving(position));
        event.update.game = game;
        return engine.receiveTelemetry(event, now);
    };
    expect(at(0.26, 1000).state.path).toEqual(['root', 'Steering']);
    expect(at(0.32, 1100).state.path).toEqual(['root', 'Steering']);
    expect(at(0.3449, 1200).state.path).toEqual(['root', 'Steering']);
    // The measured position reaches corner end at 0.35 before the reference stops turning.
    expect(at(0.35, 1300).state.path).toEqual(['root']);
    expect(at(0.4059, 1400).state.path).toEqual(['root']);
});

describe.each(['acc', 'iracing'] as const)('direction-change boundaries in %s', (game) => {
    it.each([-1, 1] as const)('rejects nearly straight ranges below 1 degree per 10 raw coordinate units in direction %s', (sign) => {
        for (const degrees of [0.1, 0.5, 0.75, 0.999, 1, 1.001, 1.5]) {
            const lap = trajectoryLap(degrees, sign, game);
            const guidance = steeringGuidance(driving(0.30), lap, sign);
            if (degrees < 1) {
                expect(guidance).toMatchObject({ steeringRange: null, brakeZoneExit: false });
                expect(guidance.steeringPhase).toBeUndefined();
            } else {
                expect(guidance).toMatchObject({ steeringRange: expect.any(Object), brakeZoneExit: false });
            }
        }
    });

    it.each([-1, 1] as const)('excludes a shallow entry adjustment and filters the tail in direction %s', (sign) => {
        const lap = trajectoryLap(6, sign, game);
        setTrajectory(lap, (i) => sign * (0.75 * Math.max(0, Math.min(5, i - 20))
            + 6 * Math.max(0, Math.min(10, i - 25)) + 0.75 * Math.max(0, Math.min(10, i - 35))));
        const read = (position: number) => steeringGuidance(driving(position), lap, sign);
        for (const position of [0.20, 0.2099, 0.21, 0.24, 0.2499, 0.36, 0.40]) {
            expect(read(position)).toMatchObject({ steeringRange: null, brakeZoneExit: false });
            expect(read(position).steeringPhase).toBeUndefined();
        }
        const entered = read(0.25).steeringRange!;
        expect(entered).toMatchObject({ turnSign: sign });
        expect(entered.start).toBeCloseTo(250, 6);
        expect(entered.end).toBeCloseTo(360, 6);
        for (const position of [0.25, 0.30, 0.3599]) {
            expect(read(position)).toMatchObject({ steeringRange: entered, steeringPhase: 'prepare', brakeZoneExit: false });
        }
        const mapSegments = referenceRangeSegments(lap, [{ start: 0.25, end: 0.55 }])!.segments;
        for (let i = 20; i < 45; i++) expect(mapSegments[i].changesDirection).toBe(i >= 25 && i < 36);
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(steeringMap(sign, game), 0, lap);
        const at = (position: number, now: number) => {
            const event = frame(driving(position));
            event.update.game = game;
            return engine.receiveTelemetry(event, now);
        };
        expect(at(0.2441, 1000).state.path).toEqual(['root']);
        expect(at(0.2451, 1100).state.path).toContain('Steering');
        expect(at(0.3449, 1200).state.path).toContain('Steering');
        expect(at(0.35, 1300).state.path).toEqual(['root']);
    });

    it.each([-1, 1] as const)('ends Steering before a long near-straight tail in direction %s', (sign) => {
        const lap = trajectoryLap(6, sign, game);
        setTrajectory(lap, (i) => sign * (6 * Math.max(0, Math.min(10, i - 25))
            + 0.75 * Math.max(0, Math.min(25, i - 35))));
        lap.y.forEach((row) => { row[3] = sign * 0.2; });
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(steeringMap(sign, game), 0, lap);
        const at = (position: number, now: number) => {
            const event = frame(driving(position));
            event.update.game = game;
            return engine.receiveTelemetry(event, now);
        };
        expect(at(0.26, 1000).state.path).toContain('Steering');
        expect(at(0.3449, 1100).state.path).toContain('Steering');
        // The reference keeps bending slightly until 0.61; that tail must not retain or re-enter Steering.
        for (const position of [0.36, 0.37, 0.45, 0.60]) {
            const guidance = steeringGuidance(driving(position), lap, sign, { start: 0.25, end: 0.70 });
            expect(guidance.steeringRange).toBeNull();
            expect(guidance.steeringPhase).toBeUndefined();
            expect(at(position, 2000 + position * 10000).state.path).toEqual(['root']);
        }
    });

    it('ends a gradual unwind at the absolute threshold and rejects a uniformly near-straight bend', () => {
        const lap = trajectoryLap(6, 1, game);
        setTrajectory(lap, (i) => i <= 35 ? 6 * Math.max(0, i - 25)
            : 60 + 6 * (1 - 0.5 ** Math.min(25, i - 35)));
        // At 0.37 the smoothed change is 1.125 degrees per 10 raw coordinate units; the next is below 1 degree.
        expect(steeringGuidance(driving(0.3799), lap).steeringRange).toEqual(expect.any(Object));
        expect(steeringGuidance(driving(0.38), lap).steeringRange).toBeNull();
        expect(steeringGuidance(driving(0.40), lap).steeringRange).toBeNull();
        const shallow = trajectoryLap(0.1, 1, game);
        expect(steeringGuidance(driving(0.40), shallow).steeringRange).toBeNull();
    });
});

it('keeps meaningful gentle entry and exit sections around a sharp bend', () => {
    const lap = trajectoryLap();
    setTrajectory(lap, (i) => 2 * Math.max(0, Math.min(5, i - 20)) + 24 * Math.max(0, Math.min(3, i - 25))
        + 2 * Math.max(0, Math.min(20, i - 28)));
    expect(steeringGuidance(driving(0.21), lap).steeringRange).toEqual(expect.any(Object));
    expect(steeringGuidance(driving(0.47), lap).steeringRange).toEqual(expect.any(Object));
});

it.each(['acc', 'iracing'] as const)('splits same-direction turns at a nearly straight section in %s', (game) => {
    for (const sign of [-1, 1] as const) {
        const lap = trajectoryLap(6, sign, game);
        setTrajectory(lap, (i) => sign * (6 * Math.max(0, Math.min(10, i - 25))
            + 0.75 * Math.max(0, Math.min(5, i - 35)) + 6 * Math.max(0, Math.min(10, i - 40))));
        const read = (position: number) => steeringGuidance(driving(position), lap, sign);
        const entered = read(0.26).steeringRange!;
        expect(entered.end).toBeCloseTo(360, 6);
        for (const position of [0.36, 0.38, 0.3999]) {
            expect(read(position)).toMatchObject({ steeringRange: null, brakeZoneExit: false });
        }
        const next = read(0.49).steeringRange!;
        expect(next.start).toBeCloseTo(400, 6);
        expect(next).not.toEqual(entered);
        // The first turn must not borrow overlap from a corner beyond the near-straight gap.
        expect(steeringGuidance(driving(0.26), lap, sign, { start: 0.40, end: 0.55 }).steeringPhase).toBeUndefined();
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        const map = steeringMap(sign, game);
        map.samples.middle_line = lap.y.map((row, i) => ({ bin: i, normalized_position: lap.x[i],
            x: row[0], y: row[1], z: row[2], sample_count: 1, updated_at: '' }));
        map.centerline_tags = [];
        map.centerline_segments = [
            { id: 'first', tags: ['corner'], start_position: 0.25, end_position: 0.36 },
            { id: 'second', tags: ['corner'], start_position: 0.40, end_position: 0.51 },
        ];
        engine.receiveMap(map, 0, lap);
        const at = (position: number, now: number) => {
            const event = frame(driving(position));
            event.update.game = game;
            return engine.receiveTelemetry(event, now);
        };
        expect(at(0.26, 1000).state.path).toContain('Steering');
        expect(at(0.38, 1100).state.path).toEqual(['root']);
        expect(at(0.41, 1200).state.path).toContain('Steering');
    }
});

it.each([false, true])('keeps a near-straight gap across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = trajectoryLap();
    setTrajectory(lap, (i) => 6 * Math.max(0, Math.min(10, i - 25))
        + 0.75 * Math.max(0, Math.min(5, i - 35)) + 6 * Math.max(0, Math.min(10, i - 40)));
    lap.y = [...lap.y.slice(38), ...lap.y.slice(0, 38)];
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const read = (position: number) => steeringGuidance(driving(position), lap, 1, { start: 0.87, end: 0.17 });
    for (const position of [0.99, 0.999, 1, 0, 0.01]) {
        expect(read(position)).toMatchObject({ steeringRange: null, brakeZoneExit: false });
    }
    const before = read(0.97).steeringRange, after = read(0.03).steeringRange;
    expect(before).toEqual(expect.any(Object));
    expect(after).toEqual(expect.any(Object));
    expect(before).not.toEqual(after);
});

describe.each([23, 30])('direction-change boundaries rotated by %s samples', (rotation) => {
    it.each([false, true])('keeps the first bend and filters the tail across start/finish (duplicated endpoint: %s)', (duplicate) => {
        const lap = trajectoryLap();
        setTrajectory(lap, (i) => 0.75 * Math.max(0, Math.min(5, i - 20))
            + 6 * Math.max(0, Math.min(10, i - 25)) + 0.75 * Math.max(0, Math.min(10, i - 35)));
        lap.y = [...lap.y.slice(rotation), ...lap.y.slice(0, rotation)];
        if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
        const shifted = (position: number) => (position - rotation / 100 + 1) % 1;
        const read = (position: number) => steeringGuidance(driving(shifted(position)), lap, 1,
            { start: shifted(0.25), end: shifted(0.55) });
        const entered = read(0.26).steeringRange!;
        expect(entered).toEqual(expect.any(Object));
        for (const position of [0.251, 0.30, 0.3599]) expect(read(position).steeringRange).toEqual(entered);
        for (const position of [0.20, 0.2099, 0.2101, 0.23, 0.2499, 0.3601, 0.40]) expect(read(position).steeringRange).toBeNull();
    });
});

it.each([false, true])('trims the tail after a start/finish crossing (duplicated endpoint: %s)', (duplicate) => {
    const lap = trajectoryLap();
    setTrajectory(lap, (i) => 6 * Math.max(0, Math.min(10, i - 25))
        + 0.75 * Math.max(0, Math.min(25, i - 35)));
    lap.y = [...lap.y.slice(30), ...lap.y.slice(0, 30)];
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const read = (position: number) => steeringGuidance(driving(position), lap, 1, { start: 0.95, end: 0.30 });
    const entered = read(0.96).steeringRange;
    expect(entered).toEqual(expect.any(Object));
    for (const position of [0.99, 1, 0, 0.059]) expect(read(position).steeringRange).toEqual(entered);
    for (const position of [0.06, 0.10, 0.29]) expect(read(position).steeringRange).toBeNull();
});

it('preserves the corner-limited onset and filtered tail with extra reference samples', () => {
    const lap = trajectoryLap();
    setTrajectory(lap, (i) => 0.75 * Math.max(0, Math.min(5, i - 20)) + 6 * Math.max(0, Math.min(10, i - 25))
        + 0.75 * Math.max(0, Math.min(25, i - 35)));
    const dense = { ...lap, x: [...lap.x], y: lap.y.map((row) => [...row]) };
    for (let i = 59; i >= 20; i--) {
        if (i >= 25 && i < 36) continue;
        const fraction = 0.86952861952862;
        const row = lap.y[i].map((value, axis) => value + (lap.y[i + 1][axis] - value) * fraction);
        dense.x.splice(i + 1, 0, lap.x[i] + 0.01 * fraction);
        dense.y.splice(i + 1, 0, row);
    }
    const original = steeringGuidance(driving(0.26), lap).steeringRange!;
    const resampled = steeringGuidance(driving(0.26), dense).steeringRange!;
    expect(resampled.start).toBeCloseTo(250, 6);
    expect(resampled.start).toBeCloseTo(original.start, 6);
    expect(resampled.end).toBeCloseTo(original.end, 6);
    expect(steeringGuidance(driving(0.2099), dense).steeringRange).toBeNull();
    expect(steeringGuidance(driving(0.2101), dense).steeringRange).toBeNull();
    expect(steeringGuidance(driving(0.2501), dense).steeringRange).toEqual(resampled);
    expect(steeringGuidance(driving(0.37), dense).steeringRange).toBeNull();
});

it.each(['reversal', 'later turn'])('exits the captured range even when the next sample is in a %s', (kind) => {
    const lap = trajectoryLap();
    setTrajectory(lap, (i) => i <= 30 ? 12 * Math.max(0, i - 25)
        : kind === 'reversal' ? 60 - 24 * Math.min(10, i - 30) : 60 + 12 * Math.max(0, Math.min(10, i - 35)));
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(steeringMap(), 0, lap);
    expect(engine.receiveTelemetry(frame(driving(0.26)), 1000).state.path).toEqual(['root', 'Steering']);
    // Both positions are still turning; the latter belongs to a different range.
    const position = kind === 'reversal' ? 0.32 : 0.36;
    expect(steeringGuidance(driving(position), lap).steeringRange).toEqual(expect.any(Object));
    expect(engine.receiveTelemetry(frame(driving(position)), 1100).state.path).toEqual(['root']);
});

describe.each([-1, 1])('Steering re-entry into a range with relative direction %s', (direction) => {
    it.each([false, true])('rearms at range exit with a short gap (reference steering applied: %s)', (steeringApplied) => {
        const lap = trajectoryLap();
        const nextStart = direction === -1 ? 35 : 38;
        setTrajectory(lap, (i) => 6 * Math.max(0, Math.min(10, i - 25))
            + direction * 6 * Math.max(0, Math.min(10, i - nextStart)));
        lap.y.forEach((row, i) => {
            row[3] = steeringApplied ? (i < nextStart ? 0.2 : direction * 0.2) : 0;
            row[4] = row[5] = 0;
            row[6] = 180;
        });
        const map: CircuitMapDto = { ...circuitMap(),
            samples: { middle_line: lap.y.map((row, i) => ({ bin: i, normalized_position: lap.x[i],
                x: row[0], y: row[1], z: row[2], sample_count: 1, updated_at: '' })) },
            centerline_segments: [
                { id: 'turn-1', tags: ['corner'], start_position: 0.25, end_position: direction === -1 ? 0.35 : 0.36 },
                { id: 'turn-2', tags: ['corner'], start_position: nextStart / 100, end_position: (nextStart + 11) / 100 },
            ],
        };
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(map, 0, lap);
        const at = (index: number) => engine.receiveTelemetry(frame(beforeDecision({ ...driving(index / 1000 + 1e-10), Physics_speed_kmh: 180 })),
            1000 + (index - 270) * 20);
        expect(at(270).state.path).toContain('Steering');
        // Samples are 1 m / 20 ms apart. The flat gap is shorter than the old 500 ms clear period.
        const exit = direction === -1 ? 375 : 385; // Measured progress is 25 m behind the decision point.
        const secondEntry = direction === -1 ? exit + 1 : 390;
        let snapshot = at(271);
        for (let i = 272; i <= 440; i++) {
            snapshot = at(i);
            if (i === exit - 1) expect(snapshot.state.path).toContain('Steering');
            if (i === exit) expect(snapshot.state.path).toEqual(['root']);
            if (direction === -1 && steeringApplied && i === secondEntry) {
                expect(snapshot.referenceSteeringPhase).toBe('turn');
                expect(snapshot.state.path).toContain('Steering');
                expect(snapshot.events).toEqual([]);
            }
            if (i === secondEntry) expect(snapshot.state.path).toContain('Steering');
        }
        expect(snapshot.state.path).toContain('Steering');
        expect(snapshot.events).toEqual([]);
    });
});

it('keeps Steering through low reference speed and missing telemetry, then exits without a reference speed channel', () => {
    const lap = trajectoryLap();
    lap.y.forEach((row, i) => { if (i >= 30) row[6] = 3.6; });
    const map = steeringMap();
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(map, 0, lap);
    expect(engine.receiveTelemetry(frame(driving(0.26)), 1000).state.path).toEqual(['root', 'Steering']);
    expect(engine.receiveTelemetry(frame(driving(0.32)), 1100).state.path).toEqual(['root', 'Steering']);
    expect(engine.evaluate(3000).state.path).toEqual(['root', 'Steering']);
    expect(engine.receiveTelemetry(frame({ ...driving(0.4), Graphics_normalized_car_position: undefined }), 3100).state.path)
        .toEqual(['root', 'Steering']);
    const missingSpeed = { ...lap, target_features: lap.target_features.map((name) => name === 'expert_optimal_speed' ? 'missing' : name) };
    engine.receiveMap(map, 3200, missingSpeed);
    expect(engine.receiveTelemetry(frame(driving(0.3449)), 3300).state.path).toEqual(['root', 'Steering']);
    expect(engine.receiveTelemetry(frame(driving(0.35)), 3400).state.path).toEqual(['root']);
});

it.each([false, true])('tracks a captured direction-change range across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = trajectoryLap();
    // Move the turn from 0.25–0.41 to 0.90–0.06 without changing its geometry.
    lap.y = [...lap.y.slice(35), ...lap.y.slice(0, 35)];
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const map = steeringMap();
    map.centerline_tags!.forEach((tag) => { if (tag.label === 'corner') tag.end_position = 0.36; });
    map.samples.middle_line = map.samples.middle_line!.map((point) => ({ ...point,
        normalized_position: (point.normalized_position + 0.65) % 1,
    })).sort((a, b) => a.normalized_position - b.normalized_position);
    map.centerline_tags = map.centerline_tags!.map((tag) => ({ ...tag,
        start_position: (tag.start_position + 0.65) % 1, end_position: (tag.end_position + 0.65) % 1,
    }));
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(map, 0, lap);
    [0.91, 0.99, 1, 0, 0.0049].forEach((position, index) => {
        const path = engine.receiveTelemetry(frame(driving(position)), 1000 + index * 100).state.path;
        expect({ position, path }).toEqual({ position, path: ['root', 'Steering'] });
    });
    expect(engine.receiveTelemetry(frame(driving(0.01)), 1500).state.path).toEqual(['root']);
});

it('interpolates across the signed heading wrap instead of treating it as a reversal', () => {
    const lap = trajectoryLap();
    setTrajectory(lap, (i) => 170 + 12 * Math.max(0, Math.min(15, i - 25)));
    expect(steeringGuidance(driving(0.26), lap).steeringPhase).toBe('prepare');
});

it.each([[0.25, 'brake', 'Physics_brake', 0.8],
    [0.7, 'throttle', 'Physics_gas', 0.8]] as const)('corrects low input at %s, but accepts matching input', (position, cue, field, value) => {
    const lap = referenceLap(), sample = driving(position), view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000)[cue]).toBe(2500);
    expect(readReferenceGuidance({ ...sample, [field]: value }, view, lap, 'acc', 1000)[cue]).toBeUndefined();
    expect(readReferenceGuidance({ ...sample, [field]: undefined }, view, lap, 'acc', 1000)[cue]).toBeUndefined();
});

it('bases corrections on interpolated speed even when pedals already match the reference', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const brake = { ...driving(0.255), Physics_speed_kmh: 106, Physics_brake: 0.8 };
    const throttle = { ...driving(0.655), Physics_speed_kmh: 94, Physics_gas: 0.8 };
    expect(readReferenceGuidance(brake, detection(lap, brake), lap, 'acc', 1000, lap, 1).moreBrake).toBe(2500);
    expect(readReferenceGuidance(throttle, detection(lap, throttle), lap, 'acc', 1000).moreThrottle).toBe(2500);
    for (const [sample, speed] of [[brake, 98], [throttle, 102]] as const) {
        const matching = { ...sample, Physics_speed_kmh: speed };
        const cues = readReferenceGuidance(matching, detection(lap, matching), lap, 'acc', 1000, lap, localBrakeDeceleration);
        expect(cues.moreBrake).toBeUndefined();
        expect(cues.moreThrottle).toBeUndefined();
    }
    expect(readReferenceGuidance({ ...brake, Physics_brake: 1 }, detection(lap, brake), lap, 'acc', 1000, lap, 1).moreBrake).toBeUndefined();
    expect(readReferenceGuidance({ ...throttle, Physics_gas: 1 }, detection(lap, throttle), lap, 'acc', 1000).moreThrottle).toBeUndefined();
});

it('does not announce throttle before pickup', () => {
    const lap = referenceLap(), sample = driving(0.59);
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(guidance).not.toHaveProperty('throttleSoon');
    expect(guidance.throttle).toBeUndefined();
    expect(guidance.moreThrottle).toBeUndefined();
});

it.each([
    [102, 0, 'throttle'], [94, 0.8, 'moreThrottle'], [102, 1, undefined], [120, 0.8, undefined],
    [102, 0.8, undefined], [94, 1, undefined], [120, 0, undefined],
] as const)('selects one throttle instruction at %s km/h with pedal %s: %s', (speed, pedal, cue) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.655), Physics_speed_kmh: speed, Physics_gas: pedal };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    const cues = Object.entries(guidance).filter(([key]) => ['throttle', 'moreThrottle', 'lessThrottle', 'releaseThrottle'].includes(key));
    expect(cues).toEqual(cue ? [[cue, 2500]] : []);
});

it.each([0.4, 0.8])('uses reference throttle %s to estimate demand at matching speed', (pedal) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { if (row[5]) row[5] = pedal; });
    const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 0.6 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(guidance.moreThrottle).toBe(pedal === 0.8 ? 2500 : undefined);
    expect(guidance.lessThrottle).toBeUndefined();
});

it.each([0.2, 1.2])('scales throttle correction using reference acceleration (%s km/h per metre)', (slope) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row, i) => { row[6] = i < 60 || i > 70 ? 100 : 102 + (i * 10 - 655) * slope; });
    const sample = { ...driving(0.655), Physics_speed_kmh: 97, Physics_gas: 0.8 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(slope === 0.2 ? 2500 : undefined);
});

it('preserves maintenance throttle on a flat speed profile', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.75), Physics_speed_kmh: 120, Physics_gas: 0.8 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000))
        .toEqual({ ready: true, lateralOffset: 0, braking: false, acceleration: 0, pedalZone: 'throttle', currentPedalZone: 'throttle',
            steeringRange: expect.any(Object), currentReferencePosition: { distance: 750, length: 1000 } });
    expect(readReferenceGuidance({ ...sample, Physics_gas: 0.4 }, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(2500);
});

it.each(['missing throttle', 'invalid throttle', 'missing pedal', 'invalid pedal', 'missing brake', 'missing live brake'])(
    'withholds throttle instructions with %s', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 0.2 as number | undefined };
        if (kind === 'missing throttle') lap.target_features[5] = 'missing throttle';
        if (kind === 'invalid throttle') lap.y[65][5] = NaN;
        if (kind === 'missing pedal') sample.Physics_gas = undefined;
        if (kind === 'invalid pedal') sample.Physics_gas = NaN;
        if (kind === 'missing brake') lap.target_features[4] = 'missing brake';
        if (kind === 'missing live brake') sample.Physics_brake = undefined;
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
        expect(['throttle', 'moreThrottle', 'lessThrottle', 'releaseThrottle'].filter((key) => key in guidance)).toEqual([]);
    });

it.each(['missing speed', 'invalid speed', 'flat speed'])('falls back to reference throttle with %s', (kind) => {
    const lap = referenceLap();
    addSpeed(lap);
    if (kind === 'missing speed') lap.target_features[6] = 'missing speed';
    if (kind === 'invalid speed') lap.y[65][6] = NaN;
    if (kind === 'flat speed') lap.y.forEach((row) => { row[6] = 102; });
    const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 0.4 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(2500);
    expect(readReferenceGuidance({ ...sample, Physics_gas: 1 }, detection(lap, sample), lap, 'acc', 1000).lessThrottle).toBeUndefined();
});

it.each([false, true])('uses reference acceleration across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y[99][5] = lap.y[0][5] = 0.8;
    lap.y[99][6] = 96;
    lap.y[0][6] = 100;
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const sample = { ...driving(0.995), Physics_speed_kmh: 98, Physics_gas: 0.8 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000))
        .toEqual({ ready: true, lateralOffset: 0, braking: false, acceleration: 98 / 3.6 * (4 / (3.6 * 10)), pedalZone: 'throttle', currentPedalZone: 'throttle',
            steeringRange: expect.any(Object), currentReferencePosition: { distance: 995, length: 1000 } });
    expect(readReferenceGuidance({ ...sample, Physics_speed_kmh: 90 }, detection(lap, sample), lap, 'acc', 1000).moreThrottle).toBe(2500);
});

it('withholds throttle corrections away from the line without overspeed and outward motion', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.655), Physics_speed_kmh: 102, Physics_gas: 1 }, view = detection(lap, sample);
    view.position!.referenceDelta[0] = 3;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).lessThrottle).toBeUndefined();
    const guidance = readReferenceGuidance({ ...sample, Physics_gas: 0 }, view, lap, 'acc', 1000);
    expect(guidance.throttle).toBeUndefined();
    expect(guidance.moreThrottle).toBeUndefined();
});

const brakeCueNames = ['brake', 'moreBrake', 'releaseBrake', 'holdBrake'];

// At 250 m: 100 -> 96 km/h over 10 m gives 3.02469 m/s².
const localBrakeDeceleration = (100 ** 2 - 96 ** 2) / (2 * 3.6 ** 2 * 10);

it.each([0, 25, 50])('withholds release without a mapped turn-in target at %s%% braking adjustment', (caution) => {
    const { engine, lap } = setup(addSpeed);
    engine.setDrivingAggression(caution, 0);
    const local = copyPhraseReference(lap, caution)!;
    const a = local.y[22][6], b = local.y[23][6];
    const targetSpeed = Math.sqrt((a ** 2 + b ** 2) / 2);
    // Between samples, before turn-in and the minimum, with no position model or rate measurement.
    const at = (speed: number, now: number) => engine.receiveTelemetry(frame(beforeDecision({
        ...driving(0.225), Physics_speed_kmh: speed, Physics_brake: 0.8,
    })), now);
    const slow = at(targetSpeed - 6, 1000);
    expect(slow.state.path).toEqual(['root', 'Brake zone']);
    expect(slow.events).toEqual([]);
    const fast = at(targetSpeed + 6, 1100);
    expect(fast.events).toEqual([]);
    const belowMinimum = at(74, 1200);
    expect(belowMinimum.events).toEqual([]);
});

it.each([
    [100, 0, undefined, false, 'brake'],
    [100, 0.8, 1, true, undefined],
    [100, 0.8, 1, false, 'moreBrake'],
    [95, 0.8, 0, false, 'moreBrake'],
    [100, 0.8, 4, true, undefined],
    [100, 0.8, 4, false, undefined],
    [100, 0.8, localBrakeDeceleration, false, 'holdBrake'],
    [100, 0.8, localBrakeDeceleration, true, 'holdBrake'],
    [106, 0.8, localBrakeDeceleration, false, undefined],
    [106, 0.8, localBrakeDeceleration, true, undefined],
    [94, 0.8, localBrakeDeceleration, false, undefined],
    [80, 0.8, 1, true, undefined],
    [80, 0, 1, false, undefined],
    [106, 1, 1, false, undefined],
    [100, 1, 4, false, undefined],
    [100, 0.8, undefined, false, undefined],
    [100, 0.8, undefined, true, undefined],
    [100, 0.8, NaN, true, undefined],
    [100, 0.8, Infinity, true, undefined],
    [100, 0.8, -1, true, undefined],
] as const)('selects one brake cue at %s km/h, pedal %s, deceleration %s, releasing %s: %s', (speed, pedal, deceleration, releasing, cue) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: speed, Physics_brake: pedal };
    const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, deceleration, undefined, undefined, undefined, releasing);
    expect(brakeCueNames.filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
    expect(guidance.moreThrottle).toBeUndefined();
});

it.each([0.2, 0.8, 1])('matches deceleration independently of reference pedal pressure %s', (pressure) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { if (row[4]) row[4] = pressure; });
    for (const [deceleration, cue] of [[1, undefined], [4, undefined], [localBrakeDeceleration, 'holdBrake']] as const) {
        const sample = { ...driving(0.25), Physics_speed_kmh: 100, Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, deceleration, undefined, undefined, undefined, true);
        expect(brakeCueNames.filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
    }
});

it.each([5, 50])('uses a 10%% deceleration band with a 0.1 m/s² floor at distance scale %s', (distanceScale) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { row[0] *= distanceScale; row[1] *= distanceScale; });
    lap.y[25][6] = 108;
    lap.y[26][6] = 72;
    const required = 25 / distanceScale;
    const band = Math.max(0.1, required * 0.1);
    for (const [delta, cue] of [[-1.01, undefined], [-0.99, 'holdBrake'], [0, 'holdBrake'],
        [0.99, 'holdBrake'], [1.01, undefined]] as const) {
        const sample = { ...driving(0.25), Physics_speed_kmh: 108, Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, required + delta * band, undefined, undefined, undefined, true);
        expect(brakeCueNames.filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
    }
});

it.each([-2, 2])('withholds Hold brake pressure for a small speed error of %s km/h', (error) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: 100 + error, Physics_brake: 0.8 };
    const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, localBrakeDeceleration, undefined, undefined, undefined, true);
    expect(brakeCueNames.filter((key) => key in guidance)).toEqual([]);
});

it.each([0, 0.099, 0.1, 1, undefined, NaN])('requires an applied brake for Hold brake pressure: %s', (pedal) => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: 100, Physics_brake: pedal };
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, localBrakeDeceleration).holdBrake)
        .toBe(pedal !== undefined && pedal >= 0.1 ? 2500 : undefined);
});

it.each(['missing speed', 'invalid speed', 'missing brake', 'invalid brake', 'outside zone'])(
    'withholds Hold brake pressure with %s', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[28][6] = NaN;
        if (kind === 'missing brake') lap.target_features[4] = 'missing';
        if (kind === 'invalid brake') lap.y[28][4] = NaN;
        if (kind === 'outside zone') lap.y[25][5] = 0.8;
        const sample = { ...driving(0.25), Physics_speed_kmh: 100, Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, localBrakeDeceleration);
        expect(guidance.holdBrake).toBeUndefined();
    });

it('keeps a light brake applied when positive deceleration is still needed', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.299), Physics_speed_kmh: 80, Physics_brake: 0.2 };
    const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, 4);
    expect(brakeCueNames.filter((key) => key in guidance)).toEqual([]);
    expect(guidance.releaseBrake).toBeUndefined();
});

it.each([0, 4])('does not release from zero local deceleration alone when reference speed rises by %s km/h', (increase) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y.forEach((row) => { row[6] = 100; });
    lap.y[26][6] += increase;
    for (const speed of [106, 100, 94]) {
        const sample = { ...driving(0.25), Physics_speed_kmh: speed, Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, null, lap, 'acc', 1000, lap, 0);
        expect(brakeCueNames.filter((key) => key in guidance)).toEqual([]);
    }
});

it.each([0, 25, 50])('accepts matching throttle on the active speed profile at %s%% caution', (caution) => {
    const lap = referenceLap();
    addSpeed(lap);
    const local = copyPhraseReference(lap, caution)!;
    const sample = beforeDecision({ ...driving(0.65), Physics_speed_kmh: local.y[local.x.indexOf(0.65)][6], Physics_gas: 0.8 });
    const engine = conditionTestEngine();
    engine.receiveMap(null, 0, lap);
    engine.setDrivingAggression(caution, 0);
    engine.receiveVision(detection(lap, sample), 1000);
    const snapshot = engine.receiveTelemetry(frame(sample), 1000);
    expect(snapshot.state.path).toEqual(['root', 'Throttle zone']);
    expect(snapshot.events).toEqual([]);
});

it.each([25, 50])('keeps cornering cues identical when the adjusted reference is supplied directly (%s%% caution)', (caution) => {
    const lap = referenceLap();
    addSpeed(lap);
    const local = copyPhraseReference(lap, caution)!;
    // Supplying the adjusted lap directly must behave just like selecting that caution in Live Phrases.
    for (const position of [0.21, 0.25, 0.29, 0.3, 0.41, 0.61, 0.65, 0.7]) {
        const targetSpeed = local.y[local.x.indexOf(position)][6];
        for (const error of [-5.1, 0, 6]) {
            for (const pedal of [0, 0.8]) {
                const sample = { ...driving(position), Physics_speed_kmh: targetSpeed + error,
                    Physics_brake: position < 0.4 ? pedal : 0, Physics_gas: position >= 0.4 ? pedal : 0 };
                const snapshots = [lap, local].map((reference, index) => {
                    const engine = conditionTestEngine();
                    engine.receiveMap(null, 0, reference);
                    if (!index) engine.setDrivingAggression(caution, 0);
                    engine.receiveVision(detection(reference, sample), 1000);
                    engine.receiveTelemetry(frame(sample), 1000);
                    engine.receiveVision(detection(reference, sample, 1300), 1300);
                    return engine.receiveTelemetry(frame(sample), 1300);
                });
                expect({ position, error, pedal, path: snapshots[0].state.path, events: snapshots[0].events })
                    .toEqual({ position, error, pedal, path: snapshots[1].state.path, events: snapshots[1].events });
            }
        }
    }
});

it('keeps braking demand active until the original minimum-speed point', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const cautious = copyPhraseReference(lap, 50)!;
    const early = { ...driving(0.21), Physics_speed_kmh: 116 };
    expect(readReferenceGuidance(early, detection(lap, early), cautious, 'acc', 1000, lap).brake).toBe(2500);
    expect(readReferenceGuidance(early, detection(lap, early), cautious, 'acc', 1000, lap).pedalZone).toBe('brake');
    const missingPedal = { ...early, Physics_brake: undefined };
    expect(readReferenceGuidance(missingPedal, detection(lap, missingPedal), cautious, 'acc', 1000, lap))
        .toEqual({ ready: true, lateralOffset: 0, braking: true, pedalZone: 'brake', currentPedalZone: 'brake', steeringRange: null,
            acceleration: expect.any(Number), currentReferencePosition: { distance: 210, length: 1000 } });
    const slowed = { ...driving(0.299), Physics_speed_kmh: 30, Physics_brake: 0.8 };
    const guidance = readReferenceGuidance(slowed, detection(lap, slowed), cautious, 'acc', 1000, lap);
    expect(guidance.releaseBrake).toBeUndefined();
    expect(guidance.moreBrake).toBeUndefined();
    expect(readReferenceGuidance({ ...slowed, Graphics_normalized_car_position: 0.3 }, null, cautious, 'acc', 1000, lap).pedalZone).toBe('none');
    expect(lap.y[19][4]).toBe(0);
    expect(lap.y[29][4]).toBe(0.8);
});

it.each([0.201, 0.205, 0.209, 0.255, 0.279])('preserves constant deceleration between samples with the minimum fixed at %s', (position) => {
    const lap = referenceLap();
    addSpeed(lap);
    // Reference: 40 -> 30 m/s over 100 m, giving 3.5 m/s² at full brake.
    lap.y.forEach((row, i) => {
        row[4] = i >= 21 && i < 30 ? 1 : 0;
        row[6] = i < 20 || i >= 70 ? 144 : i < 30 ? Math.sqrt(1600 - 70 * (i - 20)) * 3.6
            : i <= 60 ? 108 : 108 + (i - 60) * 3.6;
    });
    const cautious = copyPhraseReference(lap, 25)!;
    // Constant deceleration still reaches 30 m/s at the original 300 m endpoint.
    const speed = Math.sqrt(30 ** 2 + 2 * 3.5 * (300 - position * 1000)) * 3.6;
    for (const [speedError, deceleration, cue] of [
        [6, 3.2, 'moreBrake'], [6, 3.5, undefined], [6, 3.7, undefined],
        [-6, 3.8, undefined], [-6, 4, undefined],
    ] as const) {
        const sample = { ...driving(position), Physics_speed_kmh: speed + speedError, Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap, deceleration);
        expect(brakeCueNames.filter((key) => key in guidance)).toEqual(cue ? [cue] : []);
    }
});

it('aligns braking responses while retaining a nonzero first normalized position', () => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y = lap.y.slice(15).concat(lap.y.slice(0, 15));
    const shifted = { ...lap, x: lap.x.map((p) => p + 0.005), y: lap.y.map((row) => [...row]) };
    const cautious = copyPhraseReference(lap, 100)!, shiftedCautious = copyPhraseReference(shifted, 100)!;
    expect(shiftedCautious.x).toEqual(shifted.x);
    for (const position of [0.045, 0.05, 0.055, 0.46, 0.5]) {
        const index = cautious.x.findIndex((p) => p >= position);
        const speed = cautious.y[index][6];
        for (let pedal = 0; pedal <= 1; pedal += 0.01) {
            const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_brake: pedal, Physics_gas: pedal };
            const shiftedSample = { ...sample, Graphics_normalized_car_position: position + 0.005 };
            const { steeringRange: shiftedRange, ...shiftedGuidance } = readReferenceGuidance(shiftedSample,
                detection(shifted, shiftedSample), shiftedCautious, 'acc', 1000, shifted);
            const { steeringRange, ...guidance } = readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap);
            expect(shiftedGuidance).toEqual(guidance);
            expect(shiftedRange?.turnSign).toBe(steeringRange?.turnSign);
        }
    }
});

it.each([false, true])('uses reference deceleration across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    addSpeed(lap);
    lap.y[99][4] = lap.y[0][4] = 0.8;
    lap.y[99][6] = 100;
    lap.y[0][6] = 96;
    lap.y[1][6] = 92;
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const sample = { ...driving(0.995), Physics_speed_kmh: 98, Physics_brake: 0.8 };
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000))
        .toMatchObject({ ready: true, lateralOffset: 0, braking: true, pedalZone: 'brake', steeringRange: expect.any(Object) });
    sample.Physics_speed_kmh = 106;
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000, lap, 1).moreBrake).toBe(2500);
});

it.each(['missing brake', 'invalid brake', 'missing pedal', 'invalid pedal', 'missing speed'])(
    'handles %s without assuming missing data is a released pedal or inventing deceleration', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        const sample = { ...driving(0.255), Physics_speed_kmh: 106, Physics_brake: 0.8 as number | undefined };
        if (kind === 'missing brake') lap.target_features[4] = 'missing brake';
        if (kind === 'invalid brake') lap.y[25][4] = NaN;
        if (kind === 'missing pedal') sample.Physics_brake = undefined;
        if (kind === 'invalid pedal') sample.Physics_brake = NaN;
        if (kind === 'missing speed') lap.target_features[6] = 'missing speed';
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
        expect(brakeCueNames.filter((key) => key in guidance)).toEqual([]);
    });

it('reads speed targets at normalized position regardless of the position-model forward delta', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = { ...driving(0.25), Physics_speed_kmh: 108 }, view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    expect(readReferenceGuidance({ ...sample, Physics_brake: 0.8 }, view, lap, 'acc', 1000, lap, 1).moreBrake).toBe(2500);
    view.position!.referenceDelta[1] = 20;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    expect(readReferenceGuidance({ ...sample, Physics_brake: 0.8 }, view, lap, 'acc', 1000, lap, 1).moreBrake).toBe(2500);
    for (const position of [0.59, 0.65]) {
        const fast = { ...driving(position), Physics_speed_kmh: 120 };
        const cues = readReferenceGuidance(fast, detection(lap, fast), lap, 'acc', 1000);
        expect(cues.throttle).toBeUndefined();
        expect(cues.moreThrottle).toBeUndefined();
    }
});

it('exposes the signed position-model offset independently of steering input', () => {
    const lap = referenceLap();
    lap.y.forEach((row) => { row[3] *= -1; });
    const sample = driving(0.45), view = detection(lap, sample);
    view.position!.referenceDelta[0] = -2;
    expect(readReferenceGuidance({ ...sample, Physics_steer_angle: 0.4 }, view, lap, 'acc', 1000).lateralOffset).toBe(-2);
    expect(readReferenceGuidance({ ...sample, Physics_steer_angle: -0.4 }, view, lap, 'acc', 1000).lateralOffset).toBe(-2);
});

it('ignores the position-model forward delta for brake demand and uses lateral error to suppress acceleration', () => {
    const lap = referenceLap(), sample = driving(0.21), view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    view.position!.referenceDelta[1] = 10;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    view.position!.referenceDelta[1] = -5;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    const exit = driving(0.7), exitView = detection(lap, exit);
    exitView.position!.referenceDelta[0] = 3;
    expect(readReferenceGuidance(exit, exitView, lap, 'acc', 1000).moreThrottle).toBeUndefined();
});

it('interpolates brake demand across start/finish without a prepare-to-brake cue', () => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[4] = i === 0 ? 0.8 : 0; });
    const sample = driving(0.995), view = detection(lap, sample);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).brake).toBe(2500);
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000)).not.toHaveProperty('brakeSoon');
    const before = driving(0.98);
    expect(readReferenceGuidance(before, detection(lap, before), lap, 'acc', 1000).brake).toBeUndefined();
    expect(readReferenceGuidance({ ...sample, Physics_speed_kmh: 10 }, view, lap, 'acc', 1000).brake).toBe(2500);
});

it.each(['wrong car', 'wrong track', 'wrong game', 'bad lap', 'missing position', 'invalid position', 'out-of-range position'])(
    'rejects %s reference or telemetry inputs', (kind) => {
        const lap = referenceLap();
        let sample = driving(0.19);
        const view = detection(lap, sample);
        if (kind === 'wrong car') sample = { ...sample, Static_car_model: 'other' };
        if (kind === 'wrong track') sample = { ...sample, Static_track: 'other' };
        if (kind === 'bad lap') lap.y[2][0] = NaN;
        if (kind === 'missing position') sample = { ...sample, Graphics_normalized_car_position: undefined };
        if (kind === 'invalid position') sample = { ...sample, Graphics_normalized_car_position: NaN };
        if (kind === 'out-of-range position') sample = { ...sample, Graphics_normalized_car_position: 1.1 };
        expect(readReferenceGuidance(sample, view, lap, kind === 'wrong game' ? 'iracing' : 'acc', 1000)).toEqual({});
    });

it.each(['stale', 'future', 'wrong lap', 'jump', 'invalid delta', 'remote offset', 'no position', 'no vision'])(
    'keeps normalized reference guidance but withholds lateral cues with %s', (kind) => {
        const lap = referenceLap();
        lap.y[42][3] = 0.5;
        addSpeed(lap);
        lap.y.forEach((row) => { row[6] = 36; });
        lap.y[25][6] = 40; // Braking is needed at the live 36 km/h despite missing lateral evidence.
        setTrajectory(lap, (i) => 12 * Math.max(0, Math.min(15, i - 39)));
        for (const position of [0.25, 0.401, 0.7]) {
            const sample = driving(position), view = detection(lap, sample);
            if (kind === 'stale') view.capturedAt = 499;
            if (kind === 'future') view.capturedAt = 1001;
            if (kind === 'wrong lap') view.position!.referenceLap = referenceLap();
            if (kind === 'jump') view.position!.normalizedPosition = 0.1;
            if (kind === 'invalid delta') view.position!.referenceDelta[0] = NaN;
            if (kind === 'remote offset') view.position!.referenceDelta[0] = 14;
            if (kind === 'no position') view.position = undefined;
            const guidance = readReferenceGuidance(sample, kind === 'no vision' ? null : view, lap, 'acc', 1000,
                lap, undefined, undefined, 1, { start: 0.4, end: 0.5 });
            expect(guidance.ready).toBe(true);
            expect(guidance.lateralOffset).toBeUndefined();
            expect(guidance.pedalZone).toBe(position === 0.7 ? 'throttle' : position === 0.25 ? 'brake' : 'none');
            expect(guidance.brake).toBe(position === 0.25 ? 2500 : undefined);
            expect(guidance.steering).toBe(position === 0.401 ? 2500 : undefined);
            expect(guidance.throttle).toBeUndefined();
            expect(guidance.moreThrottle).toBeUndefined();
            expect(guidance.lessThrottle).toBeUndefined();
        }
    });

it('does not treat a missing reference channel as zero or advise throttle during braking', () => {
    const lap = referenceLap();
    lap.target_features[3] = 'missing steering';
    const sample = driving(0.45);
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).steering).toBeUndefined();
    const braking = { ...driving(0.7), Physics_brake: 0.5 };
    expect(readReferenceGuidance(braking, detection(lap, braking), lap, 'acc', 1000).moreThrottle).toBeUndefined();
});

it.each([20, 180])('follows negative trajectory turns independently of live steering at %s km/h', (speed) => {
    const lap = trajectoryLap(12, -1);
    lap.y[28][3] = lap.y[29][3] = -0.1;
    for (const [position, phase] of [[0.26, 'prepare'], [0.28, 'turn']] as const) {
        for (const liveSteering of [-0.4, 0, 0.4, undefined]) {
            const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_steer_angle: liveSteering };
            expect(steeringGuidance(sample, lap, -1)).toMatchObject({ steering: 2500, steeringPhase: phase });
        }
    }
});

it.each([0, 0.14, 0.15, 0.4, 0.66, 0.8, 1])('allows only More throttle before imminent braking with pedal %s', (pedal) => {
    const lap = referenceLap();
    lap.y.forEach((row, i) => { row[5] = i < 20 || i > 30 ? 0.8 : 0; });
    const sample = { ...driving(0.19), Physics_gas: pedal }, view = detection(lap, sample);
    const cues = readReferenceGuidance(sample, view, lap, 'acc', 1000);
    expect(cues.brake).toBeUndefined();
    expect(cues.throttle).toBeUndefined();
    expect(cues.moreThrottle).toBe(pedal === 0.15 || pedal === 0.4 ? 2500 : undefined);
});

it.each(['required', 'imminent'])('speaks More throttle in the throttle zone when braking is %s', (kind) => {
    const { lap, update } = setup((reference) => {
        reference.y.forEach((row, i) => { row[5] = i < 20 || i > 30 ? 0.8 : 0; });
        if (kind === 'required') {
            addSpeed(reference);
            reference.y[76][6] = 116;
        }
    });
    const position = kind === 'required' ? 0.75 : 0.19;
    const patch = { Physics_speed_kmh: kind === 'required' ? 120 : 36, Physics_gas: 0.4 };
    const sample = { ...driving(position), ...patch };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
    expect(guidance.braking).toBe(kind === 'required');
    expect(guidance.pedalZone).toBe('throttle');
    expect(guidance.moreThrottle).toBe(2500);
    expect(guidance.throttle).toBeUndefined();
    const result = update(position, 1000, patch);
    expect(result.events.map(({ sentence }) => sentence)).toEqual(['More gas.']);
    expect(result.state.path).toEqual(['root', 'Throttle zone', 'More throttle', 'say phrase']);
});

it.each(['applied brake', 'overspeed', 'away from line', 'brake zone'])(
    'still withholds More throttle with %s after removing braking-demand checks', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        lap.y.forEach((row, i) => { row[5] = i < 20 || i > 30 ? 0.8 : 0; });
        const sample = { ...driving(kind === 'brake zone' ? 0.25 : 0.75),
            Physics_speed_kmh: kind === 'overspeed' ? 125 : kind === 'brake zone' ? 100 : 120,
            Physics_gas: 0.15, Physics_brake: kind === 'applied brake' ? 0.1 : 0 };
        const view = detection(lap, sample);
        if (kind === 'away from line') view.position!.referenceDelta[0] = 2.01;
        expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).moreThrottle).toBeUndefined();
    });

it('reports reference braking at normalized position independently of advance cues and driver brake input', () => {
    const lap = referenceLap();
    for (const [position, braking] of [[0.19, false], [0.201, true], [0.202, true], [0.25, true]] as const) {
        const sample = { ...driving(position), Physics_brake: 0.8 };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000);
        // Correction demand is separate from reference zone membership.
        expect(guidance.braking).toBe(braking);
        expect(guidance.brake).toBeUndefined();
        expect(guidance.releaseBrake).toBeUndefined();
        expect(guidance.moreBrake).toBeUndefined();
    }
    const sample = driving(0.21), view = detection(lap, sample);
    view.position!.referenceDelta[1] = 20;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).braking).toBe(true);
    const missingBrake = referenceLap();
    missingBrake.target_features[4] = 'missing brake';
    expect(readReferenceGuidance(sample, detection(missingBrake, sample), missingBrake, 'acc', 1000).braking).toBeUndefined();
});

function setup(configure?: (lap: TopLapReferenceEntry) => void, map: CircuitMapDto | null = null) {
    const engine = conditionTestEngine(), lap = referenceLap();
    configure?.(lap);
    engine.receiveMap(map, 0, lap);
    const update = (position: number, now: number, patch: StandardTelemetrySample = {}) => {
        const sample = beforeDecision({ ...driving(position), ...patch });
        engine.receiveVision(detection(lap, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    return { engine, lap, update };
}

it('enters Brake zone without position-model captures and keeps it past the reference end without a speed target', () => {
    const { engine } = setup();
    const at = (position: number, now: number) => engine.receiveTelemetry(frame(driving(position)), now);
    expect(at(0.19, 1000).state.path).toEqual(['root']);
    expect(at(0.202, 1100).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(at(0.2949, 1200).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.2951, 1225).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.3, 1230).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.5949, 1250).state.path).toEqual(['root', 'Brake zone']);
    const afterThrottleOnset = at(0.7, 1300);
    expect(afterThrottleOnset.state.path).toEqual(['root', 'Brake zone']);
    expect(afterThrottleOnset.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(afterThrottleOnset.closures.find(({ id }) => id === 'reference-throttle')).toMatchObject({ status: 'Not evaluated' });
    const expired = engine.evaluate(2801);
    expect(expired.telemetryReady).toBe(false);
    expect(expired.state.path).toEqual(['root', 'Brake zone']);
    expect(expired.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(expired.referenceSteeringPhase).toBeUndefined();
});

it.each([false, true])('finishes its active brake phrase without exiting at the reference end (playback started: %s)', (started) => {
    const { engine, update } = setup();
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances().map((instance) => [instance.cacheKey, 500])));
    const entered = update(0.25, 1000);
    expect(entered.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    const eventId = entered.events[0].id;
    if (started) engine.speechStarted(eventId, 1050);
    const outside = update(0.7, 1100);
    expect(outside.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(outside.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const released = engine.speechFinished(eventId, 1200);
    expect(released.state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.7, 1250).state.path).toEqual(['root', 'Brake zone']);
});

it.each([0, 0.8])('keeps brake active through light pressure and past reference release to zero with live brake %s', (pedal) => {
    const { engine } = setup((lap) => {
        lap.y[26][4] = lap.y[27][4] = 0.05;
        lap.y[28][4] = lap.y[29][4] = 0.01;
        lap.y[30][4] = 0.009;
    });
    const at = (position: number, now: number, brake = pedal) => engine.receiveTelemetry(
        frame({ ...driving(position), Physics_brake: brake, Physics_gas: 1 }), now);
    // Zone membership does not depend on an earlier high-pressure entry.
    expect(at(0.265, 900, 0).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.25, 1000, 0).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(at(0.265, 1100).state.path).toEqual(['root', 'Brake zone']);
    expect(engine.evaluate(2700).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.285, 3000).state.path).toContain('Brake zone');
    // Reapplying brake before reference release to zero must not rearm Brake.
    expect(at(0.25, 3100, 0).events.filter(({ sentence }) => sentence === 'Brake.')).toHaveLength(1);
    expect(at(0.285, 3200).state.path).toContain('Brake zone');
    expect(at(0.3, 3250).state.path).toContain('Brake zone');
    expect(at(0.31, 3275).state.path).toEqual(['root', 'Brake zone']);
    const afterRelease = at(0.6, 3300);
    expect(afterRelease.state.path).toEqual(['root', 'Brake zone']);
    expect(afterRelease.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(at(0.265, 4300, 0).state.path).toEqual(['root', 'Brake zone']);
    expect(at(0.25, 4400, 0).events.filter(({ sentence }) => sentence === 'Brake.')).toHaveLength(1);
});

it('starts a fresh More brake confirmation on activation and after leaving the reference interval', () => {
    const { engine } = setup(addSpeed);
    const at = (position: number, now: number, speed: number) => engine.receiveTelemetry(frame(beforeDecision({
        ...driving(position), Physics_brake: 0.8, Physics_speed_kmh: speed,
    })), now);
    const moreBrake = (snapshot: ReturnType<typeof at>) => snapshot.closures.find(({ id }) => id === 'reference-moreBrake')!;
    at(0.19, 1000, 106);
    expect(moreBrake(at(0.19, 1400, 105.8)).conditions[1].conditionFit).toBeNull();
    expect(moreBrake(at(0.255, 1500, 102.6)).conditions[1].conditionFit).toBe(false);
    expect(moreBrake(at(0.255, 1600, 102.5)).conditions[1].conditionFit).toBe(true);
    expect(at(0.255, 2099, 102.3).events).toEqual([]);
    expect(at(0.255, 2100, 102.299).events).toEqual([
        expect.objectContaining({ sentence: 'More brake.', timestamp: 2100 }),
    ]);
    expect(at(0.61, 2200, 106).state.path).toEqual(['root', 'Brake zone']);
    expect(moreBrake(at(0.255, 2300, 102.6)).conditions[1].conditionFit).toBe(false);
    expect(moreBrake(at(0.255, 2400, 102.5)).conditions[1].conditionFit).toBe(true);
    expect(at(0.255, 2899, 102.3).events).toHaveLength(1);
    expect(moreBrake(at(0.255, 2900, 102.299)).conditions[1].conditionFit).toBe(true);
});

function throttleCornerSetup(direction: 'left' | 'right' = 'right', game: 'acc' | 'iracing' = 'acc') {
    const engine = conditionTestEngine(), lap = referenceLap();
    addSpeed(lap);
    lap.game = game;
    const sign = (direction === 'right' ? 1 : -1) * (game === 'iracing' ? -1 : 1);
    lap.y.forEach((row, i) => {
        const angle = i / 100 * 2 * Math.PI;
        row[0] = 100 * Math.cos(angle);
        row[1] = game === 'acc' ? sign * 100 * Math.sin(angle) : 0;
        row[2] = game === 'acc' ? 0 : sign * 100 * Math.sin(angle);
        row[3] = 0; // Corner direction must come from geometry, not steering.
        row[4] = 0;
        row[5] = 0.4;
    });
    engine.receiveMap(null, 0, lap);
    const update = (lateral: number, now: number, patch: StandardTelemetrySample = {}, capturedAt = now) => {
        const sample = beforeDecision({ ...driving(0.655), Physics_speed_kmh: 110, Physics_gas: 0.4, ...patch },
            20000 * Math.sin(Math.PI / 100));
        const view = detection(lap, sample, capturedAt), event = frame(sample);
        view.position!.referenceDelta[0] = lateral;
        view.telemetry!.game = event.update.game = game;
        engine.receiveVision(view, now);
        return engine.receiveTelemetry(event, now);
    };
    return { engine, lap, update };
}

it('starts outward-motion measurement only after entering the throttle parent', () => {
    let open = false;
    const { lap } = throttleCornerSetup();
    const engine = conditionTestEngine(PHRASE_DEFINITIONS, (phrases) => new Closure('root', 'Root.', () => true,
        [new Closure('parent', 'Parent.', () => open, createPhraseRoot(phrases).children)]));
    engine.receiveMap(null, 0, lap);
    const at = (lateral: number, now: number) => {
        const sample = beforeDecision({ ...driving(0.655), Physics_speed_kmh: 110, Physics_gas: 0.4 },
            20000 * Math.sin(Math.PI / 100));
        const view = detection(lap, sample, now);
        view.position!.referenceDelta[0] = lateral;
        engine.receiveVision(view, now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    expect(at(4, 1000).events).toEqual([]);
    open = true;
    expect(at(5, 1100).events).toEqual([]);
    expect(at(6, 1200).events).toEqual([
        expect.objectContaining({ sentence: 'Ease gas.', timestamp: 1200 }),
    ]);
});

it.each([-20, 0, 20])('uses normalized reference speed and model lateral movement for Less throttle with forward offset %s', (forward) => {
    const { lap } = throttleCornerSetup();
    const sample = { ...driving(0.655), Physics_speed_kmh: 110, Physics_gas: 0.4 };
    const view = detection(lap, sample, 1100);
    view.position!.referenceDelta = [5, forward, 0];
    const previous = { capturedAt: 1000, lateralOffset: 4 };
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1100, lap, undefined, previous).lessThrottle).toBe(2600);
    expect(readReferenceGuidance(sample, null, lap, 'acc', 1100, lap, undefined, previous).lessThrottle).toBeUndefined();
    view.capturedAt = 500;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1100, lap, undefined, previous).lessThrottle).toBeUndefined();
});

it.each([
    ['acc', 'left', -1], ['acc', 'right', 1], ['iracing', 'left', -1], ['iracing', 'right', 1],
] as const)('requires at least 5 units of outward line error in a %s %s corner', (game, direction, sign) => {
    for (const offset of [4.99, 5, 5.01]) {
        const { engine, update } = throttleCornerSetup(direction, game);
        expect(update(4 * sign, 1000).events).toEqual([]);
        const result = update(offset * sign, 1100);
        expect(result.events.map(({ sentence }) => sentence)).toEqual(offset >= 5 ? ['Ease gas.'] : []);
        expect(result.state.path).toEqual(offset >= 5
            ? ['root', 'Throttle zone', 'Less throttle', 'say phrase']
            : ['root', 'Throttle zone']);
        expect(engine.evaluate(1101).state.path).toEqual(['root', 'Throttle zone']);
    }
});

it.each([
    ['matching speed', 4, 5, 102, 1, 0],
    ['slower than reference', 4, 5, 94, 1, 0],
    ['speed within tolerance', 4, 5, 106.9, 0.4, 0],
    ['holding an outside line', 5, 5, 110, 0.4, 0],
    ['returning from outside', 6, 5, 110, 0.4, 0],
    ['moving inward', -4, -5, 110, 0.4, 0],
    ['returning from inside', -6, -5, 110, 0.4, 0],
    ['no applied throttle', 4, 5, 110, 0, 0],
    ['brake applied', 4, 5, 110, 0.4, 0.3],
] as const)('withholds Less throttle when %s', (_name, before, after, speed, throttle, brake) => {
    const { update, lap, engine } = throttleCornerSetup();
    // Keep the target constant throughout preparation so a lower target earlier
    // in the window cannot legitimately request a speed correction.
    lap.y.forEach((row) => { row[6] = 102; });
    engine.receiveMap(null, 0, null);
    engine.receiveMap(null, 0, lap);
    const patch = { Physics_speed_kmh: speed, Physics_gas: throttle, Physics_brake: brake };
    update(before, 1000, patch);
    expect(update(after, 1100, patch).events.some(({ sentence }) => sentence === 'Ease gas.')).toBe(false);
});

it.each(['gap', 'expired capture', 'repeated capture', 'out-of-order capture', 'pause', 'stream reset', 'reference change'])(
    'requires fresh outward movement after %s', (kind) => {
        const { engine, lap, update } = throttleCornerSetup();
        update(4, 1000);
        if (kind === 'pause') engine.receiveTelemetry({ ...frame(driving(0.655)), telemetryStatus: 1 }, 1050);
        if (kind === 'stream reset') engine.receiveTelemetry({ type: 'stream-reset', snapshot: createLiveTelemetryStore().getSnapshot() }, 1050);
        if (kind === 'reference change') engine.receiveMap(null, 1050, { ...lap });
        const now = kind === 'gap' || kind === 'expired capture' ? 1601 : 1100;
        const capturedAt = kind === 'expired capture' || kind === 'repeated capture' ? 1000
            : kind === 'out-of-order capture' ? 999 : now;
        expect(update(5, now, {}, capturedAt).events).toEqual([]);
    });

it('uses the interpolated local speed threshold even when the applied throttle is below reference demand', () => {
    const { update } = throttleCornerSetup();
    const patch = { Physics_speed_kmh: 107.1, Physics_gas: 0.15 };
    update(4, 1000, patch);
    expect(update(5, 1100, patch).events.map(({ sentence }) => sentence)).toEqual(['Ease gas.']);
});

it('preserves the corner-exit overspeed threshold when braking moves earlier', () => {
    const { engine, update } = throttleCornerSetup();
    engine.setDrivingAggression(50, 0);
    const patch = { Physics_speed_kmh: 108 }; // The exit target stays at the reference 102 km/h.
    update(4, 1000, patch);
    expect(update(5, 1100, patch).events.map(({ sentence }) => sentence)).toEqual(['Ease gas.']);
});

it.each(['straight', 'missing speed', 'invalid speed', 'missing throttle', 'brake zone'])(
    'withholds outward-drift throttle correction with %s', (kind) => {
        const { lap } = throttleCornerSetup();
        if (kind === 'straight') lap.y.forEach((row, i) => { row[0] = i * 10; row[1] = 0; });
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[65][6] = NaN;
        if (kind === 'missing throttle') lap.target_features[5] = 'missing';
        if (kind === 'brake zone') lap.y.forEach((row) => { row[4] = 0.8; });
        const sample = { ...driving(0.655), Physics_speed_kmh: 110, Physics_gas: 1 };
        const view = detection(lap, sample, 1100);
        view.position!.referenceDelta[0] = 5;
        const guidance = readReferenceGuidance(sample, view, lap, 'acc', 1100, lap, undefined,
            { capturedAt: 1000, lateralOffset: 4 });
        expect(guidance.lessThrottle).toBeUndefined();
    });

it.each([false, true])('uses local curvature across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const { lap } = throttleCornerSetup();
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    for (const position of [0, 0.005, 0.995, 1]) {
        const sample = { ...driving(position), Physics_speed_kmh: 130, Physics_gas: 0.4 };
        const view = detection(lap, sample, 1100);
        view.position!.referenceDelta[0] = 5;
        expect(readReferenceGuidance(sample, view, lap, 'acc', 1100, lap, undefined,
            { capturedAt: 1000, lateralOffset: 4 }).lessThrottle).toBe(2600);
    }
});

function steeringMap(sign = 1, game: 'acc' | 'iracing' = 'acc'): CircuitMapDto {
    const map = circuitMap();
    map.game = game;
    map.samples.middle_line = map.samples.middle_line!.map((point) => ({ ...point,
        normalized_position: (point.normalized_position + 0.15) % 1,
        y: game === 'acc' ? point.y * sign : 0, z: game === 'iracing' ? point.y * sign : 0,
    })).sort((a, b) => a.normalized_position - b.normalized_position);
    map.centerline_tags = map.centerline_tags!.map((tag) => ({ ...tag,
        start_position: (tag.start_position + 0.15) % 1, end_position: (tag.end_position + 0.15) % 1,
    }));
    return map;
}

function brakingTurn(lap: TopLapReferenceEntry) {
    addSpeed(lap);
    setTrajectory(lap, (i) => 48 + 12 * (Math.max(0, Math.min(15, i - 25)) + (i === 25 ? 0.5 : 0)));
    lap.y.forEach((row, i) => {
        row[3] = i >= 25 && i <= 28 ? (i - 25) * 0.02 : i >= 29 && i <= 40 ? 0.4 + (i - 29) * 0.04 : 0;
        // Steering starts at 250 m; 3 m/s² braking eases to 1 m/s² from 260 m before turn-in near 281 m.
        row[6] = Math.sqrt(100 ** 2 / 3.6 ** 2 - 6 * Math.max(0, Math.min(60, i * 10 - 200))
            - 2 * Math.max(0, Math.min(40, i * 10 - 260))) * 3.6;
    });
}

function easingLap(drop: number) {
    const lap = referenceLap();
    brakingTurn(lap);
    let energy = 40 ** 2;
    lap.y.forEach((row, i) => {
        if (i >= 25 && i <= 30) energy -= 20 * (i <= 26 ? 4 : 4 - drop);
        row[6] = Math.sqrt(energy) * 3.6;
        row[4] = 0.8; // Deliberately constant: pedal release rate must not drive the instance.
    });
    return lap;
}

it.each([0.1, 0.2, 0.4, 0.8])('uses reference deceleration change over time for a %s m/s² easing', (drop) => {
    const lap = easingLap(drop);
    const v = (i: number) => lap.y[i][6] / 3.6;
    const expected = drop / (10 / (v(25) + v(26)) + 10 / (v(26) + v(27)));
    for (const liveSpeed of [36, 180]) {
        const guidance = steeringGuidance({ ...driving(0.265), Physics_brake: 0.8, Physics_speed_kmh: liveSpeed }, lap);
        expect(guidance.brakeReleaseRate).toBeCloseTo(expected, 8);
    }
});

it.each([false, true])('keeps the easing rate across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = referenceLap();
    addSpeed(lap);
    // 4 m/s² just before the line, then 3.8 m/s² just after it, over two 10 m intervals.
    lap.y[99][6] = 40 * 3.6;
    lap.y[0][6] = Math.sqrt(40 ** 2 - 80) * 3.6;
    lap.y[1][6] = Math.sqrt(40 ** 2 - 80 - 76) * 3.6;
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const v = (i: number) => lap.y[i][6] / 3.6;
    const expected = 0.2 / (10 / (v(99) + v(0)) + 10 / (v(0) + v(1)));
    expect(steeringGuidance({ ...driving(0.005), Physics_brake: 0.8 }, lap).brakeReleaseRate).toBeCloseTo(expected, 8);
});

it.each(['released pedal', 'invalid pedal', 'missing speed', 'invalid speed', 'constant deceleration', 'increasing deceleration', 'straight'])(
    'does not say Ease brake with %s', (kind) => {
        const lap = easingLap(kind === 'constant deceleration' ? 0 : kind === 'increasing deceleration' ? -0.2 : 0.2);
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[26][6] = NaN;
        const engine = conditionTestEngine();
        engine.receiveMap(kind === 'straight' ? null : steeringMap(), 0, lap);
        const sample = { ...driving(0.265), Physics_brake: kind === 'released pedal' ? 0 : kind === 'invalid pedal' ? NaN : 0.8 };
        engine.receiveTelemetry(frame(sample), 1000);
        const snapshot = engine.receiveTelemetry(frame(sample), 1100);
        expect(snapshot.events.some(({ sentence }) => sentence === SLOWLY_RELEASE_BRAKE)).toBe(false);
    });

it.each([[0.1, 1], [0.2, 2 / 3], [0.4, 1 / 3], [0.8, 0]])(
    'speaks one prepared release instance without a directional cue for deceleration easing %s', (drop, timing) => {
        const lap = easingLap(drop);
        const engine = conditionTestEngine();
        engine.receiveMap(steeringMap(), 0, lap);
        const at = (position: number, now: number) => engine.receiveTelemetry(frame(beforeDecision({ ...driving(position), Physics_brake: 0.8 })), now);
        expect(at(0.255, 1000).events).toEqual([]);
        const released = at(0.265, 1100);
        expect(released.events.map(({ sentence }) => sentence)).toEqual([SLOWLY_RELEASE_BRAKE]);
        const event = released.events[0];
        expect(event).toMatchObject({ brakeRelease: true, expiresAt: 5100, instance: { inputs: { timing } } });
        expect(event.steeringDirection).toBeUndefined();
        expect(event.instance).toBeInstanceOf(PhraseInstance);
        expect(event.instance!.inputs.timing).toBe(brakeReleaseSettings(released.referenceBrakeReleaseRate!).inputs!.timing);
        expect(getPhraseSpeechInstances().some((instance) => instance.cacheKey === event.instance!.cacheKey)).toBe(true);
        expect(at(0.266, 1200).events).toEqual(released.events);
        at(0.6, 1300);
        at(0.255, 1400);
        expect(at(0.265, 1500).events.filter(({ brakeRelease }) => brakeRelease)).toHaveLength(2);
    });

it.each(['coasting', 'throttle', 'missing brake'])(
    'enters Steering directly from root with %s reference input', (kind) => {
        const { update } = setup((lap) => {
            brakingTurn(lap);
            lap.y.forEach((row) => { row[4] = 0; row[5] = kind === 'throttle' ? 0.8 : 0; });
            if (kind === 'missing brake') lap.target_features[4] = 'missing';
        }, steeringMap());
        const snapshot = update(0.285, 1000, { Physics_gas: 0.8 });
        expect(snapshot.state.path).toEqual(['root', 'Steering']);
        expect(snapshot.events).toEqual([]);
    });

it('withholds throttle while steering changes and enters after reference steering recovers', () => {
    const { update } = setup((lap) => {
        brakingTurn(lap);
        lap.y.forEach((row, i) => { row[4] = 0; row[5] = i >= 21 && i <= 50 ? 0.8 : 0; });
    }, steeringMap());
    expect(update(0.25, 1000, { Physics_speed_kmh: 180, Physics_gas: 0.8 }).state.path).toEqual(['root', 'Steering']);
    expect(update(0.285, 1100, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Steering']);
    expect(update(0.32, 1200, { Physics_gas: 0.8 }).state.path).toContain('Steering');
    expect(update(0.32, 1250, { Physics_gas: 0.8, Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0 }).state.path).toContain('Steering');
    expect(update(0.43, 1300, { Physics_gas: 0.8 }).state.path).toEqual(['root']);
    expect(update(0.43, 1400, { Physics_gas: 0.8 }).state.path.slice(0, 2)).toEqual(['root', 'Throttle zone']);
});

it.each([0, 0.8])('defers Brake zone until the active Steering range ends with live brake %s', (brake) => {
    const { update } = setup((lap) => {
        brakingTurn(lap);
        lap.y.forEach((row, i) => { row[4] = i >= 28 && i <= 50 ? 0.8 : 0; });
    }, steeringMap());
    expect(update(0.26, 1000).state.path).toEqual(['root', 'Steering']);
    const braking = update(0.285, 1100, { Physics_speed_kmh: 180, Physics_brake: brake });
    expect(braking.state.path).toEqual(brake ? ['root', 'Steering', 'say phrase'] : ['root', 'Steering']);
    expect(braking.events.map(({ sentence }) => sentence)).toEqual(brake ? [SLOWLY_RELEASE_BRAKE] : []);
    expect(update(0.410001, 1200, { Physics_speed_kmh: 180, Physics_brake: brake }).state.path).toEqual(['root']);
    expect(update(0.42, 1300, { Physics_speed_kmh: 180, Physics_brake: brake }).state.path).toContain('Brake zone');
});

it('speaks Turn on forward-aligned entry and does not exit just because Turn completes', () => {
    const lap = referenceLap();
    brakingTurn(lap);
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.setDrivingAggression(100, 0);
    engine.receiveMap(steeringMap(), 0, lap);
    const patch = { Physics_brake: 0.8, Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0 };
    const update = (position: number, now: number) => engine.receiveTelemetry(frame({ ...driving(position), ...patch }), now);
    expect(update(0.256, 1000).state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(update(0.257, 1100).state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(update(0.285, 1200).events.map(({ sentence }) => sentence)).toEqual(['Turn in', SLOWLY_RELEASE_BRAKE]);
    expect(engine.evaluate(1201).state.path).toContain('Steering');
    expect(update(0.285, 1300).state.path).toContain('Steering');
    expect(update(0.285, 4000).events.filter(({ sentence }) => sentence === 'Turn in')).toHaveLength(1);
});

it('exits an active direction correction when the captured trajectory direction-change range ends', () => {
    const lap = referenceLap();
    brakingTurn(lap);
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(steeringMap(), 0, lap);
    engine.setDrivingAggression(100, 0);
    const at = (now: number, heading: number, position = 0.285) => {
        const sample = { ...driving(position), ...localMotion(6, 10, heading) };
        engine.receiveVision(detection(lap, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    expect(at(1000, 0.1).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    expect(at(1100, 0.1).events).toHaveLength(1);
    expect(at(1499, 0.1).events).toHaveLength(1);
    expect(at(1500, 0.1).state.path).toEqual(['root', 'Steering', 'More right', 'say phrase']);
    const exited = at(1700, 0.1, 0.419);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'More right']);
    expect(at(5200, 0.1, 0.43).events).toEqual(exited.events);
});

it('requires 500 ms of deviation after Turn and speaks once until realignment', () => {
    const lap = referenceLap();
    brakingTurn(lap);
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(steeringMap(), 0, lap);
    engine.setDrivingAggression(100, 0);
    const at = (now: number, vx: number, speed = 40, position = 0.285, heading = 0.1) => {
        // Keep the pedal released so this test isolates direction cues from brake-release speech.
        const sample = beforeDecision({ ...driving(position), Physics_brake: 0, Physics_speed_kmh: speed, ...localMotion(vx, 10, heading) });
        engine.receiveVision(detection(lap, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    expect(at(1000, 2, 40, 0.26).events).toEqual([expect.objectContaining({ sentence: 'Turn in', timestamp: 1000 })]);
    // The reference tangent here is 84 degrees; small deviations on either side stay silent.
    expect(at(1050, 10 / Math.tan(80 * Math.PI / 180), 180).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    expect(at(1075, 10 / Math.tan(88 * Math.PI / 180), 180).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    expect(at(1100, 6, 180).events).toHaveLength(1);
    expect(at(1599, 6, 180).events).toHaveLength(1);
    const wide = at(1600, 6, 180);
    expect(wide.state.path).toContain('Steering');
    expect(wide.events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'More right']);
    expect(wide.events[1]).toMatchObject({ timestamp: 1600, steeringDirection: 'right', expiresAt: 3100 });
    expect(at(1700, 6, 180).events).toEqual(wide.events);
    expect(at(1800, -4, 180).events).toEqual(wide.events);
    expect(at(2299, -4, 180).events).toEqual(wide.events);
    expect(at(2300, -4, 180).events).toEqual(wide.events);
    const alignedVelocity = at(2400, 10 / Math.tan(84 * Math.PI / 180), 180);
    expect(alignedVelocity.events).toHaveLength(2);
    expect(alignedVelocity.state.path).toContain('Steering');
    const stale = engine.receiveTelemetry(frame({ ...driving(0.285), Physics_speed_kmh: 180 }), 3000);
    expect(stale.referenceSteeringDirection).toBeUndefined();
    expect(stale.events).toHaveLength(2);
    expect(stale.state.path).toContain('Steering');
    expect(at(3100, 2, 40, 0.419, 0.1).state.path).toEqual(['root']);
    expect(at(3200, 2, 180).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'More right', 'Turn in']);
    expect(at(3300, 6).events).toHaveLength(3);
    expect(at(3800, 6).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'More right', 'Turn in', 'More right']);
    expect(engine.setDrivingAggression(99, 3900).events).toEqual([]);
    expect(at(4000, 6).events).toEqual([]);
    expect(at(4100, 6).events).toEqual([]);
    expect(at(4600, -4).events).toEqual([]);
});

it.each([[6, 'More right'], [-4, 'More left']] as const)(
    'speaks %s motion correction %s once and rearms after realignment', (vx, sentence) => {
        const lap = referenceLap();
        brakingTurn(lap);
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(steeringMap(), 0, lap);
        engine.setDrivingAggression(100, 0);
        const at = (now: number, velocity = vx as number) => {
            const sample = beforeDecision({ ...driving(0.285), ...localMotion(velocity, 10) });
            engine.receiveVision(detection(lap, sample, now), now);
            return engine.receiveTelemetry(frame(sample), now);
        };
        expect(at(1000).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
        expect(at(1499).events).toHaveLength(1);
        const spoken = at(1500);
        expect(spoken.events.map(({ sentence }) => sentence)).toEqual(['Turn in', sentence]);
        for (const now of [1750, 2000, 2250, 2500]) expect(at(now).events).toEqual(spoken.events);
        const aligned = at(2600, 10 / Math.tan(84 * Math.PI / 180));
        expect(aligned.referenceSteeringDirection).toBe('aligned');
        expect(aligned.events).toEqual(spoken.events);
        expect(at(2700).events).toEqual(spoken.events);
        expect(at(3199).events).toEqual(spoken.events);
        expect(at(3200).events.map(({ sentence }) => sentence)).toEqual(['Turn in', sentence, sentence]);
        expect(at(3300).events.map(({ sentence }) => sentence)).toEqual(['Turn in', sentence, sentence]);
    });

it.each(['missing motion', 'missing capture', 'capture gap', 'telemetry gap', 'paused'])(
    'does not rearm a spoken correction after %s without realignment', (interruption) => {
        const lap = referenceLap();
        brakingTurn(lap);
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(steeringMap(), 0, lap);
        engine.setDrivingAggression(100, 0);
        const sample = beforeDecision({ ...driving(0.285), ...localMotion(6, 10) });
        const at = (now: number) => {
            engine.receiveVision(detection(lap, sample, now), now);
            return engine.receiveTelemetry(frame(sample), now);
        };
        at(1000);
        at(1100);
        const spoken = at(1500);
        expect(spoken.events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'More right']);
        if (interruption === 'missing capture') engine.receiveVision(null, 1600);
        if (interruption === 'missing motion') engine.receiveTelemetry(frame({ ...sample, Physics_local_velocity_x: undefined }), 1600);
        if (interruption === 'paused') engine.receiveTelemetry({ ...frame(sample), telemetryStatus: 1 }, 1600);
        const resume = interruption === 'capture gap' ? 2001 : interruption === 'telemetry gap' ? 3101 : 1700;
        expect(at(resume).events).toEqual(spoken.events);
        expect(at(resume + 499).events).toEqual(spoken.events);
        expect(at(resume + 500).events).toEqual(spoken.events);
        expect(at(resume + 1000).events).toEqual(spoken.events);
    });

it.each(['aligned', 'opposite', 'missing motion', 'missing capture', 'expired capture', 'paused'])(
    'restarts the 500 ms steering hold after %s input', (interruption) => {
        const lap = referenceLap();
        brakingTurn(lap);
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(steeringMap(), 0, lap);
        engine.setDrivingAggression(100, 0);
        const sampleAt = (vx = 6) => beforeDecision({ ...driving(0.285), ...localMotion(vx, 10) });
        const at = (now: number) => {
            const sample = sampleAt();
            engine.receiveVision(detection(lap, sample, now), now);
            return engine.receiveTelemetry(frame(sample), now);
        };
        at(1000);
        at(1100);
        expect(at(1400).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
        let sample = sampleAt(interruption === 'aligned' ? 10 / Math.tan(84 * Math.PI / 180)
            : interruption === 'opposite' ? -4 : 6);
        if (interruption === 'missing motion') sample = { ...sample, Physics_local_velocity_x: undefined };
        if (interruption === 'missing capture') engine.receiveVision(null, 1450);
        else if (interruption === 'expired capture') {
            // No fresh capture arrives: the previously valid capture expires during telemetry updates.
            engine.receiveTelemetry(frame(sample), 1901);
        } else {
            engine.receiveVision(detection(lap, sample, 1450), 1450);
            const event = frame(sample);
            if (interruption === 'paused') event.telemetryStatus = 1;
            engine.receiveTelemetry(event, 1450);
        }
        const restart = interruption === 'expired capture' ? 2000 : 1500;
        expect(at(restart).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
        expect(at(restart + 499).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
        expect(at(restart + 500).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'More right']);
    });

it.each([501, 1600])('does not count an unobserved %s ms gap toward the steering hold', (gap) => {
    const lap = referenceLap();
    brakingTurn(lap);
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.receiveMap(steeringMap(), 0, lap);
    engine.setDrivingAggression(100, 0);
    const at = (now: number) => {
        const sample = beforeDecision({ ...driving(0.285), ...localMotion(6, 10) });
        engine.receiveVision(detection(lap, sample, now), now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    at(1000);
    at(1100);
    expect(at(1400).events).toHaveLength(1);
    const restart = 1400 + gap;
    expect(at(restart).events).toHaveLength(1);
    expect(at(restart + 499).events).toHaveLength(1);
    expect(at(restart + 500).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'More right']);
});

it.each(['acc', 'iracing'] as const)('speaks the required correction side in either corner direction in %s', (game) => {
    for (const sign of [-1, 1] as const) {
        for (const side of ['right', 'left'] as const) {
            const lap = trajectoryLap(2, sign, game);
            const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
            engine.receiveMap(steeringMap(sign, game), 0, lap);
            engine.setDrivingAggression(100, 0);
            // At the decision point the reference tangent is sign * 6 degrees.
            const angle = (sign * 6 - (side === 'right' ? 30 : -30) * (game === 'iracing' ? -1 : 1)) * Math.PI / 180;
            const sample = beforeDecision({ ...driving(0.285), Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0,
                Physics_local_velocity_x: 10 * Math.cos(angle),
                Physics_local_velocity_y: game === 'acc' ? 10 * Math.sin(angle) : 0,
                Physics_local_velocity_z: game === 'iracing' ? 10 * Math.sin(angle) : 0 });
            const at = (now: number) => {
                const view = detection(lap, sample, now);
                view.telemetry!.game = game;
                engine.receiveVision(view, now);
                const event = frame(sample);
                event.update.game = game;
                return engine.receiveTelemetry(event, now);
            };
            at(1000);
            expect(at(1100).referenceSteeringDirection).toBe(side);
            expect(at(1499).events).toHaveLength(1);
            expect(at(1500).events.map(({ sentence }) => sentence)).toEqual(['Turn in', `More ${side}`]);
        }
    }
});

it.each([0, 50, 99, 99.9, 100])('gates all directional speech at driving aggression %s while keeping Steering active', (aggression) => {
    for (const [vx, direction, sentence] of [[6, 'right', 'More right'], [-4, 'left', 'More left']] as const) {
        const lap = referenceLap();
        brakingTurn(lap);
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(steeringMap(), 0, lap);
        engine.setDrivingAggression(aggression, 0);
        const at = (now: number, position = 0.285) => {
            const sample = { ...driving(position), ...localMotion(vx, 10, 0.1) };
            engine.receiveVision(detection(lap, sample, now), now);
            return engine.receiveTelemetry(frame(sample), now);
        };
        const entered = at(1000);
        expect(entered.state.path).toContain('Steering');
        expect(entered.events).toEqual(aggression === 100 ? [expect.objectContaining({ sentence: 'Turn in', timestamp: 1000 })] : []);
        expect(at(1100).events).toHaveLength(aggression === 100 ? 1 : 0);
        expect(at(1499).events).toHaveLength(aggression === 100 ? 1 : 0);
        const correction = at(1500);
        expect(correction.referenceSteeringDirection).toBe(direction);
        expect(correction.events.map(({ sentence }) => sentence)).toEqual(aggression === 100 ? ['Turn in', sentence] : []);
        const steering = correction.root.children.find(({ id }) => id === 'reference-steering')!;
        expect(steering.children.find(({ id, name }) => aggression === 100 ? name === sentence : id === 'reference-steering:say:0')!.conditions)
            .toContainEqual(expect.objectContaining({
                description: 'Driving aggression = 100', conditionFit: aggression === 100, inputMissing: false,
            }));
    }
});

it.each([
    [0.265, 'prepare', 0], [0.285, 'turn', 0],
    [0.265, 'prepare', 50], [0.285, 'turn', 50],
] as const)('enters Steering at %s in phase %s using %s%% caution regardless of reference speed', (position, phase, caution) => {
    const lap = referenceLap();
    brakingTurn(lap);
    const adjusted = copyPhraseReference(lap, caution)!;
    const after = adjusted.x.findIndex((point) => point > position), before = after - 1;
    // Caution can insert reference samples, so interpolate within the actual adjusted interval.
    const fraction = (position - adjusted.x[before]) / (adjusted.x[after] - adjusted.x[before]);
    const target = Math.sqrt(adjusted.y[before][6] ** 2
        + (adjusted.y[after][6] ** 2 - adjusted.y[before][6] ** 2) * fraction);
    for (const offset of [-1, 0, 0.000001, 1, 40]) {
        const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
        engine.receiveMap(steeringMap(), 0, lap);
        engine.setDrivingAggression(caution, 0);
        const entered = engine.receiveTelemetry(frame({
            ...driving(position), Physics_speed_kmh: target + offset, Physics_brake: 0.8,
        }), 1000);
        expect(entered.referenceSteeringPhase).toBe(phase);
        expect(entered.state.path).toContain('Steering');
        expect(entered.events.every(({ sentence }) => sentence === SLOWLY_RELEASE_BRAKE)).toBe(true);
    }
});

it.each(['missing', 'NaN', 'infinite', 'negative'])('keeps Brake zone past the pedal boundary with %s reference speed', (kind) => {
    const { update } = setup((lap) => {
        brakingTurn(lap);
        if (kind === 'missing') lap.target_features[6] = 'missing';
        else lap.y[25][6] = kind === 'NaN' ? NaN : kind === 'infinite' ? Infinity : -1;
    }, steeringMap());
    expect(update(0.24, 900, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.26, 1000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.30, 1050, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    const snapshot = update(0.305, 1100, { Physics_brake: 0.8 });
    expect(snapshot.state.path).toEqual(['root', 'Brake zone']);
    expect(snapshot.events).toEqual([]);
});

it('enters Steering above reference speed and keeps Turn available after speed rises', () => {
    const lap = referenceLap();
    brakingTurn(lap);
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.setDrivingAggression(100, 0);
    engine.receiveMap(steeringMap(), 0, lap);
    const entered = engine.receiveTelemetry(frame({ ...driving(0.26), Physics_speed_kmh: lap.y[26][6] + 10 }), 1000);
    expect(entered.state.path).toEqual(['root', 'Steering', 'say phrase']);
    const turned = engine.receiveTelemetry(frame({ ...driving(0.285), Physics_speed_kmh: lap.y[28][6] + 10 }), 1100);
    expect(turned.state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(turned.events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
});

it.each(['zero', 'opposite', 'missing', 'invalid'])('speaks Turn on Steering entry with %s reference steering', (kind) => {
    const lap = referenceLap();
    brakingTurn(lap);
    lap.y.forEach((row) => { row[3] = kind === 'opposite' ? -0.5 : kind === 'invalid' ? NaN : 0; });
    if (kind === 'missing') lap.target_features[3] = 'missing';
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.setDrivingAggression(100, 0);
    engine.receiveMap(steeringMap(), 0, lap);
    for (const position of [0.25, 0.254, 0.255]) {
        const snapshot = engine.receiveTelemetry(frame(driving(position)), 1000);
        expect(snapshot.state.path).toEqual(['root', 'Steering', 'say phrase']);
        expect(snapshot.events).toEqual([expect.objectContaining({ sentence: 'Turn in', timestamp: 1000 })]);
    }
    const entered = engine.receiveTelemetry(frame(driving(0.256)), 1100);
    expect(entered.state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(entered.events).toEqual([expect.objectContaining({ sentence: 'Turn in', timestamp: 1000 })]);
});

it('speaks Turn once on Steering entry and keeps pedal zones out until the trajectory range ends', () => {
    const { engine, lap, update } = setup(brakingTurn, steeringMap());
    engine.setDrivingAggression(100, 0);
    const at = (position: number, now: number, speed: number, heading = 0.1) => update(position, now, {
        Physics_speed_kmh: speed - 1, Physics_brake: 0.8, Physics_steer_angle: -0.5,
        Physics_heading: heading, Physics_pitch: 0, Physics_roll: 0,
    });
    const approaching = at(0.22, 900, lap.y[22][6]);
    expect(approaching.state.path).toEqual(['root', 'Brake zone']);
    expect(approaching.events.map(({ sentence }) => sentence)).toEqual([]);
    const released = at(0.256, 1000, lap.y[26][6]);
    expect(released.events).toEqual([expect.objectContaining({ sentence: 'Off brake.', timestamp: 1000 })]);
    const prepared = at(0.256, 1001, lap.y[26][6]);
    expect(prepared.state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(prepared.events).toEqual([...released.events, expect.objectContaining({ sentence: 'Turn in', timestamp: 1001 })]);
    const eased = at(0.26, 2100, lap.y[26][6]);
    expect(eased.events).toEqual([...prepared.events, expect.objectContaining({ sentence: SLOWLY_RELEASE_BRAKE, brakeRelease: true })]);
    expect(eased.state.path).toContain('Steering');
    expect(at(0.26, 2300, lap.y[26][6]).events).toEqual(eased.events);
    const turned = at(0.283, 2500, lap.y[28][6]);
    expect(turned.events.map(({ sentence }) => sentence)).toEqual(['Off brake.', 'Turn in', SLOWLY_RELEASE_BRAKE]);
    expect(turned.events).toEqual(eased.events);
    expect(turned.state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(at(0.285, 5500, lap.y[28][6]).events).toHaveLength(3);
    expect(engine.evaluate(5501).state.path).toContain('Steering');
    expect(at(0.299, 5600, lap.y[29][6]).state.path).toContain('Steering');
    expect(at(0.2999, 5700, lap.y[29][6]).state.path).toContain('Steering');
    const exited = at(0.6, 5800, lap.y[29][6], 0);
    expect(exited.state.path).toEqual(['root']);
    expect(at(0.6, 5900, lap.y[29][6], 0).state.path).toEqual(['root', 'Throttle zone']);
    expect(exited.events).toHaveLength(3);
    at(0.5, 7000, lap.y[28][6]);
    at(0.283, 8000, lap.y[28][6]);
    expect(at(0.283, 8100, lap.y[28][6]).events.filter(({ sentence }) => sentence === 'Turn in')).toHaveLength(2);
});

it.each([[0.26, 'prepare'], [0.283, 'turn']] as const)(
    'keeps entered Steering at %s in phase %s despite a brake zone and increased live speed', (position, phase) => {
        const { lap, update } = setup(brakingTurn, steeringMap());
        const brake = update(0.24, 1000, { Physics_speed_kmh: lap.y[24][6] }).events;
        expect(brake.map(({ sentence }) => sentence)).toEqual(['Brake.']);
        expect(update(0.24, 1050, { Physics_speed_kmh: lap.y[24][6], Physics_brake: 0.8 }).events
            .map(({ sentence }) => sentence)).toEqual(['Brake.']);

        const released = update(position, 1100, {
            Physics_speed_kmh: lap.y[Math.floor(position * 100)][6] - 1, Physics_brake: 0.8,
        });
        expect(released.state.path).toContain('Release the brake');
        const entered = update(position, 1101, {
            Physics_speed_kmh: lap.y[Math.floor(position * 100)][6] - 1, Physics_brake: 0.8,
        });
        expect(entered.referenceSteeringPhase).toBe(phase);
        expect(entered.state.path).toContain('Steering');
        expect(entered.events.some(({ sentence }) => sentence === 'Turn in')).toBe(false);

        const turning = { Physics_speed_kmh: lap.y[28][6], Physics_brake: 0.8 };
        const turned = update(0.283, 1200, turning);
        // The original Steering scope remains entered as speed changes.
        expect(turned.events.map(({ sentence }) => sentence)).toEqual([
            'Brake.', 'Off brake.', ...(phase === 'prepare' ? [SLOWLY_RELEASE_BRAKE] : []),
        ]);
        expect(turned.state.path).toContain('Steering');
        expect(update(0.283, 1300, turning).events).toEqual(turned.events);
    });

it('enters Steering while the trajectory overlaps the corner despite a later steering reversal', () => {
    const { update } = setup((lap) => {
        brakingTurn(lap);
        lap.y.forEach((row) => { row[4] = row[5] = 0; });
        [0.1, 0.2, 0.15].forEach((value, i) => { lap.y[26 + i][3] = value; });
    }, steeringMap());
    const snapshot = update(0.26, 1000, { Physics_brake: 0.8 });
    expect(snapshot.state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual([SLOWLY_RELEASE_BRAKE]);
});

it('keeps Turn available when steering unwinds but trajectory direction still qualifies', () => {
    const lap = trajectoryLap();
    [0.02, 0.04, 0.06, 0.12, 0.1].forEach((value, i) => { lap.y[25 + i][3] = value; });
    const engine = conditionTestEngine(PHRASE_DEFINITIONS.filter(({ referenceCue }) => referenceCue === 'steering'));
    engine.setDrivingAggression(100, 0);
    engine.receiveMap(steeringMap(), 0, lap);
    const at = (position: number, now: number) => engine.receiveTelemetry(frame(driving(position)), now);
    expect(at(0.26, 1000).state.path).toEqual(['root', 'Steering', 'say phrase']);
    expect(at(0.28, 1100).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
});

it.each([20, 36, 180])('prepares and turns with an overlapping trajectory turn at %s km/h', (speed) => {
    const lap = referenceLap();
    brakingTurn(lap);
    for (const [position, phase] of [[0.256, 'prepare'], [0.26, 'prepare'], [0.283, 'turn']] as const) {
        const sample = { ...driving(position), Physics_speed_kmh: speed };
        expect(steeringGuidance(sample, lap))
            .toMatchObject({ steering: 2500, steeringPhase: phase });
    }
});

it('does not enter Steering for easing deceleration before the trajectory starts turning', () => {
    const { lap, update } = setup((lap) => {
        brakingTurn(lap);
        setTrajectory(lap, (i) => 12 * Math.max(0, Math.min(15, i - 29)));
        lap.y.forEach((row, i) => { row[3] = i >= 29 && i <= 35 ? 0.4 : 0; });
    }, steeringMap());
    update(0.26, 500, { Physics_speed_kmh: lap.y[26][6] + 7.2, Physics_brake: 0.8 });
    const snapshot = update(0.26, 1000, { Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.3 });
    expect(snapshot.state.path).toEqual(['root', 'Brake zone']);
    expect(snapshot.referenceSteeringPhase).toBeUndefined();
    expect(snapshot.events).toEqual([]);
});

it('continues Steering beyond the brake boundary and exits at the trajectory range end despite expired position-model data', () => {
    const { engine, lap, update } = setup((lap) => {
        brakingTurn(lap);
        lap.y[31][5] = lap.y[32][5] = lap.y[33][5] = lap.y[34][5] = 0.8; // Throttle pickup inside the mapped corner (0.25–0.35).
    }, steeringMap());
    update(0.24, 900, { Physics_speed_kmh: lap.y[24][6], Physics_brake: 0.8 });
    update(0.256, 1000, { Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.8 });
    update(0.256, 1001, { Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.8 });
    const stale = engine.receiveTelemetry(frame(beforeDecision({ ...driving(0.260001), Physics_speed_kmh: lap.y[26][6], Physics_brake: 0.8 })), 2100);
    expect(stale.state.path).toContain('Steering');
    expect(stale.events.map(({ sentence }) => sentence)).toEqual(['Off brake.', SLOWLY_RELEASE_BRAKE]);
    const turned = engine.receiveTelemetry(frame({ ...driving(0.283), Physics_speed_kmh: lap.y[28][6] - 1, Physics_brake: 0.8 }), 2150);
    expect(turned.events.map(({ sentence }) => sentence)).toEqual(['Off brake.', SLOWLY_RELEASE_BRAKE]);
    expect(turned.state.path).toContain('Steering');
    expect(engine.receiveTelemetry(frame(driving(0.31)), 2200).state.path).toContain('Steering');
    const exited = engine.receiveTelemetry(frame(driving(0.419)), 2300);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events).toHaveLength(2);
    // Returning to a qualifying turn can re-enter Steering before the previous 2 s cooldown expires.
    expect(update(0.34, 3300).state.path).toEqual(['root', 'Steering']);
});

it.each(['missing speed', 'invalid speed', 'constant deceleration', 'increasing deceleration', 'missing steering'])(
    'prepares for turn-in independently of reference deceleration with %s', (kind) => {
        const lap = referenceLap();
        brakingTurn(lap);
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[26][6] = NaN;
        if (kind === 'missing steering') lap.target_features[3] = 'missing';
        if (kind === 'constant deceleration' || kind === 'increasing deceleration') {
            lap.y[27][6] = Math.sqrt(lap.y[26][6] ** 2 / 3.6 ** 2 - 2 * (kind === 'constant deceleration' ? 3 : 4) * 10) * 3.6;
        }
        const sample = { ...driving(0.26), Physics_speed_kmh: 80 };
        expect(steeringGuidance(sample, lap).steeringPhase).toBe('prepare');
    });

it.each([false, true])('matches the direction-change range across start/finish (duplicated endpoint: %s)', (duplicate) => {
    const lap = trajectoryLap();
    lap.y.forEach((row, i) => {
        row[0] = 10 * Math.cos(i * 2 * Math.PI / 100);
        row[1] = 10 * Math.sin(i * 2 * Math.PI / 100);
        row[3] = i === 0 ? 0.02 : 0.1;
    });
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const bounds = { start: 0.998, end: 0.02 };
    for (const position of [0.998, 0.999, 0, 1, 0.01]) {
        const guidance = steeringGuidance(driving(position), lap, 1, bounds);
        expect(guidance.steering).toBe(2500);
        expect(guidance.steeringRange).toEqual(expect.objectContaining({ turnSign: 1 }));
    }
    // A corner wholly behind the current point must not count, even on a full-lap turn.
    expect(steeringGuidance(driving(0.03), lap, 1, { start: 0.005, end: 0.02 }).steering).toBeUndefined();
});

it.each([
    [0.1999, 'none'], [0.2, 'brake'], [0.200001, 'brake'], [0.299, 'brake'], [0.3, 'none'], [0.5999, 'none'],
    [0.6, 'throttle'], [0.600001, 'throttle'], [0.908, 'throttle'], [0.909999, 'throttle'], [0.91, 'none'], [0.95, 'none'],
] as const)('uses interpolated pedal boundaries at %s for the %s zone', (position, zone) => {
    const lap = referenceLap();
    for (const pedal of [0, 0.8, 1]) {
        const sample = { ...driving(position), Physics_brake: pedal, Physics_gas: pedal };
        expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).pedalZone).toBe(zone);
    }
});

it.each([['brake', 4], ['throttle', 5]] as const)('keeps the reference %s zone across start/finish', (zone, channel) => {
    const lap = referenceLap();
    lap.y.forEach((row) => { row[3] = 0; });
    lap.y.forEach((row, i) => { row[4] = 0; row[5] = 0; row[channel] = i >= 98 || i <= 1 ? 0.8 : 0; row[channel === 4 ? 5 : 4] = i >= 4 && i < 98 ? 0.8 : 0; });
    for (const position of [0.985, 0.999, 0, 0.005, 0.015]) {
        const sample = driving(position);
        expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).pedalZone).toBe(zone);
    }
    const sample = driving(0.030001);
    expect(readReferenceGuidance(sample, detection(lap, sample), lap, 'acc', 1000).pedalZone).toBe(zone === 'brake' ? 'throttle' : 'brake');
});

it('reads zone membership from normalized position in the private aggression-adjusted reference', () => {
    const lap = referenceLap();
    addSpeed(lap);
    const sample = driving(0.21), view = detection(lap, sample);
    view.position!.referenceDelta[1] = 20;
    expect(readReferenceGuidance(sample, view, lap, 'acc', 1000).pedalZone).toBe('brake');
    const cautious = copyPhraseReference(lap, 50)!;
    expect(readReferenceGuidance(sample, view, cautious, 'acc', 1000, lap).pedalZone).toBe('brake');
    for (const [position, zone] of [[0.1899, 'none'], [0.19, 'brake'], [0.1999, 'brake'], [0.201, 'brake'], [0.269, 'brake'], [0.297, 'brake'], [0.3, 'none']] as const) {
        const braking = { ...driving(position), Physics_speed_kmh: 160 };
        expect(readReferenceGuidance(braking, detection(lap, braking), cautious, 'acc', 1000, lap).pedalZone).toBe(zone);
    }
    for (const speed of [36, 84, 160]) {
        const throttle = { ...driving(0.61), Physics_speed_kmh: speed };
        const prediction = detection(lap, throttle);
        expect(readReferenceGuidance(throttle, prediction, lap, 'acc', 1000).pedalZone).toBe('throttle');
        expect(readReferenceGuidance(throttle, prediction, cautious, 'acc', 1000, lap).pedalZone).toBe('throttle');
    }
});

// Pedal transitions use the unchanged 10 m sample spacing of this reference.
it.each([[0, 0.2, 0.3], [25, 0.19, 0.3], [50, 0.19, 0.3], [100, 0.19, 0.3]])(
    'keeps the brake endpoint fixed at %s%% caution independently of live speed and pedals', (caution, start, end) => {
        const lap = referenceLap();
        addSpeed(lap);
        const cautious = copyPhraseReference(lap, caution)!;
        for (const speed of [36, 120, 160]) {
            for (const pedal of [0, 0.8, undefined]) {
                for (const [position, zone] of [[start - 0.00001, 'none'], [start + 0.00001, 'brake'],
                    [(start + end) / 2, 'brake'], [end - 0.00001, 'brake'], [end, 'none'], [end + 0.00001, 'none'], [0.6, 'throttle']] as const) {
                    const sample = { ...driving(position), Physics_speed_kmh: speed, Physics_brake: pedal, Physics_gas: pedal };
                    const guidance = readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap);
                    expect(guidance).toMatchObject(zone === 'brake' ? { pedalZone: zone, braking: true } : { pedalZone: zone });
                }
            }
        }
        // Position-model forward error must not shift the local brake boundary.
        const sample = { ...driving(0.202), Physics_speed_kmh: 120 }, view = detection(lap, sample);
        view.position!.referenceDelta[1] = 5;
        expect(readReferenceGuidance(sample, view, cautious, 'acc', 1000, lap).pedalZone).toBe('brake');
    });

it.each([0, 0.099, 0.1, 0.101])('uses only local brake pressure and response at pressure %s', (pressure) => {
    const lap = referenceLap();
    addSpeed(lap);
    const local = copyPhraseReference(lap, 50)!;
    local.y.forEach((row) => { row[4] = pressure; row[5] = 0; });
    const sample = { ...driving(0.19), Physics_speed_kmh: 160 };
    const guidance = readReferenceGuidance(sample, detection(lap, sample), local, 'acc', 1000, lap);
    // A valid original response cannot replace an absent response in the active lap.
    expect(guidance.brake).toBe(pressure >= 0.1 ? 2500 : undefined);
    expect(guidance.pedalZone).toBe(pressure > 0 ? 'brake' : 'none');
    expect(lap.y[19][4]).toBe(0);
});

it.each(['missing speed', 'invalid speed', 'missing brake', 'invalid brake', 'no brake response'])(
    'does not invent an earlier brake zone with %s', (kind) => {
        const lap = referenceLap();
        addSpeed(lap);
        if (kind === 'missing speed') lap.target_features[6] = 'missing';
        if (kind === 'invalid speed') lap.y[0][6] = NaN;
        if (kind === 'missing brake') lap.target_features[4] = 'missing';
        if (kind === 'invalid brake') lap.y[0][4] = NaN;
        if (kind === 'no brake response') lap.y.forEach((row) => { row[4] = 0; });
        const cautious = copyPhraseReference(lap, 50)!;
        const sample = { ...driving(0.19), Physics_speed_kmh: 160 };
        const guidance = readReferenceGuidance(sample, detection(lap, sample), cautious, 'acc', 1000, lap);
        expect(guidance.pedalZone).toBe(kind.includes('brake') && kind !== 'no brake response' ? undefined : 'none');
        expect(guidance.brake).toBeUndefined();
    });

it('stays quiet in the throttle zone when overspeed calls for releasing throttle', () => {
    const { engine, update } = setup(addSpeed);
    expect(update(0.655, 1000, { Physics_speed_kmh: 102, Physics_gas: 0.8 }).state.path)
        .toEqual(['root', 'Throttle zone']);
    const correction = update(0.655, 2200, { Physics_speed_kmh: 120, Physics_gas: 0.8 });
    expect(correction.events).toEqual([]);
    expect(correction.state.path).toEqual(['root', 'Throttle zone']);
    expect(correction.closures.find(({ id }) => id === 'reference-brake')!.conditions.some((condition) => !condition.conditionFit)).toBe(true);
    expect(engine.evaluate(2201).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.655, 3400, { Physics_speed_kmh: 120, Physics_gas: 0 }).state.path)
        .toEqual(['root', 'Throttle zone']);
    expect(update(0.95, 3500, { Physics_speed_kmh: 120 }).state.path).toEqual(['root']);
});

it('ends throttle at reference zero before mapped corner exit and keeps Brake zone past throttle onset', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.95 }];
    engine.receiveMap(map, 0, lap);
    expect(update(0.908, 1000).state.path).toEqual(['root', 'Throttle zone', 'Throttle', 'say phrase']);
    const afterReference = update(0.909, 1100);
    expect(afterReference.state.path).toEqual(['root', 'Throttle zone']);
    expect(afterReference.events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
    const throttle = afterReference.root.children
        .find(({ id }) => id === 'reference-throttle-zone')!;
    expect(throttle.children[0].conditions).toEqual([
        expect.objectContaining({ description: 'The driver has left the interpolated reference throttle zone', conditionFit: false }),
    ]);
    expect(update(0.91, 1125).state.path).toEqual(['root']);
    expect(update(0.951, 1150).state.path).toEqual(['root']);
    const braking = update(0.298, 1200);
    expect(braking.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(braking.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Squeeze gas.', timestamp: 1000 }, { sentence: 'Brake.', timestamp: 1200 },
    ]);
    expect(update(0.299, 1300).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.2999, 1400).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.61, 1500).state.path).toEqual(['root', 'Brake zone']);
});

it.each(['straight', 'long straight', 'untagged'])('keeps throttle after the mapped corner before %s until immediate brake onset', (label) => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [
        { id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.7 },
        ...(label === 'untagged' ? [] : [{ id: 'after', tags: [label], start_position: 0.7, end_position: 0.9 }]),
    ];
    engine.receiveMap(map, 0, lap);
    expect(update(0.699, 1000).state.path).toEqual(['root', 'Throttle zone', 'Throttle', 'say phrase']);
    const exited = update(0.701, 1100);
    expect(exited.state.path).toEqual(['root', 'Throttle zone']);
    expect(exited.events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
    expect(exited.closures.find(({ id }) => id === 'reference-throttle')!.conditions[1].conditionFit).toBe(true);
    expect(update(0.8, 4000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(engine.evaluate(4001).events).toHaveLength(1);
    expect(update(0.200001, 4100).state.path).toEqual(['root', 'Brake zone']);
});

it('exits throttle at brake onset with fresh telemetry even when reference location expires', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.7 }];
    engine.receiveMap(map, 0, lap);
    update(0.699, 1000, { Physics_gas: 0.8 });
    expect(engine.evaluate(1501).state.path).toEqual(['root', 'Throttle zone']);
    expect(engine.receiveTelemetry(frame(driving(0.195001)), 1600).state.path).toEqual(['root', 'Brake zone']);
});

it('keeps the brake zone before mapped corner entry when reference location expires', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.7 }];
    engine.receiveMap(map, 0, lap);
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    expect(engine.evaluate(1501).state.path).toEqual(['root', 'Brake zone']);
});

it.each([0, 0.8])('enters Brake zone before the active throttle corner ends with live brake %s', (brake) => {
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.15, end_position: 0.35 }];
    const { update } = setup((lap) => {
        lap.y.forEach((row, i) => { row[5] = i >= 15 && i <= 19 ? 0.8 : 0; });
    }, map);
    expect(update(0.18, 1000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    const braking = update(0.25, 1100, { Physics_brake: brake });
    expect(braking.state.path).toEqual(brake === 0
        ? ['root', 'Brake zone', 'Brake', 'say phrase'] : ['root', 'Brake zone']);
    expect(braking.events.map(({ sentence }) => sentence)).toEqual(brake === 0 ? ['Brake.'] : []);
});

it('keeps a throttle zone past a mapped corner exit across start/finish', () => {
    const { engine, lap, update } = setup((reference) => {
        reference.y.forEach((row) => { row[3] = 0; row[4] = 0; row[5] = 0.8; });
    });
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['fast corner'], start_position: 0.98, end_position: 0.01 }];
    engine.receiveMap(map, 0, lap);
    [0.985, 0.999, 0.001, 0.009].forEach((position, index) => {
        expect(update(position, 1000 + index * 100, { Physics_gas: 0.8 }).state.path)
            .toEqual(['root', 'Throttle zone']);
    });
    expect(update(0.011, 1400, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
});

it('keeps brake ownership beyond the mapped corner and throttle pickup without a speed target', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [
        { id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.5 },
        { id: 'after', tags: ['straight'], start_position: 0.5, end_position: 0.9 },
    ];
    engine.receiveMap(map, 0, lap);
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toContain('Brake zone');
    expect(update(0.4, 2000).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.7, 3000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    const sample = driving(0.701), view = detection(lap, sample, 4000);
    view.position!.referenceDelta[0] = -4;
    engine.receiveVision(view, 4000);
    const movement = engine.receiveTelemetry(frame(sample), 4000);
    expect(movement.state.path).toEqual(['root', 'Brake zone']);
    expect(movement.events).toEqual([]);
});

it.each(['no tags', 'wrong track', 'wrong game'])('ends throttle at reference zero when the map has %s', (kind) => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.7 }];
    if (kind === 'no tags') map.centerline_segments = [];
    if (kind === 'wrong track') map.source_track_key = 'other';
    if (kind === 'wrong game') map.game = 'iracing';
    engine.receiveMap(map, 0, lap);
    expect(update(0.8, 1000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.95, 1100, { Physics_gas: 0.8 }).state.path).toEqual(['root']);
    expect(update(0.25, 1200, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
});

it('keeps zone ownership through missing captures and switches only after fresh boundary evidence', () => {
    const { engine, lap, update } = setup();
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.6, end_position: 0.9 }];
    engine.receiveMap(map, 0, lap);
    update(0.7, 1000);
    const stale = engine.evaluate(1501);
    expect(stale.state.path).toEqual(['root', 'Throttle zone']);
    expect(stale.events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
    expect(update(0.7, 2000, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.25, 2100, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
});

it('keeps a throttle closure across start/finish until reference release to zero', () => {
    const { engine, lap, update } = setup((lap) => {
        lap.y.forEach((row, i) => { row[4] = i >= 5 && i < 98 ? 0.8 : 0; row[5] = i >= 98 || i <= 1 ? 0.8 : 0; });
    });
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.98, end_position: 0.03 }];
    engine.receiveMap(map, 0, lap);
    [0.985, 0.999, 0.001, 0.015].forEach((position, index) => {
        expect(update(position, 1000 + index * 100, { Physics_gas: position === 0.015 ? 0.4 : 0.8 }).state.path.slice(0, 2))
            .toEqual(['root', 'Throttle zone']);
    });
    expect(update(0.019, 1325, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.02025, 1450, { Physics_gas: 0.8 }).state.path).toEqual(['root']);
    expect(update(0.031, 1500, { Physics_gas: 0.8 }).state.path).toEqual(['root']);
    expect(update(0.040001, 1600).state.path).toEqual(['root', 'Brake zone']);
});

it('gives overlapping interpolated pedal zones to the brake closure', () => {
    const { update } = setup((lap) => lap.y.forEach((row) => { row[5] = 0.8; }));
    const result = update(0.25, 1000, { Physics_brake: 0.8 });
    expect(result.state.path).toEqual(['root', 'Brake zone']);
    expect(result.events).toEqual([]);
});

it('transitions through brake application and pressure cues as speed and deceleration change', () => {
    const { engine, update } = setup((lap) => { addSpeed(lap); lap.y[29][6] = 80; });
    const expected: string[] = [];
    ([[0.25, 100, 0, undefined, 0, 'Brake.'],
        [0.25, 106, 0.8, 1, 0.8, 'More brake.'],
        [0.25, 100, 0.4, localBrakeDeceleration, 0.8, 'Hold brake.']] as const)
        .forEach(([position, speed, pedal, deceleration, previousPedal, sentence], index) => {
            const now = 1000 + index * 2500;
            if (sentence === 'More brake.') {
                // Establish a deficit on current telemetry, then keep it true for the full hold.
                for (const elapsed of [600, 500, 250, 1]) update(position, now - elapsed, {
                    Physics_speed_kmh: speed + elapsed * 3.6 / 1000, Physics_brake: pedal,
                });
            } else if (deceleration !== undefined) {
                const previousSpeed = speed + deceleration * 3.6 * 0.5;
                // Freeze the setup position to clear prediction history while the measured-rate
                // interval continues. The final frame supplies the full 500 ms measurement.
                for (const at of [now - 500, now - 1]) update(position - (previousSpeed - 100) / 400, at, {
                    Physics_speed_kmh: previousSpeed, Physics_brake: previousPedal,
                });
            }
            const snapshot = update(position, now, { Physics_speed_kmh: speed, Physics_brake: pedal });
            expected.push(sentence);
            expect(snapshot.events.map((event) => event.sentence)).toEqual(expected);
            expect(snapshot.events[index].expiresAt).toBe(now + 1500);
            expect(snapshot.state.path).toContain('Brake zone');
            expect(engine.evaluate(now + 1).state.path).toEqual(['root', 'Brake zone']);
        });
});

it.each([500, 640])(
    'confirms More brake for %s ms when frozen progress prevents prediction', (interval) => {
        const { engine, update } = setup(addSpeed);
        const speed = 102, deceleration = 1.5;
        // Offset the decision point so beforeDecision keeps measured progress frozen.
        const at = (elapsed: number) => update(0.255 - deceleration * elapsed / 2000000, 1000 + elapsed, {
            Physics_speed_kmh: speed - deceleration * 3.6 * elapsed / 1000, Physics_brake: 0.8,
        });
        for (const elapsed of [0, 16, 100, 250, 499]) expect(at(elapsed).events).toEqual([]);
        // Timer evaluations cannot create a measurement without a new telemetry frame.
        expect(engine.evaluate(1000 + interval).events).toEqual([]);
        // The first measured deficit is at 16 ms; the hold starts there, not at the first sample.
        if (interval === 500) expect(at(interval).events).toEqual([]);
        const result = at(16 + interval);
        expect(result.events).toEqual([
            expect.objectContaining({ sentence: 'More brake.', timestamp: 1016 + interval }),
        ]);
        expect(result.state.path).toEqual(['root', 'Brake zone', 'More brake', 'say phrase']);
    });

it('updates More brake from each measured rate without speed-based release cues', () => {
    const { update } = setup(addSpeed);
    const at = (now: number, speed: number) => update(0.250001, now, { Physics_speed_kmh: speed, Physics_brake: 0.8 });
    expect(at(1000, 104.95).events).toEqual([]);
    expect(at(1499, 100).events).toEqual([]);
    const matching = at(1500, 100);
    expect(matching.events).toEqual([expect.objectContaining({ sentence: 'Hold brake.', timestamp: 1500 })]);
    expect(at(1999, 100).events.map(({ sentence }) => sentence)).toEqual(['Hold brake.', 'More brake.']);
    const result = at(2000, 100);
    expect(result.events.map(({ sentence }) => sentence)).toEqual(['Hold brake.', 'More brake.']);
    // Current rate checks respond on every frame without an underspeed release cue.
    expect(at(2016, 99).closures.find(({ id }) => id === 'reference-moreBrake')!.conditions[1].conditionFit).toBe(false);
    const tooSlow = at(2499, 94.6);
    expect(tooSlow.closures.find(({ id }) => id === 'reference-moreBrake')!.conditions[1].conditionFit).toBe(true);
    expect(tooSlow.events.map(({ sentence }) => sentence)).toEqual(['Hold brake.', 'More brake.']);
    expect(at(2500, 94.6).closures.find(({ id }) => id === 'reference-moreBrake')!.conditions[1].conditionFit).toBe(true);
});

it.each([500, 640])('falls back to measured pedal release and deceleration over %s ms with discontinuous progress', (interval) => {
    for (const [before, after, cue] of [[0.8, 0.4, undefined],
        [0.8, 0.8, 'More brake.'], [0.4, 0.8, 'More brake.'], [0.801, 0.8, 'More brake.']] as const) {
        const { engine, update } = setup(addSpeed);
        expect(update(0.25, 1000, { Physics_speed_kmh: 100 + 3.6 * interval / 1000, Physics_brake: before }).events).toEqual([]);
        expect(engine.evaluate(1000 + interval).events).toEqual([]);
        let measured = update(0.25, 1000 + interval, { Physics_speed_kmh: 100, Physics_brake: after });
        if (cue === 'More brake.') {
            expect(measured.events).toEqual([]);
            measured = update(0.25, 1500 + interval, { Physics_speed_kmh: 98.2, Physics_brake: after });
        }
        expect(measured.events.map(({ sentence }) => sentence)).toEqual(cue ? [cue] : []);
    }
});

it.each([undefined, NaN, -0.1, 1.1])('discards release evidence across an invalid pedal sample: %s', (brake) => {
    const { update } = setup(addSpeed);
    update(0.25, 1000, { Physics_speed_kmh: 102, Physics_brake: 0.8 });
    update(0.25, 1250, { Physics_speed_kmh: 101, Physics_brake: brake });
    const recovered = update(0.25, 1500, { Physics_speed_kmh: 100, Physics_brake: 0.4 });
    expect(recovered.events).toEqual([]);
    expect(update(0.25, 1999, { Physics_speed_kmh: 99, Physics_brake: 0.4 }).events).toEqual([]);
    expect(update(0.25, 2000, { Physics_speed_kmh: 99, Physics_brake: 0.4 }).events.map(({ sentence }) => sentence)).toEqual(['More brake.']);
});

it.each(['gap', 'pause', 'stream reset', 'missing speed', 'invalid speed', 'repeated time', 'backward time', 'session change', 'stream change'])(
    'requires a new 500 ms More brake confirmation after %s', (kind) => {
        const { engine, lap, update } = setup(addSpeed);
        const patch = { Physics_speed_kmh: 102, Physics_brake: 0.8 };
        update(0.255, 1000, patch);
        const active = update(0.255, 1500, { ...patch, Physics_speed_kmh: 101.64 });
        expect(active.closures.find(({ id }) => id === 'reference-moreBrake')!.conditions[1].conditionFit).toBe(true);
        if (kind === 'pause') engine.receiveTelemetry({ ...frame({ ...driving(0.255), ...patch }), telemetryStatus: 1 }, 1600);
        if (kind === 'stream reset') engine.receiveTelemetry({ type: 'stream-reset', snapshot: createLiveTelemetryStore().getSnapshot() }, 1600);
        if (kind === 'missing speed' || kind === 'invalid speed') {
            update(0.255, 1600, { ...patch, Physics_speed_kmh: kind === 'missing speed' ? undefined : NaN });
        }
        const now = kind === 'gap' ? 3101 : kind === 'repeated time' ? 1500 : kind === 'backward time' ? 1499 : 1700;
        const sample = beforeDecision({ ...driving(0.255), ...patch, Physics_speed_kmh: 101.28 });
        const event = frame(sample), view = detection(lap, sample, now);
        if (kind === 'session change') event.sessionGeneration = view.telemetry!.sessionGeneration = 1;
        if (kind === 'stream change') event.streamGeneration = view.telemetry!.streamGeneration = 1;
        engine.receiveVision(view, now);
        const result = engine.receiveTelemetry(event, now);
        const conditions = result.closures.find(({ id }) => id === 'reference-moreBrake')!.conditions;
        expect(conditions[0].conditionFit).toBe(true);
        expect(conditions[1].conditionFit).toBe(false);
        expect(result.events).toEqual(kind === 'stream reset' ? [] : active.events);
        const before = engine.receiveTelemetry(event, now + 499);
        expect(before.closures.find(({ id }) => id === 'reference-moreBrake')!.conditions[1].conditionFit).toBe(true);
        expect(before.events).toEqual(result.events);
        // A large speed decrease on the next millisecond breaks the confirmation immediately.
        const spike = engine.receiveTelemetry({ ...event, sample: { ...sample, Physics_speed_kmh: 99.48 } }, now + 500);
        expect(spike.closures.find(({ id }) => id === 'reference-moreBrake')!.conditions[1].conditionFit).toBe(false);
        const recovered = engine.receiveTelemetry({ ...event, sample: { ...sample, Physics_speed_kmh: 99.48 } }, now + 600);
        expect(recovered.closures.find(({ id }) => id === 'reference-moreBrake')!.conditions[1].conditionFit).toBe(true);
    });

it('withholds pedal corrections outside reference pedal zones', () => {
    const { update } = setup((lap) => { addSpeed(lap); lap.y.forEach((row) => { row[4] = row[5] = 0; }); });
    const result = update(0.45, 1000, { Physics_speed_kmh: 80, Physics_brake: 0.4 });
    expect(result.events).toEqual([]);
    expect(result.state.path).toEqual(['root']);
});

it('enters the throttle zone to apply and add throttle without a preceding braking zone', () => {
    const { engine, update } = setup(addSpeed);
    const expected: string[] = [];
    ([[0.655, 102, 0, 'Squeeze gas.'], [0.655, 94, 0.8, 'More gas.']] as const)
        .forEach(([position, speed, pedal, sentence], index) => {
            const now = 1000 + index * 1200;
            expected.push(sentence);
            const snapshot = update(position, now, { Physics_speed_kmh: speed, Physics_gas: pedal });
            expect(snapshot.events.map((event) => event.sentence)).toEqual(expected);
            expect(snapshot.state.path).toEqual(['root', 'Throttle zone', sentence === 'Squeeze gas.' ? 'Throttle' : 'More throttle', 'say phrase']);
            expect(engine.evaluate(now + 1).state.path).toEqual(['root', 'Throttle zone']);
        });
});

it('gives only brake corrections in the brake zone, even when the driver applies throttle', () => {
    const { engine, update } = setup(addSpeed);
    expect(update(0.255, 1000, { Physics_speed_kmh: 98, Physics_gas: 0.8 }).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    engine.evaluate(1001);
    update(0.255, 2200, { Physics_speed_kmh: 98, Physics_gas: 0.8, Physics_brake: 0.8 });
    const snapshot = update(0.255, 2700, { Physics_speed_kmh: 98, Physics_gas: 0.8, Physics_brake: 0.8 });
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Brake.', 'More brake.']);
    expect(snapshot.state.path).toEqual(['root', 'Brake zone', 'More brake', 'say phrase']);
});

it.each([[false, 1], [true, 1], [false, 0], [true, 0]] as const)('exits at reference throttle endpoint (cooldown: %s, throttle: %s)', (cooldown, throttle) => {
    const { engine, update } = setup((lap) => {
        lap.y.forEach((row, i) => { if (i >= 70 && i <= 90) row[5] = throttle; });
    });
    const speaking = update(0.69, 1000);
    expect(speaking.state.path).toEqual(['root', 'Throttle zone', 'Throttle', 'say phrase']);
    if (cooldown) expect(engine.evaluate(1001).state.path).toEqual(['root', 'Throttle zone']);
    // Fresh normalized telemetry confirms the exit even after the position capture expires.
    const exited = engine.receiveTelemetry(frame(driving(0.7)), 1600);
    expect(exited.state.path).toEqual(['root']);
    expect(exited.events).toEqual(speaking.events);
    const later = update(0.8, 4000);
    expect(later.state.path).toEqual(['root']);
    expect(later.events).toEqual(speaking.events);
});

it('stays in the throttle zone after speech and while the driver matches the reference', () => {
    const { engine, update } = setup();
    update(0.7, 1000);
    expect(engine.evaluate(1001).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.7, 1250).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.7, 2250).state.path).toEqual(['root', 'Throttle zone']);
    expect(update(0.7, 2500, { Physics_gas: 0.8 }).state.path).toEqual(['root', 'Throttle zone']);
    expect(engine.evaluate(2501).events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
});

it('emits short, expiring cues without opponents, calibration or circuit tags and avoids repeats', () => {
    const { engine, update } = setup();
    const first = update(0.25, 1000);
    expect(first.visionReady).toBe(false);
    expect(first.events).toEqual([{ id: 1, instance: expect.any(PhraseInstance), ruleId: 'reference-brake', sentence: 'Brake.', timestamp: 1000, expiresAt: 2500 }]);
    expect(update(0.251, 1100).events).toHaveLength(1);
    expect(update(0.25, 1500).events).toHaveLength(1);
    expect(update(0.25, 2000).events.map((event) => event.sentence)).toEqual(['Brake.']);
    expect(update(0.25, 3000).events).toHaveLength(1);
    expect(engine.evaluate(3501).closures.find((closure) => closure.id === 'reference-brake'))
        .toMatchObject({ status: 'Waiting', missing: [] });
    expect(engine.evaluate(4501).closures.find((closure) => closure.id === 'reference-brake'))
        .toMatchObject({ status: 'Missing input' });
    update(0.61, 4600, { Physics_gas: 0.8 });
    expect(update(0.25, 5000).events).toHaveLength(1);
});

it('does not delay a different brake prompt after Brake', () => {
    const { lap, update } = setup(brakingTurn, steeringMap());
    expect(update(0.23, 1000, { Physics_speed_kmh: 100 }).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const released = update(0.24, 1100, { Physics_speed_kmh: lap.y[25][6], Physics_brake: 0.8 });
    expect(released.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Brake.', timestamp: 1000 }, { sentence: 'Off brake.', timestamp: 1100 },
    ]);
});

it('speaks Brake only once per brake zone despite pedal changes and expired inputs', () => {
    const { engine, lap, update } = setup();
    const first = update(0.25, 1000).events[0];
    expect(first.sentence).toBe('Brake.');
    update(0.25, 1500, { Physics_brake: 0.8 });
    update(0.25, 2000, { Physics_brake: 0.8 });
    expect(update(0.25, 3100).events).toEqual([first]);
    expect(engine.evaluate(5000).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.25, 6000).events).toEqual([first]);
    engine.receiveMap(circuitMap(), 6100, lap);
    expect(update(0.25, 6200).events).toEqual([first]);
});

it('rearms Brake after the speed exit even with a short gap before re-entry', () => {
    const { lap, update } = setup(brakingTurn, steeringMap());
    update(0.23, 1000, { Physics_speed_kmh: 100 });
    expect(update(0.23, 3000, { Physics_speed_kmh: 100 }).events).toHaveLength(1);
    expect(update(0.24, 3100, { Physics_speed_kmh: lap.y[25][6] - 1 }).state.path).toEqual(['root']);
    const reentered = update(0.23, 3200, { Physics_speed_kmh: 100 });
    expect(reentered.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(reentered.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Brake.', timestamp: 1000 },
        { sentence: 'Brake.', timestamp: 3200 },
    ]);
});

it.each([0.19, 0.39, 0.59])('stays at root outside reference pedal zones at %s', (position) => {
    const { update } = setup((lap) => lap.y.forEach((row) => { row[4] = row[5] = 0; }));
    const snapshot = update(position, 1000);
    expect(snapshot.events).toEqual([]);
    expect(snapshot.state.path).toEqual(['root']);
    for (const zone of ['brake', 'throttle']) {
        expect(snapshot.root.children.find(({ id }) => id === `reference-${zone}-zone`)!.conditions).toEqual([
            expect.objectContaining({ description: `Inside the interpolated reference ${zone} zone`, conditionFit: false, inputMissing: false }),
            ...(zone === 'brake' ? [expect.objectContaining({
                description: 'The direction-start speed condition for brake-zone exit is not met', conditionFit: true,
            })] : []),
        ]);
    }
});

it.each([0, 0.4])(
    'keeps Brake zone while coasting and after throttle onset with live pedal %s', (pedal) => {
        const { engine, lap, update } = setup();
        const map = circuitMap();
        map.centerline_segments = [
            { id: 'straight', tags: ['straight'], start_position: 0, end_position: 0.99 },
            { id: 'turn', tags: ['corner'], start_position: 0.3, end_position: 0.8 },
        ];
        engine.receiveMap(map, 0, lap);
        const entered = update(0.25, 1000, { Physics_brake: 0.8 });
        expect(entered.events).toEqual([]);
        expect(entered.state.path).toEqual(['root', 'Brake zone']);
        expect(entered.root.children.find(({ id }) => id === 'reference-brake-zone')!.conditions[0].conditionFit).toBe(true);
        expect(update(0.305, 2000).state.path).toEqual(['root', 'Brake zone']);
        expect(update(0.39, 3000).state.path).toEqual(['root', 'Brake zone']);
        expect(engine.evaluate(3001).state.path).toEqual(['root', 'Brake zone']);
        const sample = driving(0.45), view = detection(lap, sample, 4000);
        view.position!.referenceDelta[0] = -4;
        engine.receiveVision(view, 4000);
        expect(engine.receiveTelemetry(frame(sample), 4000).state.path).toEqual(['root', 'Brake zone']);
        const acceleration = update(0.7, 5000, { Physics_gas: pedal });
        expect(acceleration.state.path).toEqual(['root', 'Brake zone']);
        expect(acceleration.events).toEqual([]);
        expect(engine.evaluate(5001).state.path).toEqual(['root', 'Brake zone']);
        expect(update(0.7, 6000, { Physics_gas: pedal }).events).toEqual([]);
        expect(update(0.25, 7000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    });

it('waits until braking is needed inside the brake zone and returns to the same scope', () => {
    const { engine, update } = setup();
    expect(update(0.25, 1000, { Physics_brake: 0.8 }).state.path).toEqual(['root', 'Brake zone']);
    expect(update(0.19, 2000).events).toEqual([]);
    const nextBraking = update(0.25, 3000);
    expect(nextBraking.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(nextBraking.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(engine.evaluate(3001).state.path).toEqual(['root', 'Brake zone']);
});

it('exits the previous throttle phrase before entering the next pedal correction', () => {
    const { update } = setup();
    expect(update(0.7, 2000).state.path).toEqual(['root', 'Throttle zone', 'Throttle', 'say phrase']);
    const next = update(0.7, 3000, { Physics_gas: 0.4 });
    expect(next.state.path).toEqual(['root', 'Throttle zone', 'More throttle', 'say phrase']);
    expect(next.events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.', 'More gas.']);
});

it('withholds reference guidance while paused and resumes from telemetry after reference changes', () => {
    const { engine, lap, update } = setup();
    update(0.19, 1000);
    engine.receiveTelemetry({ type: 'stream-reset', snapshot: createLiveTelemetryStore().getSnapshot() }, 1100);
    expect(engine.receiveTelemetry(frame(driving(0.19)), 1200).events).toEqual([]);
    const sample = driving(0.19);
    const view = detection(lap, sample, 1300);
    view.telemetry!.streamGeneration = 9;
    engine.receiveVision(view, 1300);
    expect(engine.receiveTelemetry(frame(sample), 1300).events).toEqual([]);
    update(0.19, 2000);
    expect(engine.receiveTelemetry({ ...frame(sample), telemetryStatus: 1 }, 2100).closures.find((closure) => closure.id === 'reference-brake'))
        .toMatchObject({ status: 'Not evaluated' });
    engine.receiveMap(null, 2200, referenceLap());
    expect(engine.receiveTelemetry(frame(driving(0.25)), 2200).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
});

it.each([0, 0.8])('keeps a waiting overtaking closure at reference braking with driver brake %s', (brake) => {
    const { engine, lap } = setup();
    const map = circuitMap();
    engine.receiveMap(map, 0, null);
    const sample = { ...driving(0.05), Graphics_player_car_id: 1, Graphics_normalized_positions: { 1: 0.05, 2: 0.06 } };
    engine.receiveVision(trafficVision(0), 0);
    const waiting = engine.receiveTelemetry(frame(sample), 0);
    expect(waiting.state.path).toContain('One Tight Slow Corner');
    engine.receiveMap(map, 999, lap);
    const braking = { ...driving(0.25), Physics_brake: brake };
    engine.receiveVision(detection(lap, braking, 1000), 1000);
    const result = engine.receiveTelemetry(frame(braking), 1000);
    expect(result.state.path).toContain('One Tight Slow Corner');
    expect(result.state.path).not.toContain('Brake zone');
    expect(result.events).toEqual([]);
});

function movementSetup(map: CircuitMapDto | null = circuitMap(), configure?: (lap: TopLapReferenceEntry) => void) {
    const engine = conditionTestEngine(), lap = referenceLap();
    engine.setDrivingAggression(100, 0);
    // Steering and throttle channels are optional; reference braking must be known.
    lap.target_features[3] = 'missing steering';
    lap.target_features[5] = 'missing throttle';
    configure?.(lap);
    engine.receiveMap(map, 0, lap);
    const update = (lateral: number, now: number, position = 0.45, patch: StandardTelemetrySample = {}) => {
        const sample = beforeDecision({ ...driving(position), Physics_steer_angle: undefined, ...patch });
        const view = detection(lap, sample, now);
        view.position!.referenceDelta[0] = lateral;
        engine.receiveVision(view, now);
        return engine.receiveTelemetry(frame(sample), now);
    };
    return { engine, lap, update };
}

it.each(['straight', 'long straight'])('allows movements only inside the Live Map %s bounds', (label) => {
    const map = circuitMap();
    map.centerline_segments = [{ id: 'straight', tags: [label], start_position: 0.4, end_position: 0.8 }];
    expect(movementSetup(map).update(-4, 1000).events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
    expect(movementSetup(map).update(-4, 1000, 0.35).events).toEqual([]);
});

it.each(['no map', 'no tags', 'wrong track', 'wrong game', 'corner', 'missing brake', 'invalid brake', 'braking'])(
    'withholds movement with %s even when the position model requests it', (kind) => {
        const map = circuitMap();
        if (kind === 'no tags') map.centerline_segments = [];
        if (kind === 'wrong track') map.source_track_key = 'other';
        if (kind === 'wrong game') map.game = 'iracing';
        if (kind === 'corner') map.centerline_tags!.push({ id: 'overlap', label: 'corner', start_position: 0.4, end_position: 0.5 });
        const { update } = movementSetup(kind === 'no map' ? null : map, (lap) => {
            if (kind === 'missing brake') lap.target_features[4] = 'missing brake';
            if (kind === 'invalid brake') lap.y[0][4] = NaN;
            if (kind === 'braking') lap.y.forEach((row) => { row[4] = 0.8; });
        });
        const snapshot = update(-4, 1000, 0.45, { Physics_brake: kind === 'braking' ? 0.8 : 0 });
        expect(snapshot.events).toEqual([]);
        expect(snapshot.state.path).not.toContain('Turn to reference line');
    });

function brakingMovementReference(lap: TopLapReferenceEntry, throttleOnset = 0.448) {
    lap.target_features[5] = 'expert_optimal_throttle';
    lap.y.forEach((row, index) => {
        row[4] = index >= 46 && index < 50 ? 0.8 : 0;
        row[5] = index / 100 > throttleOnset && index <= 45 ? 0.8 : 0;
    });
    const index = lap.x.findIndex((position) => position > throttleOnset);
    lap.x.splice(index, 0, throttleOnset);
    lap.y.splice(index, 0, [250, throttleOnset * 1000 - 250, 0, 0, 0, 0]);
}

it('exits movement at reference brake onset and keeps movement waiting for Brake zone to exit', () => {
    const { engine, update } = movementSetup(circuitMap(), (lap) => {
        brakingMovementReference(lap);
    });
    expect(update(-4, 1000, 0.449).state.path).toContain('Turn to reference line');
    // Braking is imminent, but the reference brake zone has not started yet.
    for (let now = 1100; now <= 1400; now += 100) {
        expect(update(-4, now, 0.4495).state.path).toContain('Turn to reference line');
    }
    expect(engine.evaluate(1400).state.path).toEqual(['root', 'Turn to reference line']);
    const braking = update(-4, 1500, 0.450501, { Physics_brake: 0 });
    expect(braking.state.path).toEqual(['root', 'Brake zone']);
    expect(braking.events).toHaveLength(1);
    expect(braking.closures.find(({ id }) => id === 'reference-move-left-small')!.conditions.every((condition) => condition.conditionFit)).toBe(false);
    for (let now = 1750; now <= 3500; now += 250) update(-0.1, now, 0.47, { Physics_brake: 0.8 });
    expect(engine.evaluate(3500).events).toHaveLength(1);
    const outsideReferenceZone = update(-0.1, 3750, 0.449);
    expect(outsideReferenceZone.state.path).toEqual(['root', 'Brake zone']);
    expect(outsideReferenceZone.events).toHaveLength(1);
    const afterBraking = update(4, 4000, 0.449);
    expect(afterBraking.state.path).toEqual(['root', 'Brake zone']);
    expect(afterBraking.events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
});

it('exits a selected movement speech action at brake onset inside its speech cooldown', () => {
    const { update } = movementSetup(circuitMap(), (lap) => {
        brakingMovementReference(lap);
    });
    expect(update(-4, 1000, 0.449).state.path).toEqual([
        'root', 'Turn to reference line', 'Move to the left', 'Small movement', 'say phrase',
    ]);
    const braking = update(-4, 1100, 0.450001, { Physics_brake: 0 });
    expect(braking.state.path).toEqual(['root', 'Brake zone']);
    expect(braking.events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
});

it.each(['left straight', 'map removed', 'vision expired'])('exits a nested movement closure and clears feedback when %s', (kind) => {
    const { engine, lap, update } = movementSetup();
    expect(update(-4, 1000).state.path).toContain('Small movement');
    const exited = kind === 'left straight' ? update(-4, 1250, 0.85)
        : kind === 'map removed' ? engine.receiveMap(null, 1250, lap) : engine.evaluate(1501);
    expect(exited.state.path).toEqual(['root']);
    engine.receiveMap(circuitMap(), 1750, lap);
    expect(update(-0.1, 2000).events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
});

it.each([-8, -4, 4, 8])('allows the same movement after leaving Turn to reference line, at offset %s', (offset) => {
    const { update } = movementSetup();
    const first = update(offset, 1000);
    expect(first.events).toHaveLength(1);
    const ruleId = first.events[0].ruleId;
    // Returning from a movement child must keep the spoken flag for this visit.
    for (let now = 1250; now <= 3500; now += 250) {
        expect(update(offset, now).events).toHaveLength(1);
    }
    expect(update(offset, 3750, 0.85).state.path).toEqual(['root']);
    const reentered = update(offset, 4000);
    expect(reentered.state.path).toContain('Turn to reference line');
    expect(reentered.events.map(({ ruleId, timestamp }) => ({ ruleId, timestamp }))).toEqual([
        { ruleId, timestamp: 1000 }, { ruleId, timestamp: 4000 },
    ]);
});

it.each(['braking', 'invalid position', 'capture expiry'])('clears movement spoken flags on %s while preserving speech cooldown', (kind) => {
    const { engine, update } = movementSetup();
    expect(update(-4, 1000).events).toHaveLength(1);
    const exited = kind === 'braking' ? update(-4, 1501, 0.25)
        : kind === 'invalid position' ? update(NaN, 1501) : engine.evaluate(1501);
    expect(exited.state.path).toEqual(['root']);
    for (let now = 1750; now < 3000; now += 250) {
        const reentered = update(-4, now);
        expect(reentered.state.path).toContain('Turn to reference line');
        expect(reentered.events).toHaveLength(1);
    }
    expect(update(-4, 3000).events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Move to the left.', timestamp: 1000 },
        { sentence: 'Move to the left.', timestamp: 3000 },
    ]);
});

it.each([
    [-3, 'left', 'Small movement', 'Move to the left.'],
    [-6.99, 'left', 'Small movement', 'Move to the left.'],
    [-7, 'left', 'Medium movement', 'Further to the left.'],
    [-13, 'left', 'Medium movement', 'Further to the left.'],
    [3, 'right', 'Small movement', 'Move to the right.'],
    [6.99, 'right', 'Small movement', 'Move to the right.'],
    [7, 'right', 'Medium movement', 'Further to the right.'],
    [8.01, 'right', 'Medium movement', 'Further to the right.'],
    [13, 'right', 'Medium movement', 'Further to the right.'],
] as const)('selects movement from signed offset %s, without a steering channel', (offset, direction, level, sentence) => {
    const { update } = movementSetup();
    const snapshot = update(offset, 1000);
    expect(snapshot.events).toEqual([expect.objectContaining({ sentence, timestamp: 1000, expiresAt: 2500 })]);
    expect(snapshot.state.path).toEqual(['root', 'Turn to reference line', `Move to the ${direction}`, level, 'say phrase']);
});

it.each([0, -0.35, 0.35, -0.6, 0.6, -1.5, 1.5, -2.99, 2.99, -13.01, 13.01, NaN])('withholds movement for a near-line or invalid offset %s', (offset) => {
    expect(movementSetup().update(offset, 1000).events).toEqual([]);
});

it.each([-1, 1])('uses alternative wordings for movement and progress on side %s', (sign) => {
    const random = jest.spyOn(Math, 'random').mockReturnValue(0.5);
    const { update } = movementSetup();
    const direction = sign < 0 ? 'left' : 'right';
    update(sign * 4, 1000);
    for (let now = 1250; now <= 3000; now += 250) update(sign * (4 - (now - 1000) / 10000), now);
    update(sign * 1.4, 3250);
    update(sign, 3500);
    update(sign * 0.6, 3750);
    const snapshot = update(sign * 0.3, 4000);
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual([`Shift to the ${direction}.`, 'A bit more.']);
    expect(snapshot.events.map(({ ruleId }) => ruleId)).toEqual([
        `reference-move-${direction}-small`, `reference-move-${direction}-more`,
    ]);
    expect(random).toHaveBeenCalledTimes(2);
});

it.each([-1, 1])('nudges slow progress and stays silent on reaching the line on side %s', (sign) => {
    const { engine, update } = movementSetup();
    const direction = sign < 0 ? 'left' : 'right';
    update(sign * 4, 1000);
    for (let now = 1250; now <= 3000; now += 250) update(sign * (4 - (now - 1000) / 10000), now);
    expect(engine.evaluate(3000).events.map(({ sentence }) => sentence)).toEqual([`Move to the ${direction}.`, 'A little bit more.']);
    update(sign * 1.4, 3250);
    update(sign, 3500);
    update(sign * 0.6, 3750);
    const completed = update(sign * 0.3, 4000);
    expect(completed.events.map(({ sentence }) => sentence)).toEqual([`Move to the ${direction}.`, 'A little bit more.']);
    expect(completed.state.path).toEqual(['root', 'Turn to reference line']);
    for (let now = 4250; now <= 7000; now += 250) update(sign * 0.2, now);
    expect(engine.evaluate(7000).events).toHaveLength(2);
});

it.each([0, 1.25, -0.1])('does not request more for stopped, adequate or wrong-way progress (%s units/s)', (rate) => {
    const { engine, update } = movementSetup();
    update(4, 1000);
    for (let now = 1250; now <= 3250; now += 250) update(4 - rate * (now - 1000) / 1000, now);
    expect(engine.evaluate(3250).events.map(({ sentence }) => sentence)).toEqual(['Move to the right.']);
});

it('stays silent when reaching the line immediately after a movement prompt', () => {
    const { update } = movementSetup();
    update(-4, 1000);
    const completed = update(-0.3, 1100);
    expect(completed.events.map(({ sentence, timestamp }) => ({ sentence, timestamp }))).toEqual([
        { sentence: 'Move to the left.', timestamp: 1000 },
    ]);
});

it('stays silent when crossing the line before issuing an opposite-side correction', () => {
    const { update } = movementSetup();
    update(-4, 1000);
    for (let now = 1250; now <= 1750; now += 250) update(-4, now);
    expect(update(0.8, 2000).events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
    for (let now = 2250; now <= 4000; now += 250) {
        expect(update(0.8, now).events).toHaveLength(1);
    }
    expect(update(3, 4250).events.map(({ sentence }) => sentence)).toEqual([
        'Move to the left.', 'Move to the right.',
    ]);
});

it.each(['capture gap', 'invalid position', 'pause', 'stream reset', 'reference change'])('clears movement feedback after %s', (kind) => {
    const { engine, update } = movementSetup();
    update(-4, 1000);
    if (kind === 'invalid position') update(NaN, 1250);
    if (kind === 'pause') engine.receiveTelemetry({ ...frame(driving(0.45)), telemetryStatus: 1 }, 1250);
    if (kind === 'stream reset') engine.receiveTelemetry({ type: 'stream-reset', snapshot: createLiveTelemetryStore().getSnapshot() }, 1250);
    if (kind === 'reference change') engine.receiveMap(null, 1250, referenceLap());
    const snapshot = update(0.1, kind === 'capture gap' ? 1600 : 1500);
    expect(snapshot.events.some(({ sentence }) => sentence === 'A little bit more.')).toBe(false);
});

it('does not treat repeated or out-of-order captures or timer evaluations as movement', () => {
    const { engine, lap, update } = movementSetup();
    update(-4, 1000);
    const sample = driving(0.45);
    const view = detection(lap, sample, 1000);
    view.position!.referenceDelta[0] = 0;
    engine.receiveVision(view, 1250);
    engine.receiveTelemetry(frame(sample), 1250);
    engine.receiveVision({ ...view, capturedAt: 900 }, 1400);
    engine.receiveTelemetry(frame(sample), 1400);
    expect(engine.evaluate(1500).events.map(({ sentence }) => sentence)).toEqual(['Move to the left.']);
    expect(engine.evaluate(1501).closures.find(({ id }) => id === 'reference-move-left-small')?.status).toBe('Not evaluated');
});

it('keeps brake priority and does not confirm a movement that was never requested', () => {
    const { engine, lap } = setup();
    const sample = driving(0.25), view = detection(lap, sample, 1000);
    view.position!.referenceDelta[0] = -4;
    engine.receiveVision(view, 1000);
    expect(engine.receiveTelemetry(frame(sample), 1000).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const near = detection(lap, sample, 1250);
    near.position!.referenceDelta[0] = -0.1;
    engine.receiveVision(near, 1250);
    engine.receiveTelemetry(frame(sample), 1250);
    expect(engine.evaluate(1250).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
});

it('keeps guidance available after repeatedly rebuilding a reference with nearly coincident braking boundaries', () => {
    const lap = variableBrakingReference(), engine = conditionTestEngine();
    engine.receiveMap(circuitMap(), 0, lap);
    const sample = { ...driving(0.15), Physics_speed_kmh: 220 };
    [0, 1, 25, 50, 100, 50, 1, 0].forEach((aggression, i) => {
        const now = 1000 + i * 100;
        engine.setDrivingAggression(aggression, now);
        const snapshot = engine.receiveTelemetry(frame(sample), now);
        expect(snapshot.state.path).toContain('Brake zone');
        expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    });
});

it('keeps brake available but enables reference-line movement only at full aggression', () => {
    const braking = setup(addSpeed), movement = movementSetup(circuitMap(), addSpeed);
    [0, 1, 25, 50, 100, 25, 0].forEach((aggression, i) => {
        const now = 1000 + i * 100;
        braking.engine.setDrivingAggression(aggression, now);
        const brake = braking.update(0.21, now, { Physics_speed_kmh: 122 });
        expect(brake.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
        expect(brake.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
        movement.engine.setDrivingAggression(aggression, now);
        const move = movement.update(-4, now);
        expect(move.state.path.includes('Turn to reference line')).toBe(aggression === 100);
        expect(move.events.map(({ sentence }) => sentence)).toEqual(aggression === 100 ? ['Move to the left.'] : []);
    });
});

it('uses the private adjusted brake copy for cues and restores the baseline when caution is reset', () => {
    const { engine, lap, update } = setup(addSpeed);
    const map = circuitMap();
    map.centerline_segments = [{ id: 'turn', tags: ['corner'], start_position: 0.2, end_position: 0.7 }];
    engine.receiveMap(map, 0, lap);
    const speed = { Physics_speed_kmh: 122 };
    expect(update(0.19, 1000, speed).events).toEqual([]);
    engine.setDrivingAggression(50, 1100);
    expect(update(0.1899, 1150, speed).state.path).toEqual(['root']);
    const beforeBrake = update(0.195, 1200, speed);
    expect(beforeBrake.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(beforeBrake.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(update(0.21, 1250, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(lap.y[19][4]).toBe(0);
    expect(lap.y[19][6]).toBe(120);
    engine.setDrivingAggression(25, 1300);
    expect(update(0.21, 1400, speed).state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
    expect(update(0.21, 1450, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    // Restoring Reference must use the original local baseline, not the adjusted copy or a mutated source.
    lap.y.forEach((row) => { row[4] = 0; row[6] = 200; });
    engine.setDrivingAggression(0, 1500);
    expect(update(0.19, 1600, speed).state.path).toEqual(['root']);
    expect(update(0.25, 1650, speed).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const view = detection(referenceLap(), driving(0.19), 1700);
    engine.receiveVision(view, 1700);
    expect(engine.receiveTelemetry(frame(driving(0.19)), 1700).closures.find(({ id }) => id === 'reference-brake'))
        .toMatchObject({ status: 'Not matched', missing: [] });
});

it.each([[0, 0.1999, 0.215, 0.2999, 0.3], [25, 0.1899, 0.205, 0.2999, 0.3],
    [50, 0.1899, 0.195, 0.2999, 0.3], [100, 0.1799, 0.195, 0.2999, 0.3]])(
    'keeps the adjusted brake zone past reference end at %s%% caution, speaking Brake once', (caution, before, entry, last, exit) => {
        const { engine, lap, update } = setup(addSpeed);
        const map = circuitMap();
        map.centerline_segments = [
            { id: 'approach', tags: ['straight'], start_position: 0.9, end_position: 0.2 },
            { id: 'turn', tags: ['corner'], start_position: 0.2, end_position: 0.7 },
        ];
        engine.receiveMap(map, 0, lap);
        const speed = { Physics_speed_kmh: 120 };
        expect(update(before, 1000, speed).state.path).toEqual(['root']);
        engine.setDrivingAggression(caution, 1100);
        const approach = update(before, 1200, speed);
        expect(approach.state.path).toEqual(['root']);
        expect(approach.events).toEqual([]);
        // Normalized telemetry alone can enter the adjusted zone and speak.
        const entered = engine.receiveTelemetry(frame({ ...driving(entry), ...speed }), 1300);
        expect(entered.state.path).toEqual(['root', 'Brake zone', 'Brake', 'say phrase']);
        expect(entered.events.map(({ sentence, timestamp }) => ({ sentence, timestamp })))
            .toEqual([{ sentence: 'Brake.', timestamp: 1300 }]);
        expect(engine.evaluate(1301).state.path).toEqual(['root', 'Brake zone']);
        for (const [position, now] of [[entry + 0.02, 2000], [(entry + last) / 2, 3400]]) {
            const snapshot = update(position, now, speed);
            expect(snapshot.state.path).toEqual(['root', 'Brake zone']);
            expect(snapshot.events.filter(({ sentence }) => sentence === 'Brake.')).toHaveLength(1);
        }
        // Neither predicted nor measured reference boundaries exit the active Brake zone.
        expect(update(last, 4600, speed).state.path).toEqual(['root', 'Brake zone']);
        expect(engine.receiveTelemetry(frame({ ...driving(exit), ...speed }), 4700).state.path).toEqual(['root', 'Brake zone']);
        expect(update(0.600001, 4800, speed).state.path).toEqual(['root', 'Brake zone']);
        const next = update(entry, 6000, speed);
        expect(next.state.path).toEqual(['root', 'Brake zone']);
        expect(next.events.filter(({ sentence }) => sentence === 'Brake.')).toHaveLength(1);
        expect(lap.y[19][4]).toBe(0);
        expect(lap.y[19][6]).toBe(120);
    });

it('keeps reference exit speeds and throttle timing when deceleration is adjusted', () => {
    const { engine, lap, update } = setup(addSpeed);
    expect(update(0.61, 1000, { Physics_speed_kmh: 84 }).events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
    engine.setDrivingAggression(50, 1100);
    expect(update(0.61, 1200, { Physics_speed_kmh: 84 }).events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
    expect(lap.y[61][5]).toBe(0.8);
    engine.setDrivingAggression(0, 1400);
    expect(update(0.61, 1500, { Physics_speed_kmh: 84 }).events.map(({ sentence }) => sentence)).toEqual(['Squeeze gas.']);
});

it('uses copied centerline tags and the adjusted braking target to end movement guidance', () => {
    const map = circuitMap();
    const { engine, update } = movementSetup(map, (lap) => {
        brakingMovementReference(lap, 0.435);
        lap.target_features.push('expert_optimal_speed');
        lap.y.forEach((row, i) => {
            const position = lap.x[i];
            row[4] = position > 0.45 && position < 0.52 ? 0.8 : 0;
            row.push(position < 0.45 || position > 0.6 ? 216 : position <= 0.52
                ? Math.sqrt(216 ** 2 - (216 ** 2 - 80 ** 2) * (position - 0.45) / 0.07) : 80);
        });
    });
    map.centerline_tags!.find(({ id }) => id === 'straight')!.label = 'corner';
    expect(update(-4, 1000, 0.449, { Physics_speed_kmh: 216 }).state.path).toContain('Turn to reference line');
    engine.setDrivingAggression(50, 1100);
    const cautious = update(-4, 1500, 0.46, { Physics_speed_kmh: 216, Physics_brake: 0.5 });
    expect(cautious.state.path).toEqual(['root', 'Brake zone']);
    expect(cautious.events).toEqual([]);
    // Current progress must reach the braking reference, then the measured deficit must persist.
    update(-4, 1600, 0.49, { Physics_speed_kmh: 216, Physics_brake: 0.5 });
    expect(update(-4, 2099, 0.49, { Physics_speed_kmh: 216, Physics_brake: 0.5 }).events).toEqual([]);
    expect(update(-4, 2100, 0.49, { Physics_speed_kmh: 216, Physics_brake: 0.5 }).events).toEqual([
        expect.objectContaining({ sentence: 'More brake.', timestamp: 2100 }),
    ]);
});

it('uses a replacement reference before new position captures arrive and clears guidance when the reference is removed', () => {
    const { engine, lap, update } = setup();
    engine.setDrivingAggression(50, 0);
    update(0.17, 1000);
    const replacement = referenceLap();
    replacement.y.forEach((row) => { row[4] = 0; });
    engine.receiveMap(circuitMap(), 1100, replacement);
    const stale = update(0.17, 1200);
    expect(stale.closures.find(({ id }) => id === 'reference-brake')).toMatchObject({ status: 'Not evaluated', missing: [] });
    const sample = driving(0.17);
    engine.receiveVision(detection(replacement, sample, 1300), 1300);
    const fresh = engine.receiveTelemetry(frame(sample), 1300);
    expect(fresh.closures.find(({ id }) => id === 'reference-brake')).toMatchObject({ status: 'Not evaluated' });
    expect(fresh.referenceLoaded).toBe(true);
    expect(engine.receiveMap(null, 1400).referenceLoaded).toBe(false);
    engine.setDrivingAggression(0, 1450);
    engine.receiveVision(detection(lap, sample, 1500), 1500);
    const removed = engine.receiveTelemetry(frame(sample), 1500);
    expect(removed.mapReady).toBe(false);
    expect(removed.closures.find(({ id }) => id === 'reference-brake')).toMatchObject({ status: 'Not evaluated' });
    expect(removed.events).toEqual([]);
});
