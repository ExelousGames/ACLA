import { drivingControllerAction, drivingReward, encodeDrivingAction, NEUTRAL, readDrivingObservation } from './driving-observation';
import { DrivingControlChanges } from './driving-control-changes';
import { circuitMapFixture, brakingFixtureSpeed, brakingReferenceFixture, trainingFixture } from './training-fixtures';
import { referenceSpeedTargets } from './driving-reference';
import { getLiveMapReferenceLine } from 'views/live-session/live-map-reference';
import { DrivingMemory, drivingMemoryInputs } from './driving-memory';
import { cloneConfig, drivingFeatureSizes } from './driving-model-config';
import { IMPACT_PENALTY_WEIGHT } from './driving-impact';

const { AccAdapter } = require('../../../electron/recording/readers/acc/acc-adapter');

it('keeps the model gear target and physical shift buttons on their respective sides of the controller boundary', () => {
    const controls = { throttle: 0.5, brake: 0.2, steering: 0.3, shiftUp: true };
    const action = encodeDrivingAction(controls, 4);
    expect(action).toEqual({ pedal: -0.2, steering: 0.3, gear: 4 });
    expect(drivingControllerAction(action)).toEqual({ throttle: 0, brake: 0.2, steering: 0.3 });
});

it.each(['missing', 'distant', 'invalid', 'mismatched'])(
    'ignores %s position-model output in observations, memory and rewards', (kind) => {
        const { live, vision, referenceLap, observation: before } = trainingFixture();
        const next = trainingFixture(1200, 50, referenceLap);
        next.live.currentTelemetry.Graphics_normalized_car_position = 0.252;
        next.live.currentTelemetry.Physics_speed_kmh = 106;
        next.vision.position!.normalizedPosition = 0.252;
        const expected = readDrivingObservation(next.live, next.vision, referenceLap, 50, 1200, undefined, circuitMapFixture(referenceLap)).observation!;
        for (const frame of [vision, next.vision]) {
            if (kind === 'missing') delete frame.position;
            if (kind === 'distant') frame.position!.referenceDelta = frame === vision ? [30, 40, 0] : [-90, 0, 0];
            if (kind === 'invalid') frame.position!.referenceDelta = [NaN, Infinity, -Infinity];
            if (kind === 'mismatched') Object.assign(frame.position!, { coordinateFrame: 'world', normalizedPosition: 0.75, referenceLap: null });
        }
        const start = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation!;
        const after = readDrivingObservation(next.live, next.vision, referenceLap, 50, 1200, undefined, circuitMapFixture(referenceLap)).observation!;
        expect(start).toEqual(before);
        expect(after).toEqual(expected);
        expect(after).not.toHaveProperty('referenceError');
        expect(after).not.toHaveProperty('referenceDistance');
        const memory = new DrivingMemory();
        const expectedMemory = new DrivingMemory();
        memory.encode(start);
        expectedMemory.encode(before);
        expect(memory.encode(after)).toEqual(expectedMemory.encode(expected));
        const reward = drivingReward(start, after);
        expect(reward).toEqual(drivingReward(before, expected));
        expect(reward).not.toHaveProperty('line');
        expect(reward).not.toHaveProperty('approach');
        expect(reward.timing).toBeCloseTo(-0.074);
    });

describe('steering-change reward', () => {
    it.each([-0.8, 0, 0.8])('does not penalize holding steering at %s', (steering) => {
        const { observation } = trainingFixture();
        const before = { ...observation, controls: { ...NEUTRAL, steering } };
        const reward = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200 });
        expect(reward.steeringChange).toBe(0);
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress + reward.impact + reward.onTrack + reward.offTrack);
    });

    it.each([[-0.4, 0.4], [0.4, -0.4], [0, 0.8]])('allows an initial steering movement from %s to %s', (from, to) => {
        const { observation } = trainingFixture();
        const before = { ...observation, controls: { ...NEUTRAL, steering: from } };
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, controls: { ...NEUTRAL, steering: to } };
        const reward = drivingReward(before, after);
        expect(reward.steeringChange).toBe(0);
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress + reward.impact + reward.onTrack + reward.offTrack);
    });

    it('includes intermediate command reversals even when telemetry endpoints match, and honors weight and total limits', () => {
        const { observation: before } = trainingFixture();
        const after = { ...before, capturedAt: 1200, measuredAt: 1200 };
        const changes = { steering: 0.8 ** 2 + 0.8 ** 2, throttle: 0, brake: 0 };
        const config = cloneConfig();
        const reward = drivingReward(before, after, config, undefined, changes);
        expect(drivingReward(before, after).steeringChange).toBe(0);
        expect(reward.steeringChange).toBeCloseTo(-0.064);
        config.reward.steeringChangeWeight = 0.1;
        expect(drivingReward(before, after, config, undefined, changes).steeringChange).toBeCloseTo(reward.steeringChange * 2);
        config.reward.totalLimit = 0.01;
        expect(drivingReward(before, after, config, undefined, changes).total).toBe(-0.01);
        config.reward.steeringChangeWeight = 0;
        expect(drivingReward(before, after, config, undefined, changes).steeringChange).toBe(0);
    });
});

it.each(['throttle', 'brake'] as const)('scales the %s penalty by movement speed and configured weight', (pedal) => {
    const { observation } = trainingFixture();
    const before = { ...observation, controls: NEUTRAL };
    const key = `${pedal}Change` as const;
    const config = cloneConfig();
    const cost = (steps: number, elapsedMs: number) => {
        const movement = new DrivingControlChanges();
        let previous = before, total = 0;
        // The throttle movement is scored on reversal. Brake is scored as it moves.
        const positions = Array.from({ length: steps }, (_, i) => (i + 1) / steps);
        if (pedal === 'throttle') positions.push(0.9);
        for (const position of positions) {
            const at = previous.measuredAt + elapsedMs / steps;
            const current = { ...previous, capturedAt: at, measuredAt: at, controls: { ...NEUTRAL, [pedal]: position } };
            const changes = movement.record(previous.controls, current.controls, at - previous.measuredAt);
            total += drivingReward(previous, current, config, undefined, changes)[key];
            previous = current;
        }
        return total;
    };
    expect(cost(1, 200)).toBeCloseTo(-0.025);
    expect(cost(2, 200)).toBeCloseTo(-0.025);
    expect(cost(1, 100)).toBeCloseTo(-0.05);
    config.reward[`${pedal}ChangeWeight`] = 0;
    expect(cost(1, 100)).toBe(0);
});

describe('impact reward', () => {
    it.each(['acc', 'iracing'])('penalizes diagonal impacts in all four directions for %s', (game) => {
        const { observation } = trainingFixture();
        const velocity = (right: number, forward: number): [number, number, number] =>
            game === 'iracing' ? [right, 0, forward] : [right, forward, 0];
        const before = { ...observation, context: { ...observation.context, game },
            controls: { ...observation.controls, steering: 0.4 }, localVelocity: velocity(0, 30) };
        for (const forward of [-1, 1]) {
            for (const right of [-1, 1]) {
                const after = { ...before, capturedAt: 1100, measuredAt: 1100,
                    speed: Math.hypot(3 * right, 30 + 3 * forward) * 3.6, localVelocity: velocity(3 * right, 30 + 3 * forward) };
                const reward = drivingReward(before, after);
                expect(reward.impactEstimate.detected).toBe(true);
                expect(reward.impact).toBeLessThan(0);
                expect(reward.total).toBeCloseTo(reward.timing + reward.progress + reward.impact);
                expect(drivingReward(after, { ...after, capturedAt: 1200, measuredAt: 1200 }).impact).toBe(0);
            }
        }
    });

    it.each(['acc', 'iracing'])('penalizes an unbraked 10 km/h impact in %s without penalizing a braked stop', (game) => {
        const { observation } = trainingFixture();
        const before = { ...observation, context: { ...observation.context, game }, speed: 10,
            localVelocity: (game === 'iracing' ? [0, 0, 10 / 3.6] : [0, 10 / 3.6, 0]) as [number, number, number] };
        const after = { ...before, capturedAt: 1500, measuredAt: 1500, speed: 0,
            localVelocity: [0, 0, 0] as [number, number, number] };
        const reward = drivingReward(before, after);
        expect(reward.impactEstimate.detected).toBe(true);
        expect(reward.impact).toBeLessThan(-0.25);
        expect(reward.impact).toBeGreaterThan(-1);
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress + reward.impact);
        expect(drivingReward(before, { ...after, controls: { ...after.controls, brake: 0.5 } }).impact).toBe(0);
        expect(drivingReward(after, { ...after, capturedAt: 1700, measuredAt: 1700 }).impact).toBe(0);
    });

    it.each(['acc', 'iracing'])('penalizes smaller impacts for %s and scales up with impulse', (game) => {
        const { observation } = trainingFixture();
        const velocity = (forward: number): [number, number, number] => game === 'iracing' ? [0, 0, forward] : [0, forward, 0];
        const before = { ...observation, context: { ...observation.context, game }, localVelocity: velocity(30) };
        const after = { ...before, capturedAt: 1100, measuredAt: 1100, speed: 26 * 3.6, localVelocity: velocity(26) };
        const small = drivingReward(before, after);
        const larger = drivingReward(before, { ...after, speed: 24 * 3.6, localVelocity: velocity(24) });
        expect(small.impactEstimate.detected).toBe(true);
        expect(small.impact).toBeLessThan(-0.3);
        expect(larger.impact).toBeLessThan(-1.5);
        expect(larger.impact).toBeLessThan(small.impact);
        expect(small.total).toBeCloseTo(small.timing + small.progress + small.impact);
        expect(drivingReward(after, { ...after, capturedAt: 1200, measuredAt: 1200 }).impact).toBe(0);
    });

    it.each(['acc', 'iracing'])('uses the horizontal adapter axes for %s and ignores vertical bumps', (game) => {
        const { observation } = trainingFixture();
        const before = { ...observation, context: { ...observation.context, game },
            localVelocity: (game === 'iracing' ? [0, 0, 30] : [0, 30, 0]) as [number, number, number] };
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, speed: 0,
            localVelocity: [0, 0, 0] as [number, number, number] };
        const reward = drivingReward(before, after);
        expect(reward.impactEstimate).toMatchObject({ available: true, detected: true, confidence: 1, severity: 1 });
        expect(reward.impact).toBe(-IMPACT_PENALTY_WEIGHT);
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress + reward.impact);
        const bump = { ...after, localVelocity: (game === 'iracing' ? [0, 20, 30] : [0, 30, 20]) as [number, number, number] };
        expect(drivingReward(before, bump).impact).toBe(0);
        // The same collision with different image timing has the same impact estimate.
        expect(drivingReward(before, { ...after, capturedAt: 1400 }).impactEstimate).toEqual(reward.impactEstimate);
        expect(drivingReward(after, { ...after, capturedAt: 1400, measuredAt: 1400 }).impact).toBe(0);
        const config = cloneConfig();
        config.reward.totalLimit = 1;
        expect(drivingReward(before, after, config).total).toBe(-1);
    });

    it('does not read damage telemetry or use yaw, throttle, gear, speed or reference offsets to infer impacts', () => {
        const fixture = trainingFixture();
        for (const side of ['front', 'rear', 'left', 'right', 'center']) {
            Object.defineProperty(fixture.vision.telemetry!.currentTelemetry, `Physics_car_damage_${side}`,
                { get: () => { throw new Error('Damage telemetry must not be read.'); } });
        }
        const before = readDrivingObservation(fixture.live, fixture.vision, fixture.referenceLap, 50, 1000, undefined, circuitMapFixture(fixture.referenceLap)).observation!;
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, localVelocity: [0, 0, 0] as [number, number, number] };
        const impact = drivingReward(before, after).impactEstimate;
        expect(impact.detected).toBe(true);
        expect(drivingReward(before, { ...after, yawRate: 100, speed: 300, gear: -1,
            controls: { ...after.controls, throttle: 1 } }).impactEstimate).toEqual(impact);
        expect(drivingReward(before, { ...after, measuredAt: 1600, capturedAt: 1600 }))
            .toMatchObject({ impact: 0, impactEstimate: { available: false } });
        expect(() => drivingReward(before, { ...after, session: 'new session' })).toThrow(/boundary/);
        expect(() => drivingReward(before, { ...after, normalizedPosition: 0.5 })).toThrow(/jumped/);
    });
});

it('uses telemetry and the full-resolution camera from the capture time', () => {
    const { live, vision, referenceLap, observation } = trainingFixture();
    expect(observation.features).toHaveLength(366);
    expect(observation).toMatchObject({ gear: 3, rpm: 6000 });
    expect(observation.features.slice(57, 59)).toEqual([3 / 8, 6000 / 20000]);
    expect(observation.camera).toBe(vision.camera);
    expect(observation.camera).toMatchObject({ width: 384, height: 216 });
    expect(observation.context).not.toHaveProperty('classes');
    // Learning keeps the sample paired with the image as live telemetry advances.
    const latest = readDrivingObservation({ ...live, currentTelemetry: { ...live.currentTelemetry, Physics_speed_kmh: 200 } },
        vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap));
    expect(latest).toMatchObject({ telemetry: { speed: 200 }, observation });
    expect(latest.observation?.speed).toBe(108);
});

it.each(['missing', 'stale', 'invalid', 'unpaired'])(
    'keeps current timestamped telemetry available when the camera is %s', (fault) => {
        const { live, vision, referenceLap } = trainingFixture();
        live.receivedAt = 2000;
        live.currentTelemetry.Physics_gas = 0.7;
        live.currentTelemetry.Physics_speed_kmh = 180;
        if (fault === 'stale') vision.capturedAt = 0;
        if (fault === 'invalid') delete vision.camera;
        if (fault === 'unpaired') delete vision.telemetry;
        const result = readDrivingObservation(live, fault === 'missing' ? null : vision,
            referenceLap, 50, 2000, undefined, circuitMapFixture(referenceLap));
        expect(result.observation).toBeUndefined();
        expect(result.error).toBeTruthy();
        expect(result.telemetry).toMatchObject({ measuredAt: 2000, speed: 180, gear: 3, rpm: 6000,
            controls: { throttle: 0.7, brake: 0, steering: 0 }, context: { game: 'acc', car: 'GT3', map: 'monza' } });
        expect(result.telemetry?.session).toBe(JSON.stringify([live.sessionGeneration, live.streamGeneration, 'acc', 'GT3', 'monza']));
        expect(result.telemetry).not.toHaveProperty('camera');
        live.currentTelemetry.Physics_gas = 0.1;
        live.currentTelemetry.Physics_local_velocity_x = 90;
        expect(result.telemetry?.controls.throttle).toBe(0.7);
        expect(result.telemetry?.localVelocity).toEqual([1, 30, 0]);
    });

it('reports live telemetry separately from the older sample paired with a valid image', () => {
    const { live, vision, referenceLap, observation } = trainingFixture();
    const latest = { ...live, receivedAt: 1200, currentTelemetry: { ...live.currentTelemetry,
        Physics_speed_kmh: 200, Physics_steer_angle: -0.3, Physics_gear: 5 } };
    const result = readDrivingObservation(latest, vision, referenceLap, 50, 1200, undefined, circuitMapFixture(referenceLap));
    expect(result.observation).toEqual(observation);
    expect(result.telemetry).toMatchObject({ measuredAt: 1200, speed: 200, gear: 4, controls: { steering: -0.3 } });
});

it('reports reference errors alongside current telemetry', () => {
    const { live, vision, referenceLap } = trainingFixture();
    const result = readDrivingObservation(live, vision, { ...referenceLap, car: 'other' }, 50, 1000,
        undefined, circuitMapFixture(referenceLap));
    expect(result.error).toMatch(/does not match/);
    expect(result.telemetry).toBeDefined();
});

it.each(['stale', 'future', 'paused', 'missing-context', 'invalid-control', 'automatic-shifting'])(
    'does not collect %s telemetry independently of the camera', (fault) => {
        const { live, referenceLap } = trainingFixture();
        if (fault === 'stale') live.receivedAt = 0;
        if (fault === 'future') live.receivedAt = 1001;
        if (fault === 'paused') live.telemetryStatus = 3;
        if (fault === 'missing-context') delete live.currentTelemetry.Static_car_model;
        if (fault === 'invalid-control') live.currentTelemetry.Physics_gas = NaN;
        if (fault === 'automatic-shifting') live.currentTelemetry.Physics_autoshifter_on = true;
        expect(readDrivingObservation(live, null, referenceLap, 50, 1000).telemetry).toBeUndefined();
    });

it.each(['telemetry', 'camera', 'pair', 'paused', 'pixels', 'reference'])(
    'only retries delayed input delivery during startup (%s)', (fault) => {
        const { live, vision, referenceLap, circuitMap } = trainingFixture();
        if (fault === 'telemetry') live.receivedAt = 0;
        if (fault === 'camera') vision.capturedAt = -1000;
        if (fault === 'pair') delete vision.telemetry;
        if (fault === 'paused') live.telemetryStatus = 3;
        if (fault === 'pixels') delete vision.camera;
        const result = readDrivingObservation(live, vision, fault === 'reference' ? null : referenceLap, 50, 1000, undefined, circuitMap);
        expect(result.observation).toBeUndefined();
        expect(Boolean(result.waitingForFreshInputs)).toBe(['telemetry', 'camera', 'pair'].includes(fault));
    });

it.each(['missing', 'invalid', 'changed'])('ignores %s segmentation, depth and reconstructed scene', (fault) => {
    const { live, vision, referenceLap, observation } = trainingFixture();
    if (fault === 'missing') {
        vision.detections = {};
        delete vision.reconstructedScene;
    } else {
        vision.detections.segment!.classNames = fault === 'invalid' ? [] : ['different'];
        const segment = vision.detections.segment!;
        if (segment.task === 'segment') segment.instances[0].mask = new Uint8Array(1);
        vision.reconstructedScene = null;
        delete vision.detections.depth;
    }
    const result = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap));
    expect(result.observation!.features).toEqual(observation.features);
    expect(result.observation!.context).toEqual(observation.context);
});

it.each(['missing', 'width', 'height', 'length', 'type'])('rejects %s camera data instead of fabricating visual history', (fault) => {
    const { live, vision, referenceLap } = trainingFixture();
    if (fault === 'missing') delete vision.camera;
    if (fault === 'width') vision.camera!.width = 4;
    if (fault === 'height') vision.camera!.height = 4;
    if (fault === 'length') vision.camera!.rgb = new Uint8Array(191);
    if (fault === 'type') Object.assign(vision.camera!, { rgb: new Array(192).fill(NaN) });
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).error).toMatch(/camera pixels/);
});

it('preserves adapted ACC axes and signs in capture-time policy inputs and driving memory', () => {
    const { live, vision, referenceLap } = trainingFixture();
    Object.assign(vision.telemetry!.currentTelemetry, new AccAdapter().adapt({ type: 'sample', broadcasting: {}, sharedMemory: {
        Physics: {
            local_velocity: { x: -12, y: 2.5, z: 40 }, velocity: { x: 40, y: 2.5, z: 12 },
            g_force: { x: -1.5, y: 1, z: 2.5 },
            heading: Math.PI / 2, pitch: -0.2, roll: 0.3,
            local_angular_vel: { x: -Math.PI / 2, y: -Math.PI / 8, z: Math.PI / 4 },
        },
    } }));
    const later = { ...live, currentTelemetry: { ...live.currentTelemetry,
        Physics_local_velocity_x: 90, Physics_heading: 0, Physics_local_angular_vel_z: 1,
        Physics_g_force_x: 4, Physics_g_force_y: 3, Physics_g_force_z: -2 } };
    const observation = readDrivingObservation(later, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation!;
    expect(observation).toMatchObject({ localVelocity: [-12, 40, 2.5], gForce: [-1.5, 2.5, 1],
        yawRate: Math.PI / 8 });
    expect(observation.features.slice(0, 3)).toEqual([-0.12, 0.4, 0.025]);
    expect(observation.features[59]).toBe(0.125);
    expect(observation.features.slice(60, 63)).toEqual([-0.3, 0.5, 0.2]);
    const memory = new DrivingMemory();
    memory.encode(observation);
    const encoded = memory.encode({ ...observation, capturedAt: 1200, measuredAt: 1200 }).features;
    expect(encoded.slice(2, 5)).toEqual([-0.12, 0.4, 0.025]);
    expect(encoded[2 + 59]).toBe(0.125);
    expect(encoded.slice(2 + 60, 2 + 63)).toEqual([-0.3, 0.5, 0.2]);
    expect(encoded.slice(drivingMemoryInputs())).toEqual(observation.features);
});

it.each(['acc', 'iracing'] as const)('normalizes g-force in adapter XYZ order for %s using the configured scale', (game) => {
    const { live, vision, referenceLap } = trainingFixture();
    live.game = vision.telemetry!.game = game;
    referenceLap.game = game;
    Object.assign(vision.telemetry!.currentTelemetry, {
        Physics_g_force_x: -8, Physics_g_force_y: 1, Physics_g_force_z: 6,
    });
    const config = cloneConfig();
    config.observation.gForceScale = 4;
    const observation = readDrivingObservation(live, vision, referenceLap, 50, 1000, config, circuitMapFixture(referenceLap)).observation!;
    expect(observation.gForce).toEqual([-8, 1, 6]);
    expect(observation.features.slice(60, 63)).toEqual([-1, 0.25, 1]);
});

describe.each(['x', 'y', 'z'])('g-force %s validation', (axis) => {
    it.each([undefined, NaN, Infinity, -Infinity, '0'])('rejects invalid g-force %s', (value) => {
        const { live, vision, referenceLap } = trainingFixture();
        Object.assign(vision.telemetry!.currentTelemetry, { [`Physics_g_force_${axis}`]: value });
        expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).error).toMatch(/g-force XYZ/);
    });
});

it('ignores world-space velocity when building policy inputs', () => {
    const { live, vision, referenceLap, observation: before } = trainingFixture();
    // The fixture has local velocity only; world-space telemetry is optional and unused.
    expect(before.localVelocity).toEqual([1, 30, 0]);
    Object.assign(vision.telemetry!.currentTelemetry, {
        Physics_velocity_x: -30, Physics_velocity_y: NaN, Physics_velocity_z: Infinity,
    });
    const after = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation!;
    expect(after.localVelocity).toEqual(before.localVelocity);
    expect(after.features).toEqual(before.features);
});

describe.each(['x', 'y', 'z'])('local velocity %s validation', (axis) => {
    it.each([undefined, NaN, Infinity, -Infinity, '0'])('rejects invalid velocity %s even with valid world-space velocity', (velocity) => {
        const { live, vision, referenceLap } = trainingFixture();
        Object.assign(vision.telemetry!.currentTelemetry, {
            [`Physics_local_velocity_${axis}`]: velocity,
            Physics_velocity_x: 1, Physics_velocity_y: 30, Physics_velocity_z: 0,
        });
        expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).error).toMatch(/Local velocity/);
    });
});

it('ignores orientation angles in both policy inputs and the reward', () => {
    const { live, vision, referenceLap, observation: before } = trainingFixture();
    Object.assign(vision.telemetry!.currentTelemetry, { Physics_heading: 1.2, Physics_pitch: -0.2, Physics_roll: 0.3 });
    const after = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation!;
    expect(after.features).toEqual(before.features);
    const stable = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200 });
    const rotated = drivingReward(before, { ...after, capturedAt: 1200, measuredAt: 1200 });
    expect(rotated).toEqual(stable);
    Object.assign(vision.telemetry!.currentTelemetry, { Physics_heading: undefined, Physics_pitch: NaN, Physics_roll: undefined });
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation).toEqual(after);
});

describe.each(['acc', 'iracing'] as const)('%s yaw-only rotation inputs', (game) => {
    function fixture() {
        const result = trainingFixture();
        result.live.game = game;
        result.vision.telemetry!.game = game;
        result.referenceLap.game = game;
        return result;
    }

    it.each([-4 * Math.PI, 4 * Math.PI])('bounds the yaw feature while retaining measured rate %s', (rate) => {
        const { live, vision, referenceLap } = fixture();
        Object.assign(vision.telemetry!.currentTelemetry, {
            Physics_local_angular_vel_x: 9,
            Physics_local_angular_vel_y: game === 'iracing' ? rate : 8,
            Physics_local_angular_vel_z: game === 'iracing' ? 8 : rate,
        });
        const observation = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation!;
        expect(observation.yawRate).toBe(rate);
        expect(observation.features[59]).toBe(Math.sign(rate));
        expect(observation).not.toHaveProperty('rotationRates');
    });

    it.each([undefined, NaN, Infinity, -Infinity, '0'])('rejects invalid yaw rate %s', (rate) => {
        const { live, vision, referenceLap } = fixture();
        Object.assign(vision.telemetry!.currentTelemetry, { [`Physics_local_angular_vel_${game === 'iracing' ? 'y' : 'z'}`]: rate });
        expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).error).toMatch(/yaw rate/);
    });

    it.each([undefined, NaN, Infinity, -4, 4])('omits pitch and roll rates even when they are %s', (rate) => {
        const { live, vision, referenceLap } = fixture();
        const before = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation!;
        Object.assign(vision.telemetry!.currentTelemetry, {
            Physics_local_angular_vel_x: rate,
            [`Physics_local_angular_vel_${game === 'iracing' ? 'z' : 'y'}`]: rate,
        });
        const after = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation!;
        expect(after).toEqual(before);
        expect(new DrivingMemory().encode(after).features).toEqual(new DrivingMemory().encode(before).features);
    });
});

it.each(['telemetry', 'capture', 'session', 'velocity', 'nonfinite'])(
    'rejects unusable %s instead of training on fabricated defaults', (fault) => {
        const { live, vision, referenceLap } = trainingFixture();
        if (fault === 'telemetry') live.receivedAt = 0;
        if (fault === 'capture') vision.capturedAt = -1000;
        if (fault === 'session') live.streamGeneration++;
        if (fault === 'velocity') delete live.currentTelemetry.Physics_local_velocity_x;
        if (fault === 'nonfinite') live.currentTelemetry.Physics_speed_kmh = NaN;
        expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).error).toBeTruthy();
    },
);

it.each([['acc', undefined], ['acc', NaN], ['acc', 0.8], ['iracing', undefined], ['iracing', 0.8]] as const)(
    '%s changes only the centerline features with optional lap progress %s', (game, position) => {
        const { live, vision, referenceLap } = trainingFixture();
        live.game = vision.telemetry!.game = referenceLap.game = game;
        const map = circuitMapFixture(referenceLap);
        const config = cloneConfig();
        config.observation.features = ['centerline', ...config.observation.features.filter((name) => name !== 'centerline')];
        const width = drivingFeatureSizes(config).centerline;
        const before = readDrivingObservation(live, vision, referenceLap, 50, 1000, config, map).observation!;
        live.currentTelemetry.Graphics_normalized_car_position = position;
        const read = readDrivingObservation(live, vision, referenceLap, 50, 1000, config, map);
        const after = read.observation!;
        expect(after.features.slice(width)).toEqual(before.features.slice(width));
        expect(after.normalizedPosition).toBe(position === 0.8 ? position : null);
        expect(read.telemetry?.normalizedPosition).toBe(after.normalizedPosition);
        expect(after.centerlineAvailable).toBe(after.normalizedPosition !== null);
        if (after.normalizedPosition === null) {
            expect(after.features.slice(0, width)).toEqual(new Array(width).fill(0));
            expect(after.speedTargets).toBeNull();
            const reward = drivingReward(before, { ...after, capturedAt: 1200, measuredAt: 1200 });
            expect(reward).toMatchObject({ timing: 0, progress: 0, speedComparison: null });
            expect(Number.isFinite(reward.total)).toBe(true);
        } else {
            expect(after.features.slice(0, width)).not.toEqual(before.features.slice(0, width));
        }
    });

it.each([0, 1, 2, 4])('normalizes standard gear %s and keeps RPM paired with the image', (gear) => {
    const { live, vision, referenceLap } = trainingFixture();
    vision.telemetry!.currentTelemetry = { ...live.currentTelemetry, Physics_gear: gear, Physics_rpm: 4500 };
    const result = readDrivingObservation({ ...live, currentTelemetry: { ...live.currentTelemetry, Physics_gear: 6, Physics_rpm: 8000 } }, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap));
    expect(result.observation).toMatchObject({ gear: gear - 1, gearAvailable: true, rpm: 4500 });
    expect(result.telemetry).toMatchObject({ gear: 5, gearAvailable: true });
});

it.each([
    { Physics_gear: 0, Physics_rpm: 0 }, { Physics_gear: 1, Physics_rpm: 0 },
    { Physics_gear: 2, Physics_rpm: 0 }, { Physics_gear: -1, Physics_rpm: -1 },
    { Physics_gear: 2.5, Physics_rpm: 12.5 },
])('accepts stalled and nonstandard gear/RPM telemetry %j', (fields) => {
    const { live, vision, referenceLap } = trainingFixture();
    Object.assign(live.currentTelemetry, fields);
    const result = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap));
    expect(result.error).toBeUndefined();
    expect(result.observation).toMatchObject({ gear: fields.Physics_gear - 1,
        gearAvailable: Number.isInteger(fields.Physics_gear) && fields.Physics_gear >= 0, rpm: fields.Physics_rpm });
    expect(new DrivingMemory().encode(result.observation!).features.every(Number.isFinite)).toBe(true);
});

describe.each(['Physics_gear', 'Physics_rpm'])('unavailable %s telemetry', (field) => {
    it.each([undefined, null, NaN, Infinity, -Infinity])('keeps policy inputs usable when the reading is %s', (value) => {
        const { live, vision, referenceLap } = trainingFixture();
        Object.assign(live.currentTelemetry, { [field]: value });
        const result = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap));
        expect(result.error).toBeUndefined();
        const gear = field === 'Physics_gear' ? 0 : 3;
        const rpm = field === 'Physics_rpm' ? 0 : 6000;
        expect(result.observation).toMatchObject({ gear, gearAvailable: field !== 'Physics_gear', rpm });
        expect(result.telemetry).toMatchObject({ gear, gearAvailable: field !== 'Physics_gear' });
        expect(result.observation!.features.slice(57, 59)).toEqual([gear / 8, rpm / 20000]);
        expect(new DrivingMemory().encode(result.observation!).features.every(Number.isFinite)).toBe(true);
    });
});

it('requires automatic shifting to be disabled when reported by the simulator', () => {
    const { live, vision, referenceLap } = trainingFixture();
    live.currentTelemetry.Physics_autoshifter_on = true;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).error).toMatch(/Disable automatic gear shifting/);
    live.currentTelemetry.Physics_autoshifter_on = false;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation).toBeDefined();
});

describe('normalized position progress reward', () => {
    it('rewards covering the same distance sooner and penalizes faster backward progress', () => {
        const before = trainingFixture().observation;
        const after = { ...before, normalizedPosition: 0.252, capturedAt: 1200, measuredAt: 1200 };
        const normal = drivingReward(before, after).progress;
        const faster = drivingReward(before, { ...after, capturedAt: 1100, measuredAt: 1100 }).progress;
        const slower = drivingReward(before, { ...after, capturedAt: 1400, measuredAt: 1400 }).progress;
        expect(normal).toBeCloseTo(0.025);
        expect(faster).toBeGreaterThan(normal);
        expect(slower).toBeGreaterThan(0);
        expect(slower).toBeLessThan(normal);
        expect(drivingReward(before, { ...after, normalizedPosition: 0.254 }).progress).toBeGreaterThan(normal);
        expect(drivingReward(before, { ...after, normalizedPosition: 0.248 }).progress).toBeCloseTo(-normal);
        expect(drivingReward(before, { ...after, normalizedPosition: 0.248,
            capturedAt: 1100, measuredAt: 1100 }).progress).toBeCloseTo(-faster);
    });

    it('uses telemetry time rather than capture timing or reported speed to measure progress rate', () => {
        const before = { ...trainingFixture().observation, measuredAt: 900 };
        const after = { ...before, normalizedPosition: 0.252, capturedAt: 1200, measuredAt: 1100 };
        const reward = drivingReward(before, after).progress;
        expect(drivingReward(before, { ...after, capturedAt: 1400, speed: 200 }).progress).toBe(reward);
    });

    it.each([1, 0, -1])('preserves accumulated reward when subdividing progress in direction %s', (direction) => {
        const before = trainingFixture().observation;
        const middle = { ...before, capturedAt: 1100, measuredAt: 1100, normalizedPosition: 0.25 + direction * 0.001 };
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.25 + direction * 0.002 };
        expect(drivingReward(before, middle).progress + drivingReward(middle, after).progress)
            .toBeCloseTo(drivingReward(before, after).progress, 12);
    });

    it.each([20, 100, 500])('scales and caps the progress bonus with progress weight %s', (progressWeight) => {
        const config = cloneConfig();
        config.reward.progressWeight = progressWeight;
        const before = trainingFixture().observation;
        for (const elapsed of [1, 100, 1000]) {
            for (const change of [0.0001, 0.002, 0.05]) {
                const after = { ...before, capturedAt: 1000 + elapsed, measuredAt: 1000 + elapsed,
                    normalizedPosition: before.normalizedPosition! + change };
                const reward = drivingReward(before, after, config);
                expect(reward.progress).toBeGreaterThan(0);
                expect(reward.progress).toBeLessThan(0.25 * progressWeight * Math.min(change, config.reward.progressLimit));
            }
        }
    });

    it.each([0, 0.5, 0.999])('penalizes speed %s km/h by elapsed telemetry time', (speed) => {
        const before = trainingFixture().observation;
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, speed };
        const reward = drivingReward(before, after);
        expect(reward.progress).toBeCloseTo(-0.05);
        expect(reward.total).toBe(reward.progress + reward.timing);
        expect(drivingReward(before, { ...after, capturedAt: 1400 }).progress).toBe(reward.progress);
        expect(drivingReward(before, { ...after, capturedAt: 1400, measuredAt: 1400 }).progress)
            .toBeCloseTo(reward.progress * 2);
        const middle = { ...after, capturedAt: 1100, measuredAt: 1100 };
        expect(drivingReward(before, middle).progress + drivingReward(middle, after).progress)
            .toBeCloseTo(reward.progress, 12);
    });

    it.each([0, 0.5, 0.999].flatMap((speed) => [0.248, 0.252].map((normalizedPosition) => ({ speed, normalizedPosition }))))(
        'penalizes speed below 1 km/h regardless of position change: %j', (fields) => {
        const before = trainingFixture().observation;
        const reward = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, ...fields });
        expect(reward.progress).toBeCloseTo(-0.05);
    });

    it.each([1, 1.001, 108])('uses signed progress without a low-speed penalty at %s km/h', (speed) => {
        const before = trainingFixture().observation;
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, speed };
        const reward = drivingReward(before, after);
        expect(reward.progress).toBe(0);
        expect(reward.total).toBe(reward.timing);
        expect(drivingReward(before, { ...after, normalizedPosition: 0.252 }).progress).toBeCloseTo(0.025);
        expect(drivingReward(before, { ...after, normalizedPosition: 0.248 }).progress).toBeCloseTo(-0.025);
    });

    it('applies the stationary progress penalty in reverse gear even when lap position increases', () => {
        const before = trainingFixture().observation;
        expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200,
            normalizedPosition: 0.252, gear: -1 }).progress).toBeCloseTo(-0.05);
    });

    it.each([-1, 0, 3])('penalizes negative progress regardless of gear %s', (gear) => {
        const before = trainingFixture().observation;
        const reward = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.248, gear });
        expect(reward.progress).toBeLessThan(0);
        expect(reward.total).toBe(reward.progress + reward.timing);
        expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200,
            normalizedPosition: 0.248, gear, speed: 0 }).progress).toBeCloseTo(-0.05);
    });
});

it('does not reduce reward for repeated yaw-rate reversals', () => {
    const { observation } = trainingFixture();
    const frames = [0.3, -0.3, 0.3, -0.3].map((yawRate, i) => ({ ...observation,
        capturedAt: 1000 + i * 200, measuredAt: 1000 + i * 200,
        normalizedPosition: 0.25 + i * 0.002, yawRate }));
    for (let i = 1; i < frames.length; i++) {
        const reward = drivingReward(frames[i - 1], frames[i]);
        const steady = drivingReward({ ...frames[i - 1], yawRate: 0.3 }, { ...frames[i], yawRate: 0.3 });
        expect(reward).toEqual(steady);
        expect(reward).not.toHaveProperty('instability');
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress + reward.onTrack);
    }
});

it.each(['throttle', 'brake', 'steering'] as const)('retains progress reward for abrupt %s changes', (control) => {
    const { observation } = trainingFixture();
    const before = { ...observation, controls: NEUTRAL };
    const action = { ...NEUTRAL, [control]: 1 };
    const stationary = { ...before, capturedAt: 1200, measuredAt: 1200, speed: 0 };
    const moving = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252 };
    for (const after of [stationary, moving]) {
        const unchanged = drivingReward(before, after);
        const changed = drivingReward(before, { ...after, controls: action });
        expect(changed.progress).toEqual(unchanged.progress);
    }
});

it.each([
    { heading: 0, pitch: 0, roll: 0 },
    { heading: 1, pitch: 0.2, roll: 0.4 },
])('charges separate pedal changes when throttle and brake overlap at orientation %j', (orientation) => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, ...orientation };
    const throttleOnly = drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1 } });
    const overlap = drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1, brake: 1 } });
    expect(overlap.timing).toBe(throttleOnly.timing);
    expect(overlap.throttleChange).toBe(throttleOnly.throttleChange);
    expect(throttleOnly.brakeChange).toBe(0);
    expect(overlap.brakeChange).toBeCloseTo(-0.025);
    expect(overlap.total).toBeCloseTo(throttleOnly.total + overlap.brakeChange);
});

it.each([
    ['coasting', NEUTRAL, NEUTRAL],
    ['holding throttle', { ...NEUTRAL, throttle: 0.7 }, { ...NEUTRAL, throttle: 0.7 }],
    ['holding brake', { ...NEUTRAL, brake: 0.4 }, { ...NEUTRAL, brake: 0.4 }],
    ['steering only', { ...NEUTRAL, throttle: 0.7 }, { ...NEUTRAL, throttle: 0.7, steering: 0.5 }],
    ['shifting only', { ...NEUTRAL, throttle: 0.7 }, { ...NEUTRAL, throttle: 0.7, shiftUp: true }],
] as const)('scores measured speed and speed changes while %s', (_label, previousControls, controls) => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100, controls: previousControls };
    for (const speed of [99.2, 200]) {
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, speed, controls };
        const reward = drivingReward(before, after);
        if (speed === 99.2) expect(reward.timing).toBeCloseTo(0);
        else expect(reward.timing).toBeLessThan(0);
        expect(reward.steeringChange).toBe(0);
        expect(reward.throttleChange).toBe(0);
        expect(reward.brakeChange).toBe(0);
    }
});

it.each([
    ['pressing throttle', NEUTRAL, { ...NEUTRAL, throttle: 0.7 }],
    ['releasing throttle', { ...NEUTRAL, throttle: 0.7 }, NEUTRAL],
    ['pressing brake', NEUTRAL, { ...NEUTRAL, brake: 0.4 }],
    ['releasing brake', { ...NEUTRAL, brake: 0.4 }, NEUTRAL],
    ['changing both pedals', { ...NEUTRAL, throttle: 0.7 }, { ...NEUTRAL, brake: 0.4 }],
] as const)('scores timing when %s', (_label, previousControls, controls) => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100, controls: previousControls };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, speed: 99.2, controls };
    expect(drivingReward(before, after).timing).toBeCloseTo(0);
    expect(drivingReward(before, { ...after, speed: 200 }).timing).toBeLessThan(0);
});

it('rewards the reference speed profile instead of simply driving faster', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100 };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, controls: NEUTRAL };
    const matched = drivingReward(before, { ...after, speed: 99.2 });
    const fast = drivingReward(before, { ...after, speed: 200 });
    expect(matched.timing).toBeCloseTo(0);
    expect(fast.timing).toBeLessThan(0);
    expect(fast.total).toBeLessThan(matched.total);
    expect(fast).not.toHaveProperty('speed');
    expect(fast).not.toHaveProperty('referencePedal');
});

describe('speed gap over time', () => {
    const { observation } = trainingFixture();
    const frame = (speed: number, at = 1000, position = 0.4 + (at - 1000) / 100000) => ({
        ...observation, speed, normalizedPosition: position, capturedAt: at, measuredAt: at,
    });

    it.each([0.398, 0.4])('keeps at least the standstill speed cost without forward progress to %s', (position) => {
        const stopped = drivingReward(frame(0), frame(0, 1200, position));
        expect(stopped.timing).toBeCloseTo(-0.8);
        for (const speed of [60, 100]) {
            const approaching = drivingReward(frame(speed), frame(80, 1200, position));
            expect(approaching.speedComparison!.measured).toBe(80);
            expect(approaching.speedComparison!.speedGapTime).toBeCloseTo(16);
            expect(approaching.timing).toBeCloseTo(stopped.timing);
            expect(drivingReward(frame(speed), frame(80, 1200)).timing).toBeCloseTo(-0.1);
        }
        expect(drivingReward(frame(200), frame(200, 1200, position)).timing).toBeCloseTo(-1.2);
    });

    it('allows speed-matching credit across start/finish in the forward direction', () => {
        const forward = drivingReward(frame(120, 1000, 0.999), frame(120, 1200, 0.001));
        const backward = drivingReward(frame(120, 1000, 0.001), frame(120, 1200, 0.999));
        expect(forward.timing).toBeCloseTo(0);
        expect(backward.timing).toBeCloseTo(-1.2);
    });

    it.each(['acc', 'iracing'])('scores backward motion below standing still in %s', (game) => {
        const stopped = { ...frame(0), context: { ...observation.context, game }, offTrack: false,
            localVelocity: [0, 0, 0] as [number, number, number] };
        const next = { capturedAt: 1200, measuredAt: 1200 };
        const idle = drivingReward(stopped, { ...stopped, ...next });
        for (const speed of [2, 80]) {
            const localVelocity: [number, number, number] = game === 'iracing' ? [0, 0, -speed / 3.6] : [0, -speed / 3.6, 0];
            const backward = { ...stopped, speed, localVelocity, gear: 0 };
            const reverse = drivingReward(backward, { ...backward, ...next, normalizedPosition: 0.401 });
            expect(reverse.speedComparison!.measured).toBe(-speed);
            expect(reverse.speedComparison!.speedGapTime).toBeCloseTo((80 + speed) * 0.2);
            expect(reverse.timing).toBeLessThan(idle.timing);
            expect(reverse.progress).toBeCloseTo(idle.progress);
            expect(reverse.onTrack).toBe(0);
            expect(reverse.total).toBeLessThan(idle.total);
        }
    });

    it('uses the direction of each speed measurement across a gear change', () => {
        const before = { ...frame(10), gear: -1, localVelocity: [0, 0, 0] as [number, number, number] };
        const reward = drivingReward(before, frame(10, 1200));
        expect(reward.speedComparison!.measured).toBe(10);
        expect(reward.speedComparison!.speedGapTime).toBeCloseTo(16);
        expect(reward.speedComparison!.measuredAcceleration).toBeCloseTo(20 / (3.6 * 0.2));
    });

    it.each([-1, 1])('scales linearly with time and gap on side %s of the target', (side) => {
        const before = frame(80 + side * 20);
        const after = frame(before.speed, 1200);
        const later = frame(before.speed, 1400);
        const cost = drivingReward(before, after);
        expect(cost.speedComparison!.speedGapTime).toBeCloseTo(4);
        expect(cost.timing).toBeCloseTo(-0.2);
        expect(drivingReward(after, later).timing).toBeCloseTo(cost.timing);
        expect(drivingReward(before, later).timing).toBeCloseTo(cost.timing * 2);
        expect(drivingReward(before, { ...after, measuredAt: 1400 }).timing).toBeCloseTo(cost.timing * 2);
        expect(drivingReward(before, { ...after, capturedAt: 1400 }).timing).toBe(cost.timing);
        expect(drivingReward(frame(80 + side * 10), frame(80 + side * 10, 1200)).timing).toBeCloseTo(cost.timing / 2);
        expect(drivingReward(frame(80), frame(80, 1200)).timing).toBe(0);
        expect(drivingReward(before, { ...after, normalizedPosition: 0.41 }).timing).toBe(cost.timing);
    });

    it.each([-1, 1])('scores reaching the target early higher and leaving early lower on side %s', (side) => {
        const goal = 80, away = goal + side * 20;
        const accumulated = (first: number, middle: number, last: number) => {
            const start = frame(first), half = frame(middle, 1200), end = frame(last, 1400);
            return drivingReward(start, half).timing + drivingReward(half, end).timing;
        };
        expect(accumulated(away, goal, goal)).toBeGreaterThan(accumulated(away, away, goal));
        expect(accumulated(goal, away, away)).toBeLessThan(accumulated(goal, goal, away));
        expect(accumulated(goal, goal + side * 5, away)).toBeLessThan(accumulated(goal, goal + side, away));
        expect(accumulated(away, away, away)).toBeLessThan(accumulated(away, goal, goal));
    });

    it('accounts for both sides of a target crossing and preserves the cost when subdivided', () => {
        const before = frame(70), middle = frame(80, 1100), after = frame(100, 1300);
        const whole = drivingReward(before, after);
        expect(whole.speedComparison!.speedGapTime).toBeCloseTo(2.5);
        expect(whole.timing).toBeCloseTo(-0.125);
        expect(drivingReward(before, middle).timing + drivingReward(middle, after).timing).toBeCloseTo(whole.timing);
        expect(whole.timing).toBeLessThan(drivingReward(before, frame(80, 1300)).timing);
    });

    it('uses the saved weight, speed scale and total limit', () => {
        const before = frame(100), after = frame(100, 1200);
        const config = cloneConfig();
        config.reward.timingWeight *= 3;
        config.reward.speedTolerance *= 2;
        expect(drivingReward(before, after, config).timing).toBeCloseTo(-0.3);
        config.reward.totalLimit = 0.01;
        expect(drivingReward(before, after, config).total).toBe(-0.01);
        config.reward.timingWeight = 0;
        expect(drivingReward(before, after, config).timing).toBe(0);
    });
});

it('measures acceleration over the telemetry interval even when camera timing differs', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100, measuredAt: 900 };
    const after = { ...before, capturedAt: 1200, measuredAt: 1100, normalizedPosition: 0.252, speed: 99.2, controls: NEUTRAL };
    const reward = drivingReward(before, after);
    expect(reward.speedComparison!.measuredAcceleration).toBeCloseTo(-0.8 / (3.6 * 0.2));
    expect(reward.speedComparison!.targetAcceleration).toBeCloseTo(reward.speedComparison!.measuredAcceleration);
    const delayedCapture = drivingReward(before, { ...after, capturedAt: 1400 });
    expect(delayedCapture.speedComparison).toEqual(reward.speedComparison);
    expect(delayedCapture.timing).toBe(reward.timing);
});

it.each([900, 1000, NaN])('rejects repeated, reversed or invalid speed timestamps: %s', (measuredAt) => {
    const { observation } = trainingFixture();
    expect(() => drivingReward(observation, { ...observation, capturedAt: 1200, measuredAt })).toThrow(/telemetry timestamps/);
});

it('preserves the speed gap cost and progress when the same trajectory is sampled twice as often', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, normalizedPosition: 0.05, speed: 80 };
    const middle = { ...before, capturedAt: 1100, measuredAt: 1100, normalizedPosition: 0.051, speed: 80.5, controls: { ...before.controls, throttle: 0.4 } };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.052, speed: 81, controls: { ...before.controls, throttle: 0.5 } };
    const whole = drivingReward(before, after);
    const first = drivingReward(before, middle);
    const second = drivingReward(middle, after);
    expect(whole.timing).toBeCloseTo(-0.395);
    expect(first.timing + second.timing).toBeCloseTo(whole.timing);
    expect(first.progress + second.progress).toBeCloseTo(whole.progress);
});

it('does not reward parking or reverse progress and handles the lap boundary', () => {
    const { observation: before } = trainingFixture();
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, speed: 0 }).total).toBeLessThan(0);
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.249 }).total).toBeLessThan(0);
    const wrapped = drivingReward({ ...before, normalizedPosition: 0.999 },
        { ...before, normalizedPosition: 0.001, capturedAt: 1200, measuredAt: 1200, controls: NEUTRAL });
    expect(wrapped.timing).toBeLessThan(0); // Forward crossing still scores the outstanding speed gap.
    const reverseWrapped = drivingReward({ ...before, normalizedPosition: 0.001 },
        { ...before, normalizedPosition: 0.999, capturedAt: 1200, measuredAt: 1200, controls: NEUTRAL });
    expect(reverseWrapped.timing).toBeLessThan(0);
    expect(wrapped.progress).toBeGreaterThan(0);
    expect(reverseWrapped.progress).toBeCloseTo(-wrapped.progress);
    expect(reverseWrapped.total).toBeCloseTo(reverseWrapped.progress + reverseWrapped.timing + reverseWrapped.throttleChange);
    expect(() => drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.7 })).toThrow(/jumped/);
});

it.each([0, 50, 100])('keeps speed targets in diagnostics and aggression in the policy at %s', (aggression) => {
    const { live, vision, referenceLap, observation } = trainingFixture();
    const next = readDrivingObservation(live, vision, referenceLap, aggression, 1000, undefined, circuitMapFixture(referenceLap)).observation!;
    expect(next.features.slice(0, -3)).toEqual(observation.features.slice(0, -3));
    expect(next.features.slice(-3)[0]).toBe((aggression - 50) / 50);
    expect(next.features.slice(-2)).toEqual([0, 0]);
    expect(next.speedTargets).toEqual({ reference: 100, target: { 0: 100, 50: 100, 100: 150 }[aggression] });
    expect(next).not.toHaveProperty('pedals');
});

it('requires a usable selected reference for speed targets without a paired position estimate', () => {
    const { live, vision, referenceLap } = trainingFixture();
    expect(readDrivingObservation(live, vision, null, 50, 1000).error).toMatch(/reference lap/);
    expect(readDrivingObservation(live, vision, { ...referenceLap, target_features: [] }, 50, 1000, undefined, circuitMapFixture({ ...referenceLap, target_features: [] })).error).toMatch(/missing raw X coordinates, raw Y coordinates, raw Z coordinates, speed/);
    expect(readDrivingObservation(live, vision, { ...referenceLap, car: 'other' }, 50, 1000, undefined, circuitMapFixture({ ...referenceLap, car: 'other' })).error).toMatch(/does not match the current game, car or circuit/);
    expect(readDrivingObservation(live, vision, { ...referenceLap }, 50, 1000, undefined, circuitMapFixture({ ...referenceLap })).observation).toBeDefined();
    delete vision.position!.referenceLap;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, circuitMapFixture(referenceLap)).observation).toBeDefined();
});

it('explains missing speed even when Live Map can draw the reference line', () => {
    const { live, vision, referenceLap } = trainingFixture();
    const lap = { ...referenceLap, target_features: referenceLap.target_features.map((name) =>
        name === 'expert_optimal_speed' ? 'unrelated' : name) };
    vision.position!.referenceLap = lap;
    expect(getLiveMapReferenceLine(lap)).toHaveLength(lap.x.length);
    expect(readDrivingObservation(live, vision, lap, 50, 1000, undefined, circuitMapFixture(lap)).error).toContain('missing speed data');
});

it('allows a drawable reference lap with different recorded start and finish coordinates', () => {
    const { live, vision, referenceLap } = trainingFixture();
    const lap = { ...referenceLap, y: referenceLap.y.map((row) => [...row]) };
    lap.y[lap.y.length - 1][0] = 0.5;
    vision.position!.referenceLap = lap;
    expect(getLiveMapReferenceLine(lap)).toHaveLength(lap.x.length);
    expect(readDrivingObservation(live, vision, lap, 50, 1000, undefined, circuitMapFixture(lap)).observation).toBeDefined();
});

it('scores measured speed and acceleration independently of capped progress and pedals', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100 };
    const position = 0.28;
    const target = referenceSpeedTargets(before.reference, position, before.aggression).target;
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: position, speed: target, controls: NEUTRAL };
    const matched = drivingReward(before, after);
    const mismatched = drivingReward(before, { ...after, speed: before.speed });
    expect(matched.timing).toBeCloseTo(0);
    expect(matched.timing).toBeGreaterThan(mismatched.timing);
    expect(matched.speedComparison!.target).toEqual(target);
    expect(matched.speedComparison!.measured).toEqual(target);
    expect(matched.speedComparison!.measuredAcceleration).toBeCloseTo(-12 / (3.6 * 0.2));
    expect(matched.speedComparison!.targetAcceleration).toBeCloseTo(matched.speedComparison!.measuredAcceleration);
    const differentPedals = drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1, brake: 1 } });
    expect(differentPedals.timing).toBe(matched.timing);
    expect(differentPedals.speedComparison).toEqual(matched.speedComparison);
});

it.each([
    [0, 0.26, 0.27, 96, 92], [100, 0.26, 0.27, 144, 138],
    [0, 0.56, 0.57, 104, 108], [100, 0.56, 0.57, 156, 162],
])('rewards the requested acceleration/deceleration timing at aggression %s', (aggression, start, end, speed, nextSpeed) => {
    const { observation } = trainingFixture();
    const before = { ...observation, aggression, normalizedPosition: start, speed };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: end, speed: nextSpeed, controls: NEUTRAL };
    const matched = drivingReward(before, after);
    expect(matched.timing).toBeCloseTo(0);
    expect(matched.speedComparison!.speedGapTime).toBeCloseTo(0);
    // Matching only the end speed still leaves a gap at the start.
    expect(drivingReward({ ...before, speed: nextSpeed }, after).timing).toBeLessThan(0);
    expect(drivingReward({ ...before, speed: 2 * nextSpeed - speed }, after).timing).toBeLessThan(0);
    // Following the correct rate with a sustained gap still incurs a cost.
    const offset = nextSpeed < speed ? 40 : -40;
    expect(drivingReward({ ...before, speed: speed + offset }, { ...after, speed: nextSpeed + offset }).timing).toBeCloseTo(-0.4);
});

it.each([
    [0.19, 0.21, 120, 116, 116, 108, 120, 120],
    [0.49, 0.51, 80, 84, 84, 92, 80, 80],
])('distinguishes moving away, approaching and holding speed across reference speed change %s',
    (start, end, speed, nextSpeed, earlySpeed, earlyNext, lateSpeed, lateNext) => {
        const { observation } = trainingFixture();
        const before = { ...observation, normalizedPosition: start, speed };
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: end, speed: nextSpeed };
        const matched = drivingReward(before, after);
        expect(matched.timing).toBeCloseTo(0);
        expect(drivingReward({ ...before, speed: earlySpeed }, { ...after, speed: earlyNext }).timing).toBeLessThan(0);
        expect(drivingReward({ ...before, speed: lateSpeed }, { ...after, speed: lateNext }).timing).toBeLessThan(0);
    });

it('charges a smaller cost when approaching the higher aggressive target', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, aggression: 100, normalizedPosition: 0.46, speed: 116.5 };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.47, speed: 120 };
    const reward = drivingReward(before, after);
    expect(reward.speedComparison).toMatchObject({ reference: 80, target: 120, referenceAcceleration: 0, targetAcceleration: 0, measured: 120 });
    expect(reward.speedComparison!.measuredAcceleration).toBeGreaterThan(0);
    expect(reward.timing).toBeCloseTo(-0.0175);
    expect(drivingReward(before, { ...after, speed: before.speed }).timing).toBeLessThan(reward.timing);
    expect(drivingReward({ ...before, aggression: 50 }, after).timing).toBeLessThan(reward.timing);
});

it.each([
    { speed: 0, normalizedPosition: 0.252 }, { speed: 100, normalizedPosition: 0.25 },
    { speed: 100, normalizedPosition: 0.249 }, { speed: 100, normalizedPosition: 0.252, gear: -1 },
])('charges a speed gap while stationary or reversing: %j', (fields) => {
    const { observation: before } = trainingFixture();
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, controls: NEUTRAL, ...fields }).timing).toBeLessThan(0);
});


it.each([0, 50, 75, 100])('penalizes holding a speed above or below the aggression %s target', (aggression) => {
    const { observation } = trainingFixture();
    const position = 0.402;
    const target = referenceSpeedTargets(observation.reference, position, aggression).target;
    const score = (error: number) => {
        const before = { ...observation, aggression, normalizedPosition: 0.4, speed: target + error };
        return drivingReward(before, { ...before, normalizedPosition: position, capturedAt: 1200, measuredAt: 1200 });
    };
    expect(score(0).timing).toBe(0);
    expect(score(10).timing).toBeLessThan(0);
    expect(score(-10).timing).toBe(score(10).timing);
    expect(score(20).timing).toBeLessThan(score(10).timing);
    expect(score(10).total).toBeLessThan(score(0).total);
});


it.each([0, 25, 50])('uses Live Phrases targets for observation diagnostics and reward at aggression %s', (aggression) => {
    const { lap, map } = brakingReferenceFixture();
    const speedIndex = lap.target_features.indexOf('expert_optimal_speed');
    const i = 20;
    const observe = (index: number, now: number) => {
        const { live, vision } = trainingFixture(now, aggression, lap);
        live.currentTelemetry.Graphics_normalized_car_position = lap.x[index];
        live.currentTelemetry.Physics_speed_kmh = brakingFixtureSpeed(lap.x[index], aggression);
        return readDrivingObservation(live, vision, lap, aggression, now, cloneConfig(), map).observation!;
    };
    const before = observe(i - 1, 1000), after = observe(i, 1200);
    expect(after.features.slice(-2)).toEqual([0, 0]);
    expect(after.speedTargets?.reference).toBeCloseTo(lap.y[i][speedIndex]);
    expect(after.speedTargets?.target).toBeCloseTo(brakingFixtureSpeed(lap.x[i], aggression));
    expect(after.reference).toBe(before.reference);
    const reward = drivingReward(before, after);
    expect(reward.speedComparison!.target).toBeCloseTo(brakingFixtureSpeed(lap.x[i], aggression));
    expect(reward.speedComparison!.targetAcceleration).toBeCloseTo((brakingFixtureSpeed(lap.x[i], aggression) - brakingFixtureSpeed(lap.x[i - 1], aggression)) / 0.72);
    const held = { ...before, speed: after.speed };
    expect(reward.timing).toBeCloseTo(0);
    expect(drivingReward(held, after).timing).toBeCloseTo(-Math.abs(after.speed - before.speed) / 200);
    expect(drivingReward({ ...held, speed: after.speed + 10 }, { ...after, speed: after.speed + 10 }).timing).toBeLessThan(0);
});
