import { drivingReward, NEUTRAL, readDrivingObservation } from './driving-observation';
import { trainingFixture } from './training-fixtures';
import { referenceSpeedTargets } from './driving-reference';
import { getLiveMapReferenceLine } from 'views/live-session/live-map-reference';
import { DrivingMemory, drivingMemoryInputs } from './driving-memory';
import { cloneConfig } from './driving-model-config';
import { IMPACT_PENALTY_WEIGHT } from './driving-impact';

const { AccAdapter } = require('../../../electron/recording/readers/acc/acc-adapter');

it.each(['missing', 'distant', 'invalid', 'mismatched'])(
    'ignores %s position-model output in observations, memory and rewards', (kind) => {
        const { live, vision, referenceLap, observation: before } = trainingFixture();
        const next = trainingFixture(1200, 50, referenceLap);
        next.live.currentTelemetry.Graphics_normalized_car_position = 0.252;
        next.live.currentTelemetry.Physics_speed_kmh = 106;
        next.vision.position!.normalizedPosition = 0.252;
        const expected = readDrivingObservation(next.live, next.vision, referenceLap, 50, 1200).observation!;
        for (const frame of [vision, next.vision]) {
            if (kind === 'missing') delete frame.position;
            if (kind === 'distant') frame.position!.referenceDelta = frame === vision ? [30, 40, 0] : [-90, 0, 0];
            if (kind === 'invalid') frame.position!.referenceDelta = [NaN, Infinity, -Infinity];
            if (kind === 'mismatched') Object.assign(frame.position!, { coordinateFrame: 'world', normalizedPosition: 0.75, referenceLap: null });
        }
        const start = readDrivingObservation(live, vision, referenceLap, 50, 1000).observation!;
        const after = readDrivingObservation(next.live, next.vision, referenceLap, 50, 1200).observation!;
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
        expect(reward.timing).toBeCloseTo(0.2 * 2 / (3.6 * 0.2 * 3));
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress + reward.impact);
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
        const before = readDrivingObservation(fixture.live, fixture.vision, fixture.referenceLap, 50, 1000).observation!;
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
    expect(observation.features).toHaveLength(66);
    expect(observation).toMatchObject({ gear: 3, rpm: 6000 });
    expect(observation.features.slice(57, 59)).toEqual([3 / 8, 6000 / 20000]);
    expect(observation.camera).toBe(vision.camera);
    expect(observation.camera).toMatchObject({ width: 384, height: 216 });
    expect(observation.context).not.toHaveProperty('classes');
    // Later telemetry must not replace the sample paired with the image.
    expect(readDrivingObservation({ ...live, currentTelemetry: { ...live.currentTelemetry, Physics_speed_kmh: 200 } }, vision, referenceLap, 50, 1000)
        .observation?.speed).toBe(108);
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
    const result = readDrivingObservation(live, vision, referenceLap, 50, 1000);
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
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/camera pixels/);
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
    const observation = readDrivingObservation(later, vision, referenceLap, 50, 1000).observation!;
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
    const observation = readDrivingObservation(live, vision, referenceLap, 50, 1000, config).observation!;
    expect(observation.gForce).toEqual([-8, 1, 6]);
    expect(observation.features.slice(60, 63)).toEqual([-1, 0.25, 1]);
});

describe.each(['x', 'y', 'z'])('g-force %s validation', (axis) => {
    it.each([undefined, NaN, Infinity, -Infinity, '0'])('rejects invalid g-force %s', (value) => {
        const { live, vision, referenceLap } = trainingFixture();
        Object.assign(vision.telemetry!.currentTelemetry, { [`Physics_g_force_${axis}`]: value });
        expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/g-force XYZ/);
    });
});

it('ignores world-space velocity when building policy inputs', () => {
    const { live, vision, referenceLap, observation: before } = trainingFixture();
    // The fixture has local velocity only; world-space telemetry is optional and unused.
    expect(before.localVelocity).toEqual([1, 30, 0]);
    Object.assign(vision.telemetry!.currentTelemetry, {
        Physics_velocity_x: -30, Physics_velocity_y: NaN, Physics_velocity_z: Infinity,
    });
    const after = readDrivingObservation(live, vision, referenceLap, 50, 1000).observation!;
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
        expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/Local velocity/);
    });
});

it('ignores orientation angles in both policy inputs and the reward', () => {
    const { live, vision, referenceLap, observation: before } = trainingFixture();
    Object.assign(vision.telemetry!.currentTelemetry, { Physics_heading: 1.2, Physics_pitch: -0.2, Physics_roll: 0.3 });
    const after = readDrivingObservation(live, vision, referenceLap, 50, 1000).observation!;
    expect(after.features).toEqual(before.features);
    const stable = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200 });
    const rotated = drivingReward(before, { ...after, capturedAt: 1200, measuredAt: 1200 });
    expect(rotated).toEqual(stable);
    Object.assign(vision.telemetry!.currentTelemetry, { Physics_heading: undefined, Physics_pitch: NaN, Physics_roll: undefined });
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).observation).toEqual(after);
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
        const observation = readDrivingObservation(live, vision, referenceLap, 50, 1000).observation!;
        expect(observation.yawRate).toBe(rate);
        expect(observation.features[59]).toBe(Math.sign(rate));
        expect(observation).not.toHaveProperty('rotationRates');
    });

    it.each([undefined, NaN, Infinity, -Infinity, '0'])('rejects invalid yaw rate %s', (rate) => {
        const { live, vision, referenceLap } = fixture();
        Object.assign(vision.telemetry!.currentTelemetry, { [`Physics_local_angular_vel_${game === 'iracing' ? 'y' : 'z'}`]: rate });
        expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/yaw rate/);
    });

    it.each([undefined, NaN, Infinity, -4, 4])('omits pitch and roll rates even when they are %s', (rate) => {
        const { live, vision, referenceLap } = fixture();
        const before = readDrivingObservation(live, vision, referenceLap, 50, 1000).observation!;
        Object.assign(vision.telemetry!.currentTelemetry, {
            Physics_local_angular_vel_x: rate,
            [`Physics_local_angular_vel_${game === 'iracing' ? 'z' : 'y'}`]: rate,
        });
        const after = readDrivingObservation(live, vision, referenceLap, 50, 1000).observation!;
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
        if (fault === 'nonfinite') live.currentTelemetry.Graphics_normalized_car_position = NaN;
        expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toBeTruthy();
    },
);

it.each([0, 1, 2, 4])('normalizes standard gear %s and keeps RPM paired with the image', (gear) => {
    const { live, vision, referenceLap } = trainingFixture();
    vision.telemetry!.currentTelemetry = { ...live.currentTelemetry, Physics_gear: gear, Physics_rpm: 4500 };
    const result = readDrivingObservation({ ...live, currentTelemetry: { ...live.currentTelemetry, Physics_gear: 6, Physics_rpm: 8000 } }, vision, referenceLap, 50, 1000);
    expect(result.observation).toMatchObject({ gear: gear - 1, rpm: 4500 });
});

it.each([
    { Physics_gear: 0, Physics_rpm: 0 }, { Physics_gear: 1, Physics_rpm: 0 },
    { Physics_gear: 2, Physics_rpm: 0 }, { Physics_gear: -1, Physics_rpm: -1 },
    { Physics_gear: 2.5, Physics_rpm: 12.5 },
])('accepts stalled and nonstandard gear/RPM telemetry %j', (fields) => {
    const { live, vision, referenceLap } = trainingFixture();
    Object.assign(live.currentTelemetry, fields);
    const result = readDrivingObservation(live, vision, referenceLap, 50, 1000);
    expect(result.error).toBeUndefined();
    expect(result.observation).toMatchObject({ gear: fields.Physics_gear - 1, rpm: fields.Physics_rpm });
    expect(new DrivingMemory().encode(result.observation!).features.every(Number.isFinite)).toBe(true);
});

describe.each(['Physics_gear', 'Physics_rpm'])('unavailable %s telemetry', (field) => {
    it.each([undefined, null, NaN, Infinity, -Infinity])('keeps policy inputs usable when the reading is %s', (value) => {
        const { live, vision, referenceLap } = trainingFixture();
        Object.assign(live.currentTelemetry, { [field]: value });
        const result = readDrivingObservation(live, vision, referenceLap, 50, 1000);
        expect(result.error).toBeUndefined();
        const gear = field === 'Physics_gear' ? 0 : 3;
        const rpm = field === 'Physics_rpm' ? 0 : 6000;
        expect(result.observation).toMatchObject({ gear, rpm });
        expect(result.observation!.features.slice(57, 59)).toEqual([gear / 8, rpm / 20000]);
        expect(new DrivingMemory().encode(result.observation!).features.every(Number.isFinite)).toBe(true);
    });
});

it('requires automatic shifting to be disabled when reported by the simulator', () => {
    const { live, vision, referenceLap } = trainingFixture();
    live.currentTelemetry.Physics_autoshifter_on = true;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/Disable automatic gear shifting/);
    live.currentTelemetry.Physics_autoshifter_on = false;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).observation).toBeDefined();
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
                    normalizedPosition: before.normalizedPosition + change };
                const reward = drivingReward(before, after, config);
                expect(reward.progress).toBeGreaterThan(0);
                expect(reward.progress).toBeLessThan(0.25 * progressWeight * Math.min(change, config.reward.progressLimit));
            }
        }
    });

    it.each([
        { normalizedPosition: 0.25 },
        { normalizedPosition: 0.25, speed: 0 },
        { normalizedPosition: 0.252, speed: 0 },
    ])('penalizes standing still by elapsed telemetry time: %j', (fields) => {
        const before = trainingFixture().observation;
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, ...fields };
        const reward = drivingReward(before, after);
        expect(reward.progress).toBeCloseTo(-0.05);
        expect(reward.total).toBe(reward.progress);
        expect(drivingReward(before, { ...after, capturedAt: 1400 }).progress).toBe(reward.progress);
        expect(drivingReward(before, { ...after, capturedAt: 1400, measuredAt: 1400 }).progress)
            .toBeCloseTo(reward.progress * 2);
    });

    it('withholds a forward-progress bonus in reverse gear', () => {
        const before = trainingFixture().observation;
        expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200,
            normalizedPosition: 0.252, gear: -1 }).progress).toBe(0);
    });

    it.each([-1, 0, 3])('penalizes negative progress regardless of gear %s', (gear) => {
        const before = trainingFixture().observation;
        const reward = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.248, gear });
        expect(reward.progress).toBeLessThan(0);
        expect(reward.total).toBe(reward.progress);
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
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress);
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
])('excludes overlapping throttle and brake from reward at orientation %j', (orientation) => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, ...orientation };
    const throttleOnly = drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1 } });
    const overlap = drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1, brake: 1 } });
    // No forward progress means the timing reward is zero.
    expect(overlap.total).toEqual(throttleOnly.total);
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
        if (speed === 99.2) expect(reward.timing).toBeCloseTo(0.2 * 0.8 / (3.6 * 0.2 * 3));
        else expect(reward.timing).toBeLessThan(0);
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress);
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
    expect(drivingReward(before, after).timing).toBeCloseTo(0.2 * 0.8 / (3.6 * 0.2 * 3));
    expect(drivingReward(before, { ...after, speed: 200 }).timing).toBeLessThan(0);
});

it('rewards the reference speed profile instead of simply driving faster', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100 };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, controls: NEUTRAL };
    const matched = drivingReward(before, { ...after, speed: 99.2 });
    const fast = drivingReward(before, { ...after, speed: 200 });
    expect(matched.timing).toBeCloseTo(0.2 * 0.8 / (3.6 * 0.2 * 3));
    expect(fast.timing).toBeLessThan(0);
    expect(fast.total).toBeLessThan(matched.total);
    expect(fast).not.toHaveProperty('speed');
    expect(fast).not.toHaveProperty('referencePedal');
});

describe('speed goal progress reward', () => {
    const observation = trainingFixture().observation;
    const score = (speed: number, nextSpeed: number, position = 0.252, aggression = 50) => drivingReward(
        { ...observation, speed, aggression },
        { ...observation, speed: nextSpeed, normalizedPosition: position,
            capturedAt: 1200, measuredAt: 1200 });

    it.each([[40, 42], [160, 158]])('rewards correction from %s to %s even far outside the old speed tolerance', (speed, nextSpeed) => {
        const reward = score(speed, nextSpeed);
        expect(reward.speedComparison.speedGoalProgress).toBeCloseTo(2);
        expect(reward.timing).toBeCloseTo(0.2 * 2 / (3.6 * 0.2 * 3));
        expect(reward.timing).toBeGreaterThan(score(99.2, 99.2).timing);
        expect(score(nextSpeed, speed).timing).toBeCloseTo(-reward.timing);
    });

    it.each([40, 99.2, 160])('gives no bonus for constant speed %s, including when the target moves toward the car', (speed) => {
        for (const position of [0.252, 0.28]) {
            expect(score(speed, speed, position).timing).toBe(0);
            expect(score(speed, speed, position).speedComparison.speedGoalProgress).toBe(0);
        }
    });

    it('rewards more correction, caps its magnitude, and charges for overshooting', () => {
        // At position 0.4 the goal is 80 km/h, independent of aggression.
        const before = { ...observation, speed: 79, normalizedPosition: 0.4 };
        const reward = (speed: number) => drivingReward(before,
            { ...before, speed, normalizedPosition: 0.402, capturedAt: 1200, measuredAt: 1200 });
        expect(reward(79.5).timing).toBeGreaterThan(0);
        expect(reward(80).timing).toBeGreaterThan(reward(79.5).timing);
        expect(reward(80.5).timing).toBeCloseTo(reward(79.5).timing);
        expect(reward(81).timing).toBe(0);
        expect(reward(82).timing).toBeLessThan(0);
        expect(score(40, 90).timing).toBeCloseTo(0.2);
        expect(score(40, 10).timing).toBeCloseTo(-0.2);
    });

    it.each([[0, 1], [100, -1]])('uses aggression %s to determine the useful speed-change direction', (aggression, sign) => {
        // 108 km/h is above the cautious goal and below the aggressive goal.
        expect(Math.sign(score(108, 106, 0.252, aggression).timing)).toBe(sign);
        expect(Math.sign(score(108, 110, 0.252, aggression).timing)).toBe(-sign);
    });

    it('keeps reward scaled by telemetry time and the configured correction rate', () => {
        const before = { ...observation, speed: 40 };
        const after = { ...before, speed: 41, normalizedPosition: 0.252, capturedAt: 1200, measuredAt: 1200 };
        const config = cloneConfig();
        config.reward.accelerationTolerance *= 2;
        const normal = drivingReward(before, after).timing;
        expect(drivingReward(before, after, config).timing).toBeCloseTo(normal / 2);
        expect(drivingReward(before, { ...after, capturedAt: 1400, measuredAt: 1400 }).timing).toBeCloseTo(normal / 2);
    });

    it.each([40, 160])('caps total reward with saved settings for speed %s', (speed) => {
        const config = cloneConfig();
        config.reward.timingWeight = 1000;
        config.reward.progressWeight = 200;
        config.reward.totalLimit = 1;
        const before = { ...observation, speed: 100 };
        const after = { ...before, speed, normalizedPosition: 0.28, capturedAt: 1200, measuredAt: 1200 };
        const reward = drivingReward(before, after, config);
        expect(reward.timing).toBe(-20);
        expect(reward.total).toBe(-1);
        expect(drivingReward(before, { ...after, speed: 88 }, config).total).toBe(1);
    });
});

it('measures acceleration over the telemetry interval even when camera timing differs', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100, measuredAt: 900 };
    const after = { ...before, capturedAt: 1200, measuredAt: 1100, normalizedPosition: 0.252, speed: 99.2, controls: NEUTRAL };
    const reward = drivingReward(before, after);
    expect(reward.speedComparison.measuredAcceleration).toBeCloseTo(-0.8 / (3.6 * 0.2));
    expect(reward.speedComparison.targetAcceleration).toBeCloseTo(reward.speedComparison.measuredAcceleration);
    const delayedCapture = drivingReward(before, { ...after, capturedAt: 1400 });
    expect(delayedCapture.speedComparison).toEqual(reward.speedComparison);
    expect(delayedCapture.timing).toBe(reward.timing);
});

it.each([900, 1000, NaN])('rejects repeated, reversed or invalid speed timestamps: %s', (measuredAt) => {
    const { observation } = trainingFixture();
    expect(() => drivingReward(observation, { ...observation, capturedAt: 1200, measuredAt })).toThrow(/telemetry timestamps/);
});

it('gives the same total reward while changing throttle at different capture intervals', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, normalizedPosition: 0.05, speed: 80 };
    const middle = { ...before, capturedAt: 1100, measuredAt: 1100, normalizedPosition: 0.051, speed: 80.5, controls: { ...before.controls, throttle: 0.4 } };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.052, speed: 81, controls: { ...before.controls, throttle: 0.5 } };
    const whole = drivingReward(before, after);
    const first = drivingReward(before, middle);
    const second = drivingReward(middle, after);
    expect(whole.timing).toBeGreaterThan(0);
    expect(first.timing + second.timing).toBeCloseTo(whole.timing);
    expect(first.total + second.total).toBeCloseTo(whole.total);
});

it.each([
    { speed: 0, normalizedPosition: 0.252 },
    { speed: 200, normalizedPosition: 0.25 },
    { speed: 200, normalizedPosition: 0.249 },
    { speed: 200, normalizedPosition: 0.252, gear: -1 },
])('withholds timing rewards without forward driving: %j', (current) => {
    const { observation: before } = trainingFixture();
    const reward = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, controls: NEUTRAL, ...current });
    expect(reward.timing).toBe(0);
});

it('does not reward parking or reverse progress and handles the lap boundary', () => {
    const { observation: before } = trainingFixture();
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, speed: 0 }).total).toBeLessThan(0);
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.249 }).total).toBeLessThan(0);
    const wrapped = drivingReward({ ...before, normalizedPosition: 0.999 },
        { ...before, normalizedPosition: 0.001, capturedAt: 1200, measuredAt: 1200, controls: NEUTRAL });
    expect(Number.isFinite(wrapped.timing)).toBe(true);
    const reverseWrapped = drivingReward({ ...before, normalizedPosition: 0.001 },
        { ...before, normalizedPosition: 0.999, capturedAt: 1200, measuredAt: 1200, controls: NEUTRAL });
    expect(reverseWrapped.timing).toBe(0);
    expect(wrapped.progress).toBeGreaterThan(0);
    expect(reverseWrapped.progress).toBeCloseTo(-wrapped.progress);
    expect(reverseWrapped.total).toBe(reverseWrapped.progress);
    expect(() => drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.7 })).toThrow(/jumped/);
});

it.each([0, 50, 100])('appends aggression and targets without changing the existing features at %s', (aggression) => {
    const { live, vision, referenceLap, observation } = trainingFixture();
    const next = readDrivingObservation(live, vision, referenceLap, aggression, 1000).observation!;
    expect(next.features.slice(0, -3)).toEqual(observation.features.slice(0, -3));
    expect(next.features.slice(-3)[0]).toBe((aggression - 50) / 50);
    expect(next.features.slice(-2)[0]).toBeCloseTo(100 / 360);
    expect(next.features.slice(-2)[1]).toBeCloseTo(({ 0: 90, 50: 100, 100: 120 }[aggression]!) / 360);
    expect(next).not.toHaveProperty('pedals');
});

it('requires a usable selected reference for speed targets without a paired position estimate', () => {
    const { live, vision, referenceLap } = trainingFixture();
    expect(readDrivingObservation(live, vision, null, 50, 1000).error).toMatch(/reference lap/);
    expect(readDrivingObservation(live, vision, { ...referenceLap, target_features: [] }, 50, 1000).error).toMatch(/missing raw X coordinates, raw Y coordinates, raw Z coordinates, speed/);
    expect(readDrivingObservation(live, vision, { ...referenceLap, car: 'other' }, 50, 1000).error).toMatch(/does not match the current game, car or circuit/);
    expect(readDrivingObservation(live, vision, { ...referenceLap }, 50, 1000).observation).toBeDefined();
    delete vision.position!.referenceLap;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).observation).toBeDefined();
});

it('explains missing speed even when Live Map can draw the reference line', () => {
    const { live, vision, referenceLap } = trainingFixture();
    const lap = { ...referenceLap, target_features: referenceLap.target_features.map((name) =>
        name === 'expert_optimal_speed' ? 'unrelated' : name) };
    vision.position!.referenceLap = lap;
    expect(getLiveMapReferenceLine(lap)).toHaveLength(lap.x.length);
    expect(readDrivingObservation(live, vision, lap, 50, 1000).error).toContain('missing speed data');
});

it('allows a drawable reference lap with different recorded start and finish coordinates', () => {
    const { live, vision, referenceLap } = trainingFixture();
    const lap = { ...referenceLap, y: referenceLap.y.map((row) => [...row]) };
    lap.y[lap.y.length - 1][0] = 0.5;
    vision.position!.referenceLap = lap;
    expect(getLiveMapReferenceLine(lap)).toHaveLength(lap.x.length);
    expect(readDrivingObservation(live, vision, lap, 50, 1000).observation).toBeDefined();
});

it('scores measured speed and acceleration with capped progress regardless of pedals', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100 };
    const position = 0.28;
    const target = referenceSpeedTargets(before.reference, position, before.aggression).target;
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: position, speed: target, controls: NEUTRAL };
    const matched = drivingReward(before, after);
    const mismatched = drivingReward(before, { ...after, speed: before.speed });
    expect(matched.timing).toBeCloseTo(100 * 0.02);
    expect(matched.timing).toBeGreaterThan(mismatched.timing);
    expect(matched.speedComparison.target).toEqual(target);
    expect(matched.speedComparison.measured).toEqual(target);
    expect(matched.speedComparison.measuredAcceleration).toBeCloseTo(-12 / (3.6 * 0.2));
    expect(matched.speedComparison.targetAcceleration).toBeCloseTo(matched.speedComparison.measuredAcceleration);
    expect(drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1, brake: 1 } })).toEqual(matched);
});

it.each([
    [0, 0.16, 0.17, 117, 114], [100, 0.26, 0.27, 114, 108],
    [0, 0.56, 0.57, 254 / 3, 268 / 3], [100, 0.46, 0.47, 83.5, 87],
])('rewards the requested acceleration/deceleration timing at aggression %s', (aggression, start, end, speed, nextSpeed) => {
    const { observation } = trainingFixture();
    const before = { ...observation, aggression, normalizedPosition: start, speed };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: end, speed: nextSpeed, controls: NEUTRAL };
    const matched = drivingReward(before, after);
    expect(matched.timing).toBeCloseTo(1);
    expect(matched.speedComparison.speedGoalProgress).toBeCloseTo(Math.abs(nextSpeed - speed));
    // Reaching the goal from either side is useful; holding at it earns no bonus.
    expect(drivingReward({ ...before, speed: nextSpeed }, after).timing).toBe(0);
    expect(drivingReward({ ...before, speed: 2 * nextSpeed - speed }, after).timing).toBeCloseTo(matched.timing);
    // Useful changes still earn credit while far from the target.
    const offset = nextSpeed < speed ? 40 : -40;
    expect(drivingReward({ ...before, speed: speed + offset }, { ...after, speed: nextSpeed + offset }).timing).toBeCloseTo(matched.timing);
});

it.each([
    [0.19, 0.21, 120, 116, 116, 108, 120, 120],
    [0.49, 0.51, 80, 84, 84, 92, 80, 80],
])('distinguishes early, matching and late speed responses at reference pedal event %s',
    (start, end, speed, nextSpeed, earlySpeed, earlyNext, lateSpeed, lateNext) => {
        const { observation } = trainingFixture();
        const before = { ...observation, normalizedPosition: start, speed };
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: end, speed: nextSpeed };
        const matched = drivingReward(before, after);
        expect(matched.timing).toBeCloseTo(2);
        expect(drivingReward({ ...before, speed: earlySpeed }, { ...after, speed: earlyNext }).timing).toBeLessThan(0);
        expect(drivingReward({ ...before, speed: lateSpeed }, { ...after, speed: lateNext }).timing).toBe(0);
    });

it('checks higher and increasing speed before reference throttle pickup against the aggressive target', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, aggression: 100, normalizedPosition: 0.46, speed: 83.5 };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.47, speed: 87 };
    const reward = drivingReward(before, after);
    expect(reward.speedComparison).toMatchObject({ reference: 80, referenceAcceleration: 0, measured: 87 });
    expect(reward.speedComparison.measuredAcceleration).toBeGreaterThan(0);
    expect(reward.timing).toBeCloseTo(1);
    expect(drivingReward({ ...before, aggression: 50 }, after).timing).toBeLessThan(0);
});

it.each([
    { speed: 0, normalizedPosition: 0.252 }, { speed: 100, normalizedPosition: 0.25 },
    { speed: 100, normalizedPosition: 0.249 }, { speed: 100, normalizedPosition: 0.252, gear: -1 },
])('cannot farm timing reward while stationary or reversing: %j', (fields) => {
    const { observation: before } = trainingFixture();
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, controls: NEUTRAL, ...fields }).timing).toBe(0);
});
