import { DrivingLineProximity, drivingReward, DrivingYawInstability, NEUTRAL, readDrivingObservation } from './driving-observation';
import { trainingFixture } from './training-fixtures';
import { referenceSpeedTargets } from './driving-reference';
import { getLiveMapReferenceLine } from 'views/live-session/live-map-reference';
import { DrivingMemory, drivingMemoryInputs } from './driving-memory';
import { cloneConfig } from './driving-model-config';

const { AccAdapter } = require('../../../electron/recording/readers/acc/acc-adapter');

it('uses telemetry and the full-resolution camera from the capture time', () => {
    const { live, vision, referenceLap, observation } = trainingFixture();
    expect(observation.features).toHaveLength(66);
    expect(observation).toMatchObject({ gear: 3, rpm: 6000 });
    expect(observation.features.slice(60, 62)).toEqual([3 / 8, 6000 / 20000]);
    expect(observation.camera).toBe(vision.camera);
    expect(observation.camera).toMatchObject({ width: 768, height: 768 });
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
            heading: Math.PI / 2, pitch: -0.2, roll: 0.3,
            local_angular_vel: { x: -Math.PI / 2, y: -Math.PI / 8, z: Math.PI / 4 },
        },
    } }));
    const later = { ...live, currentTelemetry: { ...live.currentTelemetry,
        Physics_local_velocity_x: 90, Physics_heading: 0, Physics_local_angular_vel_z: 1 } };
    const observation = readDrivingObservation(later, vision, referenceLap, 50, 1000).observation!;
    expect(observation).toMatchObject({ localVelocity: [-12, 40, 2.5],
        yawRate: Math.PI / 8 });
    expect(observation.features.slice(0, 3)).toEqual([-0.12, 0.4, 0.025]);
    expect(observation.features[62]).toBe(0.125);
    const memory = new DrivingMemory();
    memory.encode(observation);
    const encoded = memory.encode({ ...observation, capturedAt: 1200, measuredAt: 1200 }).features;
    expect(encoded.slice(2, 5)).toEqual([-0.12, 0.4, 0.025]);
    expect(encoded[2 + 62]).toBe(0.125);
    expect(encoded.slice(drivingMemoryInputs())).toEqual(observation.features);
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

it('ignores orientation angles in both policy inputs and the instability reward', () => {
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
        expect(observation.features[62]).toBe(Math.sign(rate));
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

it.each([0, Math.PI / 2, -Math.PI / 2, Math.PI])('uses local offsets directly at heading %s', (heading) => {
    const { live, vision, referenceLap, observation: before } = trainingFixture();
    Object.assign(vision.telemetry!.currentTelemetry, { Physics_heading: heading, Physics_pitch: 0.2, Physics_roll: -0.1 });
    vision.position!.referenceDelta = [-3, 4, -12];
    const observation = readDrivingObservation(live, vision, referenceLap, 50, 1000).observation!;
    // These feature slots retain right/forward/up signs, without a second rotation or world translation.
    expect(observation.features.slice(57, 60)).toEqual([-3 / 20, 4 / 20, -12 / 20]);
    expect(observation.referenceError).toEqual([-3, 4]);
    expect(observation.referenceDistance).toBe(13);
    const after = { ...observation, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, controls: NEUTRAL };
    const reward = drivingReward(before, after);
    expect(reward.timing).toBeCloseTo(drivingReward(before, { ...after, referenceDistance: 0 }).timing * Math.exp(-13 / 3));
    const smoothedDistance = 13 + (before.referenceDistance - 13) * Math.exp(-200 / 500);
    expect(reward.line).toBeCloseTo(0.2 * (2 / (1 + (smoothedDistance / 10) ** 2) - 1));
});

it.each([undefined, 'world'])('rejects a position result without car-local semantics (%s)', (coordinateFrame) => {
    const { live, vision, referenceLap } = trainingFixture();
    Object.assign(vision.position!, { coordinateFrame });
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/car-local/);
});

it.each(['telemetry', 'capture', 'session', 'position', 'velocity', 'nonfinite'])(
    'rejects unusable %s instead of training on fabricated defaults', (fault) => {
        const { live, vision, referenceLap } = trainingFixture();
        if (fault === 'telemetry') live.receivedAt = 0;
        if (fault === 'capture') vision.capturedAt = -1000;
        if (fault === 'session') live.streamGeneration++;
        if (fault === 'position') delete vision.position;
        if (fault === 'velocity') delete live.currentTelemetry.Physics_local_velocity_x;
        if (fault === 'nonfinite') vision.position!.referenceDelta[0] = NaN;
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
        expect(result.observation!.features.slice(60, 62)).toEqual([gear / 8, rpm / 20000]);
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

it('rejects mismatched position estimates', () => {
    const { live, vision, referenceLap } = trainingFixture();
    vision.position!.normalizedPosition = 0.3;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/matching/);
});

it('rewards forward reference tracking', () => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, speed: 106, referenceDistance: 1, controls: NEUTRAL };
    const smooth = drivingReward(before, after);
    expect(smooth.timing).toBeGreaterThan(0);
    expect(smooth.line).toBeGreaterThan(0);
    const distant = drivingReward(before, { ...after, referenceDistance: 8 });
    expect(distant.timing).toBeLessThan(smooth.timing);
    expect(distant.total).toBeLessThan(smooth.total);
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

    it.each([20, 100, 500])('stays below a quarter of ideal line reward with line weight %s', (lineWeight) => {
        const config = cloneConfig();
        config.reward.lineWeight = lineWeight;
        const before = { ...trainingFixture().observation, referenceDistance: 0 };
        for (const elapsed of [1, 100, 1000]) {
            for (const change of [0.0001, 0.002, 0.05]) {
                const after = { ...before, capturedAt: 1000 + elapsed, measuredAt: 1000 + elapsed,
                    normalizedPosition: before.normalizedPosition + change };
                const reward = drivingReward(before, after, undefined, config);
                expect(reward.progress).toBeGreaterThan(0);
                expect(reward.progress).toBeLessThan(0.25 * reward.line);
                const distant = drivingReward({ ...before, referenceDistance: 100 },
                    { ...after, referenceDistance: 100 }, undefined, config);
                expect(distant.progress).toBe(reward.progress);
                expect(distant.total).toBeLessThan(0); // Progress cannot cancel the distant-line penalty.
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

describe('smoothed line reward', () => {
    it.each([[0, 1], [1, 99 / 101], [10, 0], [20, -0.6], [100, -99 / 101]])(
        'rewards close tracking and penalizes distant tracking at a steady distance of %s', (referenceDistance, factor) => {
            const before = { ...trainingFixture().observation, referenceDistance };
            const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252 };
            expect(drivingReward(before, after).line).toBeCloseTo(0.2 * factor, 10);
            expect(drivingReward(before, { ...after, normalizedPosition: 0.30 }).line).toBeCloseTo(2 * factor, 10);
        });

    it('turns line reward negative when forward driving moves away beyond the smoothed tolerance', () => {
        const history = new DrivingLineProximity();
        let previous = { ...trainingFixture().observation, referenceDistance: 8 };
        const rewards = [12, 16, 20, 24].map((referenceDistance) => {
            const current = { ...previous, capturedAt: previous.capturedAt + 200, measuredAt: previous.measuredAt + 200,
                normalizedPosition: previous.normalizedPosition + 0.002, referenceDistance };
            const reward = drivingReward(previous, current, undefined, undefined, history);
            previous = current;
            return reward.line;
        });
        expect(rewards[0]).toBeGreaterThan(0); // Smoothing tolerates the first small departure.
        expect(rewards.slice(1).every((line) => line < 0)).toBe(true);
        expect(rewards.every((line, i) => i === 0 || line < rewards[i - 1])).toBe(true);
    });

    it('does not penalize stationary or reverse driving solely for being far from the line', () => {
        const before = { ...trainingFixture().observation, referenceDistance: 100 };
        for (const fields of [{ normalizedPosition: 0.25 }, { normalizedPosition: 0.249 },
            { normalizedPosition: 0.252, speed: 0 }, { normalizedPosition: 0.252, gear: -1 }]) {
            const after = { ...before, capturedAt: 1200, measuredAt: 1200, ...fields };
            expect(drivingReward(before, after).line).toBe(0);
        }
    });

    it('damps one-frame noise, follows sustained distance changes, and recovers smoothly', () => {
        const config = cloneConfig();
        config.reward.lineDistanceTolerance = 8;
        config.reward.lineSmoothingMs = 400;
        const history = new DrivingLineProximity(config);
        let previous = { ...trainingFixture().observation, referenceDistance: 0 };
        const score = (referenceDistance: number) => {
            const current = { ...previous, capturedAt: previous.capturedAt + 200, referenceDistance };
            const proximity = history.proximity(previous, current);
            previous = current;
            return proximity;
        };
        const spike = score(8);
        expect(spike).toBeGreaterThan(0.85); // Raw distance would immediately halve proximity.
        expect(spike).toBeLessThan(1);
        const recovered = score(0);
        expect(recovered).toBeGreaterThan(spike);
        expect(recovered).toBeLessThan(1);
        const sustained = Array.from({ length: 20 }, () => score(8));
        expect(sustained.every((value, i) => i === 0 || value < sustained[i - 1])).toBe(true);
        expect(sustained[sustained.length - 1]).toBeCloseTo(0.5, 4);
        history.reset();
        expect(score(8)).toBeCloseTo(0.5, 10);
    });

    it('has the same smoothing response over equal elapsed time at different capture rates', () => {
        const before = { ...trainingFixture().observation, referenceDistance: 0 };
        const after = { ...before, capturedAt: 1400, referenceDistance: 10 };
        const whole = new DrivingLineProximity().proximity(before, after);
        const history = new DrivingLineProximity();
        let previous = before;
        let subdivided = 0;
        for (const capturedAt of [1100, 1200, 1300, 1400]) {
            const current = { ...after, capturedAt };
            subdivided = history.proximity(previous, current);
            previous = current;
        }
        expect(subdivided).toBeCloseTo(whole, 12);
    });
});

describe('yaw instability reward', () => {
    function rewards(rates: number[], interval = 200) {
        const { observation } = trainingFixture();
        const history = new DrivingYawInstability();
        const frames = rates.map((yawRate, i) => ({ ...observation, capturedAt: 1000 + i * interval, measuredAt: 1000 + i * interval, yawRate }));
        return frames.slice(1).map((frame, i) => drivingReward(frames[i], frame, history));
    }

    it('penalizes repeated measured yaw-rate reversals', () => {
        const [isolated, repeated, frequent] = rewards([0.3, -0.3, 0.3, -0.3]);
        expect(isolated.instability).toBe(0);
        expect(repeated.instability).toBeLessThan(0);
        expect(frequent.instability).toBeLessThan(repeated.instability);
        expect(repeated.total).toBeCloseTo(isolated.total + repeated.instability);
    });

    it.each([1, -1])('does not penalize steady or accelerating turns in direction %s', (direction) => {
        expect(rewards([0.3, 4, 2, 6, 1].map((rate) => rate * direction)).every((reward) => reward.instability === 0)).toBe(true);
    });

    it('does not penalize infrequent reversals or near-zero noise', () => {
        expect(rewards([0.3, -0.3, 0.3, -0.3], 1100).every((reward) => reward.instability === 0)).toBe(true);
        expect(rewards([0.05, -0.05, 0, 0.01, -0.01]).every((reward) => reward.instability === 0)).toBe(true);
    });

    it('counts real reversals through zero without counting noise as additional reversals', () => {
        const result = rewards([0.3, 0.02, 0, -0.02, -0.3, 0, 0.3], 100);
        expect(result.slice(0, -1).every((reward) => reward.instability === 0)).toBe(true);
        expect(result[result.length - 1].instability).toBeLessThan(0);
    });

    it('expires reversals after a second of stable turning', () => {
        const result = rewards([0.3, -0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]);
        expect(result[1].instability).toBeLessThan(0);
        expect(result[result.length - 1].instability).toBe(0);
    });

    it('weights an active penalty by elapsed time', () => {
        const { observation } = trainingFixture();
        const penalties = [100, 200].map((elapsed) => {
            const history = new DrivingYawInstability();
            const first = { ...observation, yawRate: 0.3 };
            const second = { ...first, capturedAt: 1200, measuredAt: 1200, yawRate: -0.3 };
            const third = { ...first, capturedAt: 1400, measuredAt: 1400 };
            drivingReward(first, second, history);
            drivingReward(second, third, history);
            return drivingReward(third, { ...third, capturedAt: 1400 + elapsed, measuredAt: 1400 + elapsed }, history).instability;
        });
        expect(penalties[0]).toBeLessThan(0);
        expect(penalties[1]).toBeCloseTo(2 * penalties[0]);
    });
});

it.each([0, 2, 30])('does not reward distance changes without forward progress from distance %s', (referenceDistance) => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, referenceDistance: 1 };
    const unchangedDistance = drivingReward({ ...before, referenceDistance: 1 }, after);
    const changedDistance = drivingReward({ ...before, referenceDistance }, after);
    expect(changedDistance.total).toBeCloseTo(unchangedDistance.total);
    expect(changedDistance.line).toBe(0);
});

it.each(['throttle', 'brake', 'steering'] as const)('retains line and instability rewards for abrupt %s changes', (control) => {
    const { observation } = trainingFixture();
    const before = { ...observation, controls: NEUTRAL };
    const action = { ...NEUTRAL, [control]: 1 };
    const stationary = { ...before, capturedAt: 1200, measuredAt: 1200, speed: 0 };
    const converging = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, referenceDistance: 1 };
    for (const after of [stationary, converging]) {
        const unchanged = drivingReward(before, after);
        const changed = drivingReward(before, { ...after, controls: action });
        for (const key of ['line', 'instability'] as const) expect(changed[key]).toEqual(unchanged[key]);
    }
});

it.each([
    { heading: 0, pitch: 0, roll: 0 },
    { heading: 1, pitch: 0.2, roll: 0.4 },
])('excludes overlapping throttle and brake from instability at orientation %j', (orientation) => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, ...orientation };
    const throttleOnly = drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1 } });
    const overlap = drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1, brake: 1 } });
    expect(overlap.instability).toEqual(throttleOnly.instability);
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
    const before = { ...observation, speed: 100, referenceDistance: 0, controls: previousControls };
    for (const speed of [99.2, 200]) {
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, speed, controls };
        const reward = drivingReward(before, after);
        if (speed === 99.2) expect(reward.timing).toBeCloseTo(0.2 * 0.8 / (3.6 * 0.2 * 3));
        else expect(reward.timing).toBeLessThan(0);
        expect(reward.line).toBeGreaterThan(0);
        expect(reward.total).toBeCloseTo(reward.timing + reward.line + reward.progress + reward.instability);
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
    const before = { ...observation, speed: 100, referenceDistance: 0, controls: previousControls };
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.252, speed: 99.2, controls };
    expect(drivingReward(before, after).timing).toBeCloseTo(0.2 * 0.8 / (3.6 * 0.2 * 3));
    expect(drivingReward(before, { ...after, speed: 200 }).timing).toBeLessThan(0);
});

it('rewards the reference speed profile instead of simply driving faster', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, referenceDistance: 0, speed: 100 };
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
        { ...observation, speed, aggression, referenceDistance: 0 },
        { ...observation, speed: nextSpeed, normalizedPosition: position, referenceDistance: 0,
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
        const before = { ...observation, speed: 79, normalizedPosition: 0.4, referenceDistance: 0 };
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
        const before = { ...observation, speed: 40, referenceDistance: 0 };
        const after = { ...before, speed: 41, normalizedPosition: 0.252, capturedAt: 1200, measuredAt: 1200 };
        const config = cloneConfig();
        config.reward.accelerationTolerance *= 2;
        const normal = drivingReward(before, after).timing;
        expect(drivingReward(before, after, undefined, config).timing).toBeCloseTo(normal / 2);
        expect(drivingReward(before, { ...after, capturedAt: 1400, measuredAt: 1400 }).timing).toBeCloseTo(normal / 2);
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
    const before = { ...observation, normalizedPosition: 0.05, referenceDistance: 0, speed: 80 };
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
])('withholds timing and line rewards without forward driving: %j', (current) => {
    const { observation: before } = trainingFixture();
    const reward = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, controls: NEUTRAL, ...current });
    expect(reward.timing).toBe(0);
    expect(reward.line).toBe(0);
});

it('does not reward parking or reverse progress and handles the lap boundary', () => {
    const { observation: before } = trainingFixture();
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, speed: 0 }).total).toBeLessThan(0);
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.249 }).line).toBe(0);
    const wrapped = drivingReward({ ...before, normalizedPosition: 0.999, referenceDistance: 0 },
        { ...before, normalizedPosition: 0.001, capturedAt: 1200, measuredAt: 1200, referenceDistance: 0, controls: NEUTRAL });
    expect(wrapped.line).toBeCloseTo(0.2);
    expect(Number.isFinite(wrapped.timing)).toBe(true);
    const reverseWrapped = drivingReward({ ...before, normalizedPosition: 0.001, referenceDistance: 0 },
        { ...before, normalizedPosition: 0.999, capturedAt: 1200, measuredAt: 1200, referenceDistance: 0, controls: NEUTRAL });
    expect(reverseWrapped.timing).toBe(0);
    expect(reverseWrapped.line).toBe(0);
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

it('requires a usable selected reference and the exact lap paired with Track Vision', () => {
    const { live, vision, referenceLap } = trainingFixture();
    expect(readDrivingObservation(live, vision, null, 50, 1000).error).toMatch(/reference lap/);
    expect(readDrivingObservation(live, vision, { ...referenceLap, target_features: [] }, 50, 1000).error).toMatch(/missing raw X coordinates, raw Y coordinates, raw Z coordinates, speed/);
    expect(readDrivingObservation(live, vision, { ...referenceLap, car: 'other' }, 50, 1000).error).toMatch(/does not match the current game, car or circuit/);
    expect(readDrivingObservation(live, vision, { ...referenceLap }, 50, 1000).error).toMatch(/reference changed/);
    delete vision.position!.referenceLap;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/reference changed/);
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

it('scores measured speed and acceleration with capped progress and proximity regardless of pedals', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 100 };
    const position = 0.28;
    const target = referenceSpeedTargets(before.reference, position, before.aggression).target;
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: position, speed: target, controls: NEUTRAL };
    const matched = drivingReward(before, after);
    const mismatched = drivingReward(before, { ...after, speed: before.speed });
    expect(matched.timing).toBeCloseTo(100 * 0.02 * Math.exp(-after.referenceDistance / 3));
    expect(matched.timing).toBeGreaterThan(mismatched.timing);
    expect(matched.speedComparison.target).toEqual(target);
    expect(matched.speedComparison.measured).toEqual(target);
    expect(matched.speedComparison.measuredAcceleration).toBeCloseTo(-12 / (3.6 * 0.2));
    expect(matched.speedComparison.targetAcceleration).toBeCloseTo(matched.speedComparison.measuredAcceleration);
    expect(drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1, brake: 1 } })).toEqual(matched);
    expect(drivingReward(before, { ...after, referenceDistance: 10 }).timing).toBeLessThan(matched.timing);
});

it.each([
    [0, 0.16, 0.17, 117, 114], [100, 0.26, 0.27, 114, 108],
    [0, 0.56, 0.57, 254 / 3, 268 / 3], [100, 0.46, 0.47, 83.5, 87],
])('rewards the requested acceleration/deceleration timing at aggression %s', (aggression, start, end, speed, nextSpeed) => {
    const { observation } = trainingFixture();
    const before = { ...observation, aggression, normalizedPosition: start, speed, referenceDistance: 0 };
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
        const before = { ...observation, normalizedPosition: start, speed, referenceDistance: 0 };
        const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: end, speed: nextSpeed };
        const matched = drivingReward(before, after);
        expect(matched.timing).toBeCloseTo(2);
        expect(drivingReward({ ...before, speed: earlySpeed }, { ...after, speed: earlyNext }).timing).toBeLessThan(0);
        expect(drivingReward({ ...before, speed: lateSpeed }, { ...after, speed: lateNext }).timing).toBe(0);
    });

it('checks higher and increasing speed before reference throttle pickup against the aggressive target', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, aggression: 100, normalizedPosition: 0.46, speed: 83.5, referenceDistance: 0 };
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
