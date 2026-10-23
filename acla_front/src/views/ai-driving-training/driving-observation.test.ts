import { drivingReward, NEUTRAL, readDrivingObservation } from './driving-observation';
import { trainingFixture } from './training-fixtures';
import { referencePedalTargets } from './driving-reference';
import { getLiveMapReferenceLine } from 'views/live-session/live-map-reference';

it('uses all telemetry inputs and separate spatial segmentation channels from the capture time', () => {
    const { live, vision, referenceLap, observation } = trainingFixture();
    expect(observation.features).toHaveLength(103);
    expect(observation).toMatchObject({ gear: 3, rpm: 6000 });
    expect(observation.features.slice(66, 68)).toEqual([3 / 8, 6000 / 20000]);
    expect(observation.features.slice(-35, -19)).toEqual(new Array(16).fill(1));
    expect(observation.features.slice(-19, -3)).toEqual(new Array(16).fill(0));
    // Later telemetry must not replace the sample paired with the image.
    expect(readDrivingObservation({ ...live, currentTelemetry: { ...live.currentTelemetry, Physics_speed_kmh: 200 } }, vision, referenceLap, 50, 1000)
        .observation?.speed).toBe(108);
});

it.each(['telemetry', 'capture', 'session', 'position', 'segmentation', 'velocity', 'rotation', 'gear', 'rpm', 'nonfinite'])(
    'rejects unusable %s instead of training on fabricated defaults', (fault) => {
        const { live, vision, referenceLap } = trainingFixture();
        if (fault === 'telemetry') live.receivedAt = 0;
        if (fault === 'capture') vision.capturedAt = -1000;
        if (fault === 'session') live.streamGeneration++;
        if (fault === 'position') delete vision.position;
        if (fault === 'segmentation') delete vision.detections.segment;
        if (fault === 'velocity') delete live.currentTelemetry.Physics_velocity_x;
        if (fault === 'rotation') delete live.currentTelemetry.Physics_roll;
        if (fault === 'gear') delete live.currentTelemetry.Physics_gear;
        if (fault === 'rpm') delete live.currentTelemetry.Physics_rpm;
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
    { Physics_gear: -1 }, { Physics_gear: 2.5 }, { Physics_gear: Infinity },
    { Physics_rpm: -1 }, { Physics_rpm: NaN }, { Physics_rpm: 12.5 },
])('rejects invalid gear/RPM telemetry %j', (fields) => {
    const { live, vision, referenceLap } = trainingFixture();
    Object.assign(live.currentTelemetry, fields);
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/gear and engine RPM/);
});

it('requires automatic shifting to be disabled when reported by the simulator', () => {
    const { live, vision, referenceLap } = trainingFixture();
    live.currentTelemetry.Physics_autoshifter_on = true;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/Disable automatic gear shifting/);
    live.currentTelemetry.Physics_autoshifter_on = false;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).observation).toBeDefined();
});

it('rejects mismatched position estimates and malformed masks', () => {
    const { live, vision, referenceLap } = trainingFixture();
    vision.position!.normalizedPosition = 0.3;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/matching/);
    vision.position!.normalizedPosition = 0.25;
    const segment = vision.detections.segment!;
    if (segment.task === 'segment') segment.instances[0].mask = new Uint8Array(2);
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/mask/);
});

it('rewards forward reference tracking and penalizes instability', () => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200, normalizedPosition: 0.252, referenceDistance: 1 };
    const smooth = drivingReward(before, after, NEUTRAL);
    const unstable = drivingReward(before, { ...after, heading: 1, roll: 0.4 }, { ...NEUTRAL, steering: 1 });
    expect(smooth.speed).toBeGreaterThan(0);
    expect(smooth.line).toBeGreaterThan(0);
    expect(unstable.instability).toBeLessThan(0);
    expect(unstable.total).toBeLessThan(smooth.total);
    const distant = drivingReward(before, { ...after, referenceDistance: 8 }, NEUTRAL);
    expect(distant.speed).toBeLessThan(smooth.speed);
    expect(distant.total).toBeLessThan(smooth.total);
});

it.each([0, 2, 30])('does not add a convergence bonus or divergence penalty from previous distance %s', (referenceDistance) => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200, normalizedPosition: 0.252, referenceDistance: 1 };
    const unchangedDistance = drivingReward({ ...before, referenceDistance: 1 }, after, NEUTRAL);
    const changedDistance = drivingReward({ ...before, referenceDistance }, after, NEUTRAL);
    expect(changedDistance.total).toBeCloseTo(unchangedDistance.total);
});

it.each(['throttle', 'brake', 'steering'] as const)('retains existing reward components for abrupt %s changes', (control) => {
    const { observation } = trainingFixture();
    const before = { ...observation, controls: NEUTRAL };
    const action = { ...NEUTRAL, [control]: 1 };
    const stationary = { ...before, capturedAt: 1200, speed: 0 };
    const converging = { ...before, capturedAt: 1200, normalizedPosition: 0.252, referenceDistance: 1 };
    for (const after of [stationary, converging]) {
        const unchanged = drivingReward(before, after, NEUTRAL);
        const changed = drivingReward(before, { ...after, controls: action }, action);
        for (const key of ['speed', 'line', 'instability'] as const) expect(changed[key]).toEqual(unchanged[key]);
    }
});

it('still penalizes overlapping throttle and brake', () => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200 };
    const throttleOnly = drivingReward(before, after, { ...NEUTRAL, throttle: 1 });
    const overlap = drivingReward(before, after, { ...NEUTRAL, throttle: 1, brake: 1 });
    expect(overlap.instability).toBeLessThan(throttleOnly.instability);
    expect(overlap.total).toBeLessThan(throttleOnly.total);
});

it('rewards covering the same forward section faster', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, referenceDistance: 0 };
    const slow = drivingReward(before, { ...before, capturedAt: 1400, normalizedPosition: 0.252, speed: 100 }, NEUTRAL);
    const fast = drivingReward(before, { ...before, capturedAt: 1200, normalizedPosition: 0.252, speed: 200 }, NEUTRAL);
    expect(slow.speed).toBeGreaterThan(0);
    expect(fast.speed).toBeGreaterThan(slow.speed);
    expect(fast.total).toBeGreaterThan(slow.total);
});

it('gives the same total reward for a steady drive sampled at different capture intervals', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, normalizedPosition: 0.05, referenceDistance: 0, speed: 200 };
    const middle = { ...before, capturedAt: 1100, normalizedPosition: 0.051 };
    const after = { ...before, capturedAt: 1200, normalizedPosition: 0.052 };
    const whole = drivingReward(before, after, NEUTRAL);
    const first = drivingReward(before, middle, NEUTRAL);
    const second = drivingReward(middle, after, NEUTRAL);
    expect(whole.speed).toBeGreaterThan(0);
    expect(first.speed + second.speed).toBeCloseTo(whole.speed);
    expect(first.total + second.total).toBeCloseTo(whole.total);
});

it.each([
    { speed: 0, normalizedPosition: 0.252 },
    { speed: 200, normalizedPosition: 0.25 },
    { speed: 200, normalizedPosition: 0.249 },
])('withholds the speed bonus without both speed and forward progress: %j', (current) => {
    const { observation: before } = trainingFixture();
    const reward = drivingReward(before, { ...before, capturedAt: 1200, ...current }, NEUTRAL);
    expect(reward.speed).toBe(0);
});

it('does not reward parking or reverse progress and handles the lap boundary', () => {
    const { observation: before } = trainingFixture();
    expect(drivingReward(before, { ...before, capturedAt: 1200, speed: 0 }, NEUTRAL).total).toBeLessThanOrEqual(0);
    expect(drivingReward(before, { ...before, capturedAt: 1200, normalizedPosition: 0.249 }, NEUTRAL).line).toBeLessThan(0);
    const wrapped = drivingReward({ ...before, normalizedPosition: 0.999, referenceDistance: 0 },
        { ...before, normalizedPosition: 0.001, capturedAt: 1200, referenceDistance: 0 }, NEUTRAL);
    expect(wrapped.line).toBeCloseTo(0.2);
    expect(wrapped.speed).toBeGreaterThan(0);
    const reverseWrapped = drivingReward({ ...before, normalizedPosition: 0.001, referenceDistance: 0 },
        { ...before, normalizedPosition: 0.999, capturedAt: 1200, referenceDistance: 0 }, NEUTRAL);
    expect(reverseWrapped.speed).toBe(0);
    expect(reverseWrapped.referencePedal).toBe(0);
    expect(reverseWrapped.total).toBeLessThan(0);
    expect(() => drivingReward(before, { ...before, capturedAt: 1200, normalizedPosition: 0.7 }, NEUTRAL)).toThrow(/jumped/);
});

it.each([0, 50, 100])('appends aggression and targets without changing the existing features at %s', (aggression) => {
    const { live, vision, referenceLap, observation } = trainingFixture();
    const next = readDrivingObservation(live, vision, referenceLap, aggression, 1000).observation!;
    expect(next.features.slice(0, -3)).toEqual(observation.features.slice(0, -3));
    expect(next.features.slice(-3)[0]).toBe((aggression - 50) / 50);
    expect(next.features.slice(-2)[0]).toBeCloseTo(aggression / 100);
    expect(next.features.slice(-2)[1]).toBeCloseTo(1 - aggression / 100);
});

it('requires a usable selected reference and the exact lap paired with Track Vision', () => {
    const { live, vision, referenceLap } = trainingFixture();
    expect(readDrivingObservation(live, vision, null, 50, 1000).error).toMatch(/reference lap/);
    expect(readDrivingObservation(live, vision, { ...referenceLap, target_features: [] }, 50, 1000).error).toMatch(/missing raw X coordinates, raw Y coordinates, raw Z coordinates, throttle, brake/);
    expect(readDrivingObservation(live, vision, { ...referenceLap, car: 'other' }, 50, 1000).error).toMatch(/does not match the current game, car or circuit/);
    expect(readDrivingObservation(live, vision, { ...referenceLap }, 50, 1000).error).toMatch(/reference changed/);
    delete vision.position!.referenceLap;
    expect(readDrivingObservation(live, vision, referenceLap, 50, 1000).error).toMatch(/reference changed/);
});

it.each(['throttle', 'brake'])('explains missing %s even when Live Map can draw the reference line', (pedal) => {
    const { live, vision, referenceLap } = trainingFixture();
    const lap = { ...referenceLap, target_features: referenceLap.target_features.map((name) =>
        name === `expert_optimal_${pedal}` ? 'unrelated' : name) };
    vision.position!.referenceLap = lap;
    expect(getLiveMapReferenceLine(lap)).toHaveLength(lap.x.length);
    expect(readDrivingObservation(live, vision, lap, 50, 1000).error).toContain(`missing ${pedal} data`);
});

it('allows a drawable reference lap with different recorded start and finish coordinates', () => {
    const { live, vision, referenceLap } = trainingFixture();
    const lap = { ...referenceLap, y: referenceLap.y.map((row) => [...row]) };
    lap.y[lap.y.length - 1][0] = 0.5;
    vision.position!.referenceLap = lap;
    expect(getLiveMapReferenceLine(lap)).toHaveLength(lap.x.length);
    expect(readDrivingObservation(live, vision, lap, 50, 1000).observation).toBeDefined();
});

it('scores measured pedals at the following position with capped progress and proximity', () => {
    const { observation: before } = trainingFixture();
    const position = 0.28;
    const target = referencePedalTargets(before.reference, position, before.aggression).target;
    const after = { ...before, capturedAt: 1200, normalizedPosition: position,
        controls: { ...NEUTRAL, ...target } };
    const matched = drivingReward(before, after, { ...NEUTRAL, throttle: 0, brake: 1 });
    const mismatched = drivingReward(before, { ...after, controls: { ...NEUTRAL, throttle: 1 - target.throttle, brake: 1 - target.brake } }, NEUTRAL);
    expect(matched.referencePedal).toBeCloseTo(100 * 0.02 * Math.exp(-after.referenceDistance / 3));
    expect(matched.referencePedal).toBeGreaterThan(mismatched.referencePedal);
    expect(matched.pedalComparison.target).toEqual(target);
    expect(matched.pedalComparison.measured).toEqual(target);
    expect(drivingReward(before, { ...after, referenceDistance: 10 }, NEUTRAL).referencePedal).toBeLessThan(matched.referencePedal);
});

it.each([
    { speed: 0, normalizedPosition: 0.252 }, { speed: 100, normalizedPosition: 0.25 },
    { speed: 100, normalizedPosition: 0.249 }, { speed: 100, normalizedPosition: 0.252, gear: -1 },
])('cannot farm reference pedal reward while stationary or reversing: %j', (fields) => {
    const { observation: before } = trainingFixture();
    const target = referencePedalTargets(before.reference, fields.normalizedPosition, before.aggression).target;
    expect(drivingReward(before, { ...before, capturedAt: 1200, ...fields, controls: { ...NEUTRAL, ...target } }, NEUTRAL).referencePedal).toBe(0);
});
