import { drivingReward, NEUTRAL, readDrivingObservation } from './driving-observation';
import { trainingFixture } from './training-fixtures';

it('uses all telemetry inputs and separate spatial segmentation channels from the capture time', () => {
    const { live, vision, observation } = trainingFixture();
    expect(observation.features).toHaveLength(100);
    expect(observation).toMatchObject({ gear: 3, rpm: 6000 });
    expect(observation.features.slice(66, 68)).toEqual([3 / 8, 6000 / 20000]);
    expect(observation.features.slice(-32, -16)).toEqual(new Array(16).fill(1));
    expect(observation.features.slice(-16)).toEqual(new Array(16).fill(0));
    // Later telemetry must not replace the sample paired with the image.
    expect(readDrivingObservation({ ...live, currentTelemetry: { ...live.currentTelemetry, Physics_speed_kmh: 200 } }, vision, 1000)
        .observation?.speed).toBe(108);
});

it.each(['telemetry', 'capture', 'session', 'position', 'segmentation', 'velocity', 'rotation', 'gear', 'rpm', 'nonfinite'])(
    'rejects unusable %s instead of training on fabricated defaults', (fault) => {
        const { live, vision } = trainingFixture();
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
        expect(readDrivingObservation(live, vision, 1000).error).toBeTruthy();
    },
);

it.each([0, 1, 2, 4])('normalizes standard gear %s and keeps RPM paired with the image', (gear) => {
    const { live, vision } = trainingFixture();
    vision.telemetry!.currentTelemetry = { ...live.currentTelemetry, Physics_gear: gear, Physics_rpm: 4500 };
    const result = readDrivingObservation({ ...live, currentTelemetry: { ...live.currentTelemetry, Physics_gear: 6, Physics_rpm: 8000 } }, vision, 1000);
    expect(result.observation).toMatchObject({ gear: gear - 1, rpm: 4500 });
});

it.each([
    { Physics_gear: -1 }, { Physics_gear: 2.5 }, { Physics_gear: Infinity },
    { Physics_rpm: -1 }, { Physics_rpm: NaN }, { Physics_rpm: 12.5 },
])('rejects invalid gear/RPM telemetry %j', (fields) => {
    const { live, vision } = trainingFixture();
    Object.assign(live.currentTelemetry, fields);
    expect(readDrivingObservation(live, vision, 1000).error).toMatch(/gear and engine RPM/);
});

it('requires automatic shifting to be disabled when reported by the simulator', () => {
    const { live, vision } = trainingFixture();
    live.currentTelemetry.Physics_autoshifter_on = true;
    expect(readDrivingObservation(live, vision, 1000).error).toMatch(/Disable automatic gear shifting/);
    live.currentTelemetry.Physics_autoshifter_on = false;
    expect(readDrivingObservation(live, vision, 1000).observation).toBeDefined();
});

it('rejects mismatched position estimates and malformed masks', () => {
    const { live, vision } = trainingFixture();
    vision.position!.normalizedPosition = 0.3;
    expect(readDrivingObservation(live, vision, 1000).error).toMatch(/matching/);
    vision.position!.normalizedPosition = 0.25;
    const segment = vision.detections.segment!;
    if (segment.task === 'segment') segment.instances[0].mask = new Uint8Array(2);
    expect(readDrivingObservation(live, vision, 1000).error).toMatch(/mask/);
});

it('rewards forward reference tracking, smooth convergence and penalizes instability', () => {
    const { observation: before } = trainingFixture();
    const after = { ...before, capturedAt: 1200, normalizedPosition: 0.252, referenceDistance: 1 };
    const smooth = drivingReward(before, after, NEUTRAL);
    const unstable = drivingReward(before, { ...after, heading: 1, roll: 0.4 }, { ...NEUTRAL, steering: 1 });
    expect(smooth.speed).toBeGreaterThan(0);
    expect(smooth.line).toBeGreaterThan(0);
    expect(smooth.convergence).toBeGreaterThan(0);
    expect(unstable.convergence).toBeLessThan(smooth.convergence);
    expect(unstable.instability).toBeLessThan(0);
    expect(unstable.total).toBeLessThan(smooth.total);
    const distant = drivingReward(before, { ...after, referenceDistance: 8 }, NEUTRAL);
    expect(distant.speed).toBeLessThan(smooth.speed);
    expect(distant.total).toBeLessThan(smooth.total);
});

it.each(['throttle', 'brake', 'steering'] as const)('does not reduce reward for abrupt %s changes', (control) => {
    const { observation } = trainingFixture();
    const before = { ...observation, controls: NEUTRAL };
    const action = { ...NEUTRAL, [control]: 1 };
    const stationary = { ...before, capturedAt: 1200, speed: 0 };
    const converging = { ...before, capturedAt: 1200, normalizedPosition: 0.252, referenceDistance: 1 };
    for (const after of [stationary, converging]) {
        const unchanged = drivingReward(before, after, NEUTRAL);
        const changed = drivingReward(before, { ...after, controls: action }, action);
        expect(changed).toEqual(unchanged);
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
    const before = { ...observation, referenceDistance: 0, speed: 200 };
    const middle = { ...before, capturedAt: 1100, normalizedPosition: 0.251 };
    const after = { ...before, capturedAt: 1200, normalizedPosition: 0.252 };
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
    expect(reverseWrapped.total).toBeLessThan(0);
    expect(() => drivingReward(before, { ...before, capturedAt: 1200, normalizedPosition: 0.7 }, NEUTRAL)).toThrow(/jumped/);
});
