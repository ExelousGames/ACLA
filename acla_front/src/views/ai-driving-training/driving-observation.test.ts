import { drivingReward, NEUTRAL, readDrivingObservation } from './driving-observation';
import { trainingFixture } from './training-fixtures';

it('uses all telemetry inputs and separate spatial segmentation channels from the capture time', () => {
    const { live, vision, observation } = trainingFixture();
    expect(observation.features).toHaveLength(98);
    expect(observation.features.slice(-32, -16)).toEqual(new Array(16).fill(1));
    expect(observation.features.slice(-16)).toEqual(new Array(16).fill(0));
    // Later telemetry must not replace the sample paired with the image.
    expect(readDrivingObservation({ ...live, currentTelemetry: { ...live.currentTelemetry, Physics_speed_kmh: 200 } }, vision, 1000)
        .observation?.speed).toBe(108);
});

it.each(['telemetry', 'capture', 'session', 'position', 'segmentation', 'velocity', 'rotation', 'nonfinite'])(
    'rejects unusable %s instead of training on fabricated defaults', (fault) => {
        const { live, vision } = trainingFixture();
        if (fault === 'telemetry') live.receivedAt = 0;
        if (fault === 'capture') vision.capturedAt = -1000;
        if (fault === 'session') live.streamGeneration++;
        if (fault === 'position') delete vision.position;
        if (fault === 'segmentation') delete vision.detections.segment;
        if (fault === 'velocity') delete live.currentTelemetry.Physics_velocity_x;
        if (fault === 'rotation') delete live.currentTelemetry.Physics_roll;
        if (fault === 'nonfinite') vision.position!.referenceDelta[0] = NaN;
        expect(readDrivingObservation(live, vision, 1000).error).toBeTruthy();
    },
);

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
    const smooth = drivingReward(before, after, NEUTRAL, NEUTRAL);
    const unstable = drivingReward(before, { ...after, heading: 1, roll: 0.4 }, { ...NEUTRAL, steering: 1 }, NEUTRAL);
    expect(smooth.line).toBeGreaterThan(0);
    expect(smooth.convergence).toBeGreaterThan(0);
    expect(unstable.convergence).toBeLessThan(smooth.convergence);
    expect(unstable.instability).toBeLessThan(0);
    expect(unstable.total).toBeLessThan(smooth.total);
    expect(drivingReward(before, { ...after, referenceDistance: 8 }, NEUTRAL, NEUTRAL).total).toBeLessThan(smooth.total);
});

it('does not reward parking or reverse progress and handles the lap boundary', () => {
    const { observation: before } = trainingFixture();
    expect(drivingReward(before, { ...before, capturedAt: 1200, speed: 0 }, NEUTRAL, NEUTRAL).total).toBeLessThanOrEqual(0);
    expect(drivingReward(before, { ...before, capturedAt: 1200, normalizedPosition: 0.249 }, NEUTRAL, NEUTRAL).line).toBeLessThan(0);
    const wrapped = drivingReward({ ...before, normalizedPosition: 0.999, referenceDistance: 0 },
        { ...before, normalizedPosition: 0.001, capturedAt: 1200, referenceDistance: 0 }, NEUTRAL, NEUTRAL);
    expect(wrapped.line).toBeCloseTo(0.2);
    expect(() => drivingReward(before, { ...before, capturedAt: 1200, normalizedPosition: 0.7 }, NEUTRAL, NEUTRAL)).toThrow(/jumped/);
});
