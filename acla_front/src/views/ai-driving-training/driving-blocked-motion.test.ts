import { DrivingBlockedMotionMonitor } from './driving-blocked-motion';
import type { DrivingTelemetrySample } from './driving-observation';
import { trainingFixture } from './training-fixtures';

function setup(game = 'acc') {
    const monitor = new DrivingBlockedMotionMonitor();
    const { observation } = trainingFixture();
    const sample = (measuredAt: number, forward: number, throttle: number, steering = 0, lateralG = 0) => {
        const telemetry: DrivingTelemetrySample = { ...observation, measuredAt, context: { ...observation.context, game },
            localVelocity: game === 'iracing' ? [0, 0, forward] : [0, forward, 0], gForce: [lateralG, 0, 0],
            controls: { ...observation.controls, throttle, steering } };
        monitor.observe(telemetry);
    };
    return { monitor, sample };
}

it.each(['acc', 'iracing'])('penalizes sustained throttle at a standstill in %s over elapsed time', (game) => {
    const { monitor, sample } = setup(game);
    for (let at = 1000; at <= 2000; at += 50) sample(at, 0, 0.8);
    expect(monitor.penaltyBetween(1000, 1350)).toBe(0);
    expect(monitor.penaltyBetween(1800, 2000)).toBeCloseTo(-0.4);
    expect(monitor.penaltyBetween(1000, 2000)).toBeCloseTo(-0.85);
    expect(monitor.penaltyBetween(1000, 1600) + monitor.penaltyBetween(1600, 2000)).toBeCloseTo(-0.85);
    sample(2050, 0, 0);
    expect(monitor.penaltyBetween(2000, 2050)).toBe(0);
});

it.each(['acc', 'iracing'])('detects a weak turn response in %s while allowing a normal corner', (game) => {
    const { monitor, sample } = setup(game);
    for (let at = 1000; at <= 1500; at += 100) sample(at, 20, 0.4, 0.3, 0.7);
    for (let at = 1600; at <= 2600; at += 100) sample(at, 20, 0.4, 0.3);
    expect(monitor.penaltyBetween(1000, 1500)).toBe(0);
    expect(monitor.penaltyBetween(1600, 2600)).toBeCloseTo(-0.85);
});

it('allows a launch and clears sustained evidence after a gap or reset', () => {
    const { monitor, sample } = setup();
    sample(1000, 0, 0.8);
    sample(1100, 0.1, 0.8);
    sample(1200, 0.6, 0.8);
    sample(1300, 1.2, 0.8);
    expect(monitor.penaltyBetween(1000, 1300)).toBe(0);
    for (let at = 1400; at <= 2000; at += 100) sample(at, 0, 0.8);
    expect(monitor.penaltyBetween(1800, 2000)).toBeLessThan(0);
    sample(2400, 0, 0.8);
    sample(2600, 0, 0.8);
    expect(monitor.penaltyBetween(2000, 2600)).toBe(0);
    monitor.reset();
    sample(2700, 0, 0.8);
    sample(2800, 0, 0.8);
    expect(monitor.penaltyBetween(2400, 2800)).toBe(0);
});
