import { DrivingBlockedMotionMonitor } from './driving-blocked-motion';
import type { DrivingTelemetrySample } from './driving-observation';
import { trainingFixture } from './training-fixtures';

function setup(game = 'acc') {
    const monitor = new DrivingBlockedMotionMonitor();
    const { observation } = trainingFixture();
    const sample = (measuredAt: number, forward: number, throttle: number, steering = 0,
        { lateral = 0, up = 0, yawRate = 0, brake = 0, gear = observation.gear } = {}) => {
        const telemetry: DrivingTelemetrySample = { ...observation, measuredAt, context: { ...observation.context, game },
            localVelocity: game === 'iracing' ? [lateral, up, forward] : [lateral, forward, up], yawRate, gear,
            controls: { throttle, brake, steering } };
        monitor.observe(telemetry);
    };
    return { monitor, sample };
}

it.each(['acc', 'iracing'])('allows a brief brake hold, pedal-free standstill and smooth launch in %s', (game) => {
    const { monitor, sample } = setup(game);
    for (let at = 0; at <= 2000; at += 50) {
        sample(at, Math.max(0, at - 1000) / 200, at >= 1000 ? 0.8 : 0, 0, { brake: at < 300 ? 0.8 : 0 });
    }
    expect(monitor.penaltyBetween(0, 2000)).toEqual({ vertical: 0, horizontal: 0 });
});

describe.each(['acc', 'iracing'])('%s blocked motion', (game) => {
    it.each([
        { scenario: 'throttle against a wall', throttle: 0.8, brake: 0, gear: 1 },
        { scenario: 'throttle in neutral', throttle: 0.2, brake: 0, gear: 0 },
        { scenario: 'brake alone', throttle: 0, brake: 0.8, gear: 1 },
    ])('penalizes a sustained standstill with $scenario', ({ throttle, brake, gear }) => {
        const { monitor, sample } = setup(game);
        for (let at = 0; at <= 1000; at += 50) {
            sample(at, at % 100 === 0 ? 0.05 : -0.05, throttle, 0, { brake, gear });
        }
        expect(monitor.penaltyBetween(0, 350)).toEqual({ vertical: 0, horizontal: 0 });
        const penalty = monitor.penaltyBetween(0, 1000);
        expect(penalty.vertical).toBeCloseTo(-0.85);
        expect(penalty.horizontal).toBe(0);
        sample(1050, -2, throttle, 0, { brake, gear });
        expect(monitor.penaltyBetween(1000, 1050)).toEqual({ vertical: 0, horizontal: 0 });
    });

    it.each(['vertical', 'horizontal'] as const)('ramps up only the %s penalty after repeated rebounds', (axis) => {
        const { monitor, sample } = setup(game);
        for (let at = 0; at <= 2000; at += 50) {
            const bump = at % 400 < 200 ? 1 : -1;
            sample(at, axis === 'vertical' ? bump : 20, 0.8, -0.3,
                { lateral: axis === 'horizontal' ? bump : 0, yawRate: bump * 0.3 });
        }
        // One fall and one return are insufficient. The next fall confirms repetition at 600 ms.
        expect(monitor.penaltyBetween(0, 600)).toEqual({ vertical: 0, horizontal: 0 });
        expect(monitor.penaltyBetween(600, 700)[axis]).toBeCloseTo(-0.02222);
        expect(monitor.penaltyBetween(600, 1600)[axis]).toBeCloseTo(-1.55);
        expect(monitor.penaltyBetween(600, 1000)[axis] + monitor.penaltyBetween(1000, 1600)[axis]).toBeCloseTo(-1.55);
        const fullPenalty = monitor.penaltyBetween(1500, 2000);
        expect(fullPenalty.vertical).toBeCloseTo(axis === 'vertical' ? -1 : 0);
        expect(fullPenalty.horizontal).toBeCloseTo(axis === 'horizontal' ? -1 : 0);
        for (let at = 2050; at <= 3000; at += 50) sample(at, axis === 'vertical' ? 1 : 20, 0.8, -0.3);
        expect(monitor.penaltyBetween(2800, 3000)).toEqual({ vertical: 0, horizontal: 0 });
    });

    it.each([-1, 1])('recognizes two shallow glances with light steering in direction %s', (sign) => {
        const { monitor, sample } = setup(game);
        for (let at = 0; at <= 2800; at += 50) {
            // Briefly deflect away from the wall, then drift back at the same forward speed.
            const rebounding = (at >= 300 && at < 500) || (at >= 1500 && at < 1700);
            sample(at, 20, 0.4, sign * 0.05, { lateral: sign * (rebounding ? -0.18 : 0.15) });
        }
        expect(monitor.penaltyBetween(0, 1500).horizontal).toBe(0);
        expect(monitor.penaltyBetween(1600, 2000).horizontal).toBeLessThan(0);
        expect(monitor.penaltyBetween(2650, 2800).horizontal).toBe(0);
    });

    it.each([-1, 1])('allows a bumpy corner with consistent yaw in direction %s', (sign) => {
        const { monitor, sample } = setup(game);
        // Light steering on the approach must not mask the yaw once the corner begins.
        for (let at = -1500; at < 0; at += 25) sample(at, 20, 0.4, sign * 0.05);
        for (let at = 0; at <= 2000; at += 25) {
            const lateral = Math.cos(at * Math.PI / 100);
            sample(at, 20, 0.4, sign * 0.3, { lateral, yawRate: at % 200 === 0 ? 0 : sign * 0.2 });
        }
        expect(monitor.penaltyBetween(0, 2000)).toEqual({ vertical: 0, horizontal: 0 });
    });

    it('smooths rapid velocity jitter and ignores a single jolt or up/down bouncing', () => {
        const { monitor, sample } = setup(game);
        for (let at = 0; at <= 2000; at += 10) {
            const jitter = at % 20 === 0 ? 0.6 : -0.6;
            const jolt = at >= 500 && at < 600 ? 3 : 0;
            const up = at % 400 < 200 ? 3 : -3;
            sample(at, jitter + jolt, 0.8, 0.3, { lateral: jitter + jolt, up });
        }
        expect(monitor.penaltyBetween(0, 2000)).toEqual({ vertical: 0, horizontal: 0 });
    });
});

it('requires fresh stationary evidence after a pedal release or telemetry gap', () => {
    const { monitor, sample } = setup();
    for (let at = 0; at <= 300; at += 50) sample(at, 0, 0.8);
    sample(350, 0, 0);
    for (let at = 400; at <= 650; at += 50) sample(at, 0, 0.8);
    for (let at = 950; at <= 1400; at += 50) sample(at, 0, 0.8);
    expect(monitor.penaltyBetween(0, 1300)).toEqual({ vertical: 0, horizontal: 0 });
    expect(monitor.penaltyBetween(1300, 1400).vertical).toBeLessThan(0);
});

it('requires continued steering in one direction and fresh bumps after a telemetry gap', () => {
    const { monitor, sample } = setup();
    const bump = (at: number, steer: number) => sample(at, 20, 0.4, steer, { lateral: at % 200 === 0 ? 1 : -1 });
    for (let at = 0; at <= 1200; at += 100) bump(at, 0.02);
    for (let at = 1300; at <= 2500; at += 100) bump(at, at % 600 < 300 ? 0.3 : -0.3);
    expect(monitor.penaltyBetween(0, 2500).horizontal).toBe(0);
    for (let at = 2600; at <= 4000; at += 100) bump(at, 0.3);
    expect(monitor.penaltyBetween(3500, 4000).horizontal).toBeLessThan(0);
    for (let at = 4400; at <= 5000; at += 100) bump(at, 0.3);
    expect(monitor.penaltyBetween(4000, 4700).horizontal).toBe(0);
    expect(monitor.penaltyBetween(4700, 5000).horizontal).toBeLessThan(0);
});

it.each([10, 100, 200])('charges both sustained rebound patterns with telemetry every %i ms', (interval) => {
    const { monitor, sample } = setup();
    for (let at = 0; at <= 2600; at += interval) {
        const bounce = Math.cos(at * Math.PI / 200);
        sample(at, bounce, 0.8, 0.3, { lateral: bounce });
    }
    const penalty = monitor.penaltyBetween(2000, 2600);
    expect(penalty.vertical).toBeCloseTo(-1.2);
    expect(penalty.horizontal).toBeCloseTo(-1.2);
});

it('tracks throttle and steering independently and clears evidence on reset', () => {
    const { monitor, sample } = setup();
    const bump = (at: number, throttle: number, steering: number) => {
        const bounce = at % 400 < 200 ? 1 : -1;
        sample(at, bounce, throttle, steering, { lateral: bounce });
    };
    for (let at = 0; at <= 1600; at += 100) bump(at, 0.8, 0.3);
    for (let at = 1700; at <= 1900; at += 100) bump(at, 0.8, -0.3);
    expect(monitor.penaltyBetween(1600, 1900).vertical).toBeCloseTo(-0.6);
    expect(monitor.penaltyBetween(1600, 1900).horizontal).toBe(0);
    for (let at = 2000; at <= 3000; at += 100) bump(at, 0.8, -0.3);
    expect(monitor.penaltyBetween(2900, 3000).horizontal).toBeCloseTo(-0.2);
    for (let at = 3100; at <= 3500; at += 100) bump(at, 0, -0.3);
    expect(monitor.penaltyBetween(3000, 3500)).toEqual({ vertical: 0, horizontal: -1 });
    monitor.reset();
    bump(3600, 0.8, -0.3);
    bump(3700, 0.8, -0.3);
    expect(monitor.penaltyBetween(3500, 3700)).toEqual({ vertical: 0, horizontal: 0 });
});
