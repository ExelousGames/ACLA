import { DrivingBlockedMotionMonitor } from './driving-blocked-motion';
import type { DrivingTelemetrySample } from './driving-observation';
import { trainingFixture } from './training-fixtures';

function setup(game = 'acc') {
    const monitor = new DrivingBlockedMotionMonitor();
    const { observation } = trainingFixture();
    const sample = (measuredAt: number, forward: number, throttle: number, steering = 0,
        { lateral = 0, vertical = 0, yawRate = 0, lateralG = 0 } = {}) => {
        const telemetry: DrivingTelemetrySample = { ...observation, measuredAt, context: { ...observation.context, game },
            localVelocity: game === 'iracing' ? [lateral, vertical, forward] : [lateral, forward, vertical],
            yawRate, gForce: [lateralG, 0, 0],
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

describe.each(['acc', 'iracing'])('%s wall riding', (game) => {
    it.each(['lateral', 'forward'])('penalizes repeated %s bumps with held steering and erratic yaw', (axis) => {
        const { monitor, sample } = setup(game);
        for (let at = 0; at <= 2000; at += 100) {
            const bump = at % 200 === 0 ? 1 : -1;
            sample(at, axis === 'forward' ? 20 + bump : 20, 0.4, -0.3,
                { lateral: axis === 'lateral' ? bump : 0, yawRate: bump * 0.3, lateralG: 0.9 });
        }
        // The second swing in the same direction confirms the repetition at 300 ms.
        expect(monitor.penaltyBetween(0, 300)).toBe(0);
        expect(monitor.penaltyBetween(300, 400)).toBeLessThan(0);
        expect(monitor.penaltyBetween(300, 1300)).toBeCloseTo(-1.55);
        expect(monitor.penaltyBetween(1500, 2000)).toBeCloseTo(-1);
        for (let at = 2100; at <= 3000; at += 100) sample(at, 20, 0.4, -0.3);
        expect(monitor.penaltyBetween(2400, 3000)).toBe(0);
    });

    it.each([-1, 1])('recognizes two shallow glances with light steering in direction %s', (sign) => {
        const { monitor, sample } = setup(game);
        for (let at = 0; at <= 2800; at += 50) {
            // Briefly deflect away from the wall, then drift back at the same forward speed.
            const rebounding = (at >= 300 && at < 500) || (at >= 1500 && at < 1700);
            sample(at, 20, 0.4, sign * 0.05, { lateral: sign * (rebounding ? -0.18 : 0.15) });
        }
        expect(monitor.penaltyBetween(0, 1500)).toBe(0);
        expect(monitor.penaltyBetween(1500, 1600)).toBeLessThan(0);
        expect(monitor.penaltyBetween(1600, 2000)).toBeLessThan(0);
        expect(monitor.penaltyBetween(2500, 2800)).toBe(0);
    });

    it.each([-1, 1])('allows a bumpy corner with consistent yaw in direction %s', (sign) => {
        const { monitor, sample } = setup(game);
        // Light steering on the approach must not mask the yaw once the corner begins.
        for (let at = -1500; at < 0; at += 25) sample(at, 20, 0.4, sign * 0.05);
        for (let at = 0; at <= 2000; at += 25) {
            const lateral = Math.cos(at * Math.PI / 100);
            sample(at, 20, 0.4, sign * 0.3, { lateral, yawRate: at % 200 === 0 ? 0 : sign * 0.2 });
        }
        expect(monitor.penaltyBetween(0, 2000)).toBe(0);
    });

    it('ignores a single sideways jolt, steady deceleration, small vibration and vertical bumps', () => {
        const { monitor, sample } = setup(game);
        for (let at = 0; at <= 2000; at += 50) {
            const vibration = at % 100 === 0 ? 0.1 : -0.1;
            const jolt = at === 500 ? 3 : at === 550 ? -0.75 : 0;
            sample(at, 30 - at / 200, 0, 0.3, { lateral: jolt + vibration, vertical: 20 * vibration });
        }
        expect(monitor.penaltyBetween(0, 2000)).toBe(0);
    });
});

it('requires continued steering in one direction and fresh bumps after a telemetry gap', () => {
    const { monitor, sample } = setup();
    const bump = (at: number, steer: number) => sample(at, 20, 0.4, steer, { lateral: at % 200 === 0 ? 1 : -1 });
    for (let at = 0; at <= 1200; at += 100) bump(at, 0.02);
    for (let at = 1300; at <= 2500; at += 100) bump(at, at % 600 < 300 ? 0.3 : -0.3);
    expect(monitor.penaltyBetween(0, 2500)).toBe(0);
    for (let at = 2600; at <= 4000; at += 100) bump(at, 0.3);
    expect(monitor.penaltyBetween(3500, 4000)).toBeLessThan(0);
    for (let at = 4400; at <= 5000; at += 100) bump(at, 0.3);
    expect(monitor.penaltyBetween(4000, 4700)).toBe(0);
    expect(monitor.penaltyBetween(4700, 5000)).toBeLessThan(0);
});

it.each([10, 100, 200])('detects the same sustained vibration with telemetry every %i ms', (interval) => {
    const { monitor, sample } = setup();
    for (let at = 0; at <= 2600; at += interval) {
        const lateral = Math.cos(at * Math.PI / 200);
        sample(at, 15, 0, 0.3, { lateral });
    }
    expect(monitor.penaltyBetween(2000, 2600)).toBeCloseTo(-1.2);
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
