import type { StandardTelemetrySample } from '../live-session-types';
import { compareSteeringDirection } from './phrase-steering-direction';

const motion = (x: number, forward: number, game = 'acc'): StandardTelemetrySample => ({
    Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0, Physics_local_velocity_x: x,
    Physics_local_velocity_y: game === 'acc' ? forward : 0, Physics_local_velocity_z: game === 'acc' ? 0 : forward,
});
const trajectory = (remainingDistance = 100, heading = Math.PI / 2) => ({ heading, remainingDistance });

it.each(['acc', 'iracing'])('classifies both corner directions using local velocity in %s', (game) => {
    for (const sign of [-1, 1] as const) {
        for (const scale of [0.01, 1, 100]) {
            expect(compareSteeringDirection(trajectory(), [0, 0, 0], motion(2 * sign * scale, 10 * scale, game), game, sign)).toBe('tooWide');
            expect(compareSteeringDirection(trajectory(), [0, 0, 0], motion(-2 * sign * scale, 10 * scale, game), game, sign)).toBe('tooTight');
            expect(compareSteeringDirection(trajectory(), [0, 0, 0], motion(0, 10 * scale, game), game, sign)).toBe('aligned');
        }
    }
});

it.each(['acc', 'iracing'])('shifts the start sideways and rejoins the reference at Steering end in %s', (game) => {
    // At zero heading, model right is -X for ACC and +X for iRacing.
    const rightToWorld = game === 'acc' ? -1 : 1;
    for (const lateral of [-4, 4]) {
        const desiredX = rightToWorld * lateral / 10;
        expect(compareSteeringDirection(trajectory(), [lateral, 0, 0], motion(desiredX, 10, game), game, 1)).toBe('aligned');
        expect(compareSteeringDirection(trajectory(), [lateral, 0, 0], motion(desiredX + 2, 10, game), game, 1)).toBe('tooWide');
        expect(compareSteeringDirection(trajectory(), [lateral, 0, 0], motion(desiredX - 2, 10, game), game, 1)).toBe('tooTight');
    }
});

it('spreads the correction over the remaining curve instead of a fixed distance', () => {
    expect(compareSteeringDirection(trajectory(100), [4, 0, 0], motion(0, 10), 'acc', 1)).toBe('aligned');
    expect(compareSteeringDirection(trajectory(10), [4, 0, 0], motion(0, 10), 'acc', 1)).toBe('tooWide');
    expect(compareSteeringDirection(trajectory(10), [4, 0, 0], motion(-4, 10), 'acc', 1)).toBe('aligned');
});

it.each([-2.999, 0, 2.999])('does not shift below 3 coordinate units, but still warns about deviation at %s', (lateral) => {
    expect(compareSteeringDirection(trajectory(10), [lateral, 20, 30], motion(0, 10), 'acc', 1)).toBe('aligned');
    expect(compareSteeringDirection(trajectory(10), [lateral, 20, 30], motion(2, 10), 'acc', 1)).toBe('tooWide');
    expect(compareSteeringDirection(trajectory(10), [lateral, 20, 30], motion(-2, 10), 'acc', 1)).toBe('tooTight');
});

it.each([-3, 3])('shifts at exactly %s coordinate units without converting the error to metres', (lateral) => {
    expect(compareSteeringDirection(trajectory(10), [lateral, 0, 0], motion(-lateral, 10), 'acc', 1)).toBe('aligned');
    expect(compareSteeringDirection(trajectory(10), [lateral, 0, 0], motion(0, 10), 'acc', 1))
        .toBe(lateral > 0 ? 'tooWide' : 'tooTight');
});

it('uses only lateral error to shift the starting point', () => {
    for (const forward of [-20, 0, 20]) {
        expect(compareSteeringDirection(trajectory(10), [4, forward, 30], motion(-4, 10), 'acc', 1)).toBe('aligned');
    }
});

it.each(['acc', 'iracing'])('pairs local velocity and lateral error with the capture orientation in %s', (game) => {
    for (const heading of [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, 5 * Math.PI / 2]) {
        const sign = game === 'acc' ? 1 : -1;
        const sample = { ...motion(-sign * 4, 10, game), Physics_heading: heading };
        expect(compareSteeringDirection(trajectory(10, Math.PI / 2 + sign * heading), [4, 0, 0], sample, game, 1)).toBe('aligned');
    }
});

it.each(['acc', 'iracing'])('accounts for pitch, roll and the local vertical velocity in %s', (game) => {
    const sample = { ...motion(10, 0, game), Physics_pitch: Math.PI / 2, Physics_roll: Math.PI / 2,
        Physics_local_velocity_y: game === 'acc' ? -10 : 10, Physics_local_velocity_z: game === 'acc' ? 10 : -10 };
    // ACC projects to (10,10) in XY; iRacing projects to (-10,10) in XZ.
    const desiredHeading = game === 'acc' ? Math.PI / 4 : 3 * Math.PI / 4;
    expect(compareSteeringDirection(trajectory(100, desiredHeading), [0, 0, 0], sample, game, 1)).toBe('aligned');
});

it('ignores contradictory world velocity and uses local telemetry', () => {
    expect(compareSteeringDirection(trajectory(), [0, 0, 0], {
        ...motion(2, 10), Physics_velocity_x: -20, Physics_velocity_y: 10, Physics_velocity_z: 0,
    }, 'acc', 1)).toBe('tooWide');
});

it.each(['acc', 'iracing'])('ignores small direction errors within 6 degrees in %s', (game) => {
    for (const sign of [-1, 1] as const) {
        for (const [degrees, expected] of [[-6.01, 'tooTight'], [-5.99, 'aligned'], [-4, 'aligned'],
            [0, 'aligned'], [4, 'aligned'], [5.99, 'aligned'], [6.01, 'tooWide']] as const) {
            const x = 10 * Math.tan(degrees * Math.PI / 180) * sign;
            expect(compareSteeringDirection(trajectory(), [0, 0, 0], motion(x, 10, game), game, sign)).toBe(expected);
        }
    }
});

it('handles angle wraparound on both sides of the angular tolerance', () => {
    expect(compareSteeringDirection(trajectory(100, -Math.PI + 0.05), [0, 0, 0], motion(-10, 0.5), 'acc', 1)).toBe('aligned');
    expect(compareSteeringDirection(trajectory(100, -Math.PI + 0.1), [0, 0, 0], motion(-10, 0.5), 'acc', 1)).toBe('tooWide');
});

it.each([
    { Physics_local_velocity_x: undefined }, { Physics_local_velocity_y: NaN }, { Physics_local_velocity_z: Infinity },
    { Physics_heading: undefined }, { Physics_pitch: Infinity }, { Physics_roll: undefined },
    { Physics_local_velocity_x: 0, Physics_local_velocity_y: 0 }, { Physics_local_velocity_x: 0, Physics_local_velocity_y: -10 },
])('withholds ambiguous or unavailable local motion: %j', (patch) => {
    expect(compareSteeringDirection(trajectory(), [0, 0, 0], { ...motion(0, 10), ...patch }, 'acc', 1)).toBeUndefined();
});

it('withholds an ended or invalid trajectory, invalid error and unknown corner direction', () => {
    for (const remaining of [0, -1, NaN, Infinity]) {
        expect(compareSteeringDirection(trajectory(remaining), [0, 0, 0], motion(0, 10), 'acc', 1)).toBeUndefined();
    }
    expect(compareSteeringDirection(trajectory(), [NaN, 0, 0], motion(0, 10), 'acc', 1)).toBeUndefined();
    expect(compareSteeringDirection(trajectory(), [0, 0, 0], motion(0, 10), 'acc')).toBeUndefined();
});
