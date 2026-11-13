import type { CircuitMapBinSample, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { StandardTelemetrySample } from 'views/live-session/live-session-types';
import { detectDrivingOffTrack } from './driving-off-track';
import { drivingReward, readDrivingObservation } from './driving-observation';
import { cloneConfig } from './driving-model-config';
import { trainingFixture } from './training-fixtures';

const wheels = ['front_left', 'front_right', 'rear_left', 'rear_right'] as const;
const point = (x: number, y: number, position: number): CircuitMapBinSample => ({
    x, y, z: 100, normalized_position: position, bin: Math.floor(position * 1000), sample_count: 1, updated_at: '',
});
const mapFixture = (): CircuitMapDto => ({ id: 'monza', game: 'acc', source_track_key: 'monza', circuit_name: 'Monza', resolution: 1000,
    samples: { left_boundary: [-10, 0, 10].map((y, i) => point(-5, y, 0.24 + i * 0.01)),
        right_boundary: [-10, 0, 10].map((y, i) => point(5, y, 0.24 + i * 0.01)) } });
const tyreSample = (points: [number, number][] = [[-1, 2], [1, 2], [-1, -2], [1, -2]]):
    { -readonly [K in keyof StandardTelemetrySample]: StandardTelemetrySample[K] } => ({
    Static_track: 'monza', Graphics_normalized_car_position: 0.25,
    ...Object.fromEntries(wheels.flatMap((wheel, i) => [
        [`Physics_tyre_contact_point_${wheel}_x`, points[i][0]],
        [`Physics_tyre_contact_point_${wheel}_y`, points[i][1]],
        [`Physics_tyre_contact_point_${wheel}_z`, 999],
    ])),
});

it.each([[0, true], [1, false], [2, false], [3, false], [-1, null], [undefined, null], [NaN, null],
    [Infinity, null], [0.5, null], [4, null], ['0', null], [null, null]] as const)(
    'uses only iRacing track surface %s for off-track status', (surface, expected) => {
        expect(detectDrivingOffTrack('iracing', { Graphics_player_track_surface: surface } as StandardTelemetrySample, null)).toBe(expected);
    });

it.each(Array.from({ length: 16 }, (_, mask) => mask))('requires all four ACC tyres to be outside (mask=%s)', (mask) => {
    const sample = tyreSample(wheels.map((_, i) => [mask & (1 << i) ? 6 : 1, i < 2 ? 2 : -2]));
    // iRacing-only telemetry must not override ACC geometry.
    sample.Graphics_player_track_surface = 0;
    expect(detectDrivingOffTrack('acc', sample, mapFixture())).toBe(mask === 15);
});

it.each([-1, 1])('detects an exit on boundary side %s and permits exact boundary contact', (side) => {
    const map = mapFixture();
    const sample = tyreSample(wheels.map((_, i) => [side * (i % 2 ? 8 : 6), i < 2 ? 2 : -2]));
    expect(detectDrivingOffTrack('acc', sample, map)).toBe(true);
    for (const wheel of wheels) {
        expect(detectDrivingOffTrack('acc', { ...sample, [`Physics_tyre_contact_point_${wheel}_x`]: side * 5 }, map)).toBe(false);
    }
});

it('uses world tyre contacts on the ACC X/Y plane with independently sampled, rotated boundaries', () => {
    const map = mapFixture();
    map.samples.right_boundary = [-10, -5, 5, 10].map((y) => point(5, y, 0.25 + y / 1000));
    const transform = (x: number, y: number): [number, number] => [(x - y) / Math.sqrt(2) + 400, (x + y) / Math.sqrt(2) - 600];
    for (const line of Object.values(map.samples)) for (const p of line) [p.x, p.y] = transform(p.x, p.y);
    expect(detectDrivingOffTrack('acc', tyreSample(wheels.map((_, i) => transform(6 + i % 2, i < 2 ? 2 : -2))), map)).toBe(true);
    expect(detectDrivingOffTrack('acc', tyreSample(wheels.map((_, i) => transform(i % 2, i < 2 ? 2 : -2))), map)).toBe(false);
});

it.each([0, 1, 0.999, 0.001, 0.005, 0.125, 0.495, 0.995])('handles curved boundaries and the lap seam at %s', (position) => {
    const map = mapFixture();
    for (const [side, radius] of [['left_boundary', 95], ['right_boundary', 105]] as const) {
        map.samples[side] = Array.from({ length: 100 }, (_, i) => {
            const p = (i + 0.5) / 100;
            return point(radius * Math.cos(p * 2 * Math.PI), radius * Math.sin(p * 2 * Math.PI), p);
        });
    }
    for (const [radius, expected] of [[90, true], [100, false], [110, true]] as const) {
        const angle = position * 2 * Math.PI;
        const sample = tyreSample(wheels.map((_, i) => {
            const x = radius + (i % 2) * 0.5, y = i < 2 ? 0.01 : -0.01;
            return [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle)];
        }));
        sample.Graphics_normalized_car_position = position;
        expect(detectDrivingOffTrack('acc', sample, map)).toBe(expected);
    }
});

it.each(['missing map', 'wrong track', 'wrong game', 'missing edge', 'missing tyre', 'invalid tyre',
    'invalid boundary', 'collapsed boundaries', 'boundary gap', 'outside recording'])(
    'leaves ACC detection unavailable with %s', (problem) => {
        let map: CircuitMapDto | null = mapFixture();
        const sample = tyreSample(wheels.map((_, i) => [7, i < 2 ? 2 : -2]));
        if (problem === 'missing map') map = null;
        if (problem === 'wrong track') map!.source_track_key = 'spa';
        if (problem === 'wrong game') map!.game = 'iracing';
        if (problem === 'missing edge') delete map!.samples.left_boundary;
        if (problem === 'missing tyre') delete sample.Physics_tyre_contact_point_rear_right_y;
        if (problem === 'invalid tyre') sample.Physics_tyre_contact_point_rear_right_x = NaN;
        if (problem === 'invalid boundary') map!.samples.left_boundary![1].x = NaN;
        if (problem === 'collapsed boundaries') map!.samples.right_boundary = map!.samples.left_boundary;
        if (problem === 'boundary gap') map!.samples.left_boundary = [point(-5, -100, 0.15), point(-5, 100, 0.35)];
        if (problem === 'outside recording') sample.Physics_tyre_contact_point_rear_right_y = 20;
        expect(detectDrivingOffTrack('acc', sample, map)).toBeNull();
    });

it('accepts the ACC track display name when matching a map', () => {
    const map = mapFixture();
    delete map.source_track_key;
    map.circuit_name = 'Autodromo Nazionale Monza';
    expect(detectDrivingOffTrack('acc', tyreSample(), map)).toBe(false);
});

it.each(['acc', 'iracing'] as const)('reads %s off-track data from the image-paired telemetry, not a newer live sample', (game) => {
    const fixture = trainingFixture();
    fixture.live.game = fixture.vision.telemetry!.game = game;
    fixture.referenceLap.game = game;
    fixture.vision.telemetry!.currentTelemetry = { ...fixture.live.currentTelemetry,
        ...tyreSample(wheels.map((_, i) => [7, i < 2 ? 2 : -2])), Graphics_player_track_surface: 0 };
    fixture.live.currentTelemetry = { ...fixture.live.currentTelemetry, ...tyreSample(), Graphics_player_track_surface: 3 };
    const result = readDrivingObservation(fixture.live, fixture.vision, fixture.referenceLap, 50, 1000, undefined, mapFixture());
    expect(result.error).toBeUndefined();
    expect(result.observation?.offTrack).toBe(true);
    expect(result.observation?.features).toHaveLength(fixture.observation.features.length);
});

it.each([0.1, 0.2, 0.5, 1])('charges 5 reward per second at %s-second telemetry intervals, including stationary driving', (seconds) => {
    const { observation } = trainingFixture();
    const before = { ...observation, speed: 0, offTrack: true };
    let sum = 0;
    for (let i = 0; i < Math.round(1 / seconds); i++) {
        const reward = drivingReward(before, { ...before, capturedAt: 2000, measuredAt: 1000 + seconds * 1000 });
        expect(reward.offTrack).toBeCloseTo(-5 * seconds);
        expect(reward.total).toBeCloseTo(reward.timing + reward.progress + reward.impact + reward.offTrack);
        sum += reward.offTrack;
    }
    expect(sum).toBeCloseTo(-5);
});

it('stops charging on re-entry or unavailable detection, and respects the total reward limit', () => {
    const { observation } = trainingFixture();
    const before = { ...observation, offTrack: true };
    for (const offTrack of [false, null]) {
        const reward = drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200, offTrack });
        expect(reward.offTrack).toBe(0);
        expect(reward.offTrackDetected).toBe(offTrack);
    }
    const config = cloneConfig();
    config.reward.totalLimit = 0.5;
    expect(drivingReward(before, { ...before, capturedAt: 1200, measuredAt: 1200 }, config).total).toBe(-0.5);
});
