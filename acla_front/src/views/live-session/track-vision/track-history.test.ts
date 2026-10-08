import { readVisionMotion, TrackHistory } from './track-history';
import { DEFAULT_CAMERA } from './camera-projection';
import type { BirdsEyeScene } from './birds-eye-scene';
import type { TrackVisionFrame } from './track-vision-types';
import type { StandardTelemetrySample } from '../live-session-types';

const calibration = { ...DEFAULT_CAMERA, imageWidth: 1280, imageHeight: 720 };
const frame = (capturedAt: number, patch: Partial<TrackVisionFrame> = {}): TrackVisionFrame => ({
    capturedAt, width: 1280, height: 720, calibration, detections: {}, filterConfidence: 0.65,
    motion: { receivedAt: capturedAt, generation: 0, headingRad: 0, forwardMps: 20, rightMps: 0 }, ...patch,
});
const road = (near = 8, far = 40): BirdsEyeScene => ({
    leftBoundary: [[{ x: -5, y: near, z: 0 }, { x: -5, y: far, z: 0 }]],
    rightBoundary: [[{ x: 5, y: near, z: 0 }, { x: 5, y: far, z: 0 }]],
    centerline: [[{ x: 0, y: near, z: 0 }, { x: 0, y: far, z: 0 }]], cars: [], unplacedCars: 0,
});
const empty = (): BirdsEyeScene => ({ leftBoundary: [], rightBoundary: [], centerline: [], cars: [], unplacedCars: 0 });

it('moves previously visible road into the cockpit blind area', () => {
    const history = new TrackHistory(), first = road(), current = road();
    const original = JSON.stringify(first);
    expect(history.update(frame(0), first)).toEqual(first);
    const result = history.update(frame(500), current)!;
    expect(result.leftBoundary[0]).toBe(current.leftBoundary[0]);
    expect(result.leftBoundary[1]).toEqual([
        { x: -5, y: -2, z: 0, observedAt: 0 }, { x: -5, y: 8, z: 0, observedAt: 0 },
    ]);
    expect(result.rightBoundary[1]).toEqual([
        { x: 5, y: -2, z: 0, observedAt: 0 }, { x: 5, y: 8, z: 0, observedAt: 0 },
    ]);
    expect(JSON.stringify(first)).toBe(original);
});

it('accumulates several frames, respects new edges, and never retains previous traffic', () => {
    const history = new TrackHistory(), first = road();
    first.cars.push({ classId: 3, pack: false, confidence: 0.9, position: { x: 0, y: 12, z: 0 } });
    history.update(frame(0), first);
    history.update(frame(200), road());
    history.update(frame(400), road());
    const current = road();
    current.leftBoundary[0].forEach((point) => { point.x = -4; });
    const result = history.update(frame(600), current)!;
    expect(result.cars).toEqual([]);
    expect(result.leftBoundary[0]).toEqual(current.leftBoundary[0]);
    expect(result.leftBoundary.slice(1).flat().every(({ y }) => y <= 8 && y >= -5)).toBe(true);
    expect(result.leftBoundary.slice(1).flat().some(({ y }) => y <= 0)).toBe(true);
    expect(history.update(frame(800), empty())!.leftBoundary.flat().some(({ y }) => y <= 0)).toBe(true);
});

it('accounts for lateral motion instead of leaving old boundaries centered on the car', () => {
    const history = new TrackHistory();
    const moving = (time: number) => frame(time, { motion: { ...frame(time).motion!, rightMps: 2 } });
    history.update(moving(0), road());
    const result = history.update(moving(500), road())!;
    expect(result.leftBoundary[1].map(({ x }) => x)).toEqual([-6, -6]);
    expect(result.rightBoundary[1].map(({ x }) => x)).toEqual([4, 4]);
});

it.each([[0, 0.2], [0, -0.2], [Math.PI - 0.1, 0.2], [-Math.PI + 0.1, -0.2]])(
    'rotates remembered road from heading %s through turn %s, including wrapped headings', (initial, turn) => {
    const history = new TrackHistory();
    history.update(frame(0, { motion: { ...frame(0).motion!, headingRad: initial, forwardMps: 10 } }), road(10, 15));
    const heading = Math.atan2(Math.sin(initial + turn), Math.cos(initial + turn));
    const result = history.update(frame(500, { motion: { ...frame(500).motion!, headingRad: heading, forwardMps: 10 } }), empty())!;
    // 5 m traveled along the arc, then turn the car's frame. Radius is signed by turn direction.
    const radius = 5 / turn;
    expect(result.rightBoundary[0][0]).toMatchObject({
        x: expect.closeTo(5 * Math.cos(turn) - 10 * Math.sin(turn) + radius * (1 - Math.cos(turn))),
        y: expect.closeTo(5 * Math.sin(turn) + 10 * Math.cos(turn) - radius * Math.sin(turn)), observedAt: 0,
    });
    });

it('supports reversing without moving remembered road toward the driver', () => {
    const history = new TrackHistory();
    const reverse = (time: number) => frame(time, { motion: { ...frame(time).motion!, forwardMps: -4 } });
    history.update(reverse(0), road());
    const result = history.update(reverse(500), empty())!;
    expect(result.leftBoundary[0][0].y).toBe(10);
});

it('expires original observations even when empty frames continue to arrive', () => {
    const history = new TrackHistory();
    const stationary = (time: number) => frame(time, { motion: { ...frame(time).motion!, forwardMps: 0 } });
    history.update(stationary(0), road());
    for (let time = 200; time < 4000; time += 200) {
        expect(history.update(stationary(time), empty())!.leftBoundary.length).toBeGreaterThan(0);
    }
    expect(history.update(stationary(4000), empty())).toEqual(empty());
});

it('does not bridge missing sections', () => {
    const history = new TrackHistory(), source = road();
    source.leftBoundary = [road(8, 12).leftBoundary[0], road(16, 25).leftBoundary[0]];
    history.update(frame(0), source);
    const result = history.update(frame(500), empty())!;
    expect(result.leftBoundary.map((line) => line.map(({ y }) => y))).toEqual([[-2, 2], [6, 15]]);
});

it.each([
    ['missing motion', frame(500, { motion: undefined })],
    ['stale telemetry', frame(700, { motion: frame(0).motion })],
    ['future telemetry', frame(500, { motion: frame(600).motion })],
    ['capture gap', frame(1500)],
    ['backward capture time', frame(-100)],
    ['telemetry reset', frame(500, { motion: { ...frame(500).motion!, generation: 1 } })],
    ['heading discontinuity', frame(500, { motion: { ...frame(500).motion!, headingRad: 2 } })],
    ['calibration change', frame(500, { calibration: { ...calibration, heightM: 2 } })],
    ['filter change', frame(500, { filterConfidence: 0.9 })],
])('discards incompatible history on %s', (_label, next) => {
    const history = new TrackHistory();
    history.update(frame(0), road());
    expect(history.update(next, empty())).toEqual(empty());
});

it('does not advance motion on republish, and resets on stop or missing calibration', () => {
    const history = new TrackHistory();
    history.update(frame(0), road());
    const result = history.update(frame(500), road());
    expect(history.update(frame(500), road())).toBe(result);
    expect(history.update(null, null)).toBeNull();
    expect(history.update(frame(600), empty())).toEqual(empty());
    history.update(frame(800), road());
    expect(history.update(frame(900, { calibration: undefined }), null)).toBeNull();
    expect(history.update(frame(1000), empty())).toEqual(empty());
});

it('reads only live motion and never accesses world position or lap progress', () => {
    const sample: StandardTelemetrySample = { Graphics_status: 2, Physics_heading: 0.3,
        Physics_speed_kmh: 72, Physics_gear: 3 };
    for (const field of ['Graphics_car_coordinates', 'Graphics_normalized_car_position', 'Physics_velocity_x']) {
        Object.defineProperty(sample, field, { get() { throw new Error(`Must not read ${field}`); } });
    }
    expect(readVisionMotion(sample, 100, 4)).toEqual({ receivedAt: 100, generation: 4,
        headingRad: 0.3, forwardMps: 20, rightMps: 0 });
    expect(readVisionMotion({ ...sample, Physics_local_velocity_x: 1.5, Physics_local_velocity_z: -4 }, 100, 4))
        .toMatchObject({ forwardMps: -4, rightMps: 1.5 });
    expect(readVisionMotion({ ...sample, Physics_gear: 0 }, 100, 4)?.forwardMps).toBe(-20);
});

it.each([
    { Graphics_status: 1 }, { Graphics_status: 3 }, { Graphics_status: undefined },
    { Physics_heading: NaN }, { Physics_heading: undefined }, { Physics_speed_kmh: Infinity },
    { Physics_gear: undefined }, { Physics_gear: 1 }, { Physics_speed_kmh: -10 },
])('rejects unavailable/paused motion: %s', (patch) => {
    expect(readVisionMotion({ Graphics_status: 2, Physics_heading: 0, Physics_speed_kmh: 72, Physics_gear: 3,
        ...patch }, 0, 0)).toBeUndefined();
});
