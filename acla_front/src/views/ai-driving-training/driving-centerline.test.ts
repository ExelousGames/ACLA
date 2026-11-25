import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { drivingCenterlineInput } from './driving-centerline';
import { cloneConfig, drivingFeatureSizes, drivingObservationInputs, drivingPolicyInputs } from './driving-model-config';
import { DrivingMemory } from './driving-memory';
import { assembleDrivingInput } from './driving-policy-input';
import { readDrivingObservation } from './driving-observation';
import { emptyTrainingDocument, readTrainingDocument } from './driving-policy';
import { parseRecordedDrivingSession, readRecordedDrivingObservation } from './recorded-driving-session';
import { circuitMapFixture, trainingFixture } from './training-fixtures';

it.each(['acc', 'iracing'] as const)('encodes 50 steps on each side in relative map XYZ for %s', (game) => {
    const map = circuitMapFixture();
    map.game = game;
    // Retain map-resolution steps behind the car even when the samples are sparse and unordered.
    map.samples.middle_line = map.samples.middle_line!.filter((p) => p.bin % 250 === 0).reverse()
        .map((p) => ({ ...p, x: 1000 + p.x, y: 500 + 2 * p.x + p.y, z: -100 + 3 * p.x }));
    const config = cloneConfig();
    config.observation.centerlineScale = 200;
    const speed = 12 * Math.sqrt(14);
    const features = drivingCenterlineInput(map, game, 'monza', 0.1255, speed, config)!;
    expect(features).toHaveLength(300);
    const steps = [...Array.from({ length: 50 }, (_, i) => i - 50), ...Array.from({ length: 50 }, (_, i) => i + 1)];
    steps.forEach((step, i) => {
        [0.4, 0.8, 1.2].forEach((distance, axis) => expect(features[i * 3 + axis]).toBeCloseTo(step * distance / 200, 10));
    });
    const translated: CircuitMapDto = { ...map, samples: { middle_line: map.samples.middle_line!.map((p) =>
        ({ ...p, x: p.x + 8000, y: p.y - 7000, z: p.z + 4000 })) } };
    drivingCenterlineInput(translated, game, 'monza', 0.1255, speed, config)!.forEach((value, i) =>
        expect(value).toBeCloseTo(features[i], 10));
});

it.each([0, 1])('wraps both sides at lap position %s without a duplicate origin', (position) => {
    const features = drivingCenterlineInput(circuitMapFixture(), 'acc', 'monza', position, 12)!;
    for (let i = 0; i < 50; i++) {
        expect(features.slice(i * 3, i * 3 + 3)[0]).toBeCloseTo(0);
        expect(features[i * 3 + 1]).toBeCloseTo((50 - i) * 0.004);
        expect(features[(50 + i) * 3]).toBeCloseTo((i + 1) * 0.004);
        expect(features[(50 + i) * 3 + 1]).toBeCloseTo(0);
    }
    expect(features.every(Number.isFinite)).toBe(true);
});

it('samples six seconds of arc distance ahead at capture speed on a nonuniform map', () => {
    const map = circuitMapFixture();
    const positions = [0, 0.1, 0.6, 0.9];
    map.samples.middle_line = map.samples.middle_line!.filter(({ bin }) => bin % 250 === 0)
        .map((point, i) => ({ ...point, normalized_position: positions[i] }));
    const [start] = map.samples.middle_line;
    map.samples.middle_line.push({ ...start, normalized_position: 1, bin: 1000 });

    const stationary = drivingCenterlineInput(map, 'acc', 'monza', 0.05, 0)!;
    expect(stationary.slice(150)).toEqual(new Array(150).fill(0));
    for (const speed of [18, 72]) {
        const features = drivingCenterlineInput(map, 'acc', 'monza', 0.05, speed)!;
        expect(features.slice(0, 150)).toEqual(stationary.slice(0, 150));
        for (let step = 1; step <= 50; step++) {
            const distance = speed / 3.6 * 6 * step / 50;
            const offset = 150 + (step - 1) * 3;
            expect(features[offset]).toBeCloseTo(Math.min(50, distance) / 100, 10);
            expect(features[offset + 1]).toBeCloseTo(Math.max(0, distance - 50) / 100, 10);
        }
    }
    // Start 10 m before the finish line and travel 20 m, continuing onto the next lap.
    const wrapped = drivingCenterlineInput(map, 'acc', 'monza', 0.99, 12)!;
    expect(wrapped[297]).toBeCloseTo(0.1, 10);
    expect(wrapped[298]).toBeCloseTo(-0.1, 10);
});

it('uses the selected map resolution and preserves offsets larger than the input scale', () => {
    const map = circuitMapFixture();
    map.resolution = 2000;
    const config = cloneConfig();
    config.observation.centerlineScale = 1;
    const features = drivingCenterlineInput(map, 'acc', 'monza', 0.1255, 12, config)!;
    expect(features[0]).toBeCloseTo(-10);
    expect(features[147]).toBeCloseTo(-0.2);
    expect(features[150]).toBeCloseTo(0.4);
    expect(features[297]).toBeCloseTo(20);
});

it('uses capture-time lap progress and speed and carries centerline inputs through chronological model history', () => {
    const { live, vision, referenceLap, circuitMap } = trainingFixture();
    const config = cloneConfig();
    const read = () => readDrivingObservation(live, vision, referenceLap, 50, vision.capturedAt, config, circuitMap).observation!;
    // Use the camera-paired sample for both the origin and the forward horizon.
    live.currentTelemetry = { ...live.currentTelemetry, Graphics_normalized_car_position: 0.75,
        Graphics_car_coordinates: [{ x: 10000, y: -10000, z: 2000 }], Physics_speed_kmh: 200 };
    const before = read();
    const expectedBefore = drivingCenterlineInput(circuitMap, 'acc', 'monza', 0.25, 108)!;
    expect(before.features.slice(63, 363)).toEqual(expectedBefore);
    const memory = new DrivingMemory(config);
    memory.encode(before);
    vision.capturedAt = live.receivedAt = vision.telemetry!.receivedAt = 1200;
    vision.telemetry!.currentTelemetry = { ...vision.telemetry!.currentTelemetry, Graphics_normalized_car_position: 0.26,
        Physics_speed_kmh: 54 };
    const after = read();
    const expectedAfter = drivingCenterlineInput(circuitMap, 'acc', 'monza', 0.26, 54)!;
    expect(after.features.slice(63, 363)).toEqual(expectedAfter);
    expect(expectedAfter).not.toEqual(expectedBefore);
    const input = assembleDrivingInput(memory.encode(after), config);
    const offset = 2 + 63 + config.network.cameraProjectionUnits;
    expect(input.steps).toBe(2);
    expect(input.features.slice(offset, offset + 300)).toEqual(expectedBefore);
    expect(input.features.slice(drivingPolicyInputs(config) + offset, drivingPolicyInputs(config) + offset + 300)).toEqual(expectedAfter);
    expect(drivingFeatureSizes(config).centerline).toBe(300);
    expect(drivingObservationInputs(config)).toBe(366);
    expect(drivingPolicyInputs(config)).toBe(624);
});

it('uses the recorded frame’s paired speed for the centerline horizon', () => {
    const { live, vision, referenceLap, circuitMap } = trainingFixture();
    const rows = [18, 72].map((speed, i) => JSON.stringify({ ...live.currentTelemetry,
        Physics_speed_kmh: speed, Graphics_analyst_session_time: 10 + i }));
    const recording = parseRecordedDrivingSession(rows.join('\n'), 'acc');
    const read = (seconds: number) => readRecordedDrivingObservation(recording, seconds, vision.camera!,
        referenceLap, 50, cloneConfig(), circuitMap).observation!.features.slice(360, 363);
    const slow = read(10.1), fast = read(11.1);
    expect(slow[0]).toBeCloseTo(0);
    expect(slow[1]).toBeCloseTo(0.3);
    expect(fast[0]).toBeCloseTo(-0.2);
    expect(fast[1]).toBeCloseTo(1);
});

it.each(['missing', 'empty', 'single', 'invalid coordinates', 'invalid resolution', 'wrong game', 'wrong circuit'])(
    'zero-fills centerline features for %s map data', (fault) => {
        const { live, vision, referenceLap, observation } = trainingFixture();
        let map: CircuitMapDto | null = circuitMapFixture(referenceLap);
        if (fault === 'missing') map = null;
        if (fault === 'empty') map!.samples.middle_line = [];
        if (fault === 'single') map!.samples.middle_line = map!.samples.middle_line!.slice(0, 1);
        if (fault === 'invalid coordinates') map!.samples.middle_line!.forEach((p) => { p.x = NaN; });
        if (fault === 'invalid resolution') map!.resolution = 0;
        if (fault === 'wrong game') map!.game = 'iracing';
        if (fault === 'wrong circuit') map!.source_track_key = 'spa';
        // Use a new selected map object so it has no previously prepared centerline.
        const result = readDrivingObservation(live, vision, referenceLap, 50, 1000, undefined, map ? { ...map } : null);
        expect(result.error).toBeUndefined();
        expect(result.observation?.centerlineAvailable).toBe(false);
        expect(result.observation?.features).toEqual([
            ...observation.features.slice(0, 63), ...new Array(300).fill(0), ...observation.features.slice(363),
        ]);
    });

it('matches ACC display names and reads an updated selected centerline', () => {
    const map = circuitMapFixture();
    delete map.source_track_key;
    map.circuit_name = 'Autodromo Nazionale Monza';
    const before = drivingCenterlineInput(map, 'acc', 'monza', 0.25, 12)!;
    const changed: CircuitMapDto = { ...map, samples: { middle_line: map.samples.middle_line!.map((p) => ({ ...p, x: p.x + p.y })) } };
    const after = drivingCenterlineInput(changed, 'acc', 'monza', 0.25, 12)!;
    expect(before).toHaveLength(300);
    expect(before[297]).toBeCloseTo(0, 10);
    expect(after[297]).toBeCloseTo(before[298] * Math.SQRT1_2, 10);
});

it.each(['old layout', 'missing feature', 'wrong encoding', 'wrong steps', 'zero scale'])(
    'rejects %s model configuration without migration', (fault) => {
        const document = emptyTrainingDocument();
        const observation: any = document.configuration.observation;
        if (fault === 'old layout' || fault === 'missing feature') {
            observation.features = observation.features.filter((name: string) => name !== 'centerline');
            document.configuration.memory.features = document.configuration.memory.features.filter((name) => name !== 'centerline');
        }
        if (fault === 'old layout') {
            delete observation.centerlineEncoding;
            delete observation.centerlineStepsPerSide;
            delete observation.centerlineScale;
        }
        if (fault === 'wrong encoding') observation.centerlineEncoding = 'absolute-xyz';
        if (fault === 'wrong steps') observation.centerlineStepsPerSide = 49;
        if (fault === 'zero scale') observation.centerlineScale = 0;
        const original = JSON.stringify(document);
        expect(() => readTrainingDocument(document)).toThrow(/configuration/);
        expect(JSON.stringify(document)).toBe(original);
    });
