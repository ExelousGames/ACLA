import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { drivingCenterlineInput } from './driving-centerline';
import { cloneConfig, drivingFeatureSizes, drivingObservationInputs, drivingPolicyInputs } from './driving-model-config';
import { DrivingMemory } from './driving-memory';
import { assembleDrivingInput } from './driving-policy-input';
import { readDrivingObservation } from './driving-observation';
import { emptyTrainingDocument, readTrainingDocument } from './driving-policy';
import { circuitMapFixture, trainingFixture } from './training-fixtures';

it.each(['acc', 'iracing'] as const)('encodes 50 steps on each side in relative map XYZ for %s', (game) => {
    const map = circuitMapFixture();
    map.game = game;
    // Sparse, unordered samples must retain map-resolution steps, not sample indices.
    map.samples.middle_line = map.samples.middle_line!.filter((p) => p.bin % 250 === 0).reverse()
        .map((p) => ({ ...p, x: 1000 + p.x, y: 500 + 2 * p.x + p.y, z: -100 + 3 * p.x }));
    const config = cloneConfig();
    config.observation.centerlineScale = 200;
    const features = drivingCenterlineInput(map, game, 'monza', 0.1255, config)!;
    expect(features).toHaveLength(300);
    const steps = [...Array.from({ length: 50 }, (_, i) => i - 50), ...Array.from({ length: 50 }, (_, i) => i + 1)];
    steps.forEach((step, i) => {
        [0.4, 0.8, 1.2].forEach((distance, axis) => expect(features[i * 3 + axis]).toBeCloseTo(step * distance / 200, 10));
    });
    const translated: CircuitMapDto = { ...map, samples: { middle_line: map.samples.middle_line!.map((p) =>
        ({ ...p, x: p.x + 8000, y: p.y - 7000, z: p.z + 4000 })) } };
    drivingCenterlineInput(translated, game, 'monza', 0.1255, config)!.forEach((value, i) =>
        expect(value).toBeCloseTo(features[i], 10));
});

it.each([0, 1])('wraps both sides at lap position %s without a duplicate origin', (position) => {
    const features = drivingCenterlineInput(circuitMapFixture(), 'acc', 'monza', position)!;
    for (let i = 0; i < 50; i++) {
        expect(features.slice(i * 3, i * 3 + 3)[0]).toBeCloseTo(0);
        expect(features[i * 3 + 1]).toBeCloseTo((50 - i) * 0.004);
        expect(features[(50 + i) * 3]).toBeCloseTo((i + 1) * 0.004);
        expect(features[(50 + i) * 3 + 1]).toBeCloseTo(0);
    }
    expect(features.every(Number.isFinite)).toBe(true);
});

it('uses the selected map resolution and preserves offsets larger than the input scale', () => {
    const map = circuitMapFixture();
    map.resolution = 2000;
    const config = cloneConfig();
    config.observation.centerlineScale = 1;
    const features = drivingCenterlineInput(map, 'acc', 'monza', 0.1255, config)!;
    expect(features[0]).toBeCloseTo(-10);
    expect(features[147]).toBeCloseTo(-0.2);
    expect(features[150]).toBeCloseTo(0.2);
    expect(features[297]).toBeCloseTo(10);
});

it('uses capture-time lap progress and carries centerline inputs through chronological model history', () => {
    const { live, vision, referenceLap, circuitMap } = trainingFixture();
    const config = cloneConfig();
    const read = () => readDrivingObservation(live, vision, referenceLap, 50, vision.capturedAt, config, circuitMap).observation!;
    // Absolute car position and the newer telemetry position must not change the origin.
    live.currentTelemetry = { ...live.currentTelemetry, Graphics_normalized_car_position: 0.75,
        Graphics_car_coordinates: [{ x: 10000, y: -10000, z: 2000 }] };
    const before = read();
    const expectedBefore = drivingCenterlineInput(circuitMap, 'acc', 'monza', 0.25)!;
    expect(before.features.slice(63, 363)).toEqual(expectedBefore);
    const memory = new DrivingMemory(config);
    memory.encode(before);
    vision.capturedAt = live.receivedAt = vision.telemetry!.receivedAt = 1200;
    vision.telemetry!.currentTelemetry = { ...vision.telemetry!.currentTelemetry, Graphics_normalized_car_position: 0.26 };
    const after = read();
    const expectedAfter = drivingCenterlineInput(circuitMap, 'acc', 'monza', 0.26)!;
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

it.each(['missing', 'empty', 'single', 'invalid coordinates', 'invalid resolution', 'wrong game', 'wrong circuit'])(
    'requires a usable centerline and reports %s map data', (fault) => {
        const { live, vision, referenceLap } = trainingFixture();
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
        expect(result.observation).toBeUndefined();
        expect(result.error).toMatch(/Live Map.*centerline/);
    });

it('matches ACC display names and reads an updated selected centerline', () => {
    const map = circuitMapFixture();
    delete map.source_track_key;
    map.circuit_name = 'Autodromo Nazionale Monza';
    const before = drivingCenterlineInput(map, 'acc', 'monza', 0.25)!;
    const changed: CircuitMapDto = { ...map, samples: { middle_line: map.samples.middle_line!.map((p) => ({ ...p, y: p.y * 2 })) } };
    const after = drivingCenterlineInput(changed, 'acc', 'monza', 0.25)!;
    expect(before).toHaveLength(300);
    expect(after[298]).toBeCloseTo(2 * before[298]);
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
