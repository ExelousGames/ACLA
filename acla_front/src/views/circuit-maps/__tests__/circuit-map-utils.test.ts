import {
    alignCircuitMapSamples,
    extractCircuitMapCaptureSample,
    getCircuitMapBin,
    getCircuitMapDrawSegments,
    getCircuitMapName,
    getCircuitMapTrackKey,
    mergeCircuitMapSamples,
    upsertCircuitMapSample
} from '../circuit-map-utils';
import { CircuitMapBinSample } from '../circuit-map-types';
import type { StandardTelemetrySample } from 'views/live-session/live-session-types';

const makeTelemetryRow = (normalizedPosition: number) => ({
    Graphics_status: 2,
    Graphics_normalized_car_position: normalizedPosition,
    Graphics_current_time: 1000,
    Graphics_car_coordinates: [
        { x: 10, y: 2, z: 30 },
        { x: 100, y: 20, z: 300 }
    ],
    Graphics_car_id: [42, 99],
    Graphics_player_car_id: 42
});

describe('circuit map utilities', () => {
    it('merges standard player coordinates across chunks and preserves locked bins', () => {
        const row = { ...makeTelemetryRow(0.25), Graphics_status: 2 };
        const first = mergeCircuitMapSamples([], [row]);
        const locked = { ...first.samples[0], bin: 500, locked: true };
        const result = mergeCircuitMapSamples([...first.samples, locked], [
            { ...row, Graphics_car_coordinates: [{ x: 20, y: 4, z: 60 }] },
            { ...row, Graphics_normalized_car_position: 0.5 },
            { ...row, Graphics_normalized_car_position: 1, Graphics_player_car_id: 63,
                Graphics_car_id: [63, -1], Graphics_car_coordinates: [{ x: 0, y: 0, z: 0 }] },
        ]);
        expect(result.capturedRows).toBe(3);
        expect(result.samples).toEqual([
            expect.objectContaining({ bin: 250, x: 15, y: 3, z: 45, sample_count: 2 }),
            locked,
            expect.objectContaining({ bin: 999, x: 0, y: 0, z: 0, sample_count: 1 }),
        ]);
        expect(first.samples[0].sample_count).toBe(1);
    });

    it.each([
        { Graphics_status: 0 },
        { Graphics_status: undefined },
        { Graphics_normalized_car_position: null },
        { Graphics_normalized_car_position: -1 },
        { Graphics_normalized_car_position: 1.1 },
        { Graphics_normalized_car_position: NaN },
        { Graphics_normalized_car_position: '0.25' },
        { Graphics_player_car_id: -1 },
        { Graphics_player_car_id: undefined },
        { Graphics_player_car_id: 63 },
        { Graphics_car_coordinates: undefined },
        { Graphics_car_coordinates: [null, { x: 1, y: 2, z: 3 }] },
        { Graphics_car_coordinates: [{ x: NaN, y: 0, z: 0 }] },
    ])('rejects unusable player samples in both live and file capture: %o', (fields) => {
        const row = { ...makeTelemetryRow(0.25), ...fields } as unknown as StandardTelemetrySample;
        expect(extractCircuitMapCaptureSample(row)).toBeNull();
        const result = mergeCircuitMapSamples([], [row]);
        expect(result).toEqual({ samples: [], capturedRows: 0 });
    });

    it('bins normalized positions and clamps the finish line to the final bin', () => {
        expect(getCircuitMapBin(0)).toBe(0);
        expect(getCircuitMapBin(0.123)).toBe(123);
        expect(getCircuitMapBin(1)).toBe(999);
        expect(getCircuitMapBin(-0.1)).toBeNull();
        expect(getCircuitMapBin(1.1)).toBeNull();
    });

    it('extracts the standard player coordinate for a valid capture sample', () => {
        const capture = extractCircuitMapCaptureSample(makeTelemetryRow(0.25));

        expect(capture).toEqual({
            bin: 250,
            normalizedPosition: 0.25,
            position: { x: 10, y: 2, z: 30 }
        });
    });

    it('reads track identity only from standard telemetry and keeps ACC display aliases', () => {
        expect(getCircuitMapTrackKey({ Static_track: 'spa - grandprix' })).toBe('spa - grandprix');
        expect(getCircuitMapTrackKey({})).toBeNull();
        expect(getCircuitMapTrackKey({ Static_track: ' ' })).toBeNull();
        expect(getCircuitMapName('monza', 'acc')).toBe('Autodromo Nazionale Monza');
        expect(getCircuitMapName('monza', 'iracing')).toBe('monza');
    });

    it.each([undefined, null, '', ' ', false, true, 'bad', NaN, Infinity, -0.1, 1.1])(
        'rejects capture without a valid normalized position: %s', (position) => {
            expect(extractCircuitMapCaptureSample({ ...makeTelemetryRow(0.25), Graphics_normalized_car_position: position } as unknown as StandardTelemetrySample)).toBeNull();
        }
    );

    it('averages repeated live samples in the same bin', () => {
        const first = upsertCircuitMapSample([], {
            bin: 100,
            normalizedPosition: 0.1,
            position: { x: 10, y: 0, z: 20 }
        }, '2026-01-01T00:00:00.000Z');

        const second = upsertCircuitMapSample(first, {
            bin: 100,
            normalizedPosition: 0.1,
            position: { x: 20, y: 2, z: 40 }
        }, '2026-01-01T00:00:01.000Z');

        expect(second).toEqual([{
            bin: 100,
            normalized_position: 0.1,
            x: 15,
            y: 1,
            z: 30,
            sample_count: 2,
            updated_at: '2026-01-01T00:00:01.000Z'
        }]);
    });

    it('does not overwrite locked manual bins', () => {
        const locked: CircuitMapBinSample = {
            bin: 100,
            normalized_position: 0.1,
            x: 5,
            y: 0,
            z: 8,
            sample_count: 1,
            updated_at: '2026-01-01T00:00:00.000Z',
            locked: true
        };

        const next = upsertCircuitMapSample([locked], {
            bin: 100,
            normalizedPosition: 0.1,
            position: { x: 20, y: 0, z: 40 }
        });

        expect(next).toEqual([locked]);
    });

    it('aligns boundary, middle line, and pit lane samples by bin index', () => {
        const rows = alignCircuitMapSamples({
            left_boundary: [{
                bin: 10,
                normalized_position: 0.01,
                x: 1,
                y: 0,
                z: 1,
                sample_count: 1,
                updated_at: '2026-01-01T00:00:00.000Z'
            }],
            right_boundary: [{
                bin: 10,
                normalized_position: 0.01,
                x: 3,
                y: 0,
                z: 3,
                sample_count: 1,
                updated_at: '2026-01-01T00:00:00.000Z'
            }],
            middle_line: [{
                bin: 10,
                normalized_position: 0.01,
                x: 2,
                y: 0,
                z: 2,
                sample_count: 1,
                updated_at: '2026-01-01T00:00:00.000Z'
            }],
            pit_lane: [{
                bin: 10,
                normalized_position: 0.01,
                x: 2,
                y: 0,
                z: 2,
                sample_count: 1,
                updated_at: '2026-01-01T00:00:00.000Z'
            }]
        });

        expect(rows).toHaveLength(1);
        expect(rows[0].bin).toBe(10);
        expect(rows[0].left_boundary?.x).toBe(1);
        expect(rows[0].right_boundary?.x).toBe(3);
        expect(rows[0].middle_line?.x).toBe(2);
        expect(rows[0].pit_lane?.x).toBe(2);
    });

    it('splits pit lane drawing across the lap start instead of connecting its ends', () => {
        const samples: CircuitMapBinSample[] = [
            {
                bin: 20,
                normalized_position: 0.02,
                x: 1,
                y: 0,
                z: 1,
                sample_count: 1,
                updated_at: '2026-01-01T00:00:00.000Z'
            },
            {
                bin: 35,
                normalized_position: 0.035,
                x: 2,
                y: 0,
                z: 2,
                sample_count: 1,
                updated_at: '2026-01-01T00:00:00.000Z'
            },
            {
                bin: 940,
                normalized_position: 0.94,
                x: 9,
                y: 0,
                z: 9,
                sample_count: 1,
                updated_at: '2026-01-01T00:00:00.000Z'
            },
            {
                bin: 960,
                normalized_position: 0.96,
                x: 10,
                y: 0,
                z: 10,
                sample_count: 1,
                updated_at: '2026-01-01T00:00:00.000Z'
            }
        ];

        expect(getCircuitMapDrawSegments(samples, 'pit_lane').map((segment) => segment.map((sample) => sample.bin))).toEqual([
            [20, 35],
            [940, 960]
        ]);
        expect(getCircuitMapDrawSegments(samples, 'left_boundary').map((segment) => segment.map((sample) => sample.bin))).toEqual([
            [20, 35, 940, 960]
        ]);
    });
});
