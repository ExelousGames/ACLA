import {
    alignCircuitMapSamples,
    extractAccCaptureSample,
    getCircuitMapBin,
    getCircuitMapDrawSegments,
    upsertCircuitMapSample
} from '../circuit-map-utils';
import { CircuitMapBinSample } from '../circuit-map-types';

const makeTelemetryRow = (normalizedPosition: number) => ({
    Graphics_normalized_car_position: normalizedPosition,
    Graphics_current_time: 1000,
    Graphics_car_coordinates: JSON.stringify([
        { x: 10, y: 2, z: 30 },
        { x: 100, y: 20, z: 300 }
    ]),
    Graphics_car_id: JSON.stringify([42, 99]),
    Graphics_player_car_id: 42
});

describe('circuit map utilities', () => {
    it('bins normalized positions and clamps the finish line to the final bin', () => {
        expect(getCircuitMapBin(0)).toBe(0);
        expect(getCircuitMapBin(0.123)).toBe(123);
        expect(getCircuitMapBin(1)).toBe(999);
        expect(getCircuitMapBin(-0.1)).toBeNull();
    });

    it('extracts the ACC player coordinate for a valid capture sample', () => {
        const capture = extractAccCaptureSample(makeTelemetryRow(0.25));

        expect(capture).toEqual({
            bin: 250,
            normalizedPosition: 0.25,
            position: { x: 10, y: 2, z: 30 }
        });
    });

    it('ignores invalid ACC samples', () => {
        expect(extractAccCaptureSample({ Graphics_car_coordinates: '[]' })).toBeNull();
        expect(extractAccCaptureSample({ Graphics_normalized_car_position: 'bad' })).toBeNull();
    });

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

    it('aligns boundary samples by bin index', () => {
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
