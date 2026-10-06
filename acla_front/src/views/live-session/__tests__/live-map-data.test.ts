import type { CircuitMapBinSample, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapCars, getLiveMapMiddleLine, getLiveMapPosition } from '../live-map-data';

const point = (normalized_position: number, x: number, z = 0): CircuitMapBinSample => ({
    normalized_position, x, y: 0, z, bin: Math.floor(normalized_position * 1000), sample_count: 1, updated_at: '',
});

describe('live map normalized positions', () => {
    const middleLine = [point(0.1, 0), point(0.5, 100), point(0.9, 20)];

    it('uses only the downloaded middle line, sorted by normalized position without mutating the map', () => {
        const samples = [point(0.9, 20), point(0.1, 0), point(0.5, 100), point(0.5, 999), point(-1, 0), point(0.2, NaN)];
        const map = { samples: { middle_line: samples, left_boundary: [point(0, 999)] } } as CircuitMapDto;
        expect(getLiveMapMiddleLine(map)).toEqual(middleLine);
        expect(samples[0].normalized_position).toBe(0.9);
        expect(getLiveMapMiddleLine({ samples: { left_boundary: samples } } as CircuitMapDto)).toEqual([]);
    });

    it('interpolates by lap position rather than sample index and wraps at the finish line', () => {
        expect(getLiveMapPosition(middleLine, 0.3)?.x).toBeCloseTo(50);
        expect(getLiveMapPosition(middleLine, 0.5)?.x).toBe(100);
        expect(getLiveMapPosition(middleLine, 0)?.x).toBeCloseTo(10);
        expect(getLiveMapPosition(middleLine, 1)?.x).toBeCloseTo(10);
        expect(getLiveMapPosition(middleLine, 0.95)?.x).toBeCloseTo(15);
        expect(getLiveMapPosition(middleLine, 0.05)?.x).toBeCloseTo(5);
    });

    it('handles explicit lap endpoints and rejects unavailable or invalid positions', () => {
        const endpoints = [point(0, 0), point(0.5, 100), point(1, 0)];
        expect(getLiveMapPosition(endpoints, 1)).toEqual({ x: 0, y: 0, z: 0 });
        expect(getLiveMapPosition(endpoints, 0.75)?.x).toBe(50);
        [-1, 1.1, NaN, Infinity].forEach((value) => expect(getLiveMapPosition(middleLine, value)).toBeNull());
        expect(getLiveMapPosition([], 0)).toBeNull();
        expect(getLiveMapPosition([point(0, 0)], 0)).toBeNull();
    });

    it('identifies cars by native IDs, preferring the player lap position', () => {
        const cars = getLiveMapCars({
            Graphics_player_car_id: 1052,
            Graphics_normalized_car_position: 0.5,
            Graphics_normalized_positions: { 63: 0.1, 1052: 0.9 },
        }, middleLine);
        expect(cars).toEqual([
            { key: '63', isPlayer: false, position: { x: 0, y: 0, z: 0 } },
            { key: '1052', isPlayer: true, position: { x: 100, y: 0, z: 0 } },
        ]);
    });

    it('supports player-only telemetry and never substitutes another car for a missing player', () => {
        expect(getLiveMapCars({ Graphics_normalized_car_position: 0.5 }, middleLine)).toEqual([
            { key: 'player', isPlayer: true, position: { x: 100, y: 0, z: 0 } },
        ]);
        expect(getLiveMapCars({ Graphics_player_car_id: 63, Graphics_normalized_positions: { 63: 0.5 } }, middleLine)[0].isPlayer).toBe(true);
        expect(getLiveMapCars({ Graphics_player_car_id: 63, Graphics_normalized_positions: { 0: 0.1 } }, middleLine)[0].isPlayer).toBe(false);
        expect(getLiveMapCars({}, middleLine)).toEqual([]);
    });
});
