import type { CircuitMapBinSample, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { Vec3 } from 'views/session-shared/visualization/charts/mapTelemetry';
import type { StandardTelemetrySample } from './live-session-types';

export type LiveMapCar = {
    key: string;
    isPlayer: boolean;
    position: Vec3;
};

const isNormalizedPosition = (value: unknown): value is number => (
    typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
);

export const getLiveMapMiddleLine = (map: CircuitMapDto | null): CircuitMapBinSample[] => (
    (map?.samples.middle_line || [])
        .filter((sample) => isNormalizedPosition(sample.normalized_position)
            && [sample.x, sample.y, sample.z].every((value) => typeof value === 'number' && Number.isFinite(value)))
        .sort((left, right) => left.normalized_position - right.normalized_position)
        .filter((sample, index, samples) => index === 0
            || sample.normalized_position !== samples[index - 1].normalized_position)
);

// Interpolate by lap progress, including the interval across the start/finish line.
export const getLiveMapPosition = (middleLine: CircuitMapBinSample[], position: number): Vec3 | null => {
    if (middleLine.length < 2 || !isNormalizedPosition(position)) return null;

    let low = 0;
    let high = middleLine.length;
    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (middleLine[mid].normalized_position < position) low = mid + 1;
        else high = mid;
    }
    const next = middleLine[low % middleLine.length];
    if (next.normalized_position === position) return { x: next.x, y: next.y, z: next.z };

    const previous = middleLine[(low + middleLine.length - 1) % middleLine.length];
    const start = previous.normalized_position - (low === 0 ? 1 : 0);
    const end = next.normalized_position + (low === middleLine.length ? 1 : 0);
    const fraction = (position - start) / (end - start);
    return {
        x: previous.x + (next.x - previous.x) * fraction,
        y: previous.y + (next.y - previous.y) * fraction,
        z: previous.z + (next.z - previous.z) * fraction,
    };
};

export const getLiveMapCars = (sample: StandardTelemetrySample, middleLine: CircuitMapBinSample[]): LiveMapCar[] => {
    const playerId = sample.Graphics_player_car_id;
    const playerKey = typeof playerId === 'number' && Number.isSafeInteger(playerId) && playerId >= 0
        ? String(playerId) : null;
    const positions = new Map(Object.entries(sample.Graphics_normalized_positions || {}));
    if (isNormalizedPosition(sample.Graphics_normalized_car_position)) {
        positions.set(playerKey ?? 'player', sample.Graphics_normalized_car_position);
    }

    const cars: LiveMapCar[] = [];
    positions.forEach((normalizedPosition, key) => {
        const position = getLiveMapPosition(middleLine, normalizedPosition);
        if (position) cars.push({ key, isPlayer: key === playerKey || key === 'player', position });
    });
    return cars;
};
