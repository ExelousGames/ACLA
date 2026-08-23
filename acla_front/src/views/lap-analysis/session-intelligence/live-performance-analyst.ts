import { TelemetrySample } from './types';

export type LiveSessionType = 'solo_practice' | 'traffic_or_race' | 'unknown';

const toFiniteNumber = (value: unknown): number | undefined => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
};

const countActiveCars = (value: unknown): number | undefined => {
    if (Array.isArray(value)) {
        const count = value.filter((item) => {
            const parsed = Number(item);
            return Number.isFinite(parsed) && parsed !== 0;
        }).length;
        return count > 0 ? count : undefined;
    }

    return toFiniteNumber(value);
};

export const detectLiveSessionType = (sample: TelemetrySample | null | undefined): LiveSessionType => {
    if (!sample) return 'unknown';

    const counts = [
        countActiveCars(sample.Graphics_active_cars_count),
        countActiveCars(sample.Graphics_active_cars),
        countActiveCars(sample.Graphics?.active_cars),
        countActiveCars(sample.Static_num_cars),
        countActiveCars(sample.Static?.num_cars),
    ].filter((count): count is number => count !== undefined);

    if (counts.length === 0) return 'unknown';
    if (counts.some((count) => count <= 1)) return 'solo_practice';
    return 'traffic_or_race';
};

export const getTelemetryLap = (sample: TelemetrySample | null | undefined): number => (
    toFiniteNumber(sample?.Graphics_completed_lap)
    ?? 0
);

export const getTelemetryPosition = (sample: TelemetrySample | null | undefined): number | undefined => {
    const value = toFiniteNumber(
        sample?.Graphics_normalized_car_position
        ?? sample?.graphics_normalized_car_position
        ?? sample?.normalized_car_position
        ?? sample?.car_position
    );

    if (value === undefined) return undefined;
    return Math.max(0, Math.min(1, value));
};

export const getTelemetryTrack = (sample: TelemetrySample | null | undefined): string => (
    typeof sample?.Static_track === 'string' && sample.Static_track
        ? sample.Static_track
        : typeof sample?.Static?.track === 'string'
            ? sample.Static.track
            : ''
);

export const getTelemetryCar = (sample: TelemetrySample | null | undefined): string => (
    typeof sample?.Static_car_model === 'string' && sample.Static_car_model
        ? sample.Static_car_model
        : typeof sample?.Static?.car_model === 'string'
            ? sample.Static.car_model
            : ''
);
