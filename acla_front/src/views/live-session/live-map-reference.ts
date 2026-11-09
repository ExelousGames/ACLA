import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import type { Vec3 } from 'views/session-shared/visualization/charts/mapTelemetry';

const normalizeTrack = (game: string, track: string): string => (
    game === 'acc' ? getAccTelemetryTrackKey(track) || track.trim() : track.trim()
);

export const selectLiveMapReference = (
    entries: TopLapReferenceEntry[],
    game: string | null,
    track: string | undefined,
    car: string | undefined,
    grip: number | undefined,
): TopLapReferenceEntry | null => {
    if (!game || !track || !car) return null;
    const trackKey = normalizeTrack(game, track);
    const requestedGrip = typeof grip === 'number' && Number.isFinite(grip) ? Math.max(0, Math.min(6, Math.round(grip))) : 2;
    let selected: TopLapReferenceEntry | null = null;
    entries.forEach((entry) => {
        if (entry.game !== game || normalizeTrack(game, entry.track) !== trackKey || entry.car !== car.trim()) return;
        // Match TopLapStore.predict: exact grip, otherwise the nearest available
        // bucket for this same game/track/car, preserving store order on ties.
        if (!selected || Math.abs(entry.avg_grip_int - requestedGrip) < Math.abs(selected.avg_grip_int - requestedGrip)) {
            selected = entry;
        }
    });
    return selected;
};

export const getLiveMapReferenceLine = (entry: TopLapReferenceEntry | null): Vec3[] => {
    if (!entry || entry.x.length < 2) return [];
    const indices = ['x', 'y', 'z'].map((axis) => entry.target_features.indexOf(`expert_optimal_player_pos_${axis}`));
    if (indices.some((index) => index < 0)) return [];
    return entry.y.map((row) => ({ x: row[indices[0]], y: row[indices[1]], z: row[indices[2]] }));
};

export const getLiveMapReferencePedal = (entry: TopLapReferenceEntry | null, pedal: 'brake' | 'throttle'): number[] => {
    if (!entry || entry.x.length < 2) return [];
    const index = entry.target_features.indexOf(`expert_optimal_${pedal}`);
    if (index < 0) return [];
    const values = entry.y.map((row) => row[index]);
    return values.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) ? values : [];
};

export const getLiveMapReferenceSpeed = (entry: TopLapReferenceEntry | null): number[] => {
    if (!entry || entry.x.length < 2) return [];
    const index = entry.target_features.indexOf('expert_optimal_speed');
    if (index < 0) return [];
    const values = entry.y.map((row) => row[index]);
    return values.every((value) => Number.isFinite(value) && value >= 0) ? values : [];
};

// Hit-test the rendered segments in canvas CSS pixels, independent of zoom and pixel density.
export const getLiveMapReferenceHit = (points: { x: number; y: number }[], pointer: { x: number; y: number }) => {
    let closest: { index: number; fraction: number; x: number; y: number } | null = null;
    let closestDistance = 8 ** 2;
    for (let index = 0; index < points.length - 1; index += 1) {
        const start = points[index];
        const dx = points[index + 1].x - start.x;
        const dy = points[index + 1].y - start.y;
        const lengthSquared = dx * dx + dy * dy;
        if (lengthSquared === 0) continue;
        const fraction = Math.max(0, Math.min(1, ((pointer.x - start.x) * dx + (pointer.y - start.y) * dy) / lengthSquared));
        const x = start.x + dx * fraction;
        const y = start.y + dy * fraction;
        const distance = (pointer.x - x) ** 2 + (pointer.y - y) ** 2;
        if (distance <= closestDistance && (!closest || distance < closestDistance)) {
            closest = { index, fraction, x, y };
            closestDistance = distance;
        }
    }
    return closest;
};
