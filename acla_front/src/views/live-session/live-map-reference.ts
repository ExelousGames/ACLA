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
