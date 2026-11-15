import { fetchCircuitMapById, fetchCircuitMapList } from 'services/circuitMapService';
import { loadTopLapReferenceModel, TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import type { RecordedDrivingSession } from './recorded-driving-session';

export interface RecordedDrivingReferences {
    laps: TopLapReferenceEntry[];
    maps: Map<string, CircuitMapDto>;
}

export const recordedTrackKey = (game: string, track: string) => game === 'acc'
    ? getAccTelemetryTrackKey(track) || track.trim() : track.trim();

export async function loadRecordedDrivingReferences(session: RecordedDrivingSession): Promise<RecordedDrivingReferences> {
    if (session.game !== 'acc' && session.game !== 'iracing') throw new Error('Recorded driving training supports ACC and iRacing.');
    const game = session.game;
    const [laps, summaries] = await Promise.all([loadTopLapReferenceModel(), fetchCircuitMapList(game)]);
    const tracks = new Set(session.rows.map(({ sample }) => recordedTrackKey(game, sample.Static_track || '')));
    const maps = new Map<string, CircuitMapDto>();
    for (const track of Array.from(tracks)) {
        const summary = summaries.find((candidate) => candidate.game === game
            && recordedTrackKey(game, candidate.source_track_key || '') === track);
        if (summary) maps.set(track, await fetchCircuitMapById(summary.id, game));
    }
    return { laps, maps };
}
