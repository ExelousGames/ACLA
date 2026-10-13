import apiService from 'services/api.service';
import {
    CircuitMapDto,
    CircuitMapGame,
    CircuitMapSamplesByMode,
    CircuitMapSummaryDto
} from 'views/circuit-maps/circuit-map-types';
import { CIRCUIT_MAP_BIN_RESOLUTION } from 'views/circuit-maps/circuit-map-utils';
import { getCircuitMapCenterlineSegments } from 'views/circuit-maps/centerline-segments';

const countSamples = (samplesByMode: CircuitMapSamplesByMode): number => (
    Object.values(samplesByMode).reduce((sum, samples) => sum + (samples?.length || 0), 0)
);

const normalizeGame = (game: unknown, fallback: CircuitMapGame = 'acc'): CircuitMapGame => (
    game === 'acc' || game === 'iracing' ? game : fallback
);

export const normalizeCircuitMapList = (data: any): CircuitMapSummaryDto[] => {
    const rows = Array.isArray(data) ? data : Array.isArray(data?.list) ? data.list : [];

    return rows
        .map((row: any): CircuitMapSummaryDto | null => {
            const id = String(row.id ?? row.map_id ?? row.MapId ?? '');
            const circuitName = String(row.circuit_name ?? row.name ?? row.map_name ?? '');
            const game = normalizeGame(row.game);

            if (!id || !circuitName) return null;

            return {
                id,
                game,
                circuit_name: circuitName,
                source_track_key: row.source_track_key ?? null,
                updated_at: row.updated_at ?? null,
                sample_count: Number(row.sample_count ?? 0)
            };
        })
        .filter((row: CircuitMapSummaryDto | null): row is CircuitMapSummaryDto => row !== null);
};

export const normalizeCircuitMap = (data: any, fallbackGame: CircuitMapGame = 'acc'): CircuitMapDto => {
    const rawSamples = data?.samples || {};
    const samples: CircuitMapSamplesByMode = {
        left_boundary: Array.isArray(rawSamples.left_boundary) ? rawSamples.left_boundary : [],
        middle_line: Array.isArray(rawSamples.middle_line) ? rawSamples.middle_line : [],
        right_boundary: Array.isArray(rawSamples.right_boundary) ? rawSamples.right_boundary : [],
        pit_lane: Array.isArray(rawSamples.pit_lane) ? rawSamples.pit_lane : []
    };

    return {
        id: String(data?.id ?? data?.map_id ?? ''),
        game: normalizeGame(data?.game, fallbackGame),
        circuit_name: String(data?.circuit_name ?? data?.name ?? ''),
        source_track_key: data?.source_track_key ?? null,
        updated_at: data?.updated_at ?? null,
        sample_count: Number(data?.sample_count ?? countSamples(samples)),
        resolution: Number(data?.resolution ?? CIRCUIT_MAP_BIN_RESOLUTION),
        samples,
        centerline_segments: getCircuitMapCenterlineSegments({
            centerline_segments: data?.centerline_segments,
            centerline_tags: Array.isArray(data?.centerline_tags) ? data.centerline_tags : []
        })
    };
};

export const fetchCircuitMapList = async (game: CircuitMapGame): Promise<CircuitMapSummaryDto[]> => {
    const response = await apiService.get<any>('/circuit-map/list', { game });
    return normalizeCircuitMapList(response.data);
};

export const fetchCenterlineTagOptions = async (): Promise<string[]> => {
    const response = await apiService.get<{ tags: string[] }>('/circuit-map/centerline-tags');
    return Array.isArray(response.data.tags) ? response.data.tags : [];
};

export const fetchCircuitMapById = async (
    mapId: string,
    fallbackGame: CircuitMapGame = 'acc'
): Promise<CircuitMapDto> => {
    const response = await apiService.get<any>(`/circuit-map/${encodeURIComponent(mapId)}`);
    return normalizeCircuitMap(response.data, fallbackGame);
};
