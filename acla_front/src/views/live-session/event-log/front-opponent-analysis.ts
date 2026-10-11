import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { StandardTelemetrySample } from '../live-session-types';
import type { DriverCornerRecord } from './CornerRecorder';
import type { DriverLapRecord } from './LapRecorder';
import { summarizeDriverCorners } from './corner-summary';
import { summarizeTimes, type TimingSummary } from './timing-summary';

interface ComparisonSamples {
    player_average_time_ms: number | null;
    opponent_average_time_ms: number | null;
    player_sample_count: number;
    opponent_sample_count: number;
    player_included_count: number;
    opponent_included_count: number;
}

export interface AvailableTimingComparison extends ComparisonSamples {
    status: 'available';
    player_average_time_ms: number;
    opponent_average_time_ms: number;
    delta_ms: number;
    reason: null;
}

export interface UnavailableTimingComparison extends ComparisonSamples {
    status: 'unavailable';
    delta_ms: null;
    reason: string;
}

interface CornerIdentity { corner_id: string; corner_name: string }
export type FrontOpponentCornerComparison = AvailableTimingComparison & CornerIdentity;

const COMPARISON_METHODS = {
    opponent_selection: 'nearest_positive_forward_distance_under_half_lap',
    corner_average: 'trimmed_average_complete_corner_passes',
    lap_average: 'trimmed_average_between_consecutive_interpolated_finish_line_crossings',
    clock: 'Graphics_current_time_str parsed by parseCornerTime',
    minimum_samples_per_car: 3,
    trim_rule: 'Remove max(1, floor(sample_count * 0.125)) samples from each end.',
    delta: 'player_minus_opponent_ms; rounded to milliseconds; negative means player faster',
} as const;

export type LiveFrontOpponentAiResult = {
    status: 'complete' | 'partial' | 'insufficient_data' | 'unavailable';
    player_car_id: string | null;
    opponent_car_id: string | null;
    comparison_methods: typeof COMPARISON_METHODS;
    overall_lap_comparison: AvailableTimingComparison | UnavailableTimingComparison;
    faster_corners: FrontOpponentCornerComparison[];
    slower_corners: FrontOpponentCornerComparison[];
    equal_corners: FrontOpponentCornerComparison[];
    insufficient_corners: Array<UnavailableTimingComparison & CornerIdentity>;
    reasons: string[];
};

const EMPTY_SUMMARY = summarizeTimes([]);
const normalized = (value: unknown): value is number => typeof value === 'number'
    && Number.isFinite(value) && value >= 0 && value <= 1;

function compareTimes(player = EMPTY_SUMMARY, opponent = EMPTY_SUMMARY, unit: string): AvailableTimingComparison | UnavailableTimingComparison {
    const samples: ComparisonSamples = {
        player_average_time_ms: player.averageTimeMs === null ? null : Math.round(player.averageTimeMs),
        opponent_average_time_ms: opponent.averageTimeMs === null ? null : Math.round(opponent.averageTimeMs),
        player_sample_count: player.sampleCount,
        opponent_sample_count: opponent.sampleCount,
        player_included_count: player.includedCount,
        opponent_included_count: opponent.includedCount,
    };
    if (player.averageTimeMs === null || opponent.averageTimeMs === null) return {
        ...samples, status: 'unavailable', delta_ms: null,
        reason: `At least three complete ${unit} per car are required (player: ${player.sampleCount}, opponent: ${opponent.sampleCount}).`,
    };
    return {
        ...samples, status: 'available', reason: null,
        player_average_time_ms: Math.round(player.averageTimeMs),
        opponent_average_time_ms: Math.round(opponent.averageTimeMs),
        delta_ms: Math.round(player.averageTimeMs - opponent.averageTimeMs) || 0,
    };
}

export function unavailableFrontOpponentAnalysis(reason: string, playerCarId: string | null = null): LiveFrontOpponentAiResult {
    return {
        status: 'unavailable', player_car_id: playerCarId, opponent_car_id: null,
        comparison_methods: COMPARISON_METHODS,
        overall_lap_comparison: { ...compareTimes(undefined, undefined, 'laps'), status: 'unavailable', delta_ms: null, reason },
        faster_corners: [], slower_corners: [], equal_corners: [], insufficient_corners: [], reasons: [reason],
    };
}

/** Compare only the mounted Event Log's completed timing records, never sums of corners. */
export function analyzeFrontOpponent(
    sample: StandardTelemetrySample,
    cornerRecords: DriverCornerRecord[],
    lapRecords: DriverLapRecord[],
): LiveFrontOpponentAiResult {
    const playerId = sample.Graphics_player_car_id;
    const playerKey = typeof playerId === 'number' && Number.isSafeInteger(playerId) && playerId >= 0 ? String(playerId) : null;
    const playerPosition = sample.Graphics_normalized_car_position;
    if (playerKey === null || !normalized(playerPosition)
        || (sample.Graphics_status !== undefined && sample.Graphics_status !== ACC_STATUS.ACC_LIVE)) {
        return unavailableFrontOpponentAnalysis('Live player ID and normalized position are required to select the nearest car ahead.', playerKey);
    }
    if (!sample.Graphics_normalized_positions) {
        return unavailableFrontOpponentAnalysis('Opponent normalized-position telemetry is unavailable.', playerKey);
    }
    const forwardDistance = (position: number) => (position - playerPosition + 1) % 1;
    const opponent = Object.entries(sample.Graphics_normalized_positions)
        .filter(([id, position]) => id !== playerKey && normalized(position)
            && forwardDistance(position) > 0 && forwardDistance(position) < 0.5)
        .sort(([, a], [, b]) => forwardDistance(a) - forwardDistance(b))[0];
    if (!opponent) return unavailableFrontOpponentAnalysis('No qualifying opponent is physically ahead within half a lap.', playerKey);
    const opponentKey = opponent[0];
    const result = unavailableFrontOpponentAnalysis('', playerKey);
    result.opponent_car_id = opponentKey;
    result.reasons = [];
    const lapSummary = (carId: string): TimingSummary => summarizeTimes(
        lapRecords.filter((record) => record.carId === carId).map((record) => record.lapTimeMs),
    );
    result.overall_lap_comparison = compareTimes(lapSummary(playerKey), lapSummary(opponentKey), 'laps');
    if (result.overall_lap_comparison.status === 'unavailable') {
        result.overall_lap_comparison.reason += ' Only uninterrupted Graphics_current_time_str intervals between consecutive finish-line crossings count; corner times are not substituted.';
        result.reasons.push(result.overall_lap_comparison.reason);
    }
    const summaries = summarizeDriverCorners(cornerRecords.filter((record) => record.carId === playerKey || record.carId === opponentKey));
    const playerCorners = new Map(summaries.filter((summary) => summary.carId === playerKey).map((summary) => [summary.cornerId, summary]));
    const opponentCorners = new Map(summaries.filter((summary) => summary.carId === opponentKey).map((summary) => [summary.cornerId, summary]));
    const cornerIds = Array.from(new Set(summaries.map((summary) => summary.cornerId)));
    cornerIds.forEach((cornerId) => {
        const player = playerCorners.get(cornerId);
        const opponent = opponentCorners.get(cornerId);
        const identity = { corner_id: cornerId, corner_name: (player ?? opponent)!.cornerName };
        const comparison = compareTimes(player, opponent, 'corner passes');
        if (comparison.status === 'unavailable') {
            result.insufficient_corners.push({ ...identity, ...comparison });
            result.reasons.push(`${identity.corner_name} (${cornerId}): ${comparison.reason}`);
        } else {
            const list = comparison.delta_ms < 0 ? result.faster_corners
                : comparison.delta_ms > 0 ? result.slower_corners : result.equal_corners;
            list.push({ ...identity, ...comparison });
        }
    });
    if (cornerIds.length === 0) result.reasons.push('No corner timing records for these cars. Open Live Map with tagged corners and collect three complete passes per car.');
    const byDifference = (a: FrontOpponentCornerComparison, b: FrontOpponentCornerComparison) => Math.abs(b.delta_ms) - Math.abs(a.delta_ms);
    result.faster_corners.sort(byDifference);
    result.slower_corners.sort(byDifference);
    const hasComparisons = result.overall_lap_comparison.status === 'available'
        || result.faster_corners.length + result.slower_corners.length + result.equal_corners.length > 0;
    result.status = hasComparisons ? (result.reasons.length > 0 ? 'partial' : 'complete') : 'insufficient_data';
    return result;
}
