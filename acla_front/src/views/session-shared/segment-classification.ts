import { normalizeSegmentLabels, SegmentClassificationSegment } from './visualization/charts/segmentClassificationDisplay';

export type ExpertReferenceRow = {
    raw_index: number;
    expert_optimal_time: number;
    expert_time_difference: number;
    expert_optimal_player_pos_x: number;
    expert_optimal_player_pos_y: number;
    expert_optimal_player_pos_z: number;
    Graphics_normalized_car_position: number;
    expert_optimal_throttle: number;
    expert_optimal_brake: number;
    expert_optimal_gear: number;
};

export type SegmentClassificationResult = {
    status: string;
    session_id: string;
    samples_analyzed: number;
    parent_segment_count: number;
    segments: Array<SegmentClassificationSegment & {
        expert_reference_data: ExpertReferenceRow[];
    }>;
    expert_time_available?: boolean;
};

export const normalizeSegmentClassificationResult = (
    result: Partial<SegmentClassificationResult> | null | undefined,
    sessionId: string,
): SegmentClassificationResult => {
    const segments = result && Array.isArray(result.segments)
        ? result.segments.map((segment) => ({
            ...segment,
            labels: normalizeSegmentLabels(segment.labels),
            track_section: typeof segment.track_section === 'string' ? segment.track_section : undefined,
            expert_reference_data: Array.isArray(segment.expert_reference_data)
                ? segment.expert_reference_data
                : [],
        }))
        : [];

    return {
        status: result?.status || 'success',
        session_id: result?.session_id || sessionId,
        samples_analyzed: Number(result?.samples_analyzed) || 0,
        parent_segment_count: segments.length,
        segments,
        ...(typeof result?.expert_time_available === 'boolean'
            ? { expert_time_available: result.expert_time_available }
            : {}),
    };
};
