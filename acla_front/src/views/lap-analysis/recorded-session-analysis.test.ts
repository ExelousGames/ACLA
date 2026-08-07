import { normalizeSegmentClassificationResult } from './recorded-session-analysis';

describe('normalizeSegmentClassificationResult', () => {
    it('preserves the complete expert reference array for recorded and live results', () => {
        const expertReferenceData = [{
            raw_index: 7,
            expert_optimal_time: 1_250,
            expert_time_difference: 12.5,
            expert_optimal_player_pos_x: 100,
            expert_optimal_player_pos_y: 200,
            expert_optimal_player_pos_z: 300,
            Graphics_normalized_car_position: 0.45,
            expert_optimal_throttle: 0.8,
            expert_optimal_brake: 0.1,
            expert_optimal_gear: 4,
        }];

        const normalized = normalizeSegmentClassificationResult({
            status: 'success',
            session_id: 'session-1',
            samples_analyzed: 1,
            parent_segment_count: 0,
            segments: [],
            expert_reference_data: expertReferenceData,
            expert_time_available: true,
        }, 'fallback-session');

        expect(normalized.expert_reference_data).toBe(expertReferenceData);
        expect(normalized.expert_reference_data).toEqual(expertReferenceData);
        expect(normalized.expert_time_available).toBe(true);
    });

    it('defaults missing expert reference data to an empty array', () => {
        const normalized = normalizeSegmentClassificationResult({
            segments: [],
        }, 'session-1');

        expect(normalized.expert_reference_data).toEqual([]);
    });
});
