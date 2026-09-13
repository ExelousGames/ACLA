import { normalizeSegmentClassificationResult } from './recorded-session-analysis';

describe('normalizeSegmentClassificationResult', () => {
    it('preserves repeated label names at distinct ranges without adding hierarchy', () => {
        const labels = [
            { label_name: 'MSP1', start_index: 125, end_index: 130 },
            { label_name: 'MSP1', start_index: 150, end_index: 160 },
        ];
        const result = normalizeSegmentClassificationResult({
            segments: [{
                id: 'section-id:120-180',
                labels,
                track_section: 'section-id',
                start_index: 120,
                end_index: 180,
                time_gap: { start_ms: 150, end_ms: 420, delta_ms: 270 },
                expert_reference_data: [],
            }],
        }, 'session-1');

        expect(result.segments[0].labels).toEqual(labels);
        expect(result.segments[0].time_gap).toEqual({ start_ms: 150, end_ms: 420, delta_ms: 270 });
    });

    it('normalizes expert references on each segment and preserves live availability', () => {
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
            parent_segment_count: 2,
            segments: [{
                id: 'segment-1',
                labels: [{ label_name: 'EA', start_index: 7, end_index: 8 }],
                start_index: 7,
                end_index: 8,
                expert_reference_data: expertReferenceData,
            }, {
                id: 'segment-2',
                labels: [{ label_name: 'MSP', start_index: 12, end_index: 13 }],
                start_index: 12,
                end_index: 13,
            }],
            expert_time_available: true,
        } as any, 'fallback-session');

        expect(normalized.segments[0].expert_reference_data).toBe(expertReferenceData);
        expect(normalized.segments[0].expert_reference_data).toEqual(expertReferenceData);
        expect(normalized.segments[1].expert_reference_data).toEqual([]);
        expect(normalized).not.toHaveProperty('expert_reference_data');
        expect(normalized.expert_time_available).toBe(true);
    });

    it('does not use a removed top-level expert reference array as a fallback', () => {
        const normalized = normalizeSegmentClassificationResult({
            segments: [{
                labels: [],
                start_index: 0,
                end_index: 1,
            }],
            expert_reference_data: [{ raw_index: 0 }],
        } as any, 'session-1');

        expect(normalized.segments[0].expert_reference_data).toEqual([]);
        expect(normalized).not.toHaveProperty('expert_reference_data');
    });
});
