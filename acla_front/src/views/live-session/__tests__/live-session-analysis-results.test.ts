import {
    createLiveSessionAnalysisResultPage,
} from '../live-session-analysis-results';

describe('live session analysis result pages', () => {
    it('preserves nullable lap timing in the created page metadata', () => {
        const page = createLiveSessionAnalysisResultPage({
            baseline: {
                id: 'baseline-1',
                lap_id: 0,
                lap_time_ms: 98_765,
                captured_at: 123,
                track: 'Spa',
                car: 'GT3',
                sample_count: 3,
            },
            elements: [{ id: 'result-1', labels: labelRanges('MSP') }],
        });

        expect(page.baseline.lap_time_ms).toBe(98_765);
        expect(page.baseline.lap_id).toBe(0);
        expect(page.elements).toEqual([expect.objectContaining({ id: 'result-1' })]);
    });
});

function labelRanges(...names: string[]) {
    return names.map((label_name) => ({ label_name, start_index: 0, end_index: 1 }));
}
