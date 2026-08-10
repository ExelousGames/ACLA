import { createLiveSessionAnalysisResultPage } from '../live-session-analysis-results';

describe('live session analysis result pages', () => {
    it('preserves nullable lap timing in the created page metadata', () => {
        const page = createLiveSessionAnalysisResultPage({
            baseline: {
                id: 'baseline-1',
                lap: 7,
                lap_time_ms: 98_765,
                captured_at: 123,
                track: 'Spa',
                car: 'GT3',
                sample_count: 3,
            },
            elements: [{ id: 'result-1', labels: ['MSP'] }],
        });

        expect(page.baseline.lap_time_ms).toBe(98_765);
        expect(page.elements).toEqual([expect.objectContaining({ id: 'result-1' })]);
    });
});
