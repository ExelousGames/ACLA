import {
    createLiveSessionAnalysisResultPage,
    getLiveAnalysisMistakeCount,
    type LiveSessionAnalysisResultPage,
} from '../live-session-analysis-results';

const createPage = (elements: LiveSessionAnalysisResultPage['elements']): LiveSessionAnalysisResultPage => ({
    id: 'page-latest',
    createdAt: 456,
    baseline: {
        id: 'baseline-latest',
        lap: 8,
        lap_time_ms: 98_000,
        captured_at: 123,
        track: 'Spa',
        car: 'GT3',
        sample_count: 10,
    },
    elements,
});

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

    it('counts each result once while recognizing ids, canonical names, and configured names', () => {
        const result = getLiveAnalysisMistakeCount(createPage([
            { id: 'practice-id', labels: ['MSP', 'MSP'] },
            { id: 'practice-canonical', labels: ['Mistake (Practice)'] },
            { id: 'practice-configured', labels: ['Training Error'] },
            { id: 'racing-id', labels: ['MSR'] },
            { id: 'racing-canonical', labels: ['Mistake (Racing)'] },
            { id: 'racing-configured', labels: ['Race Error'] },
            { id: 'combined', labels: ['MSP', 'MSR', 'MSP', 'MSR'] },
            { id: 'children-only', labels: ['MSP1', 'MSR1'] },
            { id: 'unrelated', labels: ['Expert Adherence'] },
        ]), (id) => (id === 'MSP' ? 'Training Error' : 'Race Error'));

        expect(result).toEqual({
            status: 'ready',
            mistake_count: 7,
            practice_mistake_count: 4,
            racing_mistake_count: 4,
            page_id: 'page-latest',
            baseline_lap: 8,
            track: 'Spa',
            car: 'GT3',
        });
    });

    it('returns successful zero counts for an analyzed empty page', () => {
        expect(getLiveAnalysisMistakeCount(createPage([]))).toMatchObject({
            status: 'ready',
            mistake_count: 0,
            practice_mistake_count: 0,
            racing_mistake_count: 0,
            page_id: 'page-latest',
        });
    });

    it('returns the explicit unavailable error when no analysis page exists', () => {
        expect(getLiveAnalysisMistakeCount(null)).toEqual({
            status: 'error',
            error: 'live_analysis_result_unavailable',
        });
    });
});
