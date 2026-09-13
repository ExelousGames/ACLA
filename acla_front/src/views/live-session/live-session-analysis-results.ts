import {
    AnalysisResultElement,
    normalizeAnalysisResultsData,
} from 'views/lap-analysis/visualization/charts/analysisResultsModel';

export interface LiveSessionBaselineMetadata {
    id: string;
    lap_id: number;
    lap_time_ms: number | null;
    captured_at: number;
    track: string;
    car: string;
    sample_count: number;
}

export interface LiveSessionAnalysisResultPage {
    id: string;
    createdAt: number;
    baseline: LiveSessionBaselineMetadata;
    elements: AnalysisResultElement[];
}

export interface AppendLiveSessionAnalysisResultPageInput {
    baseline: LiveSessionBaselineMetadata;
    elements: unknown[];
}

export interface AppendLiveSessionAnalysisResultPageResult {
    pageId: string;
    pageCount: number;
}

let generatedPageSequence = 0;

const createLiveSessionAnalysisResultPageId = (): string => (
    `baseline-analysis-page-${Date.now().toString(36)}-${(++generatedPageSequence).toString(36)}`
);

export const createLiveSessionAnalysisResultPage = (
    input: AppendLiveSessionAnalysisResultPageInput,
): LiveSessionAnalysisResultPage => ({
    id: createLiveSessionAnalysisResultPageId(),
    createdAt: Date.now(),
    baseline: { ...input.baseline },
    elements: normalizeAnalysisResultsData({ elements: input.elements }).elements,
});
