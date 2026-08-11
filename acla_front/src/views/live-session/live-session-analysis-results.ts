import {
    AnalysisResultElement,
    normalizeAnalysisResultsData,
} from 'views/lap-analysis/visualization/charts/analysisResultsModel';

export interface LiveSessionBaselineMetadata {
    id: string;
    lap: number;
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

export type LiveAnalysisMistakeCountResult = {
    status: 'ready';
    mistake_count: number;
    practice_mistake_count: number;
    racing_mistake_count: number;
    page_id: string;
    baseline_lap: number;
    track: string;
    car: string;
} | {
    status: 'error';
    error: 'live_analysis_result_unavailable';
};

export type LiveAnalysisLabelResolver = (labelId: string) => string | undefined;

const getParentLabelNames = (
    id: 'MSP' | 'MSR',
    canonicalName: string,
    resolveLabel?: LiveAnalysisLabelResolver,
): Set<string> => new Set([
    id,
    canonicalName,
    resolveLabel?.(id),
].filter((label): label is string => Boolean(label)));

export const getLiveAnalysisMistakeCount = (
    page: LiveSessionAnalysisResultPage | null,
    resolveLabel?: LiveAnalysisLabelResolver,
): LiveAnalysisMistakeCountResult => {
    if (!page) {
        return { status: 'error', error: 'live_analysis_result_unavailable' };
    }

    const practiceLabels = getParentLabelNames('MSP', 'Mistake (Practice)', resolveLabel);
    const racingLabels = getParentLabelNames('MSR', 'Mistake (Racing)', resolveLabel);
    let mistakeCount = 0;
    let practiceMistakeCount = 0;
    let racingMistakeCount = 0;

    page.elements.forEach((element) => {
        const practice = element.labels.some((label) => practiceLabels.has(label));
        const racing = element.labels.some((label) => racingLabels.has(label));
        if (practice) practiceMistakeCount += 1;
        if (racing) racingMistakeCount += 1;
        if (practice || racing) mistakeCount += 1;
    });

    return {
        status: 'ready',
        mistake_count: mistakeCount,
        practice_mistake_count: practiceMistakeCount,
        racing_mistake_count: racingMistakeCount,
        page_id: page.id,
        baseline_lap: page.baseline.lap,
        track: page.baseline.track,
        car: page.baseline.car,
    };
};

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
