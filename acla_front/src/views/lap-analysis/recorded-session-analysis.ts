import { SegmentClassificationSegment } from './visualization/charts/segmentClassificationDisplay';

export type RecordedAiAnalysisStatus = 'idle' | 'loading' | 'ready' | 'empty' | 'error';

export type SegmentClassificationResult = {
    status: string;
    session_id: string;
    samples_analyzed: number;
    segment_count: number;
    segments: SegmentClassificationSegment[];
};

export type RecordedActiveSegmentSummary = {
    segmentId?: string;
    startIndex: number;
    endIndex: number;
    parentLabel: string;
    childLabels: string[];
};

export type RecordedPlaybackSummary = {
    sessionId: string | null;
    sampleCount: number;
    durationSeconds: number;
    playbackIndex: number;
    playbackTimeSeconds: number;
    activeSegment: RecordedActiveSegmentSummary | null;
};

export type RecordedAiAnalysisState = {
    sessionId: string | null;
    status: RecordedAiAnalysisStatus;
    message?: string;
    result: SegmentClassificationResult | null;
};

export const createIdleRecordedAiAnalysis = (sessionId: string | null = null): RecordedAiAnalysisState => ({
    sessionId,
    status: 'idle',
    result: null,
});

export const createEmptyRecordedPlaybackSummary = (
    sessionId: string | null = null,
): RecordedPlaybackSummary => ({
    sessionId,
    sampleCount: 0,
    durationSeconds: 0,
    playbackIndex: 0,
    playbackTimeSeconds: 0,
    activeSegment: null,
});

export const normalizeSegmentClassificationResult = (
    result: Partial<SegmentClassificationResult> | null | undefined,
    sessionId: string,
): SegmentClassificationResult => ({
    ...(result || {}),
    segments: result && Array.isArray(result.segments) ? result.segments : [],
    segment_count: Number(result?.segment_count) || 0,
    samples_analyzed: Number(result?.samples_analyzed) || 0,
    session_id: result?.session_id || sessionId,
    status: result?.status || 'success',
});

export const getRecordedAnalysisStateForResult = (
    result: SegmentClassificationResult,
): Pick<RecordedAiAnalysisState, 'status' | 'message'> => (
    result.segment_count > 0
        ? { status: 'ready' }
        : { status: 'empty', message: 'AI analysis found no classified segments.' }
);
