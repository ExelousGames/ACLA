import type { SegmentClassificationResult } from 'views/session-shared/segment-classification';

export type RecordedAiAnalysisStatus = 'idle' | 'loading' | 'ready' | 'empty' | 'error';

export type RecordedActiveSegmentSummary = {
    segmentId?: string;
    startIndex: number;
    endIndex: number;
    trackSection: string;
    labels: string[];
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

export const getRecordedAnalysisStateForResult = (
    result: SegmentClassificationResult,
): Pick<RecordedAiAnalysisState, 'status' | 'message'> => (
    result.parent_segment_count > 0
        ? { status: 'ready' }
        : { status: 'empty', message: 'AI analysis found no classified segments.' }
);
