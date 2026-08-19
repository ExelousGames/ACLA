import React, {
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    NamedAiToolComponentHandle,
    awaitNamedComponentHandle,
    resolveNamedComponentHandle,
    useAiToolComponentRefDirectory,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import {
    AnalysisResultsVisualizationUnavailableError,
    BaselineAnalysisCancelledError,
    BaselineCollectionAlreadyStartedError,
    BaselineCollectionIncompleteError,
    BaselineCollectionNotStartedError,
    BaselineLapRecordRequiredError,
    RecordedAnalysisFailedError,
} from 'contexts/AiToolComponentError';
import apiService from 'services/api.service';
import type { AiChatHandle } from 'views/lap-analysis/ai-chat/ai-chat';
import {
    detectLiveSessionType,
    getTelemetryCar,
    getTelemetryLap,
    getTelemetryPosition,
    getTelemetryTrack,
} from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import {
    normalizeSegmentClassificationResult,
    type SegmentClassificationResult,
} from 'views/lap-analysis/recorded-session-analysis';
import type { VisualizationManagerHandle } from 'views/lap-analysis/visualization/VisualizationPanelManager';
import type { AnalysisResultsChartHandle } from 'views/lap-analysis/visualization/charts/AnalysisResultsChart';
import type { AnalysisResultElement } from 'views/lap-analysis/visualization/charts/analysisResultsModel';
import { adaptAnalysisResultsComparison } from 'views/lap-analysis/visualization/charts/analysisResultsComparisonAdapter';
import { getSegmentLabelIds } from 'views/lap-analysis/visualization/charts/segmentClassificationDisplay';
import { getSingletonVisualizationComponentName } from 'views/lap-analysis/visualization/visualization-component-names';
import BaselineProgressDisplay from './BaselineProgressDisplay';
import { LiveSessionContext } from './LiveSessionContext';
import {
    createAiToolDeferred,
    createAiToolOperation,
    createAiToolOperationFrom,
    type AiToolDeferred,
    type AiToolOperation,
} from 'components/ai-engineering-tools';
import type { AiOverlayComponentHandle } from 'views/floating-chat/ai-overlay-types';
import { OVERLAY_HOLD_MS } from 'views/floating-chat/ai-overlay-types';

export type BaselineCollectionTag = {
    status: 'waiting_for_start' | 'collecting' | 'complete';
    progress_percent: number;
    detail: string;
    track: string | null;
    car: string | null;
    current_lap: number | null;
    baseline_lap: number | null;
};

export type BaselineLapRecord = {
    id: string;
    lap: number;
    lap_time_ms: number | null;
    captured_at: number;
    track: string;
    car: string;
    sample_count: number;
    snapshot: Record<string, any>;
    records: Record<string, any>[];
};

export type BaselineCollectionPayload = {
    progress_percent: number;
    status: BaselineCollectionTag['status'];
    car: string | null;
    track: string | null;
    message: string;
};

export type BaselineAnalysisPayload = {
    status: 'ready' | 'empty';
    message: string;
    analysis: {
        status: string;
        session_id: string;
        samples_analyzed: number;
        segments: Record<string, any>[];
    };
    source: 'baseline_lap_record';
    baseline: Omit<BaselineLapRecord, 'snapshot' | 'records'>;
    chartId: string | null;
    component_name: string;
    pageId: string;
    pageCount: number;
};

export type BaselineCollectionStatus = BaselineCollectionPayload & {
    event: 'baseline_waiting' | 'baseline_collecting' | 'baseline_progress';
    milestone: number;
    skipped?: boolean;
};

export type BaselineCollectionOptions = { timeoutMs?: number };

export interface BaselineCollectionHandle extends NamedAiToolComponentHandle, AiOverlayComponentHandle<BaselineCollectionTag | null> {
    startCollection(options?: BaselineCollectionOptions): AiToolOperation<BaselineCollectionPayload, BaselineCollectionStatus>;
    restartCollection(): AiToolOperation<BaselineCollectionPayload>;
    requestAnalysis(options?: { limit?: number }): AiToolOperation<BaselineAnalysisPayload>;
    getTag(): BaselineCollectionTag | null;
    getLapRecord(): BaselineLapRecord | null;
    subscribe(listener: (tag: BaselineCollectionTag | null) => void): () => void;
}

type PendingBaselineOperation = {
    result: AiToolDeferred<BaselineCollectionPayload>;
    statuses: Array<{ milestone: number; deferred: AiToolDeferred<BaselineCollectionStatus> }>;
    timeoutId: ReturnType<typeof setTimeout> | null;
};

type BaselineRecorderState = {
    status: BaselineCollectionTag['status'];
    rows: Record<string, any>[];
    sampleKeys: Set<string>;
    startLap: number | null;
    startPosition: number;
    currentLap: number;
    currentPosition: number;
    lastPosition: number | null;
    canStartAtBoundary: boolean;
    lapCounterAdvancePending: boolean;
    track: string;
    car: string;
    completedRecord: BaselineLapRecord | null;
};

const BASELINE_START_POSITION_EPSILON = 0.005;
const BASELINE_WRAP_THRESHOLD = 0.65;

type BaselineAnalysisState = 'idle' | 'analyzing' | 'complete' | 'error';

const cloneSample = (sample: Record<string, any>): Record<string, any> => ({ ...sample });

const getSampleKey = (
    sample: Record<string, any>,
    lap: number,
    position: number,
): string => [
    lap,
    position,
    sample.Graphics_current_time ?? sample.Graphics?.current_time ?? '',
    sample.Physics_timestamp ?? sample.timestamp ?? '',
].join(':');

const isTelemetrySample = (value: unknown): value is Record<string, any> => (
    Boolean(value)
    && typeof value === 'object'
    && !Array.isArray(value)
    && Object.keys(value as Record<string, any>).length > 0
);

const getUniqueTelemetryRows = (
    rows: readonly Record<string, any>[],
): { rows: Record<string, any>[]; sampleKeys: Set<string> } => {
    const sampleKeys = new Set<string>();
    const uniqueRows = rows.filter((row) => {
        if (!isTelemetrySample(row)) return false;
        const position = getTelemetryPosition(row);
        if (position === undefined) return false;
        const key = getSampleKey(row, getTelemetryLap(row), position);
        if (sampleKeys.has(key)) return false;
        sampleKeys.add(key);
        return true;
    });

    return { rows: uniqueRows.map(cloneSample), sampleKeys };
};

const getContinuationRows = (
    rows: readonly Record<string, any>[],
    currentLap: number,
    completedLap: number,
): { rows: Record<string, any>[]; sampleKeys: Set<string> } => {
    const currentLapRows = rows.filter((row) => (
        isTelemetrySample(row)
        && getTelemetryLap(row) === currentLap
        && getTelemetryPosition(row) !== undefined
    ));
    let continuationRows = currentLapRows;

    if (currentLap === completedLap) {
        let mostRecentWrapIndex = -1;
        for (let index = 1; index < currentLapRows.length; index += 1) {
            const previousPosition = getTelemetryPosition(currentLapRows[index - 1]);
            const position = getTelemetryPosition(currentLapRows[index]);
            if (
                previousPosition !== undefined
                && position !== undefined
                && previousPosition - position > BASELINE_WRAP_THRESHOLD
            ) {
                mostRecentWrapIndex = index;
            }
        }
        continuationRows = mostRecentWrapIndex >= 0
            ? currentLapRows.slice(mostRecentWrapIndex)
            : [];
    }

    return getUniqueTelemetryRows(continuationRows);
};

const createEmptyRecorderState = (
    currentTelemetry?: Record<string, any> | null,
): BaselineRecorderState => {
    const sample = isTelemetrySample(currentTelemetry) ? currentTelemetry : null;
    const lap = sample ? getTelemetryLap(sample) : 0;
    const position = sample ? getTelemetryPosition(sample) ?? 0 : 0;

    return {
        status: 'waiting_for_start',
        rows: [],
        sampleKeys: new Set(sample ? [getSampleKey(sample, lap, position)] : []),
        startLap: null,
        startPosition: 0,
        currentLap: lap,
        currentPosition: position,
        lastPosition: sample ? position : null,
        canStartAtBoundary: !sample || position > BASELINE_START_POSITION_EPSILON,
        lapCounterAdvancePending: false,
        track: sample ? getTelemetryTrack(sample) : '',
        car: sample ? getTelemetryCar(sample) : '',
        completedRecord: null,
    };
};

const hasCompletedRecordingLap = (
    state: BaselineRecorderState,
    lap: number,
    position: number,
): boolean => (
    state.startLap !== null
    && state.rows.length > 0
    && (
        lap > state.startLap
        || (
            state.lastPosition !== null
            && state.lastPosition - position > BASELINE_WRAP_THRESHOLD
        )
    )
);

const getCollectionProgress = (state: BaselineRecorderState): number => {
    if (state.status === 'complete') return 100;
    if (state.status !== 'collecting') return 0;

    const rawProgress = state.currentPosition >= state.startPosition
        ? state.currentPosition - state.startPosition
        : 1 - state.startPosition + state.currentPosition;

    return Math.max(1, Math.min(99, Math.round(rawProgress * 100)));
};

const buildRecorderSnapshot = (state: BaselineRecorderState): Record<string, any> => ({
    status: 'ready',
    track: state.track,
    car: state.car,
    current_lap: state.currentLap,
    completed_laps: state.currentLap,
    normalized_position: state.currentPosition,
    sample_count: state.rows.length,
    live_session_type: state.rows.length > 0
        ? detectLiveSessionType(state.rows[state.rows.length - 1])
        : 'unknown',
    baseline_ready: state.status === 'complete',
    baseline_collection_started: state.status !== 'waiting_for_start',
    baseline_progress_percent: getCollectionProgress(state),
    baseline_lap: state.startLap,
    completed_lap_count: state.status === 'complete' ? 1 : 0,
});

const toNullableFiniteNumber = (value: unknown): number | null => {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
};

const toPositiveFiniteNumber = (value: unknown): number | null => {
    if (typeof value !== 'number' && typeof value !== 'string') return null;
    const parsed = toNullableFiniteNumber(value);
    return parsed !== null && parsed > 0 ? parsed : null;
};

export const getCompletedBaselineLapTimeMs = (
    completionSample: Record<string, any>,
    recordedSamples: readonly Record<string, any>[],
): number | null => {
    const exactLastLapTime = toPositiveFiniteNumber(completionSample.Graphics_last_time)
        ?? toPositiveFiniteNumber(completionSample.Graphics?.last_time);
    if (exactLastLapTime !== null) return exactLastLapTime;

    return recordedSamples.reduce<number | null>((highest, sample) => {
        const currentLapTime = toPositiveFiniteNumber(sample.Graphics_current_time)
            ?? toPositiveFiniteNumber(sample.Graphics?.current_time);
        if (currentLapTime === null) return highest;
        return highest === null ? currentLapTime : Math.max(highest, currentLapTime);
    }, null);
};

const getAnalysisLimit = (value: unknown): number => {
    const parsed = Math.floor(Number(value));
    return Number.isFinite(parsed) && parsed > 0 ? Math.min(parsed, 50) : 8;
};

const getLabelText = (chat: AiChatHandle, labelId?: string | null): string | null => (
    labelId ? chat.getLabelName(labelId) || labelId : null
);

const compactSegment = (segment: any, chat: AiChatHandle) => ({
    id: segment.id ?? null,
    start_index: segment.start_index,
    end_index: segment.end_index,
    track_section: getLabelText(chat, segment.track_section),
    labels: getSegmentLabelIds(segment)
        .map((labelId) => getLabelText(chat, labelId))
        .filter(Boolean),
    ...(segment.time_gap ? { time_gap: segment.time_gap } : {}),
});

const getBaselinePosition = (records: Record<string, any>[], index: number): number | null => {
    const row = records[Math.max(0, Math.min(records.length - 1, Math.trunc(index)))];
    const parsed = Number(
        row?.Graphics_normalized_car_position
        ?? row?.normalized_position
        ?? row?.normalizedPosition,
    );
    return Number.isFinite(parsed) ? parsed : null;
};

const buildAnalysisElements = (
    result: SegmentClassificationResult,
    chat: AiChatHandle,
    records: Record<string, any>[],
): AnalysisResultElement[] => result.segments.map((segment, index) => {
    const start = getBaselinePosition(records, segment.start_index);
    const end = getBaselinePosition(records, segment.end_index);
    const comparison = segment.expert_reference_data.length
        ? adaptAnalysisResultsComparison({
            baselineRecords: records,
            expertReferenceData: segment.expert_reference_data,
        })
        : undefined;
    return {
        id: segment.id || `${result.session_id}:segment:${index}`,
        labels: getSegmentLabelIds(segment)
            .map((labelId) => chat.getLabelName(labelId) || labelId),
        ...(segment.track_section ? {
            section: chat.getLabelName(segment.track_section) || segment.track_section,
        } : {}),
        ...(start !== null && end !== null ? {
            normalizedPositionRange: { start, end },
        } : {}),
        ...(comparison?.samples.length ? { comparison } : {}),
        metadata: {
            source: 'ai_classifier',
            start_index: segment.start_index,
            end_index: segment.end_index,
        },
    };
});

const ensureAnalysisResultsChart = async (
    directory: AiToolComponentRefDirectory,
) => {
    const manager = resolveNamedComponentHandle<VisualizationManagerHandle>(
        directory,
        AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER,
    );
    const name = getSingletonVisualizationComponentName('analysis-results');
    const mountedHandle = directory.findComponentRef<AnalysisResultsChartHandle>(name)?.current;
    if (mountedHandle) {
        const instance = manager.getCurrentVisualizations()
            .find((visualization) => visualization.name === name);
        return {
            success: true as const,
            chartId: instance?.id ?? null,
            componentName: name,
            handle: mountedHandle,
        };
    }

    const requested = manager.requestVisualization({
        name,
        type: 'analysis-results',
    });
    const mountedName = requested.componentName || name;
    const handle = await awaitNamedComponentHandle<AnalysisResultsChartHandle>(directory, mountedName);
    return {
        success: true as const,
        chartId: requested.chartId ?? null,
        componentName: mountedName,
        handle,
    };
};

export const buildBaselineCollectionTag = (
    snapshot: Record<string, any>,
): BaselineCollectionTag => {
    const progress = Math.max(0, Math.min(100, Number(snapshot.baseline_progress_percent ?? 0)));
    const ready = snapshot.baseline_ready === true;
    const status = ready
        ? 'complete'
        : snapshot.baseline_collection_started
            ? 'collecting'
            : 'waiting_for_start';
    const detail = ready
        ? 'Baseline complete. Classifier request is ready.'
        : snapshot.baseline_collection_started
            ? `Lap ${Number(snapshot.current_lap ?? 0) + 1} baseline`
            : 'Waiting for the next lap start';

    return {
        status,
        progress_percent: progress,
        detail,
        track: typeof snapshot.track === 'string' && snapshot.track ? snapshot.track : null,
        car: typeof snapshot.car === 'string' && snapshot.car ? snapshot.car : null,
        current_lap: toNullableFiniteNumber(snapshot.current_lap),
        baseline_lap: toNullableFiniteNumber(snapshot.baseline_lap),
    };
};

export const buildBaselineCollectionToolPayload = (
    tag: BaselineCollectionTag | null,
    record: BaselineLapRecord | null,
): BaselineCollectionPayload => {
    const message = record
        ? 'Baseline complete. Cached lap record is ready.'
        : tag?.detail ?? 'Waiting for baseline collection to start.';

    return {
        progress_percent: record ? 100 : tag?.progress_percent ?? 0,
        status: record ? 'complete' : tag?.status ?? 'waiting_for_start',
        car: record?.car ?? tag?.car ?? null,
        track: record?.track ?? tag?.track ?? null,
        message,
    };
};

const BaselineCollection = ({ name }: { name: string }) => {
    const {
        currentTelemetry,
        sessionIntelligence,
        appendAnalysisResultPage,
    } = useContext(LiveSessionContext);
    const componentRefs = useAiToolComponentRefDirectory();
    const currentTelemetryRef = useRef(currentTelemetry);
    currentTelemetryRef.current = currentTelemetry;

    const [enabled, setEnabled] = useState(false);
    const [tag, setTag] = useState<BaselineCollectionTag | null>(null);
    const [analysisState, setAnalysisState] = useState<BaselineAnalysisState>('idle');
    const enabledRef = useRef(false);
    const tagRef = useRef<BaselineCollectionTag | null>(null);
    const lapRecordRef = useRef<BaselineLapRecord | null>(null);
    const recorderRef = useRef<BaselineRecorderState>(createEmptyRecorderState());
    const tagListenersRef = useRef<Set<(tag: BaselineCollectionTag | null) => void>>(new Set());
    const pendingCollectionOperationsRef = useRef<Set<PendingBaselineOperation>>(new Set());
    const analysisRequestRef = useRef<Promise<BaselineAnalysisPayload> | null>(null);
    const collectionGenerationRef = useRef(0);

    const publishTag = useCallback((nextTag: BaselineCollectionTag | null) => {
        tagRef.current = nextTag;
        setTag(nextTag);
        tagListenersRef.current.forEach((listener) => listener(nextTag));
    }, []);

    const settleCollectionStatus = useCallback((nextTag: BaselineCollectionTag) => {
        const payload = buildBaselineCollectionToolPayload(nextTag, lapRecordRef.current);
        pendingCollectionOperationsRef.current.forEach((operation) => {
            operation.statuses.forEach(({ milestone, deferred }) => {
                if (deferred.settled) return;
                const reached = milestone === 0
                    ? nextTag.status === 'waiting_for_start'
                    : milestone === 1
                        ? nextTag.status === 'collecting' || nextTag.status === 'complete'
                        : nextTag.progress_percent >= milestone;
                if (!reached) return;
                deferred.resolve({
                    ...payload,
                    event: milestone === 0
                        ? 'baseline_waiting'
                        : milestone === 1
                            ? 'baseline_collecting'
                            : 'baseline_progress',
                    milestone,
                });
            });
        });
    }, []);

    const settlePendingCollectionOperations = useCallback((
        outcome: BaselineCollectionPayload | Error,
    ) => {
        const operations = Array.from(pendingCollectionOperationsRef.current);
        pendingCollectionOperationsRef.current.clear();
        operations.forEach((operation) => {
            if (operation.timeoutId) clearTimeout(operation.timeoutId);
            const payload = outcome instanceof Error
                ? buildBaselineCollectionToolPayload(tagRef.current, lapRecordRef.current)
                : outcome;
            operation.statuses.forEach(({ milestone, deferred }) => {
                if (!deferred.settled) {
                    deferred.resolve({
                        ...payload,
                        event: milestone === 0
                            ? 'baseline_waiting'
                            : milestone === 1
                                ? 'baseline_collecting'
                                : 'baseline_progress',
                        milestone,
                        skipped: true,
                    });
                }
            });
            if (outcome instanceof Error) operation.result.reject(outcome);
            else operation.result.resolve(outcome);
        });
    }, []);

    const beginCollection = useCallback((nextRecorder: BaselineRecorderState) => {
        collectionGenerationRef.current += 1;
        recorderRef.current = nextRecorder;
        enabledRef.current = true;
        lapRecordRef.current = null;
        analysisRequestRef.current = null;
        setAnalysisState('idle');
        const nextTag = buildBaselineCollectionTag(buildRecorderSnapshot(nextRecorder));
        publishTag(nextTag);
        settleCollectionStatus(nextTag);
        setEnabled(true);
        return buildBaselineCollectionToolPayload(nextTag, null);
    }, [publishTag, settleCollectionStatus]);

    const beginFreshCollection = useCallback(() => beginCollection(
        createEmptyRecorderState(currentTelemetryRef.current),
    ), [beginCollection]);

    const beginContinuedCollection = useCallback((
        completedRecord: BaselineLapRecord,
    ): BaselineCollectionPayload | null => {
        const sample = currentTelemetryRef.current;
        if (!isTelemetrySample(sample)) return null;

        const currentLap = getTelemetryLap(sample);
        const currentPosition = getTelemetryPosition(sample);
        if (currentPosition === undefined) return null;

        const seeded = getContinuationRows(
            sessionIntelligence.getRowsForLap(currentLap),
            currentLap,
            completedRecord.lap,
        );
        if (seeded.rows.length === 0) return null;

        return beginCollection({
            status: 'collecting',
            rows: seeded.rows,
            sampleKeys: seeded.sampleKeys,
            startLap: currentLap,
            startPosition: 0,
            currentLap,
            currentPosition,
            lastPosition: currentPosition,
            canStartAtBoundary: true,
            lapCounterAdvancePending: currentLap === completedRecord.lap,
            track: getTelemetryTrack(sample) || completedRecord.track,
            car: getTelemetryCar(sample) || completedRecord.car,
            completedRecord: null,
        });
    }, [beginCollection, sessionIntelligence]);

    const subscribe = useCallback((listener: (tag: BaselineCollectionTag | null) => void) => {
        tagListenersRef.current.add(listener);
        return () => {
            tagListenersRef.current.delete(listener);
        };
    }, []);

    const startCollection = useCallback((options: BaselineCollectionOptions = {}) => {
        const status = tagRef.current?.status ?? recorderRef.current.status;
        const collectionInProgress = enabledRef.current
            && (status === 'waiting_for_start' || status === 'collecting');
        if (collectionInProgress) {
            return createAiToolOperationFrom<BaselineCollectionPayload>(() => {
                throw new BaselineCollectionAlreadyStartedError(
                    name,
                    'Baseline collection is already in progress.',
                );
            });
        }

        const completedRecord = status === 'complete' ? lapRecordRef.current : null;
        if (!completedRecord || !beginContinuedCollection(completedRecord)) {
            beginFreshCollection();
        }
        const result = createAiToolDeferred<BaselineCollectionPayload>();
        const statuses = [0, 1, 25, 50, 75, 100].map((milestone) => ({
            milestone,
            deferred: createAiToolDeferred<BaselineCollectionStatus>(),
        }));
        const timeoutMs = Number.isFinite(options.timeoutMs) && Number(options.timeoutMs) > 0
            ? Number(options.timeoutMs)
            : 600000;
        const pending: PendingBaselineOperation = {
            result,
            statuses,
            timeoutId: setTimeout(() => {
                pendingCollectionOperationsRef.current.delete(pending);
                const error = new BaselineCollectionIncompleteError(
                    name,
                    'Baseline collection did not complete before the timeout.',
                );
                const current = buildBaselineCollectionToolPayload(tagRef.current, lapRecordRef.current);
                statuses.forEach(({ milestone, deferred }) => {
                    if (!deferred.settled) deferred.resolve({
                        ...current,
                        event: milestone <= 1 ? 'baseline_collecting' : 'baseline_progress',
                        milestone,
                        skipped: true,
                    });
                });
                result.reject(error);
            }, timeoutMs),
        };
        pendingCollectionOperationsRef.current.add(pending);
        if (tagRef.current) settleCollectionStatus(tagRef.current);
        return createAiToolOperation(result.promise, statuses.map((status) => status.deferred.promise));
    }, [beginContinuedCollection, beginFreshCollection, name, settleCollectionStatus]);

    const restartCollection = useCallback(() => {
        try {
            const status = tagRef.current?.status ?? recorderRef.current.status;
            const collectionInProgress = enabledRef.current
                && (status === 'waiting_for_start' || status === 'collecting');
            if (!collectionInProgress) {
                throw new BaselineCollectionNotStartedError(
                    name,
                    'Baseline collection is not in progress. Start a new collection instead.',
                );
            }
            settlePendingCollectionOperations(new BaselineAnalysisCancelledError(
                name,
                'Baseline collection was cancelled because collection restarted.',
            ));
            return createAiToolOperation(beginFreshCollection());
        } catch (error) {
            return createAiToolOperationFrom(() => { throw error; });
        }
    }, [beginFreshCollection, name, settlePendingCollectionOperations]);

    const requestAnalysis = useCallback((
        options: { limit?: number } = {},
    ): Promise<BaselineAnalysisPayload> => {
        if (analysisRequestRef.current) return analysisRequestRef.current;

        const baseline = lapRecordRef.current;
        if (!baseline?.records?.length) {
            setAnalysisState('error');
            return Promise.reject(new BaselineLapRecordRequiredError(
                name,
                'Live recorded analysis requires a recorded baseline lap before it can run.',
            ));
        }

        const generation = collectionGenerationRef.current;
        let request!: Promise<BaselineAnalysisPayload>;
        request = (async () => {
            try {
                setAnalysisState('analyzing');
                let result: SegmentClassificationResult;
                try {
                    const response = await apiService.post(
                        '/racing-session/analyze-live-recorded-analysis',
                        {
                            track: baseline.track,
                            car: baseline.car,
                            baseline_lap: baseline.lap,
                            records: baseline.records,
                        },
                        { timeout: 120000 },
                    );
                    result = normalizeSegmentClassificationResult(response.data as any, baseline.id);
                } catch (error: any) {
                    if (collectionGenerationRef.current === generation) setAnalysisState('error');
                    throw new RecordedAnalysisFailedError(
                        name,
                        error?.data?.message
                            || error?.message
                            || 'Failed to run live baseline analysis.',
                        { cause: error },
                    );
                }

                if (collectionGenerationRef.current !== generation) {
                    throw new BaselineAnalysisCancelledError(
                        name,
                        'Baseline analysis was cancelled because baseline collection restarted.',
                    );
                }

                const chat = resolveNamedComponentHandle<AiChatHandle>(
                    componentRefs,
                    AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
                );
                const elements = buildAnalysisElements(result, chat, baseline.records);
                let chart: Awaited<ReturnType<typeof ensureAnalysisResultsChart>>;
                try {
                    chart = await ensureAnalysisResultsChart(componentRefs);
                } catch (error) {
                    if (collectionGenerationRef.current === generation) setAnalysisState('error');
                    throw new AnalysisResultsVisualizationUnavailableError(
                        name,
                        error instanceof Error && error.message
                            ? error.message
                            : 'The analysis-results visualization is unavailable.',
                        { cause: error },
                    );
                }
                if (collectionGenerationRef.current !== generation) {
                    throw new BaselineAnalysisCancelledError(
                        name,
                        'Baseline analysis was cancelled because baseline collection restarted.',
                    );
                }
                const page = appendAnalysisResultPage({
                    elements,
                    baseline: {
                        id: baseline.id,
                        lap: baseline.lap,
                        lap_time_ms: baseline.lap_time_ms,
                        captured_at: baseline.captured_at,
                        track: baseline.track,
                        car: baseline.car,
                        sample_count: baseline.sample_count,
                    },
                });
                await chart.handle.waitForAnalysisResultPage(page.pageId);
                if (collectionGenerationRef.current !== generation) {
                    throw new BaselineAnalysisCancelledError(
                        name,
                        'Baseline analysis was cancelled because baseline collection restarted.',
                    );
                }
                const payload: BaselineAnalysisPayload = {
                    status: result.segments.length > 0 ? 'ready' : 'empty',
                    message: result.segments.length > 0
                        ? 'Telemetry analysis is ready.'
                        : 'Telemetry analysis found no classified segments.',
                    analysis: {
                        status: result.status,
                        session_id: result.session_id,
                        samples_analyzed: result.samples_analyzed,
                        segments: result.segments
                            .slice(0, getAnalysisLimit(options.limit))
                            .map((segment) => compactSegment(segment, chat)),
                    },
                    source: 'baseline_lap_record',
                    baseline: {
                        id: baseline.id,
                        lap: baseline.lap,
                        lap_time_ms: baseline.lap_time_ms,
                        captured_at: baseline.captured_at,
                        track: baseline.track,
                        car: baseline.car,
                        sample_count: baseline.sample_count,
                    },
                    chartId: chart.chartId,
                    component_name: chart.componentName,
                    pageId: page.pageId,
                    pageCount: page.pageCount,
                };
                if (collectionGenerationRef.current === generation) setAnalysisState('complete');
                return payload;
            } catch (error) {
                if (collectionGenerationRef.current === generation) setAnalysisState('error');
                throw error;
            } finally {
                if (analysisRequestRef.current === request) analysisRequestRef.current = null;
            }
        })();
        analysisRequestRef.current = request;
        return request;
    }, [appendAnalysisResultPage, componentRefs, name]);

    const requestAnalysisFromButton = useCallback(() => {
        void requestAnalysis().catch(() => undefined);
    }, [requestAnalysis]);

    const handle = useMemo<BaselineCollectionHandle>(() => ({
        getComponentName: () => name,
        startCollection,
        restartCollection,
        requestAnalysis: (options) => createAiToolOperation(requestAnalysis(options)),
        getTag: () => tagRef.current,
        getComponentType: () => 'baseline_progress',
        getSnapshot: () => tagRef.current,
        getOverlayBehavior: (next) => ({
            placement: 'flow',
            requestedStatus: 'expanded',
            foldAfterMs: OVERLAY_HOLD_MS,
            remove: next === null,
        }),
        getOverlayMetadata: () => ({}),
        handleOverlayRendererEvent: () => undefined,
        getLapRecord: () => lapRecordRef.current,
        subscribe,
    }), [name, requestAnalysis, restartCollection, startCollection, subscribe]);
    const componentRef = useRef<BaselineCollectionHandle | null>(handle);
    componentRef.current = handle;
    useRegisterAiToolComponentRef(componentRef);

    useEffect(() => {
        if (!enabledRef.current || !enabled) return;
        if (!isTelemetrySample(currentTelemetry)) {
            publishTag(buildBaselineCollectionTag(buildRecorderSnapshot(recorderRef.current)));
            return;
        }

        const state = recorderRef.current;
        const lap = getTelemetryLap(currentTelemetry);
        const position = getTelemetryPosition(currentTelemetry) ?? state.currentPosition;
        const sampleKey = getSampleKey(currentTelemetry, lap, position);
        if (state.sampleKeys.has(sampleKey)) return;

        state.track = getTelemetryTrack(currentTelemetry) || state.track;
        state.car = getTelemetryCar(currentTelemetry) || state.car;
        state.currentLap = lap;
        state.currentPosition = position;
        let completedRecordToEmit: BaselineLapRecord | null = null;

        if (state.status === 'waiting_for_start') {
            if (position > BASELINE_START_POSITION_EPSILON) {
                state.canStartAtBoundary = true;
            } else if (state.canStartAtBoundary) {
                state.status = 'collecting';
                state.startLap = lap;
                state.startPosition = 0;
                state.rows = [cloneSample(currentTelemetry)];
            }
        } else if (state.status === 'collecting') {
            const positionWrapped = state.lastPosition !== null
                && state.lastPosition - position > BASELINE_WRAP_THRESHOLD;
            if (
                state.lapCounterAdvancePending
                && state.startLap !== null
                && lap > state.startLap
                && !positionWrapped
            ) {
                state.startLap = lap;
                state.lapCounterAdvancePending = false;
            }

            if (hasCompletedRecordingLap(state, lap, position)) {
                const snapshot = buildRecorderSnapshot({
                    ...state,
                    status: 'complete',
                    currentLap: lap,
                    currentPosition: position,
                });
                const completedRecord: BaselineLapRecord = {
                    id: [
                        state.track,
                        state.car,
                        String(state.startLap ?? 0),
                        String(state.rows.length),
                    ].join(':'),
                    lap: state.startLap ?? 0,
                    lap_time_ms: getCompletedBaselineLapTimeMs(currentTelemetry, state.rows),
                    captured_at: Date.now(),
                    track: state.track,
                    car: state.car,
                    sample_count: state.rows.length,
                    snapshot,
                    records: state.rows.map(cloneSample),
                };
                state.status = 'complete';
                state.completedRecord = completedRecord;
                lapRecordRef.current = completedRecord;
                completedRecordToEmit = completedRecord;
            } else {
                state.rows.push(cloneSample(currentTelemetry));
            }
        }

        state.lastPosition = position;
        state.sampleKeys.add(sampleKey);
        const snapshot = state.completedRecord?.snapshot ?? buildRecorderSnapshot(state);
        const nextTag = buildBaselineCollectionTag(snapshot);
        publishTag(nextTag);
        settleCollectionStatus(nextTag);
        if (completedRecordToEmit) {
            settlePendingCollectionOperations(buildBaselineCollectionToolPayload(null, completedRecordToEmit));
        }
    }, [currentTelemetry, enabled, publishTag, settleCollectionStatus, settlePendingCollectionOperations]);

    useEffect(() => () => {
        enabledRef.current = false;
        recorderRef.current = createEmptyRecorderState();
        tagRef.current = null;
        lapRecordRef.current = null;
        settlePendingCollectionOperations(new BaselineAnalysisCancelledError(
            name,
            'Baseline collection was cancelled because the component unmounted.',
        ));
        tagListenersRef.current.clear();
        analysisRequestRef.current = null;
        collectionGenerationRef.current += 1;
    }, [name, settlePendingCollectionOperations]);

    const analysisButtonLabel = analysisState === 'analyzing'
        ? 'Analyzing Baseline…'
        : analysisState === 'complete'
            ? 'Analysis Complete'
            : analysisState === 'error'
                ? 'Retry Analysis'
                : 'Request Analysis';

    return (
        <div className="baseline-collection" data-testid="baseline-collection">
            {tag ? (
                <>
                    <BaselineProgressDisplay tag={tag} />
                    {tag.status === 'complete' && lapRecordRef.current && (
                        <button
                            type="button"
                            className="baseline-collection__start"
                            onClick={requestAnalysisFromButton}
                            disabled={analysisState === 'analyzing' || analysisState === 'complete'}
                            aria-busy={analysisState === 'analyzing'}
                        >
                            {analysisButtonLabel}
                        </button>
                    )}
                </>
            ) : (
                <div className="baseline-collection__idle" role="status">
                    <strong>Ready for baseline collection</strong>
                    <span>Start collection, then keep this panel open until the lap is complete.</span>
                    <button
                        type="button"
                        className="baseline-collection__start"
                            onClick={() => { void startCollection().result.catch(() => undefined); }}
                    >
                        Start Baseline Collection
                    </button>
                </div>
            )}
        </div>
    );
};

export default BaselineCollection;
