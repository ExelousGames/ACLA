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
import { resolveAnalysisResultsComparison } from 'views/lap-analysis/visualization/charts/analysisResultsComparisonAdapter';
import { getSegmentLabelIds } from 'views/lap-analysis/visualization/charts/segmentClassificationDisplay';
import { getSingletonVisualizationComponentName } from 'views/lap-analysis/visualization/visualization-component-names';
import BaselineProgressDisplay from './BaselineProgressDisplay';
import './baseline-collection.css';
import { LiveSessionContext } from './LiveSessionContext';
import { liveTelemetryStore } from './live-telemetry-store';
import {
    createAiToolDeferred,
    createControlledAiToolOperation,
    createAiToolOperation,
    createAiToolOperationFrom,
    type AiToolDeferred,
    type ControlledAiToolOperation,
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
    baseline_lap_id: number | null;
};

export type BaselineLapRecord = {
    id: string;
    lap_id: number;
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

export type BaselineCollectionPreset = 'full_lap';

export type BaselineTelemetryCondition = {
    field: string;
    operator: 'eq' | 'neq' | 'lt' | 'lte' | 'gt' | 'gte';
    value: number;
};

export type BaselineCollectionQuery =
    | {
        preset: BaselineCollectionPreset;
        start_query?: never;
        end_query?: never;
    }
    | {
        preset?: never;
        start_query: BaselineTelemetryCondition;
        end_query: BaselineTelemetryCondition;
    };

export type BaselineCollectionOptions = {
    timeoutMs?: number;
    query?: BaselineCollectionQuery;
};

export interface BaselineCollectionHandle extends NamedAiToolComponentHandle, AiOverlayComponentHandle<BaselineCollectionTag | null> {
    startCollection(options?: BaselineCollectionOptions): AiToolOperation<BaselineCollectionPayload, BaselineCollectionStatus>;
    restartCollection(): AiToolOperation<BaselineCollectionPayload>;
    requestAnalysis(options?: { limit?: number }): AiToolOperation<BaselineAnalysisPayload>;
    getTag(): BaselineCollectionTag | null;
    getLapRecord(): BaselineLapRecord | null;
    subscribe(listener: (tag: BaselineCollectionTag | null) => void): () => void;
}

type PendingBaselineOperation = {
    controller: ControlledAiToolOperation<
        BaselineCollectionPayload,
        BaselineCollectionStatus,
        'complete' | 'timed_out' | 'cancelled'
    >;
    statuses: Array<{ milestone: number; deferred: AiToolDeferred<BaselineCollectionStatus> }>;
    timeoutId: ReturnType<typeof setTimeout> | null;
};

type BaselineRecorderState = {
    status: BaselineCollectionTag['status'];
    rows: Record<string, any>[];
    sampleKeys: Set<string>;
    startLap: number | null;
    currentLap: number;
    currentPosition: number;
    lastPosition: number | null;
    lapCounterAdvancePending: boolean;
    query: ResolvedBaselineCollectionQuery;
    track: string;
    car: string;
    completedRecord: BaselineLapRecord | null;
};

type BaselineTelemetryCache = {
    identity: string;
    lap: number | null;
    rows: Record<string, any>[];
    sampleKeys: Set<string>;
};

const BASELINE_START_POSITION_EPSILON = 0.005;
const BASELINE_WRAP_THRESHOLD = 0.65;
const NORMALIZED_POSITION_FIELD = 'Graphics_normalized_car_position';

type ResolvedBaselineCollectionQuery = {
    preset: BaselineCollectionPreset | null;
    startQuery: BaselineTelemetryCondition;
    endQuery: BaselineTelemetryCondition;
};

const FULL_LAP_QUERY: ResolvedBaselineCollectionQuery = {
    preset: 'full_lap',
    startQuery: {
        field: NORMALIZED_POSITION_FIELD,
        operator: 'eq',
        value: 0,
    },
    endQuery: {
        field: NORMALIZED_POSITION_FIELD,
        operator: 'eq',
        value: 1,
    },
};

const resolveBaselineCollectionQuery = (
    query: BaselineCollectionQuery | undefined,
): ResolvedBaselineCollectionQuery => {
    if (!query || query.preset === 'full_lap') return FULL_LAP_QUERY;
    return {
        preset: null,
        startQuery: query.start_query,
        endQuery: query.end_query,
    };
};

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

const createEmptyTelemetryCache = (): BaselineTelemetryCache => ({
    identity: '',
    lap: null,
    rows: [],
    sampleKeys: new Set(),
});

const getTelemetryConditionValue = (
    sample: Record<string, any>,
    field: string,
): number | null => {
    const value = field === NORMALIZED_POSITION_FIELD
        ? getTelemetryPosition(sample)
        : sample[field];
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
};

const matchesTelemetryCondition = (
    sample: Record<string, any>,
    condition: BaselineTelemetryCondition,
): boolean => {
    const actual = getTelemetryConditionValue(sample, condition.field);
    if (actual === null) return false;

    switch (condition.operator) {
        case 'eq': return actual === condition.value;
        case 'neq': return actual !== condition.value;
        case 'lt': return actual < condition.value;
        case 'lte': return actual <= condition.value;
        case 'gt': return actual > condition.value;
        case 'gte': return actual >= condition.value;
        default: return false;
    }
};

const matchesBaselineStart = (
    query: ResolvedBaselineCollectionQuery,
    sample: Record<string, any>,
    position: number,
): boolean => query.preset === 'full_lap'
    ? position <= BASELINE_START_POSITION_EPSILON
    : matchesTelemetryCondition(sample, query.startQuery);

const cacheCurrentLapTelemetry = (
    cache: BaselineTelemetryCache,
    sample: Record<string, any>,
): void => {
    const position = getTelemetryPosition(sample);
    if (position === undefined) return;

    const lap = getTelemetryLap(sample);
    const track = getTelemetryTrack(sample);
    const car = getTelemetryCar(sample);
    const identity = track && car ? `${track}:${car}` : '';
    const identityChanged = Boolean(identity && cache.identity && cache.identity !== identity);
    if (cache.lap !== lap || identityChanged) {
        cache.identity = identity;
        cache.lap = lap;
        cache.rows = [];
        cache.sampleKeys = new Set();
    } else if (identity) {
        cache.identity = identity;
    }

    const sampleKey = getSampleKey(sample, lap, position);
    if (cache.sampleKeys.has(sampleKey)) return;
    cache.sampleKeys.add(sampleKey);
    cache.rows.push(cloneSample(sample));
};

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
    query: ResolvedBaselineCollectionQuery = FULL_LAP_QUERY,
    startWhenMatched = false,
): BaselineRecorderState => {
    const sample = isTelemetrySample(currentTelemetry) ? currentTelemetry : null;
    const lap = sample ? getTelemetryLap(sample) : 0;
    const position = sample ? getTelemetryPosition(sample) ?? 0 : 0;
    const collecting = Boolean(
        startWhenMatched
        && sample
        && matchesBaselineStart(query, sample, position),
    );

    return {
        status: collecting ? 'collecting' : 'waiting_for_start',
        rows: collecting && sample ? [cloneSample(sample)] : [],
        sampleKeys: new Set(sample ? [getSampleKey(sample, lap, position)] : []),
        startLap: collecting ? lap : null,
        currentLap: lap,
        currentPosition: position,
        lastPosition: sample ? position : null,
        lapCounterAdvancePending: false,
        query,
        track: sample ? getTelemetryTrack(sample) : '',
        car: sample ? getTelemetryCar(sample) : '',
        completedRecord: null,
    };
};

const hasCompletedRecording = (
    state: BaselineRecorderState,
    sample: Record<string, any>,
    lap: number,
    position: number,
): boolean => (
    state.startLap !== null
    && state.rows.length > 0
    && (state.query.preset === 'full_lap'
        ? (
            position >= 1 - BASELINE_START_POSITION_EPSILON
            || lap > state.startLap
            || (
                state.lastPosition !== null
                && state.lastPosition - position > BASELINE_WRAP_THRESHOLD
            )
        )
        : matchesTelemetryCondition(sample, state.query.endQuery))
);

const getCollectionProgress = (state: BaselineRecorderState): number => {
    if (state.status === 'complete') return 100;
    if (state.status !== 'collecting') return 0;

    if (
        state.query.startQuery.field !== NORMALIZED_POSITION_FIELD
        || state.query.endQuery.field !== NORMALIZED_POSITION_FIELD
    ) return 1;

    const start = state.query.startQuery.value;
    const end = state.query.endQuery.value;
    const range = end >= start ? end - start : 1 - start + end;
    if (range <= 0) return 1;
    const travelled = state.currentPosition >= start
        ? state.currentPosition - start
        : 1 - start + state.currentPosition;
    const rawProgress = travelled / range;

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
    baseline_lap_id: state.startLap,
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
    const comparisonResolution = resolveAnalysisResultsComparison({
        baselineRecords: records,
        expertReferenceData: segment.expert_reference_data,
    });
    const comparison = comparisonResolution.comparison;
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
        ...(comparisonResolution.diagnostics.length > 0
            ? { comparisonDiagnostics: comparisonResolution.diagnostics }
            : {}),
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
            ? `Collecting baseline on lap ${Number(snapshot.current_lap ?? 0) + 1}`
            : 'Waiting for the baseline start condition';

    return {
        status,
        progress_percent: progress,
        detail,
        track: typeof snapshot.track === 'string' && snapshot.track ? snapshot.track : null,
        car: typeof snapshot.car === 'string' && snapshot.car ? snapshot.car : null,
        current_lap: toNullableFiniteNumber(snapshot.current_lap),
        baseline_lap_id: toNullableFiniteNumber(snapshot.baseline_lap_id),
    };
};

export const buildBaselineCollectionToolPayload = (
    tag: BaselineCollectionTag | null,
    record: BaselineLapRecord | null,
): BaselineCollectionPayload => {
    const message = record
        ? 'Baseline complete. Cached baseline record is ready.'
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
        appendAnalysisResultPage,
    } = useContext(LiveSessionContext);
    const componentRefs = useAiToolComponentRefDirectory();
    const currentTelemetryRef = useRef(liveTelemetryStore.getSnapshot().currentTelemetry);

    const [, setEnabled] = useState(false);
    const [tag, setTag] = useState<BaselineCollectionTag | null>(null);
    const [analysisState, setAnalysisState] = useState<BaselineAnalysisState>('idle');
    const [selectedPreset, setSelectedPreset] = useState<BaselineCollectionPreset>('full_lap');
    const enabledRef = useRef(false);
    const tagRef = useRef<BaselineCollectionTag | null>(null);
    const lapRecordRef = useRef<BaselineLapRecord | null>(null);
    const recorderRef = useRef<BaselineRecorderState>(createEmptyRecorderState());
    const telemetryCacheRef = useRef<BaselineTelemetryCache>(createEmptyTelemetryCache());
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
        terminationStatus: 'complete' | 'cancelled',
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
            if (outcome instanceof Error) operation.controller.reject(terminationStatus, outcome);
            else operation.controller.resolve(terminationStatus, outcome);
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

    const beginFreshCollection = useCallback((query: ResolvedBaselineCollectionQuery) => beginCollection(
        createEmptyRecorderState(currentTelemetryRef.current, query, true),
    ), [beginCollection]);

    const beginContinuedCollection = useCallback((
        completedRecord: BaselineLapRecord,
        query: ResolvedBaselineCollectionQuery,
    ): BaselineCollectionPayload | null => {
        if (query.preset !== 'full_lap') return null;
        const sample = currentTelemetryRef.current;
        if (!isTelemetrySample(sample)) return null;

        const currentLap = getTelemetryLap(sample);
        const currentPosition = getTelemetryPosition(sample);
        if (currentPosition === undefined) return null;

        cacheCurrentLapTelemetry(telemetryCacheRef.current, sample);

        const seeded = getContinuationRows(
            telemetryCacheRef.current.rows,
            currentLap,
            completedRecord.lap_id,
        );
        if (seeded.rows.length === 0) return null;

        return beginCollection({
            status: 'collecting',
            rows: seeded.rows,
            sampleKeys: seeded.sampleKeys,
            startLap: currentLap,
            currentLap,
            currentPosition,
            lastPosition: currentPosition,
            lapCounterAdvancePending: currentLap === completedRecord.lap_id,
            query,
            track: getTelemetryTrack(sample) || completedRecord.track,
            car: getTelemetryCar(sample) || completedRecord.car,
            completedRecord: null,
        });
    }, [beginCollection]);

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
            }, 'failed');
        }

        const completedRecord = status === 'complete' ? lapRecordRef.current : null;
        const query = resolveBaselineCollectionQuery(options.query);
        if (!completedRecord || !beginContinuedCollection(completedRecord, query)) {
            beginFreshCollection(query);
        }
        const statuses = [0, 1, 25, 50, 75, 100].map((milestone) => ({
            milestone,
            deferred: createAiToolDeferred<BaselineCollectionStatus>(),
        }));
        const timeoutMs = Number.isFinite(options.timeoutMs) && Number(options.timeoutMs) > 0
            ? Number(options.timeoutMs)
            : 600000;
        const controller = createControlledAiToolOperation<
            BaselineCollectionPayload,
            BaselineCollectionStatus,
            'complete' | 'timed_out' | 'cancelled'
        >(statuses.map((status) => status.deferred.promise));
        const pending: PendingBaselineOperation = {
            controller,
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
                controller.reject('timed_out', error);
            }, timeoutMs),
        };
        pendingCollectionOperationsRef.current.add(pending);
        if (tagRef.current) settleCollectionStatus(tagRef.current);
        return controller.operation;
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
            ), 'cancelled');
            return createAiToolOperation(beginFreshCollection(recorderRef.current.query), 'complete');
        } catch (error) {
            return createAiToolOperationFrom(() => { throw error; }, 'failed');
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
                'Live recorded analysis requires a recorded baseline before it can run.',
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
                            baseline_lap_id: baseline.lap_id,
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
                        lap_id: baseline.lap_id,
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
                        lap_id: baseline.lap_id,
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
        requestAnalysis: (options) => createAiToolOperation(requestAnalysis(options), 'complete'),
        getTag: () => tagRef.current,
        getComponentType: () => 'baseline_progress',
        getSnapshot: () => tagRef.current,
        getOverlayBehavior: (next) => ({
            placement: 'flow',
            requestedStatus: 'expanded',
            foldAfterMs: OVERLAY_HOLD_MS,
            remove: next === null || next.status === 'complete',
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
        return liveTelemetryStore.subscribeEvents((event) => {
            if (event.type === 'session-reset') {
                currentTelemetryRef.current = {};
                telemetryCacheRef.current = createEmptyTelemetryCache();
                return;
            }
            if (event.type !== 'frame') return;

            const sample = event.sample;
            currentTelemetryRef.current = sample;
            if (isTelemetrySample(sample)) {
                cacheCurrentLapTelemetry(telemetryCacheRef.current, sample);
            }
            if (!enabledRef.current || !isTelemetrySample(sample)) return;

            const state = recorderRef.current;
            const lap = getTelemetryLap(sample);
            const position = getTelemetryPosition(sample) ?? state.currentPosition;
            const sampleKey = getSampleKey(sample, lap, position);
            if (state.sampleKeys.has(sampleKey)) return;

            state.track = getTelemetryTrack(sample) || state.track;
            state.car = getTelemetryCar(sample) || state.car;
            state.currentLap = lap;
            state.currentPosition = position;
            let completedRecordToEmit: BaselineLapRecord | null = null;

            if (state.status === 'waiting_for_start') {
                if (matchesBaselineStart(state.query, sample, position)) {
                    state.status = 'collecting';
                    state.startLap = lap;
                    state.rows = [cloneSample(sample)];
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

                if (hasCompletedRecording(state, sample, lap, position)) {
                    const includeCompletionSample = state.query.preset !== 'full_lap'
                        || (!positionWrapped && (state.startLap === null || lap <= state.startLap));
                    const completedRows = includeCompletionSample
                        ? [...state.rows, cloneSample(sample)]
                        : state.rows;
                    const snapshot = buildRecorderSnapshot({
                        ...state,
                        status: 'complete',
                        currentLap: lap,
                        currentPosition: position,
                        rows: completedRows,
                    });
                    const completedRecord: BaselineLapRecord = {
                        id: [
                            state.track,
                            state.car,
                            String(state.startLap ?? 0),
                            String(completedRows.length),
                        ].join(':'),
                        lap_id: state.startLap ?? 0,
                        lap_time_ms: getCompletedBaselineLapTimeMs(sample, completedRows),
                        captured_at: Date.now(),
                        track: state.track,
                        car: state.car,
                        sample_count: completedRows.length,
                        snapshot,
                        records: completedRows.map(cloneSample),
                    };
                    state.rows = completedRows;
                    state.status = 'complete';
                    state.completedRecord = completedRecord;
                    lapRecordRef.current = completedRecord;
                    completedRecordToEmit = completedRecord;
                } else {
                    state.rows.push(cloneSample(sample));
                }
            }

            state.lastPosition = position;
            state.sampleKeys.add(sampleKey);
            const snapshot = state.completedRecord?.snapshot ?? buildRecorderSnapshot(state);
            const nextTag = buildBaselineCollectionTag(snapshot);
            publishTag(nextTag);
            settleCollectionStatus(nextTag);
            if (completedRecordToEmit) {
                settlePendingCollectionOperations(
                    buildBaselineCollectionToolPayload(null, completedRecordToEmit),
                    'complete',
                );
            }
        }, { replayLatest: true });
    }, [publishTag, settleCollectionStatus, settlePendingCollectionOperations]);

    useEffect(() => () => {
        enabledRef.current = false;
        recorderRef.current = createEmptyRecorderState();
        telemetryCacheRef.current = createEmptyTelemetryCache();
        tagRef.current = null;
        lapRecordRef.current = null;
        settlePendingCollectionOperations(new BaselineAnalysisCancelledError(
            name,
            'Baseline collection was cancelled because the component unmounted.',
        ), 'cancelled');
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
            <BaselineProgressDisplay
                tag={tag}
                action={tag?.status === 'complete' && lapRecordRef.current ? (
                    <button
                        type="button"
                        className="baseline-timeline__button baseline-timeline__button--analysis"
                        onClick={requestAnalysisFromButton}
                        disabled={analysisState === 'analyzing' || analysisState === 'complete'}
                        aria-busy={analysisState === 'analyzing'}
                    >
                        {analysisButtonLabel}
                    </button>
                ) : !tag ? (
                    <div className="baseline-timeline__start-controls">
                        <label className="baseline-timeline__preset">
                            <span>Preset</span>
                            <select
                                aria-label="Baseline collection preset"
                                value={selectedPreset}
                                onChange={(event) => setSelectedPreset(
                                    event.target.value as BaselineCollectionPreset,
                                )}
                            >
                                <option value="full_lap">Full lap</option>
                            </select>
                        </label>
                        <button
                            type="button"
                            className="baseline-timeline__button baseline-timeline__button--start"
                            onClick={() => {
                                void startCollection({
                                    query: { preset: selectedPreset },
                                }).result.catch(() => undefined);
                            }}
                        >
                            Start
                            <span className="baseline-timeline__button-icon" aria-hidden="true">▶</span>
                        </button>
                    </div>
                ) : undefined}
            />
        </div>
    );
};

export default BaselineCollection;
