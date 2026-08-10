import React, {
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import {
    NamedAiToolComponentHandle,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import {
    detectLiveSessionType,
    getTelemetryCar,
    getTelemetryLap,
    getTelemetryPosition,
    getTelemetryTrack,
} from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import {
    createToolOutputController,
    type ToolOutputController,
    type ToolOutputEmitter,
    type ToolOutputEnvelope,
} from 'views/lap-analysis/ai-chat/ai-tool-base';
import BaselineProgressDisplay from './BaselineProgressDisplay';
import { LiveSessionContext } from './LiveSessionContext';

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

export interface BaselineCollectionHandle extends NamedAiToolComponentHandle {
    startCollection(): BaselineCollectionPayload;
    restartCollection(): BaselineCollectionPayload;
    getTag(): BaselineCollectionTag | null;
    getLapRecord(): BaselineLapRecord | null;
    getToolOutput(): ToolOutputEnvelope | null;
    subscribeToolOutput(listener: ToolOutputEmitter): () => void;
}

type BaselineRecorderState = {
    status: BaselineCollectionTag['status'];
    rows: Record<string, any>[];
    startLap: number | null;
    startPosition: number;
    currentLap: number;
    currentPosition: number;
    lastPosition: number | null;
    lastSampleKey: string | null;
    canStartAtBoundary: boolean;
    track: string;
    car: string;
    completedRecord: BaselineLapRecord | null;
};

const BASELINE_START_POSITION_EPSILON = 0.005;
const BASELINE_WRAP_THRESHOLD = 0.65;

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

const createEmptyRecorderState = (
    currentTelemetry?: Record<string, any> | null,
): BaselineRecorderState => {
    const sample = isTelemetrySample(currentTelemetry) ? currentTelemetry : null;
    const lap = sample ? getTelemetryLap(sample) : 0;
    const position = sample ? getTelemetryPosition(sample) ?? 0 : 0;

    return {
        status: 'waiting_for_start',
        rows: [],
        startLap: null,
        startPosition: 0,
        currentLap: lap,
        currentPosition: position,
        lastPosition: sample ? position : null,
        lastSampleKey: sample ? getSampleKey(sample, lap, position) : null,
        canStartAtBoundary: !sample || position > BASELINE_START_POSITION_EPSILON,
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
    section_count: 0,
});

const toNullableFiniteNumber = (value: unknown): number | null => {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
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

const createBaselineToolRunId = (): string =>
    `collect_live_baseline-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

const BaselineCollection = ({ name }: { name: string }) => {
    const { currentTelemetry } = useContext(LiveSessionContext);
    const currentTelemetryRef = useRef(currentTelemetry);
    currentTelemetryRef.current = currentTelemetry;

    const [enabled, setEnabled] = useState(false);
    const [tag, setTag] = useState<BaselineCollectionTag | null>(null);
    const enabledRef = useRef(false);
    const tagRef = useRef<BaselineCollectionTag | null>(null);
    const lapRecordRef = useRef<BaselineLapRecord | null>(null);
    const recorderRef = useRef<BaselineRecorderState>(createEmptyRecorderState());
    const toolOutputRef = useRef<ToolOutputEnvelope | null>(null);
    const toolOutputControllerRef = useRef<ToolOutputController | null>(null);
    const toolOutputListenersRef = useRef<Set<ToolOutputEmitter>>(new Set());
    const completionEmittedRef = useRef(false);

    const publishTag = useCallback((nextTag: BaselineCollectionTag | null) => {
        tagRef.current = nextTag;
        setTag(nextTag);
    }, []);

    const beginFreshCollection = useCallback(() => {
        const nextRecorder = createEmptyRecorderState(currentTelemetryRef.current);
        recorderRef.current = nextRecorder;
        enabledRef.current = true;
        lapRecordRef.current = null;
        toolOutputRef.current = null;
        toolOutputControllerRef.current = null;
        completionEmittedRef.current = false;
        const nextTag = buildBaselineCollectionTag(buildRecorderSnapshot(nextRecorder));
        publishTag(nextTag);
        setEnabled(true);
        return buildBaselineCollectionToolPayload(nextTag, null);
    }, [publishTag]);

    const startCollection = useCallback(() => {
        if (enabledRef.current) {
            return buildBaselineCollectionToolPayload(tagRef.current, lapRecordRef.current);
        }
        return beginFreshCollection();
    }, [beginFreshCollection]);

    const restartCollection = useCallback(() => beginFreshCollection(), [beginFreshCollection]);

    const subscribeToolOutput = useCallback((listener: ToolOutputEmitter) => {
        toolOutputListenersRef.current.add(listener);
        return () => {
            toolOutputListenersRef.current.delete(listener);
        };
    }, []);

    const emitCompletion = useCallback((record: BaselineLapRecord) => {
        if (completionEmittedRef.current) return;
        completionEmittedRef.current = true;
        const payload = buildBaselineCollectionToolPayload(null, record);
        if (!toolOutputControllerRef.current) {
            toolOutputControllerRef.current = createToolOutputController(
                'collect_live_baseline',
                createBaselineToolRunId(),
                (envelope, options) => {
                    toolOutputRef.current = envelope;
                    toolOutputListenersRef.current.forEach((listener) => listener(envelope, options));
                },
            );
        }
        toolOutputControllerRef.current.final(payload, { message: payload.message });
    }, []);

    const handle = useMemo<BaselineCollectionHandle>(() => ({
        getComponentName: () => name,
        startCollection,
        restartCollection,
        getTag: () => tagRef.current,
        getLapRecord: () => lapRecordRef.current,
        getToolOutput: () => toolOutputRef.current,
        subscribeToolOutput,
    }), [name, restartCollection, startCollection, subscribeToolOutput]);
    useRegisterAiToolComponentRef(name, handle);

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
        if (state.lastSampleKey === sampleKey) return;

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
        state.lastSampleKey = sampleKey;
        const snapshot = state.completedRecord?.snapshot ?? buildRecorderSnapshot(state);
        publishTag(buildBaselineCollectionTag(snapshot));
        if (completedRecordToEmit) emitCompletion(completedRecordToEmit);
    }, [currentTelemetry, emitCompletion, enabled, publishTag]);

    useEffect(() => () => {
        enabledRef.current = false;
        recorderRef.current = createEmptyRecorderState();
        tagRef.current = null;
        lapRecordRef.current = null;
        toolOutputRef.current = null;
        toolOutputControllerRef.current = null;
        toolOutputListenersRef.current.clear();
        completionEmittedRef.current = false;
    }, []);

    return (
        <div className="baseline-collection" data-testid="baseline-collection">
            {tag ? (
                <BaselineProgressDisplay tag={tag} />
            ) : (
                <div className="baseline-collection__idle" role="status">
                    <strong>Ready for baseline collection</strong>
                    <span>Ask the assistant to start. Keep this panel open until the lap is complete.</span>
                </div>
            )}
        </div>
    );
};

export default BaselineCollection;
