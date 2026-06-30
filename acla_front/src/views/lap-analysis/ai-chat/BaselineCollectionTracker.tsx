import { useCallback, useEffect, useRef } from 'react';
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
} from './ai-tool-base';

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

type BaselineCollectionTrackerProps = {
    enabled: boolean;
    liveData: Record<string, any> | null | undefined;
    sessionMode: 'live' | 'recorded' | 'user_summary';
    onTagChange: (tag: BaselineCollectionTag | null) => void;
    onLapRecordChange: (record: BaselineLapRecord | null) => void;
    onToolOutput?: ToolOutputEmitter;
};

type BaselineRecorderState = {
    status: 'waiting_for_start' | 'collecting' | 'complete';
    rows: Record<string, any>[];
    startLap: number | null;
    startPosition: number;
    currentLap: number;
    currentPosition: number;
    lastLap: number | null;
    lastPosition: number | null;
    lastSampleKey: string | null;
    track: string;
    car: string;
    completedRecord: BaselineLapRecord | null;
};

const BASELINE_START_POSITION_EPSILON = 0.005;
const BASELINE_WRAP_THRESHOLD = 0.65;

const createEmptyRecorderState = (): BaselineRecorderState => ({
    status: 'waiting_for_start',
    rows: [],
    startLap: null,
    startPosition: 0,
    currentLap: 0,
    currentPosition: 0,
    lastLap: null,
    lastPosition: null,
    lastSampleKey: null,
    track: '',
    car: '',
    completedRecord: null,
});

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

const crossedLapStart = (
    previousLap: number | null,
    currentLap: number,
    previousPosition: number | null,
    currentPosition: number,
): boolean => (
    currentPosition <= BASELINE_START_POSITION_EPSILON
    || (previousLap !== null && currentLap > previousLap)
    || (
        previousPosition !== null
        && previousPosition - currentPosition > BASELINE_WRAP_THRESHOLD
    )
);

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
            : 'Start at normalized position 0';

    return {
        status,
        progress_percent: progress,
        detail,
        track: typeof snapshot.track === 'string' ? snapshot.track : null,
        car: typeof snapshot.car === 'string' ? snapshot.car : null,
        current_lap: Number.isFinite(Number(snapshot.current_lap))
            ? Number(snapshot.current_lap)
            : null,
        baseline_lap: Number.isFinite(Number(snapshot.baseline_lap))
            ? Number(snapshot.baseline_lap)
            : null,
    };
};

export const buildBaselineCollectionToolPayload = (
    tag: BaselineCollectionTag | null,
    record: BaselineLapRecord | null,
) => {
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

export const BaselineCollectionTracker = ({
    enabled,
    liveData,
    sessionMode,
    onTagChange,
    onLapRecordChange,
    onToolOutput,
}: BaselineCollectionTrackerProps) => {
    const recorderRef = useRef<BaselineRecorderState>(createEmptyRecorderState());
    const toolOutputRef = useRef<ToolOutputController | null>(null);
    const toolOutputEmitterRef = useRef<ToolOutputEmitter | undefined>(onToolOutput);
    const lastToolOutputKeyRef = useRef<string>('');

    useEffect(() => {
        toolOutputEmitterRef.current = onToolOutput;
    }, [onToolOutput]);

    const resetToolOutput = useCallback(() => {
        toolOutputRef.current = null;
        lastToolOutputKeyRef.current = '';
    }, []);

    const getToolOutput = useCallback(() => {
        if (!toolOutputRef.current) {
            toolOutputRef.current = createToolOutputController(
                'collect_live_baseline',
                createBaselineToolRunId(),
                (envelope, options) => toolOutputEmitterRef.current?.(envelope, options),
            );
        }
        return toolOutputRef.current;
    }, []);

    const emitBaselineToolOutput = useCallback((record: BaselineLapRecord | null) => {
        if (!toolOutputEmitterRef.current || !record) return;

        const payload = buildBaselineCollectionToolPayload(null, record);
        const outputKey = [
            'final',
            payload.status,
            record.id,
        ].join(':');
        if (outputKey === lastToolOutputKeyRef.current) return;
        lastToolOutputKeyRef.current = outputKey;

        getToolOutput().final(payload, { message: payload.message });
    }, [getToolOutput]);

    useEffect(() => {
        const resetRecorder = () => {
            recorderRef.current = createEmptyRecorderState();
            onLapRecordChange(null);
            resetToolOutput();
        };

        if (sessionMode !== 'live' || !enabled) {
            onTagChange(null);
            resetRecorder();
            return;
        }

        if (!liveData || typeof liveData !== 'object' || Object.keys(liveData).length === 0) {
            const snapshot = buildRecorderSnapshot(recorderRef.current);
            const tag = buildBaselineCollectionTag(snapshot);
            onTagChange(tag);
            return;
        }

        const sample = liveData as Record<string, any>;
        const lap = getTelemetryLap(sample);
        const position = getTelemetryPosition(sample) ?? recorderRef.current.currentPosition;
        const track = getTelemetryTrack(sample) || recorderRef.current.track;
        const car = getTelemetryCar(sample) || recorderRef.current.car;
        const sampleKey = getSampleKey(sample, lap, position);
        const state = recorderRef.current;

        if (state.lastSampleKey !== sampleKey) {
            state.track = track;
            state.car = car;
            state.currentLap = lap;
            state.currentPosition = position;

            if (state.status === 'waiting_for_start') {
                if (crossedLapStart(state.lastLap, lap, state.lastPosition, position)) {
                    state.status = 'collecting';
                    state.startLap = lap;
                    state.startPosition = position <= BASELINE_START_POSITION_EPSILON ? 0 : position;
                    state.rows = [cloneSample(sample)];
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
                    onLapRecordChange(completedRecord);
                } else {
                    state.rows.push(cloneSample(sample));
                }
            }

            state.lastLap = lap;
            state.lastPosition = position;
            state.lastSampleKey = sampleKey;
        }

        const snapshot = state.completedRecord?.snapshot ?? buildRecorderSnapshot(state);
        const tag = buildBaselineCollectionTag(snapshot);
        onTagChange(tag);
        emitBaselineToolOutput(state.completedRecord);
    }, [
        enabled,
        liveData,
        onLapRecordChange,
        onTagChange,
        emitBaselineToolOutput,
        resetToolOutput,
        sessionMode,
    ]);

    return null;
};
