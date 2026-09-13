import { useSyncExternalStore } from 'react';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type {
    LiveSessionStaticData,
    RecordingViewUpdate,
    StandardTelemetrySample,
} from './live-session-types';

export interface LiveTelemetrySnapshot {
    graphicsTelemetry: StandardTelemetrySample;
    physicsTelemetry: StandardTelemetrySample;
    currentTelemetry: StandardTelemetrySample;
    sampleIndex: number;
    telemetryStatus: ACC_STATUS | null;
    committedSampleCount: number;
    sessionGeneration: number;
    streamGeneration: number;
}

export interface LiveTelemetryFrameEvent {
    type: 'frame';
    update: RecordingViewUpdate;
    sample: StandardTelemetrySample;
    sampleIndex: number;
    telemetryStatus: ACC_STATUS | null;
    committedSampleCount: number;
    sessionGeneration: number;
    streamGeneration: number;
}

export interface LiveTelemetryStreamResetEvent {
    type: 'stream-reset';
    snapshot: LiveTelemetrySnapshot;
}

export interface LiveTelemetrySessionResetEvent {
    type: 'session-reset';
    snapshot: LiveTelemetrySnapshot;
}

export type LiveTelemetryEvent =
    | LiveTelemetryFrameEvent
    | LiveTelemetryStreamResetEvent
    | LiveTelemetrySessionResetEvent;

export type LiveTelemetryEventListener = (event: LiveTelemetryEvent) => void;
export type LiveTelemetrySnapshotListener = () => void;

export interface LiveTelemetryEventSubscriptionOptions {
    replayLatest?: boolean;
}

export interface LiveTelemetryStore {
    getSnapshot: () => LiveTelemetrySnapshot;
    subscribeEvents: (
        listener: LiveTelemetryEventListener,
        options?: LiveTelemetryEventSubscriptionOptions | boolean,
    ) => () => void;
    subscribeSnapshot: (listener: LiveTelemetrySnapshotListener) => () => void;
    subscribeSelector: <T>(
        selector: (snapshot: LiveTelemetrySnapshot) => T,
        listener: (value: T) => void,
        isEqual?: (left: T, right: T) => boolean,
    ) => () => void;
    publishFrame: (update: RecordingViewUpdate, lockedStaticData?: LiveSessionStaticData) => boolean;
    beginStream: () => void;
    resetSession: () => void;
    restoreCommittedSampleCount: (count: number) => void;
    finalizeCommittedSampleCount: (count: number) => void;
    restoreCommittedCount: (count: number) => void;
    finalizeCommittedCount: (count: number) => void;
}

const EMPTY_SAMPLE: StandardTelemetrySample = Object.freeze({});

const normalizeStatus = (value: unknown): ACC_STATUS | null => {
    const numeric = typeof value === 'string' ? Number(value) : value;
    if (typeof numeric !== 'number' || Number.isNaN(numeric)) return null;
    return ACC_STATUS[numeric as ACC_STATUS] !== undefined ? numeric as ACC_STATUS : null;
};

const selectFields = (sample: StandardTelemetrySample, prefix: 'Graphics_' | 'Physics_') => {
    const selected: StandardTelemetrySample = {};
    Object.entries(sample).forEach(([key, value]) => {
        if (key.startsWith(prefix)) selected[key] = value;
    });
    return Object.freeze(selected);
};

const getInitialSnapshot = (sessionGeneration = 0, streamGeneration = 0): LiveTelemetrySnapshot => ({
    graphicsTelemetry: EMPTY_SAMPLE,
    physicsTelemetry: EMPTY_SAMPLE,
    currentTelemetry: EMPTY_SAMPLE,
    sampleIndex: -1,
    telemetryStatus: null,
    committedSampleCount: 0,
    sessionGeneration,
    streamGeneration,
});

const isValidCount = (count: number): boolean => Number.isSafeInteger(count) && count >= 0;

export const createLiveTelemetryStore = (): LiveTelemetryStore => {
    let snapshot = getInitialSnapshot();
    let lockedStaticData: StandardTelemetrySample = EMPTY_SAMPLE;
    let latestFrameEvent: LiveTelemetryFrameEvent | null = null;
    const eventListeners = new Set<LiveTelemetryEventListener>();
    const snapshotListeners = new Set<LiveTelemetrySnapshotListener>();

    const notifyEvent = (event: LiveTelemetryEvent) => {
        Array.from(eventListeners).forEach((listener) => {
            try {
                listener(event);
            } catch {
                // A faulty processor must not prevent lossless delivery to other processors.
            }
        });
    };

    const notifySnapshot = () => {
        Array.from(snapshotListeners).forEach((listener) => {
            try {
                listener();
            } catch {
                // Keep snapshot observers isolated for the same reason as event observers.
            }
        });
    };

    const updateCommittedSampleCount = (count: number) => {
        if (!isValidCount(count)) throw new Error('Committed telemetry count must be a non-negative safe integer.');
        if (snapshot.committedSampleCount === count) return;
        snapshot = { ...snapshot, committedSampleCount: count };
        notifySnapshot();
    };

    const store: LiveTelemetryStore = {
        getSnapshot: () => snapshot,
        subscribeEvents: (listener, options = false) => {
            eventListeners.add(listener);
            const replayLatest = typeof options === 'boolean' ? options : options.replayLatest === true;
            if (replayLatest && latestFrameEvent) {
                try {
                    listener(latestFrameEvent);
                } catch {
                    // Replay follows the same listener-isolation contract as live delivery.
                }
            }
            return () => eventListeners.delete(listener);
        },
        subscribeSnapshot: (listener) => {
            snapshotListeners.add(listener);
            return () => snapshotListeners.delete(listener);
        },
        subscribeSelector: (selector, listener, isEqual = Object.is) => {
            let selected = selector(snapshot);
            return store.subscribeSnapshot(() => {
                const next = selector(snapshot);
                if (isEqual(selected, next)) return;
                selected = next;
                listener(next);
            });
        },
        publishFrame: (update, nextLockedStaticData = {}) => {
            const sampleIndex = update.sequence - 1;
            if (!Number.isSafeInteger(sampleIndex) || sampleIndex < 0 || sampleIndex <= snapshot.sampleIndex) {
                return false;
            }

            const nextStatic: StandardTelemetrySample = { ...lockedStaticData };
            Object.entries(nextLockedStaticData).forEach(([key, value]) => {
                if (key.startsWith('Static_') && !Object.prototype.hasOwnProperty.call(nextStatic, key)) {
                    nextStatic[key] = value as StandardTelemetrySample[string];
                }
            });
            lockedStaticData = Object.freeze(nextStatic);

            const graphicsTelemetry = selectFields(update.sample, 'Graphics_');
            const physicsTelemetry = selectFields(update.sample, 'Physics_');
            const currentTelemetry = Object.freeze({
                ...lockedStaticData,
                ...graphicsTelemetry,
                ...physicsTelemetry,
            });
            const telemetryStatus = normalizeStatus(graphicsTelemetry.Graphics_status);
            snapshot = {
                graphicsTelemetry,
                physicsTelemetry,
                currentTelemetry,
                sampleIndex,
                telemetryStatus: telemetryStatus ?? snapshot.telemetryStatus,
                committedSampleCount: update.committedCount,
                sessionGeneration: snapshot.sessionGeneration,
                streamGeneration: snapshot.streamGeneration,
            };
            latestFrameEvent = {
                type: 'frame',
                update,
                sample: currentTelemetry,
                sampleIndex,
                telemetryStatus: snapshot.telemetryStatus,
                committedSampleCount: update.committedCount,
                sessionGeneration: snapshot.sessionGeneration,
                streamGeneration: snapshot.streamGeneration,
            };
            notifyEvent(latestFrameEvent);
            notifySnapshot();
            return true;
        },
        beginStream: () => {
            latestFrameEvent = null;
            snapshot = {
                ...snapshot,
                graphicsTelemetry: EMPTY_SAMPLE,
                physicsTelemetry: EMPTY_SAMPLE,
                currentTelemetry: lockedStaticData,
                sampleIndex: -1,
                telemetryStatus: null,
                streamGeneration: snapshot.streamGeneration + 1,
            };
            notifyEvent({ type: 'stream-reset', snapshot });
            notifySnapshot();
        },
        resetSession: () => {
            lockedStaticData = EMPTY_SAMPLE;
            latestFrameEvent = null;
            snapshot = getInitialSnapshot(snapshot.sessionGeneration + 1, snapshot.streamGeneration + 1);
            notifyEvent({ type: 'session-reset', snapshot });
            notifySnapshot();
        },
        restoreCommittedSampleCount: updateCommittedSampleCount,
        finalizeCommittedSampleCount: updateCommittedSampleCount,
        restoreCommittedCount: updateCommittedSampleCount,
        finalizeCommittedCount: updateCommittedSampleCount,
    };

    return store;
};

export const liveTelemetryStore = createLiveTelemetryStore();

export const useLiveTelemetrySelector = <T,>(
    selector: (snapshot: LiveTelemetrySnapshot) => T,
): T => useSyncExternalStore(
    liveTelemetryStore.subscribeSnapshot,
    () => selector(liveTelemetryStore.getSnapshot()),
    () => selector(liveTelemetryStore.getSnapshot()),
);

export const useCurrentTelemetry = () => useLiveTelemetrySelector((snapshot) => snapshot.currentTelemetry);
export const useTelemetryStatus = () => useLiveTelemetrySelector((snapshot) => snapshot.telemetryStatus);
export const useTelemetrySampleIndex = () => useLiveTelemetrySelector((snapshot) => snapshot.sampleIndex);
export const useCommittedSampleCount = () => useLiveTelemetrySelector((snapshot) => snapshot.committedSampleCount);

// Compatibility aliases keep call sites explicit while the runtime context sheds these fields.
export const useLiveTelemetry = useCurrentTelemetry;
export const useLiveTelemetryStatus = useTelemetryStatus;
export const useLiveTelemetrySampleIndex = useTelemetrySampleIndex;
export const useRecordedSampleCount = useCommittedSampleCount;
