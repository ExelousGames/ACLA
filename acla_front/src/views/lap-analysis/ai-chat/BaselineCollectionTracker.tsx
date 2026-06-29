import { useEffect, useRef } from 'react';
import type { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';

export const BASELINE_PROGRESS_MESSAGE_ID = 'live-baseline-progress';
export const BASELINE_COLLECTION_SUBSCRIBER = 'baseline_collection';

export type BaselineCollectionTag = {
    id: typeof BASELINE_PROGRESS_MESSAGE_ID;
    subscriber: typeof BASELINE_COLLECTION_SUBSCRIBER;
    status: 'waiting_for_start' | 'collecting' | 'complete';
    ready: boolean;
    progress_percent: number;
    detail: string;
    snapshot: Record<string, any>;
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

type BaselineProgressMessage = {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    kind?: 'chat' | 'tool' | 'progress';
    progress?: {
        label: string;
        value: number;
        detail?: string;
        startMarkerLabel?: string;
        startMarkerValue?: number;
    };
};

type BaselineCollectionTrackerProps = {
    enabled: boolean;
    liveData: Record<string, any> | null | undefined;
    sessionMode: 'live' | 'recorded' | 'user_summary';
    sessionIntelligence: SessionIntelligence | null | undefined;
    onTagChange: (tag: BaselineCollectionTag | null) => void;
    onLapRecordChange: (record: BaselineLapRecord | null) => void;
    updateAgentMessages: (
        updater: (messages: BaselineProgressMessage[]) => BaselineProgressMessage[],
    ) => void;
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
            : 'Start at normalized position 0';

    return {
        id: BASELINE_PROGRESS_MESSAGE_ID,
        subscriber: BASELINE_COLLECTION_SUBSCRIBER,
        status,
        ready,
        progress_percent: progress,
        detail,
        snapshot,
    };
};

const upsertBaselineProgressMessage = (
    messages: BaselineProgressMessage[],
    tag: BaselineCollectionTag,
): BaselineProgressMessage[] => {
    const progressMessage: BaselineProgressMessage = {
        id: BASELINE_PROGRESS_MESSAGE_ID,
        content: 'Collecting baseline',
        isUser: false,
        timestamp: new Date(),
        kind: 'progress',
        progress: {
            label: 'Collecting baseline',
            value: tag.progress_percent,
            detail: tag.detail,
            startMarkerLabel: 'Start',
            startMarkerValue: 0,
        },
    };

    const existingIndex = messages.findIndex((message) => message.id === BASELINE_PROGRESS_MESSAGE_ID);
    if (existingIndex === -1) {
        return messages.concat(progressMessage);
    }

    const existingProgress = messages[existingIndex].progress;
    if (
        existingProgress?.value === progressMessage.progress?.value
        && existingProgress?.detail === progressMessage.progress?.detail
        && existingProgress?.label === progressMessage.progress?.label
        && existingProgress?.startMarkerLabel === progressMessage.progress?.startMarkerLabel
        && existingProgress?.startMarkerValue === progressMessage.progress?.startMarkerValue
    ) {
        return messages;
    }

    const next = messages.slice();
    next[existingIndex] = {
        ...next[existingIndex],
        progress: progressMessage.progress,
    };
    return next;
};

export const BaselineCollectionTracker = ({
    enabled,
    liveData,
    sessionMode,
    sessionIntelligence,
    onTagChange,
    onLapRecordChange,
    updateAgentMessages,
}: BaselineCollectionTrackerProps) => {
    const lapRecordRef = useRef<BaselineLapRecord | null>(null);

    useEffect(() => {
        const clearLapRecord = () => {
            if (lapRecordRef.current) {
                lapRecordRef.current = null;
                onLapRecordChange(null);
            }
        };

        if (sessionMode !== 'live' || !enabled) {
            onTagChange(null);
            clearLapRecord();
            updateAgentMessages((messages) => messages.filter((message) => message.id !== BASELINE_PROGRESS_MESSAGE_ID));
            return;
        }

        const snapshot = sessionIntelligence?.getLiveSessionSnapshot?.();
        if (!snapshot || snapshot.status === 'empty') {
            onTagChange(null);
            clearLapRecord();
            return;
        }

        const tag = buildBaselineCollectionTag(snapshot);
        onTagChange(tag);
        updateAgentMessages((messages) => upsertBaselineProgressMessage(messages, tag));

        if (!tag.ready) {
            clearLapRecord();
            return;
        }

        const lap = Number(snapshot.baseline_lap);
        const rows = sessionIntelligence?.getLastCompletedLapRows?.() ?? [];
        if (!Number.isFinite(lap) || rows.length === 0) {
            clearLapRecord();
            return;
        }

        const id = [
            String(snapshot.track || ''),
            String(snapshot.car || ''),
            String(lap),
            String(rows.length),
        ].join(':');
        if (lapRecordRef.current?.id === id) {
            return;
        }

        const nextRecord: BaselineLapRecord = {
            id,
            lap,
            captured_at: Date.now(),
            track: String(snapshot.track || ''),
            car: String(snapshot.car || ''),
            sample_count: rows.length,
            snapshot: { ...snapshot },
            records: rows.map((row) => ({ ...(row as Record<string, any>) })),
        };
        lapRecordRef.current = nextRecord;
        onLapRecordChange(nextRecord);
    }, [
        enabled,
        liveData,
        onLapRecordChange,
        onTagChange,
        sessionIntelligence,
        sessionMode,
        updateAgentMessages,
    ]);

    return null;
};
