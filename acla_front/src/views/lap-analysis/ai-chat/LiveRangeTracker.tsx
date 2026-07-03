import React, {
    forwardRef,
    useEffect,
    useImperativeHandle,
    useRef,
    useState,
} from 'react';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';

export type LiveRangeLifecycleStatus = 'pending' | 'triggered' | 'classifying' | 'classified' | 'error';
export type LiveRangeTrackerStatus = 'open' | 'closed';

export type LiveRangeSegment = {
    labels: string[];
    start_index: number;
    end_index: number;
};

export type LiveTrackedRange = {
    id: string;
    label?: string;
    start_position: number;
    end_position: number;
    lifecycle_status: LiveRangeLifecycleStatus;
    classifier_status?: string;
    parent_segment?: LiveRangeSegment;
    child_segments: LiveRangeSegment[];
    start_sample_idx?: number;
    end_sample_idx?: number;
    lap?: number;
    triggered_at?: number;
    error?: string;
};

export type LiveRangeTrackerState = {
    status: LiveRangeTrackerStatus;
    ranges: LiveTrackedRange[];
    created_at: number;
    updated_at: number;
};

export type LiveRangeTrackerToolResult = {
    status: 'ready' | 'closed' | 'empty' | 'error';
    tracker: LiveRangeTrackerState | null;
    error?: string;
    message?: string;
};

export type LiveRangeTrackerHandle = {
    setTracker: (args: Record<string, unknown>) => LiveRangeTrackerToolResult;
    updateTracker: (args: Record<string, unknown>) => LiveRangeTrackerToolResult;
    getTracker: () => LiveRangeTrackerToolResult;
};

type LiveRangeTrackerProps = {
    liveData?: Record<string, any> | null;
    sessionMode?: 'front_desk' | 'live' | 'recorded' | 'user_summary';
    sessionIntelligence?: SessionIntelligence | null;
    sendToolStatus?: (data: Record<string, unknown>) => boolean;
    resolveLabel?: (labelId: string) => string | undefined;
    onStateChange?: (tracker: LiveRangeTrackerState | null) => void;
};

const TRACKER_EMPTY_RESULT: LiveRangeTrackerToolResult = {
    status: 'empty',
    tracker: null,
    message: 'No live range tracker is active.',
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const getNormalizedCarPos = (telemetry: Record<string, any> | null | undefined): number | undefined => {
    if (!telemetry) return undefined;
    const keys = [
        'Graphics_normalized_car_position',
        'graphics_normalized_car_position',
        'normalized_car_position',
        'car_position',
    ];
    for (const key of keys) {
        if (key in telemetry) {
            const value = Number(telemetry[key]);
            if (Number.isFinite(value)) return Math.max(0, Math.min(1, value));
        }
    }
    return undefined;
};

const getTelemetryLap = (telemetry: Record<string, any> | null | undefined): number => {
    const raw = telemetry?.Graphics_completed_laps
        ?? telemetry?.Graphics_completed_lap
        ?? telemetry?.Graphics?.completed_laps
        ?? 0;
    const parsed = Math.floor(Number(raw));
    return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
};

export const crossedNormalizedPosition = (
    lastPos: number,
    currentPos: number,
    targetPos: number,
): boolean => {
    if (currentPos >= lastPos) {
        return lastPos < targetPos && currentPos >= targetPos;
    }
    return lastPos < targetPos || currentPos >= targetPos;
};

const clampNormalizedPosition = (value: unknown): number | null => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return Math.max(0, Math.min(1, parsed));
};

const getRangeInputArray = (args: Record<string, unknown>): unknown[] => {
    const raw = args.ranges ?? args.sections ?? args.tracked_ranges;
    return Array.isArray(raw) ? raw : [];
};

const normalizeLabels = (value: unknown): string[] => {
    if (Array.isArray(value)) {
        return value.map(String).map((label) => label.trim()).filter(Boolean);
    }
    if (typeof value === 'string' && value.trim()) {
        return value.split(',').map((label) => label.trim()).filter(Boolean);
    }
    return [];
};

const normalizeSegment = (
    value: unknown,
    fallbackStart?: number,
    fallbackEnd?: number,
): LiveRangeSegment | null => {
    const record = isRecord(value) ? value : {};
    const labels = normalizeLabels(record.labels ?? record.label_ids ?? record.labelIds);
    const start = Math.floor(Number(record.start_index ?? record.startIndex ?? fallbackStart));
    const end = Math.floor(Number(record.end_index ?? record.endIndex ?? fallbackEnd));

    if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
    return {
        labels,
        start_index: Math.max(0, start),
        end_index: Math.max(0, end),
    };
};

const normalizeRangeInput = (value: unknown, index: number): LiveTrackedRange | null => {
    if (!isRecord(value)) return null;

    const start = clampNormalizedPosition(value.start_position ?? value.startPosition ?? value.start);
    const end = clampNormalizedPosition(value.end_position ?? value.endPosition ?? value.end);
    if (start === null || end === null) return null;

    const id = typeof value.id === 'string' && value.id.trim()
        ? value.id.trim()
        : `range-${index + 1}`;
    const label = typeof value.label === 'string' && value.label.trim()
        ? value.label.trim()
        : undefined;

    return {
        id,
        label,
        start_position: start,
        end_position: end,
        lifecycle_status: 'pending',
        child_segments: [],
    };
};

const resultForTracker = (tracker: LiveRangeTrackerState | null): LiveRangeTrackerToolResult => {
    if (!tracker) return TRACKER_EMPTY_RESULT;
    return {
        status: tracker.status === 'closed' ? 'closed' : 'ready',
        tracker,
    };
};

const errorResult = (message: string, tracker: LiveRangeTrackerState | null): LiveRangeTrackerToolResult => ({
    status: 'error',
    tracker,
    error: message,
    message,
});

const formatPosition = (value: number): string => value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');

type LiveRangeTrackerDisplayProps = {
    tracker: LiveRangeTrackerState | null;
    surface?: 'chat' | 'pill';
    resolveLabel?: (labelId: string) => string | undefined;
};

export const LiveRangeTrackerDisplay: React.FC<LiveRangeTrackerDisplayProps> = ({
    tracker,
    surface = 'chat',
    resolveLabel,
}) => {
    if (!tracker || tracker.ranges.length === 0) {
        return null;
    }

    const ranges = surface === 'pill'
        ? tracker.ranges.slice(0, 3)
        : tracker.ranges;

    return (
        <div className={`ai-chat__range-tracker ai-chat__range-tracker--${surface}`} aria-label="Live tracked ranges">
            <div className="ai-chat__range-tracker-head">
                <div>
                    <span className="ai-chat__range-tracker-kicker">RANGE TRACKER</span>
                    <div className="ai-chat__range-tracker-title">
                        {tracker.ranges.length} tracked range{tracker.ranges.length === 1 ? '' : 's'}
                    </div>
                </div>
                <span className={`ai-chat__range-tracker-state ai-chat__range-tracker-state--${tracker.status}`}>
                    {tracker.status}
                </span>
            </div>
            <ul className="ai-chat__range-list">
                {ranges.map((range) => (
                    <li key={range.id} className={`ai-chat__range-item ai-chat__range-item--${range.lifecycle_status}`}>
                        <div className="ai-chat__range-item-main">
                            <span className="ai-chat__range-item-name">{range.label || range.id}</span>
                            <span className="ai-chat__range-item-pos">
                                {formatPosition(range.start_position)}-{formatPosition(range.end_position)}
                            </span>
                            <span className="ai-chat__range-item-status">{range.lifecycle_status}</span>
                        </div>
                        {surface === 'chat' && range.parent_segment && (
                            <div className="ai-chat__range-segment">
                                <span>Parent</span>
                                <strong>{range.parent_segment.labels.map((label) => resolveLabel?.(label) || label).join(', ') || 'Unlabeled'}</strong>
                                <em>{range.parent_segment.start_index}-{range.parent_segment.end_index}</em>
                            </div>
                        )}
                        {surface === 'chat' && range.child_segments.length > 0 && (
                            <div className="ai-chat__range-children">
                                {range.child_segments.map((child, index) => (
                                    <div className="ai-chat__range-segment" key={`${range.id}-${index}-${child.start_index}-${child.end_index}`}>
                                        <span>Child</span>
                                        <strong>{child.labels.map((label) => resolveLabel?.(label) || label).join(', ') || 'Unlabeled'}</strong>
                                        <em>{child.start_index}-{child.end_index}</em>
                                    </div>
                                ))}
                            </div>
                        )}
                        {range.error && (
                            <div className="ai-chat__range-error">{range.error}</div>
                        )}
                    </li>
                ))}
            </ul>
        </div>
    );
};

const LiveRangeTracker = forwardRef<LiveRangeTrackerHandle, LiveRangeTrackerProps>(({
    liveData,
    sessionMode = 'live',
    sessionIntelligence,
    sendToolStatus,
    resolveLabel,
    onStateChange,
}, ref) => {
    const [tracker, setTracker] = useState<LiveRangeTrackerState | null>(null);
    const trackerRef = useRef<LiveRangeTrackerState | null>(null);
    const lastPositionRef = useRef<number | undefined>(undefined);
    const onStateChangeRef = useRef<typeof onStateChange>(onStateChange);

    useEffect(() => {
        onStateChangeRef.current = onStateChange;
    }, [onStateChange]);

    const commitTracker = (next: LiveRangeTrackerState | null): LiveRangeTrackerToolResult => {
        trackerRef.current = next;
        setTracker(next);
        onStateChangeRef.current?.(next);
        return resultForTracker(next);
    };

    const getTracker = (): LiveRangeTrackerToolResult => resultForTracker(trackerRef.current);

    const setNewTracker = (args: Record<string, unknown>): LiveRangeTrackerToolResult => {
        const ranges = getRangeInputArray(args)
            .map(normalizeRangeInput)
            .filter((range): range is LiveTrackedRange => Boolean(range));

        if (ranges.length === 0) {
            return errorResult('Provide at least one valid tracked range with start_position and end_position.', trackerRef.current);
        }

        const now = Date.now();
        lastPositionRef.current = undefined;
        return commitTracker({
            status: 'open',
            ranges,
            created_at: now,
            updated_at: now,
        });
    };

    const updateTracker = (args: Record<string, unknown>): LiveRangeTrackerToolResult => {
        const current = trackerRef.current;
        if (!current) return TRACKER_EMPTY_RESULT;

        const action = typeof args.action === 'string' ? args.action : 'update_ranges';
        const now = Date.now();

        if (action === 'close') {
            return commitTracker({
                ...current,
                status: 'closed',
                updated_at: now,
            });
        }

        if (action === 'remove_ranges' || action === 'remove_sections') {
            const rawIds = args.range_ids ?? args.rangeIds ?? args.ids;
            const ids = Array.isArray(rawIds)
                ? rawIds.map(String)
                : typeof rawIds === 'string'
                    ? [rawIds]
                    : [];
            if (ids.length === 0) return errorResult('Provide range_ids to remove.', current);

            return commitTracker({
                ...current,
                ranges: current.ranges.filter((range) => !ids.includes(range.id)),
                updated_at: now,
            });
        }

        if (action === 'record_classification') {
            const rangeId = String(args.range_id ?? args.rangeId ?? args.id ?? '').trim();
            if (!rangeId) return errorResult('Provide range_id for record_classification.', current);

            const parentSegment = normalizeSegment(
                args.parent_segment ?? args.parentSegment ?? {
                    labels: args.parent_labels ?? args.parentLabels ?? args.labels,
                    start_index: args.start_index ?? args.startIndex,
                    end_index: args.end_index ?? args.endIndex,
                },
            );
            const rawChildren = args.child_segments ?? args.childSegments ?? [];
            const childSegments = Array.isArray(rawChildren)
                ? rawChildren
                    .map((child) => normalizeSegment(child))
                    .filter((child): child is LiveRangeSegment => Boolean(child))
                : [];
            const classifierStatus = typeof args.classifier_status === 'string'
                ? args.classifier_status
                : typeof args.classifierStatus === 'string'
                    ? args.classifierStatus
                    : 'classified';

            return commitTracker({
                ...current,
                ranges: current.ranges.map((range) => (
                    range.id === rangeId
                        ? {
                            ...range,
                            lifecycle_status: 'classified',
                            classifier_status: classifierStatus,
                            parent_segment: parentSegment ?? range.parent_segment,
                            child_segments: childSegments,
                            error: undefined,
                        }
                        : range
                )),
                updated_at: now,
            });
        }

        if (action === 'update_ranges' || action === 'update_sections') {
            const updates = getRangeInputArray(args)
                .map(normalizeRangeInput)
                .filter((range): range is LiveTrackedRange => Boolean(range));
            if (updates.length === 0) return errorResult('Provide ranges to update.', current);

            const byId = new Map(current.ranges.map((range) => [range.id, range]));
            updates.forEach((range) => {
                const existing = byId.get(range.id);
                byId.set(range.id, existing
                    ? {
                        ...existing,
                        label: range.label,
                        start_position: range.start_position,
                        end_position: range.end_position,
                        lifecycle_status: 'pending',
                        classifier_status: undefined,
                        parent_segment: undefined,
                        child_segments: [],
                        start_sample_idx: undefined,
                        end_sample_idx: undefined,
                        lap: undefined,
                        triggered_at: undefined,
                        error: undefined,
                    }
                    : range);
            });

            return commitTracker({
                ...current,
                ranges: Array.from(byId.values()),
                updated_at: now,
            });
        }

        return errorResult(`Unsupported live range tracker action: ${action}`, current);
    };

    useImperativeHandle(ref, () => ({
        setTracker: setNewTracker,
        updateTracker,
        getTracker,
    }));

    useEffect(() => {
        if (sessionMode !== 'live') {
            lastPositionRef.current = undefined;
            return;
        }

        const currentPos = getNormalizedCarPos(liveData);
        if (currentPos === undefined) return;

        const lastPos = lastPositionRef.current;
        lastPositionRef.current = currentPos;
        if (lastPos === undefined) return;

        const current = trackerRef.current;
        if (!current || current.status !== 'open') return;

        const triggeredRanges = current.ranges.filter((range) => (
            range.lifecycle_status === 'pending'
            && crossedNormalizedPosition(lastPos, currentPos, range.end_position)
        ));
        if (triggeredRanges.length === 0) return;

        let nextTracker = current;
        triggeredRanges.forEach((range) => {
            const lap = getTelemetryLap(liveData);
            const windowResult = sessionIntelligence?.getTelemetryWindowForNormalizedRange({
                start_position: range.start_position,
                end_position: range.end_position,
                lap,
            });
            const triggeredAt = Date.now();
            const hasWindow = windowResult?.status === 'ready';
            const toolStatus = {
                source: 'live_range_tracker',
                event: 'live_range_classification_requested',
                range_id: range.id,
                label: range.label ?? null,
                start_position: range.start_position,
                end_position: range.end_position,
                lap,
                start_sample_idx: windowResult?.startSampleIdx ?? null,
                end_sample_idx: windowResult?.endSampleIdx ?? null,
                telemetry_rows: windowResult?.rows ?? [],
                telemetry_row_count: windowResult?.rows.length ?? 0,
                message: 'Classify this tracked range and update it with update_live_range_tracker action=record_classification.',
            };
            const sent = hasWindow && sendToolStatus?.(toolStatus) === true;

            nextTracker = {
                ...nextTracker,
                ranges: nextTracker.ranges.map((candidate) => (
                    candidate.id === range.id
                        ? {
                            ...candidate,
                            lifecycle_status: sent ? 'classifying' : 'error',
                            lap,
                            start_sample_idx: windowResult?.startSampleIdx,
                            end_sample_idx: windowResult?.endSampleIdx,
                            triggered_at: triggeredAt,
                            error: sent
                                ? undefined
                                : hasWindow
                                    ? 'AI session is not connected.'
                                    : 'No telemetry rows were available for this tracked range.',
                        }
                        : candidate
                )),
                updated_at: triggeredAt,
            };
        });

        commitTracker(nextTracker);
    }, [liveData, sendToolStatus, sessionIntelligence, sessionMode]);

    return (
        <LiveRangeTrackerDisplay
            tracker={tracker}
            resolveLabel={resolveLabel}
        />
    );
});

export default LiveRangeTracker;
