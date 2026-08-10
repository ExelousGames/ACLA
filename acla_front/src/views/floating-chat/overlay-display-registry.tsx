import React from 'react';
import {
    DriverExpertComparisonGraph,
    getDriverExpertReplayDurationMs,
    normalizeDriverExpertComparisonData,
} from 'components/driver-expert-comparison';
import { LiveRangeTodoListDisplay } from 'views/live-session/LiveRangeTodoList';
import AiMapToolDisplay from 'views/lap-analysis/ai-chat/AiMapToolDisplay';
import BaselineProgressDisplay from 'views/live-session/BaselineProgressDisplay';
import ProcedurePlanDisplay from 'views/lap-analysis/ai-chat/ProcedurePlanDisplay';
import ToolMessageDisplay from 'views/lap-analysis/ai-chat/ToolMessageDisplay';
import {
    OVERLAY_COMPARISON_COMPLETION_PAUSE_MS,
    OVERLAY_HOLD_MS,
    type OverlayCardinality,
    type OverlayComponentEvent,
    type OverlayDisplayType,
    type OverlayPolicy,
    type OverlaySnapshotByType,
    type OverlayUpsertOptions,
} from './overlay-display-types';

const TYPE_INTERVAL_MS = 28;
const ALL_POLICIES: readonly OverlayPolicy[] = [
    'pinned_top',
    'fold_until_update',
    'visible_until_exit',
    'transient',
];

export interface OverlayDisplayDimensions {
    width: number;
    height: number;
}

export interface OverlayDisplayRenderProps<TSnapshot> {
    snapshot: TSnapshot;
    revision: number;
    emitComponentEvent: (event: OverlayComponentEvent) => void;
}

export interface OverlayDisplayLifecycleDirective {
    exitAfterMs?: number;
    foldAfterMs?: number;
}

export interface OverlayDisplayDefinition<TSnapshot = unknown> {
    type: OverlayDisplayType;
    cardinality: OverlayCardinality;
    validateSnapshot: (snapshot: unknown) => snapshot is TSnapshot;
    resolveKey?: (snapshot: TSnapshot, options?: OverlayUpsertOptions) => string | null;
    initialPolicy: OverlayPolicy;
    permittedTransitions: Readonly<Record<OverlayPolicy, readonly OverlayPolicy[]>>;
    manualDismiss: boolean;
    dimensions: {
        expanded: OverlayDisplayDimensions;
        folded: OverlayDisplayDimensions;
    };
    pulseDurationMs?: number;
    transientDurationMs?: (snapshot: TSnapshot) => number | null;
    renderExpanded: (props: OverlayDisplayRenderProps<TSnapshot>) => React.ReactNode;
    renderFullSize?: (props: OverlayDisplayRenderProps<TSnapshot>) => React.ReactNode;
    renderSummary: (snapshot: TSnapshot) => React.ReactNode;
    lifecycleReducer?: (
        snapshot: TSnapshot,
        event: OverlayComponentEvent,
    ) => OverlayDisplayLifecycleDirective | null;
}

const transitionTable = (): Readonly<Record<OverlayPolicy, readonly OverlayPolicy[]>> => ({
    pinned_top: ALL_POLICIES,
    fold_until_update: ALL_POLICIES,
    visible_until_exit: ALL_POLICIES,
    transient: ALL_POLICIES,
});

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const isNonEmptyString = (value: unknown): value is string => (
    typeof value === 'string' && Boolean(value.trim())
);

const isFiniteOrNull = (value: unknown): boolean => value === null
    || (typeof value === 'number' && Number.isFinite(value));

const AiMessageBody: React.FC<OverlayDisplayRenderProps<OverlaySnapshotByType['ai_message']>> = ({
    snapshot,
    revision,
    emitComponentEvent,
}) => {
    const [text, setText] = React.useState('');

    React.useEffect(() => {
        const target = snapshot.text.trim();
        setText('');
        if (!target) {
            emitComponentEvent('visual_complete');
            return undefined;
        }
        let index = 0;
        const timer = window.setInterval(() => {
            index += 1;
            setText(target.slice(0, index));
            if (index >= target.length) {
                window.clearInterval(timer);
                emitComponentEvent('visual_complete');
            }
        }, TYPE_INTERVAL_MS);
        return () => window.clearInterval(timer);
    }, [emitComponentEvent, revision, snapshot.text]);

    return (
        <div className="overlay-card__message" data-testid="overlay-ai-message">
            {text}
            {text.length < snapshot.text.trim().length && <span className="overlay-card__caret" />}
        </div>
    );
};

const registry: { [K in OverlayDisplayType]: OverlayDisplayDefinition<OverlaySnapshotByType[K]> } = {
    ai_message: {
        type: 'ai_message',
        cardinality: 'singleton',
        validateSnapshot: (snapshot): snapshot is OverlaySnapshotByType['ai_message'] => (
            isRecord(snapshot) && isNonEmptyString(snapshot.text)
        ),
        initialPolicy: 'transient',
        permittedTransitions: transitionTable(),
        manualDismiss: true,
        dimensions: { expanded: { width: 420, height: 92 }, folded: { width: 280, height: 58 } },
        transientDurationMs: () => null,
        renderExpanded: (props) => <AiMessageBody {...props} />,
        renderSummary: (snapshot) => snapshot.text,
        lifecycleReducer: (_snapshot, event) => event === 'visual_complete'
            ? { exitAfterMs: OVERLAY_HOLD_MS }
            : null,
    },
    tool_status: {
        type: 'tool_status',
        cardinality: 'keyed',
        validateSnapshot: (snapshot): snapshot is OverlaySnapshotByType['tool_status'] => (
            isRecord(snapshot)
            && isNonEmptyString(snapshot.runId)
            && isNonEmptyString(snapshot.name)
            && isNonEmptyString(snapshot.title)
            && (snapshot.status === 'started' || snapshot.status === 'completed')
        ),
        resolveKey: (snapshot, options) => {
            const key = options?.key ?? snapshot.runId;
            return key === snapshot.runId && isNonEmptyString(key) ? key : null;
        },
        initialPolicy: 'transient',
        permittedTransitions: transitionTable(),
        manualDismiss: true,
        dimensions: { expanded: { width: 420, height: 118 }, folded: { width: 300, height: 58 } },
        transientDurationMs: () => OVERLAY_HOLD_MS,
        renderExpanded: ({ snapshot }) => <ToolMessageDisplay tool={snapshot} surface="pill" />,
        renderSummary: (snapshot) => `${snapshot.status === 'started' ? 'Running' : 'Finished'}: ${snapshot.title}`,
    },
    map: {
        type: 'map',
        cardinality: 'singleton',
        validateSnapshot: (snapshot): snapshot is OverlaySnapshotByType['map'] => (
            isRecord(snapshot) && (snapshot.status === 'ready' || snapshot.status === 'unavailable')
        ),
        initialPolicy: 'transient',
        permittedTransitions: transitionTable(),
        manualDismiss: true,
        dimensions: { expanded: { width: 420, height: 260 }, folded: { width: 300, height: 58 } },
        transientDurationMs: () => OVERLAY_HOLD_MS,
        renderExpanded: ({ snapshot }) => <AiMapToolDisplay display={snapshot} surface="pill" />,
        renderSummary: (snapshot) => snapshot.title || snapshot.map?.circuit_name || 'Circuit map',
    },
    driver_expert_comparison: {
        type: 'driver_expert_comparison',
        cardinality: 'multiple',
        validateSnapshot: (snapshot): snapshot is OverlaySnapshotByType['driver_expert_comparison'] => (
            isRecord(snapshot)
            && isNonEmptyString(snapshot.title)
            && Boolean(normalizeDriverExpertComparisonData(snapshot.comparison))
            && (
                snapshot.game === undefined
                || snapshot.game === null
                || snapshot.game === 'ac'
                || snapshot.game === 'acc'
                || snapshot.game === 'iracing'
            )
        ),
        initialPolicy: 'transient',
        permittedTransitions: transitionTable(),
        manualDismiss: true,
        dimensions: { expanded: { width: 760, height: 500 }, folded: { width: 360, height: 58 } },
        transientDurationMs: (snapshot) => Math.max(
            OVERLAY_HOLD_MS,
            getDriverExpertReplayDurationMs(snapshot.comparison)
                + OVERLAY_COMPARISON_COMPLETION_PAUSE_MS,
        ),
        renderExpanded: ({ snapshot }) => (
            <DriverExpertComparisonGraph
                className="floating-pill-comparison"
                data={snapshot.comparison}
                game={snapshot.game}
                title={snapshot.title}
                layout={{ trajectoryHeight: 280 }}
            />
        ),
        renderFullSize: ({ snapshot }) => (
            <DriverExpertComparisonGraph
                className="floating-pill-comparison floating-pill-comparison--full-size"
                data={snapshot.comparison}
                game={snapshot.game}
                title={snapshot.title}
                layout={{ trajectoryHeight: '100%' }}
            />
        ),
        renderSummary: (snapshot) => snapshot.title,
    },
    baseline_progress: {
        type: 'baseline_progress',
        cardinality: 'singleton',
        validateSnapshot: (snapshot): snapshot is OverlaySnapshotByType['baseline_progress'] => (
            isRecord(snapshot)
            && ['waiting_for_start', 'collecting', 'complete'].includes(String(snapshot.status))
            && typeof snapshot.progress_percent === 'number'
            && Number.isFinite(snapshot.progress_percent)
            && typeof snapshot.detail === 'string'
        ),
        initialPolicy: 'fold_until_update',
        permittedTransitions: transitionTable(),
        manualDismiss: true,
        dimensions: { expanded: { width: 420, height: 136 }, folded: { width: 300, height: 58 } },
        pulseDurationMs: OVERLAY_HOLD_MS,
        renderExpanded: ({ snapshot }) => <BaselineProgressDisplay tag={snapshot} surface="pill" />,
        renderSummary: (snapshot) => `Baseline ${Math.round(snapshot.progress_percent)}% - ${snapshot.detail}`,
    },
    procedure_plan: {
        type: 'procedure_plan',
        cardinality: 'singleton',
        validateSnapshot: (snapshot): snapshot is OverlaySnapshotByType['procedure_plan'] => (
            isRecord(snapshot)
            && isNonEmptyString(snapshot.goal)
            && Array.isArray(snapshot.requests)
            && snapshot.requests.every((request) => isRecord(request) && isNonEmptyString(request.title))
            && typeof snapshot.currentStep === 'number'
            && Number.isInteger(snapshot.currentStep)
        ),
        initialPolicy: 'visible_until_exit',
        permittedTransitions: transitionTable(),
        manualDismiss: true,
        dimensions: { expanded: { width: 420, height: 220 }, folded: { width: 320, height: 58 } },
        renderExpanded: ({ snapshot }) => <ProcedurePlanDisplay plan={snapshot} surface="pill" />,
        renderSummary: (snapshot) => snapshot.requests[snapshot.currentStep]?.title || snapshot.goal,
    },
    live_range_todo: {
        type: 'live_range_todo',
        cardinality: 'singleton',
        validateSnapshot: (snapshot): snapshot is OverlaySnapshotByType['live_range_todo'] => (
            isRecord(snapshot)
            && Array.isArray(snapshot.events)
            && snapshot.events.length > 0
            && snapshot.events.every((event) => (
                isRecord(event)
                && isNonEmptyString(event.id)
                && isRecord(event.content)
                && isNonEmptyString(event.content.title)
            ))
            && isFiniteOrNull(snapshot.current_position)
            && isFiniteOrNull(snapshot.rolling_rate)
        ),
        initialPolicy: 'pinned_top',
        permittedTransitions: transitionTable(),
        manualDismiss: true,
        dimensions: { expanded: { width: 420, height: 210 }, folded: { width: 340, height: 58 } },
        renderExpanded: ({ snapshot }) => <LiveRangeTodoListDisplay snapshot={snapshot} surface="pill" />,
        renderSummary: (snapshot) => `${snapshot.events.length} live range event${snapshot.events.length === 1 ? '' : 's'}`,
    },
};

export const overlayDisplayRegistry = registry;

export const getOverlayDisplayDefinition = <T extends OverlayDisplayType>(
    type: T,
): OverlayDisplayDefinition<OverlaySnapshotByType[T]> => registry[type];
