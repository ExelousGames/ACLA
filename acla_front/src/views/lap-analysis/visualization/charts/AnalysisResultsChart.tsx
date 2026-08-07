import React from 'react';
import { Badge, Box, Card, Flex, HoverCard, ScrollArea, Text } from '@radix-ui/themes';
import type { VisualizationProps } from '../VisualizationRegistry';
import {
    AnalysisResultElement,
    normalizeAnalysisResultsData,
} from './analysisResultsModel';
import { useAiLabels } from 'contexts/AiLabelsContext';
import { DataGraph, GraphRecord, GraphSpec } from 'components/data-graphs';
import {
    DriverExpertComparisonGraph,
    getDriverExpertReplayDurationMs,
    hasComparableDriverExpertData,
} from 'components/driver-expert-comparison';
import type { DriverExpertComparisonData } from 'components/driver-expert-comparison';
import {
    FLOATING_PILL_COMPARISON_COMPLETION_PAUSE_MS,
    FLOATING_PILL_RICH_CONTENT_HOLD_MS,
    broadcastFloatingPillPayload,
} from 'views/floating-chat/floating-pill-bridge';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import type {
    JsonValue,
    LiveRangeTodoEventInput,
    LiveRangeTodoListHandle,
} from 'views/live-session/live-range-todo-list-types';
import styles from './AnalysisResultsChart.module.css';

const formatPosition = (value: number): string => `${(value * 100).toFixed(1)}%`;
const LIVE_RANGE_TODO_LIST_TYPE = 'live-range-todo-list';
const LIVE_RANGE_TODO_LIST_MOUNT_TIMEOUT_MS = 2000;

let queuedComparisonEventSequence = 0;

const createQueuedComparisonEventId = (): string => (
    `analysis-comparison-${Date.now().toString(36)}-${(++queuedComparisonEventSequence).toString(36)}`
);

interface ComparisonPillPayload {
    title: string;
    comparison: DriverExpertComparisonData;
}

interface LeadingMistakeOccurrence {
    element: AnalysisResultElement;
    matchedLeadingLabels: string[];
}

interface MostCommonMistakeQueuePlan {
    eligible: LeadingMistakeOccurrence[];
    skippedCount: number;
}

interface PendingComparisonQueue {
    requestId: number;
    events: LiveRangeTodoEventInput[];
    skippedCount: number;
}

const showComparisonPayloadForHold = (
    payload: ComparisonPillPayload,
    signal: AbortSignal,
): Promise<void> => new Promise((resolve) => {
    let firstFrame: number | null = null;
    let secondFrame: number | null = null;
    let fallbackTimer: number | null = null;
    let holdTimer: number | null = null;
    let finished = false;

    const finish = () => {
        if (finished) return;
        finished = true;
        if (firstFrame !== null) window.cancelAnimationFrame(firstFrame);
        if (secondFrame !== null) window.cancelAnimationFrame(secondFrame);
        if (fallbackTimer !== null) window.clearTimeout(fallbackTimer);
        if (holdTimer !== null) window.clearTimeout(holdTimer);
        signal.removeEventListener('abort', finish);
        resolve();
    };
    const broadcast = () => {
        firstFrame = null;
        secondFrame = null;
        fallbackTimer = null;
        if (signal.aborted) {
            finish();
            return;
        }
        broadcastFloatingPillPayload({
            kind: 'driver_expert_comparison',
            text: payload.title,
            data: payload,
        });
        const holdDurationMs = Math.max(
            FLOATING_PILL_RICH_CONTENT_HOLD_MS,
            getDriverExpertReplayDurationMs(payload.comparison)
                + FLOATING_PILL_COMPARISON_COMPLETION_PAUSE_MS,
        );
        holdTimer = window.setTimeout(finish, holdDurationMs);
    };

    signal.addEventListener('abort', finish, { once: true });
    if (signal.aborted) {
        finish();
        return;
    }

    // The queue first commits its running state. Two paint frames let that
    // list lifecycle render and broadcast before the richer comparison wins.
    if (typeof window.requestAnimationFrame === 'function') {
        firstFrame = window.requestAnimationFrame(() => {
            firstFrame = null;
            secondFrame = window.requestAnimationFrame(broadcast);
        });
    } else {
        fallbackTimer = window.setTimeout(broadcast, 0);
    }
});

const HIDDEN_METADATA_KEYS = new Set(['source', 'start_index', 'end_index']);

type AnalysisResultsSortMode = 'original' | 'most-frequent-sub-label' | 'most-time-lost';
type AnalysisResultsMainLabelFilter = 'MSP' | 'MSR';

interface MainLabelFilterOption {
    value: AnalysisResultsMainLabelFilter;
    label: string;
    resolvedLabel: string;
    sortDisplayName: string;
}

const MAIN_LABEL_FILTER_OPTIONS: readonly MainLabelFilterOption[] = [
    {
        value: 'MSP',
        label: 'Training Mistake',
        resolvedLabel: 'Mistake (Practice)',
        sortDisplayName: 'Most common training mistake',
    },
    {
        value: 'MSR',
        label: 'Racing Mistake',
        resolvedLabel: 'Mistake (Racing)',
        sortDisplayName: 'Most common racing mistake',
    },
];

interface AnalysisResultsChartProps extends VisualizationProps {
    showElementId?: boolean;
}

interface IndexedAnalysisResult {
    element: AnalysisResultElement;
    originalIndex: number;
}

interface RecognizedSubLabel {
    id: string;
    label: string;
}

type RecognizedSubLabels = ReadonlyMap<string, RecognizedSubLabel>;

const compareLabelText = (left: string, right: string): number => (
    left.localeCompare(right, undefined, { sensitivity: 'base' }) || left.localeCompare(right)
);

const getSubLabels = (
    element: AnalysisResultElement,
    recognizedSubLabels: RecognizedSubLabels,
): RecognizedSubLabel[] => {
    const matches = new Map<string, RecognizedSubLabel>();
    element.labels.forEach((label) => {
        const recognized = recognizedSubLabels.get(label);
        if (recognized) matches.set(recognized.id, recognized);
    });
    return Array.from(matches.values());
};

export const buildMistakeFrequencyData = (
    elements: readonly AnalysisResultElement[],
    recognizedSubLabels: ReadonlyMap<string, RecognizedSubLabel>,
): GraphRecord[] => {
    const frequencies = new Map<string, { label: string; occurrences: number }>();
    elements.forEach((element) => {
        getSubLabels(element, recognizedSubLabels).forEach((subLabel) => {
            const current = frequencies.get(subLabel.id);
            frequencies.set(subLabel.id, {
                label: subLabel.label,
                occurrences: (current?.occurrences ?? 0) + 1,
            });
        });
    });

    return Array.from(frequencies.values())
        .filter(({ occurrences }) => occurrences > 0)
        .sort((left, right) => (
            right.occurrences - left.occurrences
            || compareLabelText(left.label, right.label)
        ));
};

const buildMostCommonMistakeQueuePlan = (
    elements: readonly AnalysisResultElement[],
    recognizedSubLabels: RecognizedSubLabels,
    frequencyData: readonly GraphRecord[],
): MostCommonMistakeQueuePlan => {
    const leadingFrequency = frequencyData.reduce((highest, row) => {
        const occurrences = Number(row.occurrences);
        return Number.isFinite(occurrences) ? Math.max(highest, occurrences) : highest;
    }, 0);
    if (leadingFrequency <= 0) return { eligible: [], skippedCount: 0 };

    const leadingLabels = new Set(frequencyData
        .filter((row) => Number(row.occurrences) === leadingFrequency)
        .map((row) => String(row.label)));
    const occurrences: LeadingMistakeOccurrence[] = elements.flatMap((element) => {
        const matchedLeadingLabels = getSubLabels(element, recognizedSubLabels)
            .map(({ label }) => label)
            .filter((label) => leadingLabels.has(label))
            .sort(compareLabelText);
        return matchedLeadingLabels.length > 0 ? [{ element, matchedLeadingLabels }] : [];
    });
    const eligible = occurrences.filter(({ element }) => {
        const start = element.normalizedPositionRange?.start;
        return typeof start === 'number'
            && Number.isFinite(start)
            && start >= 0
            && start <= 1
            && hasComparableDriverExpertData(element.comparison);
    });

    return {
        eligible,
        skippedCount: occurrences.length - eligible.length,
    };
};

const createMostCommonMistakeEvents = (
    plan: MostCommonMistakeQueuePlan,
): LiveRangeTodoEventInput[] => plan.eligible.map(({ element, matchedLeadingLabels }) => {
    const position = element.normalizedPositionRange!.start;
    const comparison = element.comparison!;
    const title = element.title || matchedLeadingLabels[0];
    const context = {
        section: element.section ?? null,
        position,
        source_result_id: element.id,
        matched_leading_labels: matchedLeadingLabels,
    };

    return {
        id: createQueuedComparisonEventId(),
        normalized_position: position,
        lead_time_seconds: 0,
        content: {
            title,
            detail: `${element.section || 'Unknown section'} · ${formatPosition(position)}`,
            metadata: context as JsonValue,
        },
        data: {
            title,
            comparison,
            context,
        } as unknown as JsonValue,
        callback: ({ signal }) => showComparisonPayloadForHold({ title, comparison }, signal),
    };
});

export const getMistakeFrequencyGraphHeight = (categoryCount: number): number => (
    160 + (Math.max(1, categoryCount) * 36)
);

const sortAnalysisResults = (
    elements: AnalysisResultElement[],
    sortMode: AnalysisResultsSortMode,
    recognizedSubLabels: RecognizedSubLabels,
): AnalysisResultElement[] => {
    const indexedElements: IndexedAnalysisResult[] = elements.map((element, originalIndex) => ({
        element,
        originalIndex,
    }));

    if (sortMode === 'original') {
        return indexedElements.map(({ element }) => element);
    }

    if (sortMode === 'most-time-lost') {
        return [...indexedElements]
            .sort((left, right) => {
                const leftDelta = left.element.timeGap?.deltaMs;
                const rightDelta = right.element.timeGap?.deltaMs;
                const leftIsValid = typeof leftDelta === 'number' && Number.isFinite(leftDelta);
                const rightIsValid = typeof rightDelta === 'number' && Number.isFinite(rightDelta);

                if (leftIsValid !== rightIsValid) return leftIsValid ? -1 : 1;
                if (leftIsValid && rightIsValid && leftDelta !== rightDelta) {
                    return rightDelta - leftDelta;
                }
                return left.originalIndex - right.originalIndex;
            })
            .map(({ element }) => element);
    }

    const subLabelsByElement = indexedElements.map(({ element }) => (
        getSubLabels(element, recognizedSubLabels)
    ));
    const subLabelCounts = new Map<string, number>();
    subLabelsByElement.forEach((subLabels) => {
        subLabels.forEach((subLabel) => {
            subLabelCounts.set(subLabel.id, (subLabelCounts.get(subLabel.id) ?? 0) + 1);
        });
    });
    const rankings = subLabelsByElement.map((subLabels) => (
        subLabels.reduce((ranking, subLabel) => {
            const count = subLabelCounts.get(subLabel.id) ?? 0;
            if (
                count > ranking.count
                || (count === ranking.count && (!ranking.label || compareLabelText(subLabel.label, ranking.label) < 0))
            ) {
                return { count, label: subLabel.label };
            }
            return ranking;
        }, { count: 0, label: '' })
    ));

    return [...indexedElements]
        .sort((left, right) => {
            const leftRanking = rankings[left.originalIndex];
            const rightRanking = rankings[right.originalIndex];
            return rightRanking.count - leftRanking.count
                || compareLabelText(leftRanking.label, rightRanking.label)
                || left.originalIndex - right.originalIndex;
        })
        .map(({ element }) => element);
};

const formatMilliseconds = (value: unknown): string | null => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? `${parsed.toFixed(0)} ms` : null;
};

const stringifyValue = (value: unknown): string => {
    if (value === null) return 'null';
    if (value === undefined) return '-';
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') {
        return String(value);
    }
    try {
        return JSON.stringify(value);
    } catch {
        return '[unavailable]';
    }
};

const TimeGap: React.FC<{ element: AnalysisResultElement }> = ({ element }) => {
    if (!element.timeGap) return null;
    const start = formatMilliseconds(element.timeGap.startMs);
    const end = formatMilliseconds(element.timeGap.endMs);
    const delta = formatMilliseconds(element.timeGap.deltaMs);
    const text = start && end
        ? `${start} – ${end}${delta ? ` (${delta})` : ''}`
        : delta || stringifyValue(element.timeGap);
    return <Text size="1" color="gray">Time gap: {text}</Text>;
};

const AnalysisResultCard: React.FC<{
    element: AnalysisResultElement;
    resultNumber: number;
    showElementId: boolean;
}> = ({ element, resultNumber, showElementId }) => {
    const [comparisonOpen, setComparisonOpen] = React.useState(false);
    const metadataEntries = Object.entries(element.metadata ?? {})
        .filter(([key]) => !HIDDEN_METADATA_KEYS.has(key));
    const hasComparison = hasComparableDriverExpertData(element.comparison);

    const card = (
        <Box
            className={[styles.element, hasComparison ? styles.comparisonTrigger : '']
                .filter(Boolean).join(' ')}
            data-testid={`analysis-result-${element.id}`}
            tabIndex={hasComparison ? 0 : undefined}
        >
            <Flex justify="between" align="start" gap="2" wrap="wrap">
                <Flex className={styles.heading} align="start" gap="2">
                    <Text
                        className={styles.number}
                        size="1"
                        weight="bold"
                        aria-label={`Analysis result ${resultNumber}`}
                    >
                        {resultNumber}
                    </Text>
                    <Box>
                        {element.title && <Text size="2" weight="bold" as="div">{element.title}</Text>}
                        {showElementId && <Text size="1" className={styles.id} as="div">{element.id}</Text>}
                    </Box>
                </Flex>
                <Flex gap="1" wrap="wrap" justify="end">
                    {element.labels.length > 0
                        ? element.labels.map((label, index) => (
                            <Badge className={styles.label} variant="soft" key={`${label}-${index}`}>{label}</Badge>
                        ))
                        : <Badge color="gray" variant="outline">Unlabeled</Badge>}
                </Flex>
            </Flex>

            {(element.section || element.normalizedPositionRange || element.timeGap) && (
                <Box className={styles.context}>
                    {element.section && <Text size="1" color="gray">Section: {element.section}</Text>}
                    {element.normalizedPositionRange && (
                        <Text size="1" color="gray">
                            Position: {formatPosition(element.normalizedPositionRange.start)} – {formatPosition(element.normalizedPositionRange.end)}
                        </Text>
                    )}
                    <TimeGap element={element} />
                </Box>
            )}

            {metadataEntries.length > 0 && (
                <Box className={styles.metadata}>
                    {metadataEntries.map(([key, value]) => (
                        <Text size="1" color="gray" as="div" key={key}>
                            {key}: {stringifyValue(value)}
                        </Text>
                    ))}
                </Box>
            )}
            <Text
                className={hasComparison ? styles.comparisonHint : styles.comparisonUnavailable}
                size="1"
                color="gray"
                as="div"
            >
                {hasComparison
                    ? 'Hover or focus for Driver and Expert comparison'
                    : 'Expert comparison unavailable'}
            </Text>
        </Box>
    );

    if (!hasComparison || !element.comparison) return card;

    return (
        <HoverCard.Root open={comparisonOpen} onOpenChange={setComparisonOpen}>
            <HoverCard.Trigger>{card}</HoverCard.Trigger>
            <HoverCard.Content
                className={styles.comparisonContent}
                side="right"
                align="start"
                sideOffset={10}
                avoidCollisions
                collisionPadding={12}
            >
                {comparisonOpen && (
                    <DriverExpertComparisonGraph
                        data={element.comparison}
                        title={element.title
                            ? `${element.title}: Driver vs Expert`
                            : 'Driver vs Expert'}
                        layout={{
                            chartHeight: 180,
                            trajectoryHeight: 210,
                            minColumnWidth: 300,
                        }}
                    />
                )}
            </HoverCard.Content>
        </HoverCard.Root>
    );
};

const AnalysisResultsChart: React.FC<AnalysisResultsChartProps> = ({
    id,
    data,
    width = '100%',
    height = '100%',
    showElementId = true,
}) => {
    const [sortMode, setSortMode] = React.useState<AnalysisResultsSortMode>('original');
    const [mainLabelFilter, setMainLabelFilter] = React.useState<AnalysisResultsMainLabelFilter>('MSP');
    const [pendingQueue, setPendingQueue] = React.useState<PendingComparisonQueue | null>(null);
    const [queueInProgress, setQueueInProgress] = React.useState(false);
    const [queueStatus, setQueueStatus] = React.useState('');
    const liveSession = React.useContext(LiveSessionContext);
    const pendingQueueRef = React.useRef<PendingComparisonQueue | null>(null);
    const mountTimeoutRef = React.useRef<number | null>(null);
    const queueRequestSequenceRef = React.useRef(0);
    const { getCategoryLabels, getLabelName } = useAiLabels();
    const { elements } = React.useMemo(() => normalizeAnalysisResultsData(data), [data]);
    const selectedFilter = MAIN_LABEL_FILTER_OPTIONS.find(({ value }) => value === mainLabelFilter)!;
    const recognizedParentLabels = React.useMemo(() => new Set([
        selectedFilter.value,
        selectedFilter.resolvedLabel,
        getLabelName(selectedFilter.value),
    ].filter((label): label is string => Boolean(label))), [getLabelName, selectedFilter]);
    const filteredElements = React.useMemo(
        () => elements.filter((element) => (
            element.labels.some((label) => recognizedParentLabels.has(label))
        )),
        [elements, recognizedParentLabels],
    );
    const recognizedSubLabels = React.useMemo(() => {
        const matches = new Map<string, RecognizedSubLabel>();
        getCategoryLabels(mainLabelFilter).forEach((childId) => {
            const child = { id: childId, label: getLabelName(childId) ?? childId };
            matches.set(childId, child);
            matches.set(child.label, child);
        });
        return matches;
    }, [getCategoryLabels, getLabelName, mainLabelFilter]);
    const sortedElements = React.useMemo(
        () => sortAnalysisResults(filteredElements, sortMode, recognizedSubLabels),
        [filteredElements, recognizedSubLabels, sortMode],
    );
    const mistakeFrequencyData = React.useMemo(
        () => buildMistakeFrequencyData(filteredElements, recognizedSubLabels),
        [filteredElements, recognizedSubLabels],
    );
    const mostCommonQueuePlan = React.useMemo(
        () => buildMostCommonMistakeQueuePlan(
            filteredElements,
            recognizedSubLabels,
            mistakeFrequencyData,
        ),
        [filteredElements, mistakeFrequencyData, recognizedSubLabels],
    );
    const graphSubject = mainLabelFilter === 'MSP' ? 'Training' : 'Racing';
    const mistakeFrequencySpec = React.useMemo<GraphSpec>(() => ({
        type: 'bar',
        data: mistakeFrequencyData,
        categoryKey: 'label',
        series: [{ key: 'occurrences', label: 'Occurrences' }],
        orientation: 'horizontal',
        title: `${graphSubject} mistake frequency`,
        xAxisLabel: 'Occurrences',
        showLegend: false,
        colors: ['#00e676'],
        accessibleLabel: `${graphSubject} mistake frequency by recognized taxonomy sub-label`,
        height: getMistakeFrequencyGraphHeight(mistakeFrequencyData.length),
        emptyStateText: `No recognized ${graphSubject.toLowerCase()} mistakes to graph.`,
    }), [graphSubject, mistakeFrequencyData]);

    const clearMountTimeout = React.useCallback(() => {
        if (mountTimeoutRef.current !== null) {
            window.clearTimeout(mountTimeoutRef.current);
            mountTimeoutRef.current = null;
        }
    }, []);

    const finishPendingQueue = React.useCallback((requestId: number) => {
        if (pendingQueueRef.current?.requestId !== requestId) return;
        pendingQueueRef.current = null;
        clearMountTimeout();
        setPendingQueue(null);
        setQueueInProgress(false);
    }, [clearMountTimeout]);

    const failPendingQueue = React.useCallback((requestId: number, message: string) => {
        if (pendingQueueRef.current?.requestId !== requestId) return;
        finishPendingQueue(requestId);
        setQueueStatus(message);
    }, [finishPendingQueue]);

    const drainPendingQueue = React.useCallback((
        prepared: PendingComparisonQueue,
        handle: LiveRangeTodoListHandle,
    ) => {
        if (pendingQueueRef.current?.requestId !== prepared.requestId) return;
        let queuedCount = 0;
        let failedCount = 0;
        prepared.events.forEach((event) => {
            try {
                const result = handle.addEvent(event);
                if (result.status === 'error') failedCount += 1;
                else queuedCount += 1;
            } catch {
                failedCount += 1;
            }
        });
        finishPendingQueue(prepared.requestId);
        const skippedCount = prepared.skippedCount + failedCount;
        setQueueStatus(`Queued: ${queuedCount}. Skipped: ${skippedCount}.`);
    }, [finishPendingQueue]);

    React.useEffect(() => {
        if (!pendingQueue || !liveSession.liveRangeTodoListHandle) return;
        drainPendingQueue(pendingQueue, liveSession.liveRangeTodoListHandle);
    }, [drainPendingQueue, liveSession.liveRangeTodoListHandle, pendingQueue]);

    React.useEffect(() => () => {
        pendingQueueRef.current = null;
        clearMountTimeout();
    }, [clearMountTimeout]);

    React.useEffect(() => {
        if (!pendingQueueRef.current) setQueueStatus('');
    }, [elements, mainLabelFilter]);

    const handleQueueMostCommonMistakes = React.useCallback(() => {
        if (queueInProgress || mostCommonQueuePlan.eligible.length === 0) return;
        const requestId = ++queueRequestSequenceRef.current;
        const prepared: PendingComparisonQueue = {
            requestId,
            events: createMostCommonMistakeEvents(mostCommonQueuePlan),
            skippedCount: mostCommonQueuePlan.skippedCount,
        };
        pendingQueueRef.current = prepared;
        setPendingQueue(prepared);
        setQueueInProgress(true);
        setQueueStatus('Opening Live Range To-do List…');

        mountTimeoutRef.current = window.setTimeout(() => {
            failPendingQueue(
                requestId,
                'Live Range To-do List did not open within two seconds. Nothing was queued.',
            );
        }, LIVE_RANGE_TODO_LIST_MOUNT_TIMEOUT_MS);

        // Even an already-mounted handle drains from the effect above, after
        // the disabled state has rendered. The timeout also covers a handle
        // disappearing while this click is being prepared.
        if (liveSession.liveRangeTodoListHandle) return;

        void import('../VisualizationController').then(({ visualizationController }) => {
            if (pendingQueueRef.current?.requestId !== requestId) return;
            const alreadyOpen = visualizationController.getCurrentInstances()
                .some((instance) => instance.type === LIVE_RANGE_TODO_LIST_TYPE);
            if (alreadyOpen) return;
            const opened = visualizationController.openVisualization(LIVE_RANGE_TODO_LIST_TYPE);
            if (!opened.success) {
                failPendingQueue(
                    requestId,
                    'Unable to open Live Range To-do List. Nothing was queued.',
                );
            }
        }).catch(() => {
            failPendingQueue(
                requestId,
                'Unable to open Live Range To-do List. Nothing was queued.',
            );
        });
    }, [
        failPendingQueue,
        liveSession.liveRangeTodoListHandle,
        mostCommonQueuePlan,
        queueInProgress,
    ]);

    const queueButtonDisabled = queueInProgress || mostCommonQueuePlan.eligible.length === 0;
    const visibleQueueStatus = queueStatus;

    return (
        <Card className={styles.chart} style={{ width, height }}>
            <Flex className={styles.summary} justify="between" align="center" gap="2" wrap="wrap">
                <Flex className={styles.summaryText} align="center" justify="between" gap="2">
                    <Text size="2" weight="medium">Labeled elements</Text>
                    <Text size="2" weight="bold" className={styles.count}>
                        {filteredElements.length} of {elements.length} total
                    </Text>
                </Flex>
                <Flex className={styles.controls} align="center" gap="2" wrap="wrap">
                    <label className={styles.filterControl}>
                        <Text size="1" color="gray" as="span">Showing</Text>
                        <select
                            className={styles.filterSelect}
                            value={mainLabelFilter}
                            onChange={(event) => (
                                setMainLabelFilter(event.target.value as AnalysisResultsMainLabelFilter)
                            )}
                        >
                            {MAIN_LABEL_FILTER_OPTIONS.map((option) => (
                                <option value={option.value} key={option.value}>{option.label}</option>
                            ))}
                        </select>
                    </label>
                    <label className={styles.sortControl}>
                        <Text size="1" color="gray" as="span">Sort by</Text>
                        <select
                            className={styles.sortSelect}
                            value={sortMode}
                            onChange={(event) => setSortMode(event.target.value as AnalysisResultsSortMode)}
                        >
                            <option value="original">Original order</option>
                            <option value="most-frequent-sub-label">{selectedFilter.sortDisplayName}</option>
                            <option value="most-time-lost">Most time lost</option>
                        </select>
                    </label>
                    <Box className={styles.queueAction}>
                        <button
                            type="button"
                            className={styles.queueButton}
                            disabled={queueButtonDisabled}
                            onClick={handleQueueMostCommonMistakes}
                            aria-describedby={visibleQueueStatus ? `${id}-queue-status` : undefined}
                            title={mostCommonQueuePlan.eligible.length === 0
                                ? 'No leading-category result has both a valid position and comparison.'
                                : undefined}
                        >
                            {queueInProgress ? 'Sending…' : 'Send most common mistakes'}
                        </button>
                        {visibleQueueStatus && (
                            <Text
                                id={`${id}-queue-status`}
                                className={styles.queueStatus}
                                size="1"
                                as="span"
                                role="status"
                                aria-live="polite"
                            >
                                {visibleQueueStatus}
                            </Text>
                        )}
                    </Box>
                </Flex>
            </Flex>
            <ScrollArea type="hover" className={styles.list}>
                <Box className={styles.graph}>
                    <DataGraph spec={mistakeFrequencySpec} />
                </Box>
                {filteredElements.length === 0 ? (
                    <Box className={styles.empty} data-testid="analysis-results-empty-state">
                        <Text color="gray">No {selectedFilter.label} results yet.</Text>
                    </Box>
                ) : (
                    <Box className={styles.cards}>
                        {sortedElements.map((element, index) => (
                            <AnalysisResultCard
                                element={element}
                                key={element.id}
                                resultNumber={index + 1}
                                showElementId={showElementId}
                            />
                        ))}
                    </Box>
                )}
            </ScrollArea>
        </Card>
    );
};

export default AnalysisResultsChart;
