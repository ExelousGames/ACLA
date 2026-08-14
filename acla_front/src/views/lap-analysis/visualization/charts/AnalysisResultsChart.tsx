import React from 'react';
import { Badge, Box, Card, Flex, HoverCard, ScrollArea, Text } from '@radix-ui/themes';
import type { VisualizationProps } from '../VisualizationRegistry';
import {
    AnalysisResultElement,
    AnalysisResultMutationResult,
    appendAnalysisResultElement,
    countAnalysisResultMistakes,
    getAnalysisResultMistakeParentLabels,
    normalizeAnalysisResultsData,
    removeAnalysisResultElement,
    updateAnalysisResultElement,
} from './analysisResultsModel';
import { useAiLabels } from 'contexts/AiLabelsContext';
import { DataGraph, GraphRecord, GraphSpec } from 'components/data-graphs';
import {
    DriverExpertComparisonGraph,
    hasComparableDriverExpertData,
} from 'components/driver-expert-comparison';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import styles from './AnalysisResultsChart.module.css';
import {
    AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS,
    NamedAiToolComponentHandle,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import {
    AnalysisResultsVisualizationNotReadyError,
    ComponentDisableFailedError,
    VisualizationComponentError,
    VisualizationControlFailedError,
    VisualizationUpdateFailedError,
} from 'contexts/AiToolComponentError';
import { runVisualizationBooleanCallback } from '../visualization-component-callbacks';
import {
    resolvedAiToolOperation,
    type AiToolOperation,
} from 'components/ai-engineering-tools';
import type {
    AnalysisResultQueryData,
    QueryAnalysisResultArguments,
    QueryAnalysisResultResult,
} from 'views/lap-analysis/ai-chat/ai-command-registry';

const formatPosition = (value: number): string => `${(value * 100).toFixed(1)}%`;

const HIDDEN_METADATA_KEYS = new Set(['source', 'start_index', 'end_index']);

type AnalysisResultsSortMode = 'original' | 'most-frequent-sub-label' | 'most-time-lost';
type AnalysisResultsMainLabelFilter = 'MSP' | 'MSR';

interface MainLabelFilterOption {
    value: AnalysisResultsMainLabelFilter;
    label: string;
    sortDisplayName: string;
}

const MAIN_LABEL_FILTER_OPTIONS: readonly MainLabelFilterOption[] = [
    {
        value: 'MSP',
        label: 'Training Mistake',
        sortDisplayName: 'Most common training mistake',
    },
    {
        value: 'MSR',
        label: 'Racing Mistake',
        sortDisplayName: 'Most common racing mistake',
    },
];

interface AnalysisResultsChartProps extends VisualizationProps {
    showElementId?: boolean;
    sessionGame?: DesktopGame | null;
    pagination?: AnalysisResultsPagination;
}

export interface AnalysisResultsPaginationPage {
    id: string;
    createdAt: number;
    baseline: {
        lap: number;
        lap_time_ms: number | null;
        track: string;
        car: string;
    };
    elements: AnalysisResultElement[];
}

export interface AnalysisResultsPagination {
    pages: readonly AnalysisResultsPaginationPage[];
    activePageId: string | null;
    onSelectPage: (pageId: string) => void;
}

export interface AnalysisResultsChartHandle extends NamedAiToolComponentHandle {
    waitForAnalysisResultPage(pageId: string): Promise<void>;
    queryAnalysisResult<TQuery extends keyof AnalysisResultQueryData>(
        args: QueryAnalysisResultArguments<TQuery>,
    ): AiToolOperation<QueryAnalysisResultResult<TQuery>>;
    replaceAnalysisResults(data: unknown): true;
    appendAnalysisResult(element: unknown): AnalysisResultControlResult;
    updateAnalysisResult(id: unknown, changes: unknown): AnalysisResultControlResult;
    removeAnalysisResult(id: unknown): AnalysisResultControlResult;
    disableAnalysisResults(): true;
}

export type AnalysisResultControlResult = Omit<AnalysisResultMutationResult, 'success'> & {
    success: true;
};

type AnalysisResultPageWaiter = {
    resolve: () => void;
    reject: (error: VisualizationComponentError) => void;
    timeoutId: ReturnType<typeof setTimeout>;
};

const createAnalysisResultsReadinessError = (
    componentName: string,
    pageId: string,
    reason: 'timeout' | 'unmounted',
): AnalysisResultsVisualizationNotReadyError => new AnalysisResultsVisualizationNotReadyError(
    componentName,
    reason === 'timeout'
        ? `Analysis Results page '${pageId}' was not committed within ${AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS}ms.`
        : `Analysis Results unmounted before page '${pageId}' was committed.`,
);

interface IndexedAnalysisResult {
    element: AnalysisResultElement;
    originalIndex: number;
}

export interface RecognizedSubLabel {
    id: string;
    label: string;
}

export type RecognizedSubLabels = ReadonlyMap<string, RecognizedSubLabel>;

export type MistakeTrendDirection = 'increasing' | 'decreasing' | 'stable' | 'insufficient';

export interface MistakeTrendLapSummary {
    pageId: string;
    label: string;
    totalCount: number;
    categoryCounts: ReadonlyMap<string, number>;
}

export interface MistakeTrendCategory {
    id: string;
    label: string;
    occurrences: number;
}

export interface MistakeTrendData {
    laps: MistakeTrendLapSummary[];
    categories: MistakeTrendCategory[];
}

export type LapTimeTrendDirection = 'improving' | 'regressing' | 'stable' | 'insufficient';

export interface LapTimeTrendLapSummary {
    pageId: string;
    label: string;
    lapTimeMs: number | null;
    bestFitLapTimeMs: number | null;
}

export interface LapTimeTrendData {
    laps: LapTimeTrendLapSummary[];
    slopeMsPerAnalysis: number | null;
    direction: LapTimeTrendDirection;
}

const EFFECTIVELY_ZERO_SLOPE = 1e-9;
const INSUFFICIENT_TREND_MESSAGE = 'Not enough analyzed laps to determine a trend.';

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

export const calculateLeastSquaresSlope = (values: readonly number[]): number | null => {
    if (values.length < 2) return null;
    const xMean = (values.length - 1) / 2;
    const yMean = values.reduce((sum, value) => sum + value, 0) / values.length;
    let numerator = 0;
    let denominator = 0;
    values.forEach((value, index) => {
        const xDelta = index - xMean;
        numerator += xDelta * (value - yMean);
        denominator += xDelta * xDelta;
    });
    return denominator === 0 ? null : numerator / denominator;
};

const calculateLeastSquaresRegression = (
    points: readonly { x: number; y: number }[],
): { slope: number; intercept: number } | null => {
    if (points.length < 2) return null;
    const xMean = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const yMean = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    let numerator = 0;
    let denominator = 0;
    points.forEach((point) => {
        const xDelta = point.x - xMean;
        numerator += xDelta * (point.y - yMean);
        denominator += xDelta * xDelta;
    });
    if (denominator === 0) return null;
    const slope = numerator / denominator;
    return { slope, intercept: yMean - slope * xMean };
};

export const getLapTimeTrendDirection = (slopeMsPerAnalysis: number | null): LapTimeTrendDirection => {
    if (slopeMsPerAnalysis === null) return 'insufficient';
    if (Math.abs(slopeMsPerAnalysis) <= EFFECTIVELY_ZERO_SLOPE) return 'stable';
    return slopeMsPerAnalysis < 0 ? 'improving' : 'regressing';
};

export const buildLapTimeTrendData = (
    pages: readonly AnalysisResultsPaginationPage[],
): LapTimeTrendData => {
    const orderedPages = pages
        .map((page, originalIndex) => ({ page, originalIndex }))
        .sort((left, right) => (
            left.page.createdAt - right.page.createdAt
            || left.originalIndex - right.originalIndex
        ));
    const laps = orderedPages.map(({ page }, index) => {
        const parsedLapTime = Number(page.baseline.lap_time_ms);
        return {
            pageId: page.id,
            label: `Analysis ${index + 1} \u00b7 Lap ${page.baseline.lap}`,
            lapTimeMs: Number.isFinite(parsedLapTime) && parsedLapTime > 0 ? parsedLapTime : null,
            bestFitLapTimeMs: null,
        };
    });
    const timedPoints = laps.flatMap((lap, index) => (
        lap.lapTimeMs === null ? [] : [{ x: index, y: lap.lapTimeMs }]
    ));
    const regression = calculateLeastSquaresRegression(timedPoints);
    const firstTimedIndex = timedPoints[0]?.x ?? -1;
    const lastTimedIndex = timedPoints[timedPoints.length - 1]?.x ?? -1;

    return {
        laps: laps.map((lap, index) => ({
            ...lap,
            bestFitLapTimeMs: regression && index >= firstTimedIndex && index <= lastTimedIndex
                ? regression.intercept + regression.slope * index
                : null,
        })),
        slopeMsPerAnalysis: regression?.slope ?? null,
        direction: getLapTimeTrendDirection(regression?.slope ?? null),
    };
};

export const formatRacingTime = (milliseconds: number): string => {
    if (!Number.isFinite(milliseconds) || milliseconds < 0) return '--:--.---';
    const roundedMilliseconds = Math.round(milliseconds);
    const minutes = Math.floor(roundedMilliseconds / 60_000);
    const remainder = roundedMilliseconds % 60_000;
    const seconds = Math.floor(remainder / 1_000);
    const millis = remainder % 1_000;
    return `${minutes}:${String(seconds).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
};

const describeLapTimeDelta = (latestMs: number, referenceMs: number): string => {
    const deltaMs = latestMs - referenceMs;
    if (Math.abs(deltaMs) <= EFFECTIVELY_ZERO_SLOPE) return 'unchanged';
    return `${formatRacingTime(Math.abs(deltaMs))} ${deltaMs < 0 ? 'faster' : 'slower'}`;
};

export const describeLapTimeTrend = (trend: LapTimeTrendData): string => {
    const timedLaps = trend.laps.filter((lap): lap is LapTimeTrendLapSummary & { lapTimeMs: number } => (
        lap.lapTimeMs !== null
    ));
    if (timedLaps.length === 0) {
        return 'Latest lap time unavailable. Versus previous timed lap: unavailable. '
            + 'Versus first timed lap: unavailable. Overall direction: not enough timed laps.';
    }

    const latest = timedLaps[timedLaps.length - 1];
    const previous = timedLaps[timedLaps.length - 2];
    const first = timedLaps[0];
    const previousDelta = previous ? describeLapTimeDelta(latest.lapTimeMs, previous.lapTimeMs) : 'unavailable';
    const firstDelta = describeLapTimeDelta(latest.lapTimeMs, first.lapTimeMs);
    const direction = trend.direction === 'insufficient'
        ? 'not enough timed laps'
        : trend.direction;
    return `Latest lap time: ${formatRacingTime(latest.lapTimeMs)}. `
        + `Versus previous timed lap: ${previousDelta}. `
        + `Versus first timed lap: ${firstDelta}. Overall direction: ${direction}.`;
};

export const getMistakeTrendDirection = (
    values: readonly number[],
): MistakeTrendDirection => {
    const slope = calculateLeastSquaresSlope(values);
    if (slope === null) return 'insufficient';
    if (Math.abs(slope) <= EFFECTIVELY_ZERO_SLOPE) return 'stable';
    return slope < 0 ? 'decreasing' : 'increasing';
};

export const buildMistakeTrendData = (
    pages: readonly AnalysisResultsPaginationPage[],
    recognizedParentLabels: ReadonlySet<string>,
    recognizedSubLabels: RecognizedSubLabels,
): MistakeTrendData => {
    const categoryTotals = new Map<string, MistakeTrendCategory>();
    const orderedPages = pages
        .map((page, originalIndex) => ({ page, originalIndex }))
        .sort((left, right) => (
            left.page.createdAt - right.page.createdAt
            || left.originalIndex - right.originalIndex
        ));
    const laps = orderedPages.map(({ page }, index) => {
        const filteredElements = page.elements.filter((element) => (
            element.labels.some((label) => recognizedParentLabels.has(label))
        ));
        const categoryCounts = new Map<string, number>();
        filteredElements.forEach((element) => {
            getSubLabels(element, recognizedSubLabels).forEach((subLabel) => {
                categoryCounts.set(subLabel.id, (categoryCounts.get(subLabel.id) ?? 0) + 1);
                const aggregate = categoryTotals.get(subLabel.id);
                categoryTotals.set(subLabel.id, {
                    id: subLabel.id,
                    label: subLabel.label,
                    occurrences: (aggregate?.occurrences ?? 0) + 1,
                });
            });
        });
        return {
            pageId: page.id,
            label: `Analysis ${index + 1} · Lap ${page.baseline.lap}`,
            totalCount: filteredElements.length,
            categoryCounts,
        };
    });

    return {
        laps,
        categories: Array.from(categoryTotals.values()).sort((left, right) => (
            compareLabelText(left.label, right.label)
        )),
    };
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
    sessionGame?: DesktopGame | null;
}> = ({ element, resultNumber, showElementId, sessionGame }) => {
    const [comparisonOpen, setComparisonOpen] = React.useState(false);
    const metadataEntries = Object.entries(element.metadata ?? {})
        .filter(([key]) => !HIDDEN_METADATA_KEYS.has(key));
    const hasComparison = hasComparableDriverExpertData(element.comparison, sessionGame ?? null);

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
                        game={sessionGame}
                        title={element.title
                            ? `${element.title}: Driver vs Expert`
                            : 'Driver vs Expert'}
                        layout={{ trajectoryHeight: 300 }}
                    />
                )}
            </HoverCard.Content>
        </HoverCard.Root>
    );
};

const describeTrendDirection = (direction: MistakeTrendDirection): string => {
    if (direction === 'decreasing') return 'Trending downward — fewer mistakes.';
    if (direction === 'increasing') return 'Trending upward — more mistakes.';
    if (direction === 'stable') return 'Trend is stable.';
    return INSUFFICIENT_TREND_MESSAGE;
};

const describeLatestTrendCount = (
    values: readonly number[],
    singular: string,
    plural: string,
): string => {
    if (values.length === 0) return `Latest count unavailable. ${INSUFFICIENT_TREND_MESSAGE}`;
    const latest = values[values.length - 1];
    const countLabel = latest === 1 ? singular : plural;
    return `Latest: ${latest} ${countLabel}. ${describeTrendDirection(getMistakeTrendDirection(values))}`;
};

interface OverallMistakeTrendProps {
    id: string;
    lapTimeTrendData: LapTimeTrendData;
    mainLabelFilter: AnalysisResultsMainLabelFilter;
    onMainLabelFilterChange: (filter: AnalysisResultsMainLabelFilter) => void;
    selectedFilter: MainLabelFilterOption;
    graphSubject: string;
    trendData: MistakeTrendData;
    selectedCategory: MistakeTrendCategory | null;
    onSelectedCategoryChange: (categoryId: string) => void;
}

const OverallMistakeTrend: React.FC<OverallMistakeTrendProps> = ({
    id,
    lapTimeTrendData,
    mainLabelFilter,
    onMainLabelFilterChange,
    selectedFilter,
    graphSubject,
    trendData,
    selectedCategory,
    onSelectedCategoryChange,
}) => {
    const totalCounts = trendData.laps.map(({ totalCount }) => totalCount);
    const specificCounts = selectedCategory
        ? trendData.laps.map(({ categoryCounts }) => categoryCounts.get(selectedCategory.id) ?? 0)
        : [];
    const lapTimeTrendSpec = React.useMemo<GraphSpec>(() => ({
        type: 'line',
        data: lapTimeTrendData.laps.map((lap) => ({
            analysis: lap.label,
            lapTimeSeconds: lap.lapTimeMs === null ? null : lap.lapTimeMs / 1_000,
            bestFitSeconds: lap.bestFitLapTimeMs === null ? null : lap.bestFitLapTimeMs / 1_000,
        })),
        xKey: 'analysis',
        xAxisType: 'category',
        series: [
            { key: 'lapTimeSeconds', label: 'Actual lap time' },
            { key: 'bestFitSeconds', label: 'Best-fit trend' },
        ],
        title: 'Lap time by analyzed lap (lower is faster).',
        yAxisLabel: 'Lap time (seconds)',
        showLegend: true,
        showPoints: true,
        colors: ['#00e676', '#62a8ff'],
        accessibleLabel: 'Actual lap time and best-fit improvement trend by analyzed lap',
        height: 260,
        emptyStateText: 'No valid lap timing is available yet.',
    }), [lapTimeTrendData.laps]);
    const totalTrendSpec = React.useMemo<GraphSpec>(() => ({
        type: 'line',
        data: trendData.laps.map((lap) => ({
            analysis: lap.label,
            totalCount: lap.totalCount,
        })),
        xKey: 'analysis',
        xAxisType: 'category',
        series: [{ key: 'totalCount', label: 'Recognized mistake elements' }],
        title: `Total ${graphSubject.toLowerCase()} mistakes by analyzed lap`,
        yAxisLabel: 'Recognized mistake elements',
        showLegend: false,
        showPoints: true,
        colors: ['#00e676'],
        accessibleLabel: `Total recognized ${graphSubject.toLowerCase()} mistake elements per analyzed lap`,
        height: 230,
        emptyStateText: 'No analyzed laps are available yet.',
    }), [graphSubject, trendData.laps]);
    const specificTrendSpec = React.useMemo<GraphSpec>(() => ({
        type: 'line',
        data: selectedCategory ? trendData.laps.map((lap) => ({
            analysis: lap.label,
            specificCount: lap.categoryCounts.get(selectedCategory.id) ?? 0,
        })) : [],
        xKey: 'analysis',
        xAxisType: 'category',
        series: [{ key: 'specificCount', label: 'Occurrences' }],
        title: selectedCategory
            ? `${selectedCategory.label} occurrences by analyzed lap`
            : `Specific ${graphSubject.toLowerCase()} mistake by analyzed lap`,
        yAxisLabel: 'Occurrences',
        showLegend: false,
        showPoints: true,
        colors: ['#62a8ff'],
        accessibleLabel: selectedCategory
            ? `${selectedCategory.label} occurrences per analyzed lap`
            : `Specific ${graphSubject.toLowerCase()} mistake occurrences per analyzed lap`,
        height: 230,
        emptyStateText: `No recognized ${graphSubject.toLowerCase()} mistake categories have been observed.`,
    }), [graphSubject, selectedCategory, trendData.laps]);

    const guidance = trendData.laps.length === 0
        ? 'No analyzed laps yet. Analyze at least two baseline laps to see a trend.'
        : trendData.laps.length === 1
            ? INSUFFICIENT_TREND_MESSAGE
            : null;

    return (
        <ScrollArea type="hover" className={styles.trendContent}>
            <Box className={styles.trendHeading}>
                <Text size="3" weight="bold" as="div">Overall Trends</Text>
                <Text size="1" color="gray" as="div">
                    Trends follow analysis creation order across every retained baseline page.
                </Text>
            </Box>
            <Box className={styles.trendChartSection}>
                <Text size="3" weight="bold" as="div">Lap Time Improvement</Text>
                <Text
                    className={styles.trendStatus}
                    data-testid="lap-time-trend-status"
                    size="2"
                    weight="medium"
                    as="div"
                    role="status"
                >
                    {describeLapTimeTrend(lapTimeTrendData)}
                </Text>
                <DataGraph spec={lapTimeTrendSpec} />
            </Box>
            <Flex className={styles.mistakeTrendHeader} justify="between" align="center" gap="2" wrap="wrap">
                <Box className={styles.trendHeading}>
                    <Text size="3" weight="bold" as="div">Overall Mistake Trend</Text>
                    <Text size="1" color="gray" as="div">
                        Counts follow analysis creation order across every retained baseline page.
                    </Text>
                </Box>
                <label className={styles.filterControl}>
                    <Text size="1" color="gray" as="span">Showing</Text>
                    <select
                        className={styles.filterSelect}
                        value={mainLabelFilter}
                        onChange={(event) => onMainLabelFilterChange(
                            event.target.value as AnalysisResultsMainLabelFilter,
                        )}
                    >
                        {MAIN_LABEL_FILTER_OPTIONS.map((option) => (
                            <option value={option.value} key={option.value}>{option.label}</option>
                        ))}
                    </select>
                </label>
            </Flex>
            {guidance && (
                <Text
                    className={styles.trendGuidance}
                    data-testid="overall-trend-guidance"
                    size="2"
                    color="gray"
                    as="div"
                >
                    {guidance}
                </Text>
            )}
            <Box className={styles.trendChartSection}>
                <Text
                    className={styles.trendStatus}
                    data-testid="overall-total-trend-status"
                    size="2"
                    weight="medium"
                    as="div"
                    role="status"
                >
                    {describeLatestTrendCount(
                        totalCounts,
                        'recognized mistake element',
                        'recognized mistake elements',
                    )}
                </Text>
                <DataGraph spec={totalTrendSpec} />
            </Box>
            <Box className={styles.trendChartSection}>
                <Flex className={styles.specificTrendHeader} justify="between" align="center" gap="2" wrap="wrap">
                    <Text
                        className={styles.trendStatus}
                        data-testid="specific-mistake-trend-status"
                        size="2"
                        weight="medium"
                        as="div"
                        role="status"
                    >
                        {selectedCategory
                            ? `${selectedCategory.label}. ${describeLatestTrendCount(
                                specificCounts,
                                'occurrence',
                                'occurrences',
                            )}`
                            : `No recognized ${selectedFilter.label.toLowerCase()} sub-labels have been observed yet.`}
                    </Text>
                    <label className={styles.specificMistakeControl}>
                        <Text size="1" color="gray" as="span">Specific mistake</Text>
                        <select
                            id={`${id}-specific-mistake`}
                            className={styles.specificMistakeSelect}
                            value={selectedCategory?.id ?? ''}
                            disabled={trendData.categories.length === 0}
                            onChange={(event) => onSelectedCategoryChange(event.target.value)}
                        >
                            {trendData.categories.length === 0 && (
                                <option value="">No observed categories</option>
                            )}
                            {trendData.categories.map((category) => (
                                <option value={category.id} key={category.id}>{category.label}</option>
                            ))}
                        </select>
                    </label>
                </Flex>
                <DataGraph spec={specificTrendSpec} />
            </Box>
        </ScrollArea>
    );
};

const AnalysisResultsChart = React.forwardRef<AnalysisResultsChartHandle, AnalysisResultsChartProps>(({
    name,
    id,
    data,
    width = '100%',
    height = '100%',
    showElementId = true,
    sessionGame,
    pagination,
    onUpdate,
    onDisable,
}, forwardedRef) => {
    const [sortMode, setSortMode] = React.useState<AnalysisResultsSortMode>('original');
    const [mainLabelFilter, setMainLabelFilter] = React.useState<AnalysisResultsMainLabelFilter>('MSP');
    const [showOverallTrend, setShowOverallTrend] = React.useState(true);
    const [selectedTrendSubLabelId, setSelectedTrendSubLabelId] = React.useState<string | null>(null);
    const committedPageIdsRef = React.useRef<Set<string>>(new Set());
    const pendingPageWaitersRef = React.useRef<Map<string, Set<AnalysisResultPageWaiter>>>(new Map());
    const mountedRef = React.useRef(false);
    const { getCategoryLabels, getLabelName } = useAiLabels();
    const chronologicalPages = React.useMemo(() => {
        if (!pagination) return [];
        return pagination.pages
            .map((page, originalIndex) => ({ page, originalIndex }))
            .sort((left, right) => (
                left.page.createdAt - right.page.createdAt
                || left.originalIndex - right.originalIndex
            ))
            .map(({ page }) => page);
    }, [pagination]);
    const activePageIndex = React.useMemo(() => {
        if (!pagination || chronologicalPages.length === 0) return -1;
        const selectedIndex = chronologicalPages.findIndex((page) => page.id === pagination.activePageId);
        return selectedIndex >= 0 ? selectedIndex : 0;
    }, [chronologicalPages, pagination]);
    const activePage = activePageIndex >= 0 ? chronologicalPages[activePageIndex] : null;
    const isOverallTrend = Boolean(pagination) && (showOverallTrend || !activePage);
    const activeData = activePage ?? data;
    const { elements } = React.useMemo(() => normalizeAnalysisResultsData(activeData), [activeData]);
    const analysisResultQueryData = React.useMemo<AnalysisResultQueryData>(() => ({
        result_count: elements.length,
        mistake_count: countAnalysisResultMistakes(elements, getLabelName),
    }), [elements, getLabelName]);
    const waitForAnalysisResultPage = React.useCallback((pageId: string): Promise<void> => {
        if (committedPageIdsRef.current.has(pageId)) return Promise.resolve();
        if (!mountedRef.current) {
            return Promise.reject(createAnalysisResultsReadinessError(name, pageId, 'unmounted'));
        }

        return new Promise((resolve, reject) => {
            const waiter: AnalysisResultPageWaiter = {
                resolve,
                reject,
                timeoutId: setTimeout(() => {
                    const pendingForPage = pendingPageWaitersRef.current.get(pageId);
                    pendingForPage?.delete(waiter);
                    if (pendingForPage?.size === 0) pendingPageWaitersRef.current.delete(pageId);
                    reject(createAnalysisResultsReadinessError(name, pageId, 'timeout'));
                }, AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS),
            };
            const pendingForPage = pendingPageWaitersRef.current.get(pageId)
                ?? new Set<AnalysisResultPageWaiter>();
            pendingForPage.add(waiter);
            pendingPageWaitersRef.current.set(pageId, pendingForPage);
        });
    }, [name]);

    React.useLayoutEffect(() => {
        const pendingPageWaiters = pendingPageWaitersRef.current;
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            pendingPageWaiters.forEach((waiters, pageId) => {
                waiters.forEach((waiter) => {
                    clearTimeout(waiter.timeoutId);
                    waiter.reject(createAnalysisResultsReadinessError(name, pageId, 'unmounted'));
                });
            });
            pendingPageWaiters.clear();
            committedPageIdsRef.current.clear();
        };
    }, [name]);

    React.useLayoutEffect(() => {
        const committedPageIds = new Set(pagination?.pages.map((page) => page.id) ?? []);
        committedPageIdsRef.current = committedPageIds;
        committedPageIds.forEach((pageId) => {
            const waiters = pendingPageWaitersRef.current.get(pageId);
            if (!waiters) return;
            waiters.forEach((waiter) => {
                clearTimeout(waiter.timeoutId);
                waiter.resolve();
            });
            pendingPageWaitersRef.current.delete(pageId);
        });
    }, [pagination?.pages]);

    const handle = React.useMemo<AnalysisResultsChartHandle>(() => ({
        getComponentName: () => name,
        waitForAnalysisResultPage,
        queryAnalysisResult: ({ query }) => resolvedAiToolOperation({
            status: 'ready',
            data: analysisResultQueryData[query],
        }),
        replaceAnalysisResults: (nextData) => runVisualizationBooleanCallback(
            name,
            VisualizationUpdateFailedError,
            `Failed to update chart '${name}'.`,
            onUpdate ? () => onUpdate(nextData) : undefined,
        ),
        appendAnalysisResult: (element) => {
            const mutation = appendAnalysisResultElement(activeData, element);
            if (!mutation.result.success) {
                throw new VisualizationControlFailedError(
                    name,
                    mutation.result.message,
                );
            }
            runVisualizationBooleanCallback(
                name,
                VisualizationControlFailedError,
                mutation.result.message,
                onUpdate ? () => onUpdate(mutation.data) : undefined,
            );
            return { ...mutation.result, success: true as const };
        },
        updateAnalysisResult: (elementId, changes) => {
            const mutation = updateAnalysisResultElement(activeData, elementId, changes);
            if (!mutation.result.success) {
                throw new VisualizationControlFailedError(
                    name,
                    mutation.result.message,
                );
            }
            runVisualizationBooleanCallback(
                name,
                VisualizationControlFailedError,
                mutation.result.message,
                onUpdate ? () => onUpdate(mutation.data) : undefined,
            );
            return { ...mutation.result, success: true as const };
        },
        removeAnalysisResult: (elementId) => {
            const mutation = removeAnalysisResultElement(activeData, elementId);
            if (!mutation.result.success) {
                throw new VisualizationControlFailedError(
                    name,
                    mutation.result.message,
                );
            }
            runVisualizationBooleanCallback(
                name,
                VisualizationControlFailedError,
                mutation.result.message,
                onUpdate ? () => onUpdate(mutation.data) : undefined,
            );
            return { ...mutation.result, success: true as const };
        },
        disableAnalysisResults: () => runVisualizationBooleanCallback(
            name,
            ComponentDisableFailedError,
            `Component '${name}' could not be disabled.`,
            onDisable,
        ),
    }), [activeData, analysisResultQueryData, name, onDisable, onUpdate, waitForAnalysisResultPage]);
    React.useImperativeHandle(forwardedRef, () => handle, [handle]);
    useRegisterAiToolComponentRef(name, handle);
    const selectedFilter = MAIN_LABEL_FILTER_OPTIONS.find(({ value }) => value === mainLabelFilter)!;
    const recognizedParentLabels = React.useMemo(() => (
        getAnalysisResultMistakeParentLabels(selectedFilter.value, getLabelName)
    ), [getLabelName, selectedFilter]);
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
    const trendData = React.useMemo(() => buildMistakeTrendData(
        chronologicalPages,
        recognizedParentLabels,
        recognizedSubLabels,
    ), [chronologicalPages, recognizedParentLabels, recognizedSubLabels]);
    const lapTimeTrendData = React.useMemo(
        () => buildLapTimeTrendData(chronologicalPages),
        [chronologicalPages],
    );
    const selectedTrendCategory = React.useMemo(() => {
        const retained = trendData.categories.find(({ id: categoryId }) => (
            categoryId === selectedTrendSubLabelId
        ));
        if (retained) return retained;
        return [...trendData.categories].sort((left, right) => (
            right.occurrences - left.occurrences
            || compareLabelText(left.label, right.label)
        ))[0] ?? null;
    }, [selectedTrendSubLabelId, trendData.categories]);
    React.useEffect(() => {
        const nextId = selectedTrendCategory?.id ?? null;
        if (selectedTrendSubLabelId !== nextId) setSelectedTrendSubLabelId(nextId);
    }, [selectedTrendCategory, selectedTrendSubLabelId]);
    const sortedElements = React.useMemo(
        () => sortAnalysisResults(filteredElements, sortMode, recognizedSubLabels),
        [filteredElements, recognizedSubLabels, sortMode],
    );
    const mistakeFrequencyData = React.useMemo(
        () => buildMistakeFrequencyData(filteredElements, recognizedSubLabels),
        [filteredElements, recognizedSubLabels],
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

    const displayedPageIndex = isOverallTrend ? 0 : activePageIndex + 1;
    const displayedPageCount = chronologicalPages.length + 1;

    const handlePreviousPage = () => {
        if (!pagination || isOverallTrend) return;
        if (activePageIndex <= 0) {
            setShowOverallTrend(true);
            return;
        }
        pagination.onSelectPage(chronologicalPages[activePageIndex - 1].id);
    };

    const handleNextPage = () => {
        if (!pagination) return;
        if (isOverallTrend) {
            const firstPage = chronologicalPages[0];
            if (!firstPage) return;
            pagination.onSelectPage(firstPage.id);
            setShowOverallTrend(false);
            return;
        }
        const nextPage = chronologicalPages[activePageIndex + 1];
        if (nextPage) pagination.onSelectPage(nextPage.id);
    };

    return (
        <Card className={styles.chart} style={{ width, height }}>
            {pagination && (
                <Flex className={styles.pagination} justify="between" align="center" gap="2" wrap="wrap">
                    <Flex className={styles.pageNavigation} align="center" gap="2">
                        <button
                            type="button"
                            className={styles.pageButton}
                            disabled={isOverallTrend}
                            onClick={handlePreviousPage}
                        >
                            Previous
                        </button>
                        <Text
                            className={styles.pageCounter}
                            size="1"
                            as="span"
                            role="status"
                            aria-live="polite"
                        >
                            Page {displayedPageIndex + 1} of {displayedPageCount}
                        </Text>
                        <button
                            type="button"
                            className={styles.pageButton}
                            disabled={displayedPageIndex === displayedPageCount - 1}
                            onClick={handleNextPage}
                        >
                            Next
                        </button>
                    </Flex>
                    <Text className={styles.baselineContext} size="1" color="gray" as="span">
                        {isOverallTrend
                            ? `Overall Trends · ${chronologicalPages.length} analyzed ${chronologicalPages.length === 1 ? 'lap' : 'laps'}`
                            : <>Baseline: {activePage!.baseline.track || 'Unknown track'} ·{' '}
                                {activePage!.baseline.car || 'Unknown car'} · Lap {activePage!.baseline.lap}</>}
                    </Text>
                </Flex>
            )}
            {isOverallTrend ? (
                <OverallMistakeTrend
                    id={id}
                    lapTimeTrendData={lapTimeTrendData}
                    mainLabelFilter={mainLabelFilter}
                    onMainLabelFilterChange={setMainLabelFilter}
                    selectedFilter={selectedFilter}
                    graphSubject={graphSubject}
                    trendData={trendData}
                    selectedCategory={selectedTrendCategory}
                    onSelectedCategoryChange={setSelectedTrendSubLabelId}
                />
            ) : <>
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
                                sessionGame={sessionGame}
                            />
                        ))}
                    </Box>
                )}
                </ScrollArea>
            </>}
        </Card>
    );
});

AnalysisResultsChart.displayName = 'AnalysisResultsChart';

export default AnalysisResultsChart;
