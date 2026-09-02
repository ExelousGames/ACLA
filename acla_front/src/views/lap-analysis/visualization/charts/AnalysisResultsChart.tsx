import React from 'react';
import { Badge, Box, Card, Flex, HoverCard, ScrollArea, Text } from '@radix-ui/themes';
import type { VisualizationProps } from '../VisualizationRegistry';
import {
    AnalysisResultElement,
    AnalysisResultMutationResult,
    appendAnalysisResultElement,
    normalizeAnalysisResultsData,
    removeAnalysisResultElement,
    updateAnalysisResultElement,
} from './analysisResultsModel';
import { useAiLabels } from 'contexts/AiLabelsContext';
import { DataGraph, GraphRecord, GraphSpec } from 'components/data-graphs';
import {
    DriverExpertComparisonGraph,
    getDriverExpertComparisonUnavailableDiagnostics,
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
    createAiToolOperationFrom,
    type AiToolOperation,
} from 'components/ai-engineering-tools';
import {
    buildActivePageQueryTemplates,
    buildOverallTrendQueryExpression,
    evaluateAllAnalysisResultsQuery,
    evaluateAnalysisResultsQuery,
    normalizeOverallTrendQueryInput,
    resolveActivePageQueryResult,
    resolveOverallTrendQueryResult,
    toAnalysisResultsQueryError,
    AnalysisResultsQueryError,
    type ActivePageQueryTemplate,
    type ActivePageQueryTemplateKey,
    type ApplyAnalysisResultQueryInput,
    type ApplyAnalysisResultQueryOutput,
    type AnalysisResultsQueryErrorDetail,
    type OverallTrendQueryCategoryResult,
    type OverallTrendQueryLapResult,
    type OverallTrendQueryResult,
    type OverallTrendQueryTaxonomy,
    type QueryAnalysisResultInput,
    type QueryAnalysisResultOutput,
} from './analysisResultsQuery';

const formatPosition = (value: number): string => `${(value * 100).toFixed(1)}%`;

const HIDDEN_METADATA_KEYS = new Set(['source', 'start_index', 'end_index']);

export type MistakeTrendParent = 'MSP' | 'MSR';

interface TrendParentOption {
    value: MistakeTrendParent;
    label: string;
    fallbackName: string;
}

export const TREND_PARENT_OPTIONS: readonly TrendParentOption[] = [
    {
        value: 'MSP',
        label: 'Training Mistake',
        fallbackName: 'Mistake (Practice)',
    },
    {
        value: 'MSR',
        label: 'Racing Mistake',
        fallbackName: 'Mistake (Racing)',
    },
];

export type ActivePageQueryViewKey = ActivePageQueryTemplateKey | 'custom';

const CUSTOM_ACTIVE_PAGE_VIEW = Object.freeze({
    key: 'custom' as const,
    label: 'Custom',
});

interface AnalysisResultsChartProps extends VisualizationProps {
    showElementId?: boolean;
    sessionGame?: DesktopGame | null;
    pagination?: AnalysisResultsPagination;
}

export interface AnalysisResultsPaginationPage {
    id: string;
    createdAt: number;
    baseline: {
        lap_id: number;
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

const EMPTY_ANALYSIS_RESULTS_PAGES: readonly AnalysisResultsPaginationPage[] = Object.freeze([]);
const EMPTY_ANALYSIS_RESULT_ELEMENTS: AnalysisResultElement[] = [];

export interface FilteredAnalysisSegmentsSnapshot {
    readonly status: 'ready' | 'empty' | 'busy';
    readonly activePageId: string | null;
    readonly appliedView: ActivePageQueryViewKey | null;
    readonly committedQuery: string | null;
    readonly segments: readonly AnalysisResultElement[];
}

const cloneAndFreeze = <T,>(value: T): T => {
    if (Array.isArray(value)) {
        return Object.freeze(value.map((entry) => cloneAndFreeze(entry))) as T;
    }
    if (value && typeof value === 'object') {
        const clone = Object.fromEntries(Object.entries(value).map(([key, entry]) => (
            [key, cloneAndFreeze(entry)]
        )));
        return Object.freeze(clone) as T;
    }
    return value;
};

export interface AnalysisResultsChartHandle extends NamedAiToolComponentHandle {
    waitForAnalysisResultPage(pageId: string): Promise<void>;
    getFilteredSegments(): FilteredAnalysisSegmentsSnapshot;
    applyAnalysisResultQuery(
        args: ApplyAnalysisResultQueryInput,
    ): AiToolOperation<ApplyAnalysisResultQueryOutput>;
    queryAnalysisResult(
        args: QueryAnalysisResultInput,
    ): AiToolOperation<QueryAnalysisResultOutput>;
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

type AnalysisResultSelectionWaiter = AnalysisResultPageWaiter & {
    pageId: string | null;
    operationGeneration: number;
};

type RenderedAnalysisResultSelection = {
    pageId: string | null;
    isLapResults: boolean;
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

const createAnalysisResultsApplyError = (
    componentName: string,
    message: string,
): VisualizationControlFailedError => new VisualizationControlFailedError(componentName, message);

export interface RecognizedSubLabel {
    id: string;
    label: string;
}

export type RecognizedSubLabels = ReadonlyMap<string, RecognizedSubLabel>;

export type MistakeTrendDirection = 'increasing' | 'decreasing' | 'stable' | 'insufficient';

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
    rows: readonly OverallTrendQueryLapResult[],
): LapTimeTrendData => {
    const laps = rows.map((row) => {
        const parsedLapTime = Number(row.lapTimeMs);
        return {
            pageId: row.pageId,
            label: row.label,
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

export const buildLabelFrequencyData = (
    elements: readonly AnalysisResultElement[],
    recognizedSubLabels?: ReadonlyMap<string, RecognizedSubLabel>,
): GraphRecord[] => {
    const frequencies = new Map<string, { label: string; occurrences: number }>();
    elements.forEach((element) => {
        if (recognizedSubLabels) {
            getSubLabels(element, recognizedSubLabels).forEach((subLabel) => {
                const current = frequencies.get(subLabel.id);
                frequencies.set(subLabel.id, {
                    label: subLabel.label,
                    occurrences: (current?.occurrences ?? 0) + 1,
                });
            });
            return;
        }

        element.labels.forEach((label) => {
            const current = frequencies.get(label);
            frequencies.set(label, {
                label,
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

export const getLabelFrequencyGraphHeight = (categoryCount: number): number => (
    160 + (Math.max(1, categoryCount) * 36)
);

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
    const comparisonWarningFingerprintRef = React.useRef<string | null>(null);
    const metadataEntries = Object.entries(element.metadata ?? {})
        .filter(([key]) => !HIDDEN_METADATA_KEYS.has(key));
    const hasComparison = hasComparableDriverExpertData(element.comparison, sessionGame ?? null);
    const comparisonDiagnostics = React.useMemo(() => {
        if (hasComparison) return [];
        const diagnostics = [
            ...(element.comparisonDiagnostics ?? []),
            ...getDriverExpertComparisonUnavailableDiagnostics(
                element.comparison,
                sessionGame ?? null,
            ),
        ];
        const diagnosticsByCode = new Map<string, typeof diagnostics[number]>();
        diagnostics.forEach((diagnostic) => diagnosticsByCode.set(diagnostic.code, diagnostic));
        return Array.from(diagnosticsByCode.values());
    }, [element.comparison, element.comparisonDiagnostics, hasComparison, sessionGame]);
    const comparisonWarningFingerprint = React.useMemo(() => JSON.stringify({
        segment_id: element.id,
        game: sessionGame ?? null,
        reason_codes: comparisonDiagnostics.map((diagnostic) => diagnostic.code),
    }), [comparisonDiagnostics, element.id, sessionGame]);

    React.useEffect(() => {
        if (hasComparison) {
            comparisonWarningFingerprintRef.current = null;
            return;
        }
        const isClassifierResult = element.metadata?.source === 'ai_classifier';
        if (!isClassifierResult && !element.comparisonDiagnostics?.length) return;
        if (comparisonWarningFingerprintRef.current === comparisonWarningFingerprint) return;
        comparisonWarningFingerprintRef.current = comparisonWarningFingerprint;
        console.warn('[driver-expert-comparison] Expert comparison unavailable.', {
            segment_id: element.id,
            section: element.section ?? null,
            game: sessionGame ?? null,
            reason_codes: comparisonDiagnostics.map((diagnostic) => diagnostic.code),
            reasons: comparisonDiagnostics,
        });
    }, [
        comparisonDiagnostics,
        comparisonWarningFingerprint,
        element.comparisonDiagnostics,
        element.id,
        element.metadata?.source,
        element.section,
        hasComparison,
        sessionGame,
    ]);

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
    analyzedLapCount: number;
    lapTimeTrendData: LapTimeTrendData;
    trendParent: MistakeTrendParent;
    onTrendParentChange: (parent: MistakeTrendParent) => void;
    selectedParent: TrendParentOption;
    trendSubject: string;
    trendData: OverallTrendQueryResult;
    selectedCategory: OverallTrendQueryCategoryResult | null;
    onSelectedCategoryChange: (categoryId: string) => void;
    error: AnalysisResultsQueryErrorDetail | null;
    isEvaluating: boolean;
}

const OverallMistakeTrend: React.FC<OverallMistakeTrendProps> = ({
    id,
    analyzedLapCount,
    lapTimeTrendData,
    trendParent,
    onTrendParentChange,
    selectedParent,
    trendSubject,
    trendData,
    selectedCategory,
    onSelectedCategoryChange,
    error,
    isEvaluating,
}) => {
    const totalCounts = trendData.laps.map(({ totalCount }) => totalCount);
    const specificCounts = selectedCategory
        ? trendData.laps.map(({ categoryCounts }) => (
            categoryCounts.find(({ id: categoryId }) => categoryId === selectedCategory.id)?.count ?? 0
        ))
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
        title: `Total ${trendSubject.toLowerCase()} mistakes by analyzed lap`,
        yAxisLabel: 'Recognized mistake elements',
        showLegend: false,
        showPoints: true,
        colors: ['#00e676'],
        accessibleLabel: `Total recognized ${trendSubject.toLowerCase()} mistake elements per analyzed lap`,
        height: 230,
        emptyStateText: 'No analyzed laps are available yet.',
    }), [trendData.laps, trendSubject]);
    const specificTrendSpec = React.useMemo<GraphSpec>(() => ({
        type: 'line',
        data: selectedCategory ? trendData.laps.map((lap) => ({
            analysis: lap.label,
            specificCount: lap.categoryCounts.find(({ id: categoryId }) => (
                categoryId === selectedCategory.id
            ))?.count ?? 0,
        })) : [],
        xKey: 'analysis',
        xAxisType: 'category',
        series: [{ key: 'specificCount', label: 'Occurrences' }],
        title: selectedCategory
            ? `${selectedCategory.label} occurrences by analyzed lap`
            : `Specific ${trendSubject.toLowerCase()} mistake by analyzed lap`,
        yAxisLabel: 'Occurrences',
        showLegend: false,
        showPoints: true,
        colors: ['#62a8ff'],
        accessibleLabel: selectedCategory
            ? `${selectedCategory.label} occurrences per analyzed lap`
            : `Specific ${trendSubject.toLowerCase()} mistake occurrences per analyzed lap`,
        height: 230,
        emptyStateText: `No recognized ${trendSubject.toLowerCase()} mistake categories have been observed.`,
    }), [selectedCategory, trendData.laps, trendSubject]);

    const guidance = analyzedLapCount === 0
        ? 'No analyzed laps yet. Analyze at least two baseline laps to see a trend.'
        : analyzedLapCount === 1
            ? INSUFFICIENT_TREND_MESSAGE
            : null;

    return (
        <ScrollArea
            type="hover"
            className={styles.trendContent}
            aria-busy={isEvaluating}
        >
            <Box className={styles.trendHeading}>
                <Text size="3" weight="bold" as="div">Overall Trends</Text>
                <Text size="1" color="gray" as="div">
                Trends follow retained-page array order across every baseline page.
                </Text>
            </Box>
            {error && (
                <Text
                    className={styles.trendError}
                    data-testid="overall-trend-query-error"
                    size="2"
                    as="div"
                    role="alert"
                >
                    Overall Trends query failed ({error.code}
                    {error.position === undefined ? '' : ` at position ${error.position}`}
                    {error.token === undefined ? '' : ` near ${error.token}`}): {error.message}
                </Text>
            )}
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
                    Counts follow retained-page array order across every baseline page.
                    </Text>
                </Box>
                <label className={styles.trendParentControl}>
                    <Text size="1" color="gray" as="span">Showing</Text>
                    <select
                        className={styles.trendParentSelect}
                        value={trendParent}
                        onChange={(event) => onTrendParentChange(
                            event.target.value as MistakeTrendParent,
                        )}
                    >
                        {TREND_PARENT_OPTIONS.map((option) => (
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
                            : `No recognized ${selectedParent.label.toLowerCase()} sub-labels have been observed yet.`}
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

type ActivePageQueryEvaluationResult =
    | { status: 'applied'; matchedElements: AnalysisResultElement[] }
    | { status: 'failed'; error: AnalysisResultsQueryError }
    | { status: 'stale' };

interface ActivePageQueryState {
    selectedView: ActivePageQueryViewKey;
    matchedElements: AnalysisResultElement[];
    getCommittedSnapshot: () => {
        isEvaluating: boolean;
        committedView: ActivePageQueryViewKey;
        committedExpression: string;
        matchedElements: AnalysisResultElement[];
    };
    selectView: (view: ActivePageQueryViewKey) => void;
    applyExpression: (
        expression: string,
        isCurrent?: () => boolean,
    ) => Promise<ActivePageQueryEvaluationResult>;
}

const useActivePageQueryState = (
    elements: AnalysisResultElement[],
    templates: readonly ActivePageQueryTemplate[],
): ActivePageQueryState => {
    const initialTemplate = templates.find(({ key }) => key === 'mistakes') ?? templates[0];
    const initialExpression = initialTemplate?.expression ?? 'elements';
    const [selectedView, setSelectedView] = React.useState<ActivePageQueryViewKey>('mistakes');
    const [matchedElements, setMatchedElements] = React.useState<AnalysisResultElement[]>([]);
    const isEvaluatingRef = React.useRef(false);
    const generationRef = React.useRef(0);
    const matchedElementsSourceRef = React.useRef<AnalysisResultElement[] | null>(null);
    const matchedElementsRef = React.useRef<AnalysisResultElement[]>([]);
    const committedExpressionRef = React.useRef(initialExpression);
    const committedViewRef = React.useRef<ActivePageQueryViewKey>('mistakes');
    const lastCustomExpressionRef = React.useRef(initialExpression);
    const skipNextAutomaticRef = React.useRef<{
        elements: AnalysisResultElement[];
        view: ActivePageQueryViewKey;
    } | null>(null);

    const templateByKey = React.useMemo(() => new Map(
        templates.map((template) => [template.key, template]),
    ), [templates]);
    const selectedTemplate = selectedView === 'custom'
        ? null
        : templateByKey.get(selectedView) ?? null;

    const evaluate = React.useCallback(async (
        expression: string,
        options: {
            failClosed: boolean;
            custom: boolean;
            view: ActivePageQueryViewKey;
            isCurrent?: () => boolean;
        },
    ): Promise<ActivePageQueryEvaluationResult> => {
        const generation = generationRef.current + 1;
        generationRef.current = generation;
        isEvaluatingRef.current = true;
        if (options.failClosed) setMatchedElements([]);

        try {
            const result = await evaluateAnalysisResultsQuery(expression, { elements });
            const resolved = resolveActivePageQueryResult(result, elements);
            if (
                generation !== generationRef.current
                || options.isCurrent?.() === false
            ) return { status: 'stale' };

            committedExpressionRef.current = expression;
            committedViewRef.current = options.view;
            if (options.custom) {
                lastCustomExpressionRef.current = expression;
            }
            matchedElementsSourceRef.current = elements;
            matchedElementsRef.current = resolved;
            setMatchedElements(resolved);
            return { status: 'applied', matchedElements: resolved };
        } catch (evaluationError) {
            if (
                generation !== generationRef.current
                || options.isCurrent?.() === false
            ) return { status: 'stale' };
            const queryError = toAnalysisResultsQueryError(evaluationError);
            if (options.failClosed) {
                matchedElementsSourceRef.current = elements;
                matchedElementsRef.current = [];
                setMatchedElements([]);
            }
            return { status: 'failed', error: queryError };
        } finally {
            if (generation === generationRef.current) {
                isEvaluatingRef.current = false;
            }
        }
    }, [elements]);

    React.useEffect(() => {
        const skipped = skipNextAutomaticRef.current;
        if (skipped?.elements === elements && skipped.view === selectedView) {
            skipNextAutomaticRef.current = null;
            return;
        }
        skipNextAutomaticRef.current = null;

        const expression = selectedTemplate?.expression ?? lastCustomExpressionRef.current;
        void evaluate(expression, {
            failClosed: true,
            custom: selectedView === 'custom',
            view: selectedView,
        });
    }, [elements, evaluate, selectedTemplate, selectedView]);

    React.useEffect(() => () => {
        generationRef.current += 1;
    }, []);

    const selectView = React.useCallback((view: ActivePageQueryViewKey) => {
        generationRef.current += 1;
        matchedElementsSourceRef.current = null;
        matchedElementsRef.current = [];
        setMatchedElements([]);
        isEvaluatingRef.current = true;
        setSelectedView(view);
    }, []);

    const applyExpression = React.useCallback(async (
        expression: string,
        isCurrent?: () => boolean,
    ): Promise<ActivePageQueryEvaluationResult> => {
        const matchingTemplate = selectedView === 'custom'
            ? null
            : templateByKey.get(selectedView);
        const nextView: ActivePageQueryViewKey = matchingTemplate?.expression === expression
            ? matchingTemplate.key
            : 'custom';
        const result = await evaluate(expression, {
            failClosed: false,
            custom: nextView === 'custom',
            view: nextView,
            isCurrent,
        });
        if (result.status !== 'applied' || nextView === selectedView) return result;

        skipNextAutomaticRef.current = { elements, view: nextView };
        setSelectedView(nextView);
        return result;
    }, [elements, evaluate, selectedView, templateByKey]);

    const matchedElementsAreCurrent = matchedElementsSourceRef.current === elements;
    const getCommittedSnapshot = React.useCallback(() => ({
        isEvaluating: isEvaluatingRef.current || matchedElementsSourceRef.current !== elements,
        committedView: committedViewRef.current,
        committedExpression: committedExpressionRef.current,
        matchedElements: matchedElementsRef.current,
    }), [elements]);
    return {
        selectedView,
        matchedElements: matchedElementsAreCurrent ? matchedElements : EMPTY_ANALYSIS_RESULT_ELEMENTS,
        getCommittedSnapshot,
        selectView,
        applyExpression,
    };
};

interface OverallTrendQueryState {
    result: OverallTrendQueryResult;
    error: AnalysisResultsQueryErrorDetail | null;
    isEvaluating: boolean;
}

const emptyOverallTrendQueryResult = (): OverallTrendQueryResult => ({
    laps: [],
    categories: [],
});

const useOverallTrendQueryState = (
    pages: readonly AnalysisResultsPaginationPage[],
    taxonomy: OverallTrendQueryTaxonomy,
): OverallTrendQueryState => {
    const [result, setResult] = React.useState<OverallTrendQueryResult>(emptyOverallTrendQueryResult);
    const [error, setError] = React.useState<AnalysisResultsQueryErrorDetail | null>(null);
    const [isEvaluating, setIsEvaluating] = React.useState(false);
    const generationRef = React.useRef(0);

    React.useEffect(() => {
        const generation = generationRef.current + 1;
        generationRef.current = generation;
        setResult(emptyOverallTrendQueryResult());
        setError(null);
        setIsEvaluating(true);

        void (async () => {
            try {
                const input = normalizeOverallTrendQueryInput({ pages });
                const expression = buildOverallTrendQueryExpression(taxonomy);
                const rawResult = await evaluateAnalysisResultsQuery(expression, input);
                const resolvedResult = resolveOverallTrendQueryResult(rawResult, input, taxonomy);
                if (generation !== generationRef.current) return;

                setResult(resolvedResult);
                setError(null);
            } catch (evaluationError) {
                if (generation !== generationRef.current) return;
                setResult(emptyOverallTrendQueryResult());
                setError(toAnalysisResultsQueryError(evaluationError).detail);
            } finally {
                if (generation === generationRef.current) setIsEvaluating(false);
            }
        })();

        return () => {
            if (generation === generationRef.current) generationRef.current += 1;
        };
    }, [pages, taxonomy]);

    return { result, error, isEvaluating };
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
    const [trendParent, setTrendParent] = React.useState<MistakeTrendParent>('MSP');
    const [showOverallTrend, setShowOverallTrend] = React.useState(true);
    const [selectedTrendSubLabelId, setSelectedTrendSubLabelId] = React.useState<string | null>(null);
    const committedPageIdsRef = React.useRef<Set<string>>(new Set());
    const pendingPageWaitersRef = React.useRef<Map<string, Set<AnalysisResultPageWaiter>>>(new Map());
    const pendingSelectionWaitersRef = React.useRef<Set<AnalysisResultSelectionWaiter>>(new Set());
    const renderedSelectionRef = React.useRef<RenderedAnalysisResultSelection>({
        pageId: null,
        isLapResults: false,
    });
    const applyOperationGenerationRef = React.useRef(0);
    const mountedRef = React.useRef(false);
    const { getCategoryLabels, getLabelName } = useAiLabels();
    const retainedPages = pagination?.pages ?? EMPTY_ANALYSIS_RESULTS_PAGES;
    const activePageIndex = React.useMemo(() => {
        if (!pagination || retainedPages.length === 0) return -1;
        const selectedIndex = retainedPages.findIndex((page) => page.id === pagination.activePageId);
        return selectedIndex >= 0 ? selectedIndex : 0;
    }, [pagination, retainedPages]);
    const activePage = activePageIndex >= 0 ? retainedPages[activePageIndex] : null;
    const isOverallTrend = Boolean(pagination) && (showOverallTrend || !activePage);
    const activeData = activePage ?? data;
    const { elements } = React.useMemo(() => normalizeAnalysisResultsData(activeData), [activeData]);
    const activeMistakeCatalog = React.useMemo(() => {
        const categoryLabels = {
            MSP: getCategoryLabels('MSP'),
            MSR: getCategoryLabels('MSR'),
        };
        const recognizedSubLabels = new Map<string, RecognizedSubLabel>();
        (['MSP', 'MSR'] as const).forEach((parentId) => {
            categoryLabels[parentId].forEach((childId) => {
                const child = { id: childId, label: getLabelName(childId) ?? childId };
                recognizedSubLabels.set(childId, child);
                recognizedSubLabels.set(child.label, child);
            });
        });
        return { categoryLabels, recognizedSubLabels };
    }, [getCategoryLabels, getLabelName]);
    const activeQueryTemplates = React.useMemo(() => buildActivePageQueryTemplates({
        getCategoryLabels: (parentId) => activeMistakeCatalog.categoryLabels[parentId],
        getLabelName,
    }), [activeMistakeCatalog.categoryLabels, getLabelName]);
    const activeQuery = useActivePageQueryState(elements, activeQueryTemplates);
    const activeQueryRef = React.useRef(activeQuery);
    activeQueryRef.current = activeQuery;
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

    const waitForRenderedSelection = React.useCallback((
        pageId: string | null,
        operationGeneration: number,
    ): Promise<void> => {
        if (operationGeneration !== applyOperationGenerationRef.current) {
            return Promise.reject(createAnalysisResultsApplyError(
                name,
                'A newer Analysis Results query apply operation replaced this one.',
            ));
        }
        const renderedSelection = renderedSelectionRef.current;
        if (
            renderedSelection.pageId === pageId
            && renderedSelection.isLapResults
        ) return Promise.resolve();
        if (!mountedRef.current) {
            return Promise.reject(createAnalysisResultsApplyError(
                name,
                'Analysis Results unmounted before the requested page selection rendered.',
            ));
        }

        return new Promise((resolve, reject) => {
            const waiter: AnalysisResultSelectionWaiter = {
                pageId,
                operationGeneration,
                resolve,
                reject,
                timeoutId: setTimeout(() => {
                    pendingSelectionWaitersRef.current.delete(waiter);
                    reject(createAnalysisResultsApplyError(
                        name,
                        `Analysis Results page selection did not render within ${AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS}ms.`,
                    ));
                }, AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS),
            };
            pendingSelectionWaitersRef.current.add(waiter);
        });
    }, [name]);

    React.useLayoutEffect(() => {
        const pendingPageWaiters = pendingPageWaitersRef.current;
        const pendingSelectionWaiters = pendingSelectionWaitersRef.current;
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            applyOperationGenerationRef.current += 1;
            pendingPageWaiters.forEach((waiters, pageId) => {
                waiters.forEach((waiter) => {
                    clearTimeout(waiter.timeoutId);
                    waiter.reject(createAnalysisResultsReadinessError(name, pageId, 'unmounted'));
                });
            });
            pendingPageWaiters.clear();
            pendingSelectionWaiters.forEach((waiter) => {
                clearTimeout(waiter.timeoutId);
                waiter.reject(createAnalysisResultsApplyError(
                    name,
                    'Analysis Results unmounted before the requested page selection rendered.',
                ));
            });
            pendingSelectionWaiters.clear();
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

    React.useEffect(() => {
        const renderedSelection: RenderedAnalysisResultSelection = {
            pageId: pagination ? activePage?.id ?? null : null,
            isLapResults: !isOverallTrend,
        };
        renderedSelectionRef.current = renderedSelection;
        pendingSelectionWaitersRef.current.forEach((waiter) => {
            if (waiter.operationGeneration !== applyOperationGenerationRef.current) {
                clearTimeout(waiter.timeoutId);
                pendingSelectionWaitersRef.current.delete(waiter);
                waiter.reject(createAnalysisResultsApplyError(
                    name,
                    'A newer Analysis Results query apply operation replaced this one.',
                ));
                return;
            }
            if (
                waiter.pageId !== renderedSelection.pageId
                || !renderedSelection.isLapResults
            ) return;
            clearTimeout(waiter.timeoutId);
            pendingSelectionWaitersRef.current.delete(waiter);
            waiter.resolve();
        });
    }, [activePage?.id, isOverallTrend, name, pagination]);

    const handle = React.useMemo<AnalysisResultsChartHandle>(() => ({
        getComponentName: () => name,
        waitForAnalysisResultPage,
        getFilteredSegments: () => {
            if (isOverallTrend) {
                return cloneAndFreeze({
                    status: 'empty' as const,
                    activePageId: null,
                    appliedView: null,
                    committedQuery: null,
                    segments: [],
                });
            }
            const appliedQuery = activeQueryRef.current.getCommittedSnapshot();
            const segments = appliedQuery.isEvaluating
                ? EMPTY_ANALYSIS_RESULT_ELEMENTS
                : appliedQuery.matchedElements;
            return cloneAndFreeze({
                status: appliedQuery.isEvaluating
                    ? 'busy' as const
                    : segments.length > 0 ? 'ready' as const : 'empty' as const,
                activePageId: activePage?.id ?? null,
                appliedView: appliedQuery.committedView,
                committedQuery: appliedQuery.committedExpression,
                segments,
            });
        },
        applyAnalysisResultQuery: (args) => {
            const operationGeneration = applyOperationGenerationRef.current + 1;
            applyOperationGenerationRef.current = operationGeneration;
            pendingSelectionWaitersRef.current.forEach((waiter) => {
                clearTimeout(waiter.timeoutId);
                waiter.reject(createAnalysisResultsApplyError(
                    name,
                    'A newer Analysis Results query apply operation replaced this one.',
                ));
            });
            pendingSelectionWaitersRef.current.clear();

            return createAiToolOperationFrom(async () => {
                const requestedPageNumber = args.page_number ?? null;
                const pageCount = pagination ? retainedPages.length : 1;
                if (pagination && pageCount === 0) {
                    throw createAnalysisResultsApplyError(
                        name,
                        'Cannot apply an Analysis Results query because no retained pages exist.',
                    );
                }
                const requestedPageExists = requestedPageNumber !== null
                    && requestedPageNumber >= 1
                    && requestedPageNumber <= pageCount;
                const appliedPageNumber = requestedPageExists
                    ? requestedPageNumber
                    : pageCount;
                const appliedPage = pagination
                    ? retainedPages[appliedPageNumber - 1]
                    : null;

                setShowOverallTrend(false);
                if (pagination && appliedPage) pagination.onSelectPage(appliedPage.id);
                await waitForRenderedSelection(appliedPage?.id ?? null, operationGeneration);
                if (operationGeneration !== applyOperationGenerationRef.current) {
                    throw createAnalysisResultsApplyError(
                        name,
                        'A newer Analysis Results query apply operation replaced this one.',
                    );
                }

                const applied = await activeQueryRef.current.applyExpression(
                    args.query,
                    () => operationGeneration === applyOperationGenerationRef.current,
                );
                if (
                    operationGeneration !== applyOperationGenerationRef.current
                    || applied.status === 'stale'
                ) {
                    throw createAnalysisResultsApplyError(
                        name,
                        'A newer Analysis Results query apply operation replaced this one.',
                    );
                }
                if (applied.status === 'failed') throw applied.error;

                return {
                    status: 'ready' as const,
                    data: applied.matchedElements.length,
                    applied_query: args.query,
                    applied_page_id: appliedPage?.id ?? null,
                    applied_page_number: appliedPageNumber,
                    requested_page_number: requestedPageNumber,
                    used_most_recent_fallback: !requestedPageExists,
                };
            }, 'ready');
        },
        queryAnalysisResult: ({ query }) => createAiToolOperationFrom(async () => ({
            status: 'ready' as const,
            data: await evaluateAllAnalysisResultsQuery(query, {
                analyses: pagination ? retainedPages : [{
                    id,
                    createdAt: null,
                    baseline: null,
                    elements,
                }],
            }),
        }), 'ready'),
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
    }), [
        activeData,
        activePage?.id,
        elements,
        id,
        isOverallTrend,
        name,
        onDisable,
        onUpdate,
        pagination,
        retainedPages,
        waitForAnalysisResultPage,
        waitForRenderedSelection,
    ]);
    React.useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandleRef = React.useRef(handle);
    registeredHandleRef.current = handle;
    useRegisterAiToolComponentRef(registeredHandleRef);
    const selectedTrendParent = TREND_PARENT_OPTIONS.find(({ value }) => value === trendParent)!;
    const trendTaxonomy = React.useMemo<OverallTrendQueryTaxonomy>(() => ({
        parent: {
            id: selectedTrendParent.value,
            fallbackName: selectedTrendParent.fallbackName,
            resolvedName: getLabelName(selectedTrendParent.value),
        },
        categories: getCategoryLabels(trendParent).map((categoryId) => ({
            id: categoryId,
            fallbackName: categoryId,
            resolvedName: getLabelName(categoryId),
        })),
    }), [getCategoryLabels, getLabelName, selectedTrendParent, trendParent]);
    const trendQuery = useOverallTrendQueryState(
        retainedPages,
        trendTaxonomy,
    );
    const trendData = trendQuery.result;
    const lapTimeTrendData = React.useMemo(
        () => buildLapTimeTrendData(trendData.laps),
        [trendData.laps],
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
        if (trendQuery.isEvaluating) return;
        const nextId = selectedTrendCategory?.id ?? null;
        if (selectedTrendSubLabelId !== nextId) setSelectedTrendSubLabelId(nextId);
    }, [selectedTrendCategory, selectedTrendSubLabelId, trendQuery.isEvaluating]);
    const isMistakeTemplate = activeQuery.selectedView !== 'all-results'
        && activeQuery.selectedView !== 'custom';
    const selectedActiveTemplate = activeQuery.selectedView === 'custom'
        ? CUSTOM_ACTIVE_PAGE_VIEW
        : activeQueryTemplates.find(({ key }) => key === activeQuery.selectedView)
            ?? CUSTOM_ACTIVE_PAGE_VIEW;
    const labelFrequencyData = React.useMemo(
        () => buildLabelFrequencyData(
            activeQuery.matchedElements,
            isMistakeTemplate ? activeMistakeCatalog.recognizedSubLabels : undefined,
        ),
        [activeMistakeCatalog.recognizedSubLabels, activeQuery.matchedElements, isMistakeTemplate],
    );
    const trendSubject = trendParent === 'MSP' ? 'Training' : 'Racing';
    const labelFrequencySpec = React.useMemo<GraphSpec>(() => ({
        type: 'bar',
        data: labelFrequencyData,
        categoryKey: 'label',
        series: [{ key: 'occurrences', label: 'Occurrences' }],
        orientation: 'horizontal',
        title: `Label frequency — ${selectedActiveTemplate.label}`,
        xAxisLabel: 'Occurrences',
        showLegend: false,
        colors: ['#00e676'],
        accessibleLabel: isMistakeTemplate
            ? `${selectedActiveTemplate.label} frequency by recognized taxonomy child label`
            : `${selectedActiveTemplate.label} frequency by exact label`,
        height: getLabelFrequencyGraphHeight(labelFrequencyData.length),
        emptyStateText: isMistakeTemplate
            ? 'No recognized mistake labels in the current query result to graph.'
            : 'No labels in the current query result to graph.',
    }), [isMistakeTemplate, labelFrequencyData, selectedActiveTemplate.label]);

    const displayedPageIndex = activePageIndex + 1;
    const displayedPageCount = retainedPages.length;

    const handlePreviousPage = () => {
        if (!pagination || isOverallTrend) return;
        if (activePageIndex <= 0) return;
        pagination.onSelectPage(retainedPages[activePageIndex - 1].id);
    };

    const handleNextPage = () => {
        if (!pagination || isOverallTrend) return;
        const nextPage = retainedPages[activePageIndex + 1];
        if (nextPage) pagination.onSelectPage(nextPage.id);
    };

    return (
        <Card className={styles.chart} style={{ width, height }}>
            {pagination && (
                <Flex className={styles.navigation} justify="between" align="center" gap="2" wrap="wrap">
                    <Flex
                        className={styles.viewNavigation}
                        align="center"
                        gap="1"
                        role="group"
                        aria-label="Analysis result view"
                    >
                        <button
                            type="button"
                            className={`${styles.viewButton} ${isOverallTrend ? styles.viewButtonActive : ''}`}
                            aria-pressed={isOverallTrend}
                            onClick={() => setShowOverallTrend(true)}
                        >
                            Overall Trends
                        </button>
                        <button
                            type="button"
                            className={`${styles.viewButton} ${!isOverallTrend ? styles.viewButtonActive : ''}`}
                            aria-pressed={!isOverallTrend}
                            disabled={!activePage}
                            onClick={() => setShowOverallTrend(false)}
                        >
                            Lap Results
                        </button>
                    </Flex>
                    {isOverallTrend ? (
                        <Text className={styles.baselineContext} size="1" color="gray" as="span">
                            {retainedPages.length} analyzed {retainedPages.length === 1 ? 'lap' : 'laps'}
                        </Text>
                    ) : (
                        <>
                            <Flex className={styles.pageNavigation} align="center" gap="2">
                                <button
                                    type="button"
                                    className={styles.pageButton}
                                    disabled={activePageIndex <= 0}
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
                                    Page {displayedPageIndex} of {displayedPageCount}
                                </Text>
                                <button
                                    type="button"
                                    className={styles.pageButton}
                                    disabled={activePageIndex >= displayedPageCount - 1}
                                    onClick={handleNextPage}
                                >
                                    Next
                                </button>
                            </Flex>
                            <Text className={styles.baselineContext} size="1" color="gray" as="span">
                                Baseline: {activePage!.baseline.track || 'Unknown track'} ·{' '}
                                {activePage!.baseline.car || 'Unknown car'} · Lap {activePage!.baseline.lap_id}
                            </Text>
                        </>
                    )}
                </Flex>
            )}
            {isOverallTrend ? (
                <OverallMistakeTrend
                    id={id}
                    analyzedLapCount={retainedPages.length}
                    lapTimeTrendData={lapTimeTrendData}
                    trendParent={trendParent}
                    onTrendParentChange={setTrendParent}
                    selectedParent={selectedTrendParent}
                    trendSubject={trendSubject}
                    trendData={trendData}
                    selectedCategory={selectedTrendCategory}
                    onSelectedCategoryChange={setSelectedTrendSubLabelId}
                    error={trendQuery.error}
                    isEvaluating={trendQuery.isEvaluating}
                />
            ) : <>
                <Flex className={styles.summary} justify="between" align="center" gap="2" wrap="wrap">
                <Flex className={styles.summaryText} align="center" justify="between" gap="2">
                    <Text size="2" weight="medium">Labeled elements</Text>
                    <Text size="2" weight="bold" className={styles.count}>
                        {activeQuery.matchedElements.length} of {elements.length} total
                    </Text>
                </Flex>
                <Flex className={styles.controls} align="center" gap="2" wrap="wrap">
                    <label className={styles.viewControl}>
                        <Text size="1" color="gray" as="span">View</Text>
                        <select
                            className={styles.viewSelect}
                            value={activeQuery.selectedView}
                            onChange={(event) => activeQuery.selectView(
                                event.target.value as ActivePageQueryViewKey,
                            )}
                        >
                            {activeQueryTemplates.map((template) => (
                                <option value={template.key} key={template.key}>{template.label}</option>
                            ))}
                            <option value={CUSTOM_ACTIVE_PAGE_VIEW.key}>{CUSTOM_ACTIVE_PAGE_VIEW.label}</option>
                        </select>
                    </label>
                </Flex>
                </Flex>
                <ScrollArea type="hover" className={styles.list}>
                <Box className={styles.graph}>
                    <DataGraph spec={labelFrequencySpec} />
                </Box>
                {activeQuery.matchedElements.length === 0 ? (
                    <Box className={styles.empty} data-testid="analysis-results-empty-state">
                        <Text color="gray">No results match the {selectedActiveTemplate.label} view.</Text>
                    </Box>
                ) : (
                    <Box className={styles.cards}>
                        {activeQuery.matchedElements.map((element, index) => (
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
