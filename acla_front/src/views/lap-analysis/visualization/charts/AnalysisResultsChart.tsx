import React from 'react';
import { Badge, Box, Card, Flex, ScrollArea, Text } from '@radix-ui/themes';
import { VisualizationProps } from '../VisualizationRegistry';
import {
    AnalysisResultElement,
    normalizeAnalysisResultsData,
} from './analysisResultsModel';
import { useAiLabels } from 'contexts/AiLabelsContext';
import styles from './AnalysisResultsChart.module.css';

const formatPosition = (value: number): string => `${(value * 100).toFixed(1)}%`;

const HIDDEN_METADATA_KEYS = new Set(['source', 'start_index', 'end_index']);

type AnalysisResultsSortMode = 'original' | 'most-frequent' | 'most-time-lost';

interface IndexedAnalysisResult {
    element: AnalysisResultElement;
    originalIndex: number;
}

interface RecognizedSubMistake {
    id: string;
    label: string;
}

type RecognizedSubMistakes = ReadonlyMap<string, RecognizedSubMistake>;

const MISTAKE_PARENT_LABEL_IDS = ['MSP', 'MSR'] as const;

const compareLabelText = (left: string, right: string): number => (
    left.localeCompare(right, undefined, { sensitivity: 'base' }) || left.localeCompare(right)
);

const getSubMistakes = (
    element: AnalysisResultElement,
    recognizedSubMistakes: RecognizedSubMistakes,
): RecognizedSubMistake[] => {
    const matches = new Map<string, RecognizedSubMistake>();
    element.labels.forEach((label) => {
        const recognized = recognizedSubMistakes.get(label);
        if (recognized) matches.set(recognized.id, recognized);
    });
    return Array.from(matches.values());
};

const sortAnalysisResults = (
    elements: AnalysisResultElement[],
    sortMode: AnalysisResultsSortMode,
    recognizedSubMistakes: RecognizedSubMistakes,
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

    const mistakesByElement = indexedElements.map(({ element }) => (
        getSubMistakes(element, recognizedSubMistakes)
    ));
    const mistakeCounts = new Map<string, number>();
    mistakesByElement.forEach((mistakes) => {
        mistakes.forEach((mistake) => {
            mistakeCounts.set(mistake.id, (mistakeCounts.get(mistake.id) ?? 0) + 1);
        });
    });
    const rankings = mistakesByElement.map((mistakes) => (
        mistakes.reduce((ranking, mistake) => {
            const count = mistakeCounts.get(mistake.id) ?? 0;
            if (
                count > ranking.count
                || (count === ranking.count && (!ranking.label || compareLabelText(mistake.label, ranking.label) < 0))
            ) {
                return { count, label: mistake.label };
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

const AnalysisResultCard: React.FC<{ element: AnalysisResultElement }> = ({ element }) => {
    const metadataEntries = Object.entries(element.metadata ?? {})
        .filter(([key]) => !HIDDEN_METADATA_KEYS.has(key));

    return (
        <Box className={styles.element} data-testid={`analysis-result-${element.id}`}>
            <Flex justify="between" align="start" gap="2" wrap="wrap">
                <Box>
                    {element.title && <Text size="2" weight="bold" as="div">{element.title}</Text>}
                    <Text size="1" className={styles.id} as="div">{element.id}</Text>
                </Box>
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
        </Box>
    );
};

const AnalysisResultsChart: React.FC<VisualizationProps> = ({ data, width = '100%', height = '100%' }) => {
    const [sortMode, setSortMode] = React.useState<AnalysisResultsSortMode>('original');
    const { getCategoryLabels, getLabelName } = useAiLabels();
    const { elements } = React.useMemo(() => normalizeAnalysisResultsData(data), [data]);
    const recognizedSubMistakes = React.useMemo(() => {
        const matches = new Map<string, RecognizedSubMistake>();
        MISTAKE_PARENT_LABEL_IDS.forEach((parentId) => {
            getCategoryLabels(parentId).forEach((childId) => {
                const child = { id: childId, label: getLabelName(childId) ?? childId };
                matches.set(childId, child);
                matches.set(child.label, child);
            });
        });
        return matches;
    }, [getCategoryLabels, getLabelName]);
    const sortedElements = React.useMemo(
        () => sortAnalysisResults(elements, sortMode, recognizedSubMistakes),
        [elements, recognizedSubMistakes, sortMode],
    );

    return (
        <Card className={styles.chart} style={{ width, height }}>
            <Flex className={styles.summary} justify="between" align="center" gap="2" wrap="wrap">
                <Flex className={styles.summaryText} align="center" justify="between" gap="2">
                    <Text size="2" weight="medium">Labeled elements</Text>
                    <Text size="2" weight="bold" className={styles.count}>{elements.length} total</Text>
                </Flex>
                <label className={styles.sortControl}>
                    <Text size="1" color="gray" as="span">Sort by</Text>
                    <select
                        className={styles.sortSelect}
                        value={sortMode}
                        onChange={(event) => setSortMode(event.target.value as AnalysisResultsSortMode)}
                    >
                        <option value="original">Original order</option>
                        <option value="most-frequent">Most frequent mistake</option>
                        <option value="most-time-lost">Most time lost</option>
                    </select>
                </label>
            </Flex>
            {elements.length === 0 ? (
                <Box className={styles.empty} data-testid="analysis-results-empty-state">
                    <Text color="gray">No analysis results yet.</Text>
                </Box>
            ) : (
                <ScrollArea type="hover" className={styles.list}>
                    {sortedElements.map((element) => (
                        <AnalysisResultCard element={element} key={element.id} />
                    ))}
                </ScrollArea>
            )}
        </Card>
    );
};

export default AnalysisResultsChart;
