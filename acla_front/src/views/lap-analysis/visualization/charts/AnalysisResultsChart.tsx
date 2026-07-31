import React from 'react';
import { Badge, Box, Card, Flex, ScrollArea, Text } from '@radix-ui/themes';
import { VisualizationProps } from '../VisualizationRegistry';
import {
    AnalysisResultElement,
    normalizeAnalysisResultsData,
} from './analysisResultsModel';
import styles from './AnalysisResultsChart.module.css';

const formatPosition = (value: number): string => `${(value * 100).toFixed(1)}%`;

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

const AnalysisResultCard: React.FC<{ element: AnalysisResultElement }> = ({ element }) => (
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

        {element.metadata && Object.keys(element.metadata).length > 0 && (
            <Box className={styles.metadata}>
                {Object.entries(element.metadata).map(([key, value]) => (
                    <Text size="1" color="gray" as="div" key={key}>
                        {key}: {stringifyValue(value)}
                    </Text>
                ))}
            </Box>
        )}
    </Box>
);

const AnalysisResultsChart: React.FC<VisualizationProps> = ({ data, width = '100%', height = '100%' }) => {
    const { elements } = normalizeAnalysisResultsData(data);

    return (
        <Card className={styles.chart} style={{ width, height }}>
            <Flex className={styles.summary} justify="between" align="center">
                <Text size="2" weight="medium">Labeled elements</Text>
                <Text size="2" weight="bold" className={styles.count}>{elements.length} total</Text>
            </Flex>
            {elements.length === 0 ? (
                <Box className={styles.empty} data-testid="analysis-results-empty-state">
                    <Text color="gray">No analysis results yet.</Text>
                </Box>
            ) : (
                <ScrollArea type="hover" className={styles.list}>
                    {elements.map((element) => (
                        <AnalysisResultCard element={element} key={element.id} />
                    ))}
                </ScrollArea>
            )}
        </Card>
    );
};

export default AnalysisResultsChart;
