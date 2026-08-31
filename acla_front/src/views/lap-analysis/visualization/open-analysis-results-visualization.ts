import {
    AI_TOOL_COMPONENT_NAMES,
    awaitNamedComponentHandle,
    resolveNamedComponentHandle,
    type AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import type { SegmentClassificationResult } from 'views/lap-analysis/recorded-session-analysis';
import type { VisualizationManagerHandle } from './VisualizationPanelManager';
import type { AnalysisResultsChartHandle } from './charts/AnalysisResultsChart';
import type { AnalysisResultElement } from './charts/analysisResultsModel';
import { resolveAnalysisResultsComparison } from './charts/analysisResultsComparisonAdapter';
import { getSegmentLabelIds } from './charts/segmentClassificationDisplay';
import { getSingletonVisualizationComponentName } from './visualization-component-names';

const getNormalizedPosition = (
    records: Record<string, any>[],
    index: number,
): number | null => {
    const row = records[Math.max(0, Math.min(records.length - 1, Math.trunc(index)))];
    const value = Number(
        row?.Graphics_normalized_car_position
        ?? row?.normalized_position
        ?? row?.normalizedPosition,
    );
    return Number.isFinite(value) ? value : null;
};

const buildAnalysisElements = (
    result: SegmentClassificationResult,
    records: Record<string, any>[],
    getLabelName: (labelId: string) => string | undefined,
): AnalysisResultElement[] => result.segments.map((segment, index) => {
    const start = records.length ? getNormalizedPosition(records, segment.start_index) : null;
    const end = records.length ? getNormalizedPosition(records, segment.end_index) : null;
    const comparisonResolution = resolveAnalysisResultsComparison({
        baselineRecords: records,
        expertReferenceData: segment.expert_reference_data,
    });
    const comparison = comparisonResolution.comparison;
    return {
        id: segment.id || `${result.session_id}:segment:${index}`,
        labels: getSegmentLabelIds(segment)
            .map((labelId) => getLabelName(labelId) || labelId),
        ...(segment.track_section ? {
            section: getLabelName(segment.track_section) || segment.track_section,
        } : {}),
        ...(start !== null && end !== null ? {
            normalizedPositionRange: { start, end },
        } : {}),
        ...(comparison?.samples.length ? { comparison } : {}),
        ...(comparisonResolution.diagnostics.length > 0
            ? { comparisonDiagnostics: comparisonResolution.diagnostics }
            : {}),
        metadata: {
            source: 'ai_classifier',
            start_index: segment.start_index,
            end_index: segment.end_index,
        },
    };
});

export const openAnalysisResultsVisualization = async ({
    directory,
    managerName,
    result,
    records,
}: {
    directory: AiToolComponentRefDirectory;
    managerName: string;
    result: SegmentClassificationResult;
    records: Record<string, any>[];
}) => {
    const name = getSingletonVisualizationComponentName('analysis-results');
    const manager = resolveNamedComponentHandle<VisualizationManagerHandle>(directory, managerName);
    const elements = buildAnalysisElements(
        result,
        records,
        (labelId) => resolveAnalysisLabel(directory, labelId),
    );
    const existing = directory.findComponentRef<AnalysisResultsChartHandle>(name)?.current;

    if (existing) {
        existing.replaceAnalysisResults({ elements });
        const instance = manager.getCurrentVisualizations()
            .find((visualization) => visualization.name === name);
        return {
            chart_id: instance?.id ?? null,
            component_name: name,
        };
    }

    const requested = manager.requestVisualization({
        name,
        type: 'analysis-results',
        data: { elements },
    });
    const mountedName = requested.componentName || name;
    const chart = await awaitNamedComponentHandle<AnalysisResultsChartHandle>(
        directory,
        mountedName,
    );
    chart.replaceAnalysisResults({ elements });
    return {
        chart_id: requested.chartId ?? null,
        component_name: mountedName,
    };
};

export const resolveAnalysisLabel = (
    directory: AiToolComponentRefDirectory | null,
    labelId: string,
): string | undefined => directory?.findComponentRef<{
    getComponentName(): string;
    getLabelName(id: string): string | undefined;
}>(AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT)?.current?.getLabelName(labelId);
