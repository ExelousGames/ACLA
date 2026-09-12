import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { synthesizeTtsPack } from 'components/tts';

jest.mock('components/tts/tts-service', () => ({
    ...jest.requireActual('components/tts/tts-service'),
    synthesizeTtsPack: jest.fn(),
}));

let mockOverlayPresentation: { presentationId: string } | null = null;
let mockOverlayComponentDirectory: any = null;
let mockFloatingChatClosedListener: (() => void) | null = null;
const mockOverlaySessionListeners = new Set<(
    presentation: { presentationId: string } | null,
) => void>();

jest.mock('views/floating-chat/overlay-display-client', () => ({
    overlaySessionClient: {
        current: () => mockOverlayPresentation,
        subscribe: (listener: (presentation: { presentationId: string } | null) => void) => {
            mockOverlaySessionListeners.add(listener);
            listener(mockOverlayPresentation);
            return () => mockOverlaySessionListeners.delete(listener);
        },
    },
}));

jest.mock('contexts/OperationComponentRefContext', () => {
    const actual = jest.requireActual('contexts/OperationComponentRefContext');
    return {
        ...actual,
        useOptionalOperationComponentRefDirectory: () => mockOverlayComponentDirectory,
    };
});

jest.mock('contexts/DesktopGameContext', () => ({
    useDesktopGame: () => ({
        detectedGame: null,
        detectionStatus: 'not-detected',
        error: null,
    }),
}));

jest.mock('@radix-ui/themes', () => {
    const Component = ({ as: Tag = 'div', children, ...props }: any) => (
        <Tag {...props}>{children}</Tag>
    );
    return {
        Badge: Component,
        Box: Component,
        Card: Component,
        Flex: Component,
        ScrollArea: Component,
        Text: Component,
    };
});

const mockCategoryLabels: Record<string, string[]> = {
    MSP: ['MSP1', 'MSP2'],
    MSR: ['MSR1', 'MSR2'],
    EA: ['EA1'],
    RM: ['RM7'],
};
const mockLabelNames: Record<string, string> = {
    MSP: 'Training Error',
    MSP1: 'Late turn-in',
    MSP2: 'Wheel lock',
    MSR: 'Race Error',
    MSR1: 'Failed overtake attempt',
    MSR2: 'Contact',
    EA1: 'Matches expert line',
    RM7: 'Merge back to expert line',
};
const mockDefaultGetCategoryLabels = (category: string) => mockCategoryLabels[category] ?? [];
const mockDefaultGetLabelName = (labelId: string) => mockLabelNames[labelId];
let mockGetCategoryLabels = mockDefaultGetCategoryLabels;
let mockGetLabelName = mockDefaultGetLabelName;

jest.mock('contexts/AiLabelsContext', () => ({
    useAiLabels: () => ({
        getCategoryLabels: mockGetCategoryLabels,
        getLabelName: mockGetLabelName,
    }),
}));

jest.mock('components/data-graphs', () => ({
    DataGraph: ({ spec }: any) => {
        const seriesKey = spec.series?.[0]?.key;
        const testId = spec.type === 'bar'
            ? 'label-frequency-graph'
            : seriesKey === 'lapTimeSeconds'
                ? 'lap-time-trend-graph'
                : seriesKey === 'totalCount'
                ? 'overall-total-trend-graph'
                : 'specific-mistake-trend-graph';
        return (
            <div
                data-testid={testId}
                data-graph-data={JSON.stringify(spec.data)}
                data-graph-height={String(spec.height)}
                data-graph-orientation={spec.orientation}
                data-graph-value-axis-label={spec.xAxisLabel ?? spec.yAxisLabel}
                data-graph-colors={JSON.stringify(spec.colors)}
            >
                <span>{spec.title}</span>
                {spec.data.length === 0 && <span role="status">{spec.emptyStateText}</span>}
            </div>
        );
    },
}));

import AnalysisResultsChart, {
    buildLapTimeTrendData,
    calculateLeastSquaresSlope,
    formatRacingTime,
    getMistakeTrendDirection,
    type AnalysisResultsChartHandle,
    type AnalysisResultsPaginationPage,
} from './AnalysisResultsChart';
import {
    createOperationComponentRefDirectory,
} from 'contexts/OperationComponentRefContext';
import {
    VisualizationControlFailedError,
} from 'contexts/OperationComponentError';
import {
    appendAnalysisResultElement,
    normalizeAnalysisResultsData,
    removeAnalysisResultElement,
    updateAnalysisResultElement,
} from './analysisResultsModel';
import * as analysisResultsQuery from './analysisResultsQuery';
import { ProcedurePlanRunner } from 'components/ai-operations/ProcedurePlan';
import { RepeatablePlanRunner } from 'components/ai-operations/RepeatablePlan';
import { LiveRangeTodoListRunner } from 'components/ai-operations/LiveRangeTodoList';
import { createAiCommandRegistry, createWorkflowToolDispatcher } from '../../ai-chat/ai-command-registry';
import { normalizeOperationError, serializeError } from 'errors/OperationError';
import { buildFormattedToolResultFrame } from '../../ai-chat/voice-tool-result-formatter';

const ALL_ANALYSES_COUNT_QUERY = '$count(analyses)';
const ALL_RESULTS_COUNT_QUERY = '$count(analyses.elements)';
const MISTAKE_COUNT_QUERY = '$count(analyses.elements[labels[label_name in ["MSP", "Mistake (Practice)", "Training Error", "MSR", "Mistake (Racing)", "Race Error"]]])';

const renderedResultIds = (): string[] => (
    screen.queryAllByTestId(/^analysis-result-/).map((element) => (
        element.getAttribute('data-testid')?.replace('analysis-result-', '') ?? ''
    ))
);

const selectView = (value: string): void => {
    fireEvent.change(screen.getByRole('combobox', { name: 'View' }), { target: { value } });
};

const selectTrendParent = (value: string): void => {
    fireEvent.change(screen.getByRole('combobox', { name: 'Showing' }), { target: { value } });
};

const renderedFrequencyData = (): Array<{ label: string; occurrences: number }> => (
    JSON.parse(screen.getByTestId('label-frequency-graph').getAttribute('data-graph-data') ?? '[]')
);

const renderedTrendData = (testId: 'overall-total-trend-graph' | 'specific-mistake-trend-graph') => (
    JSON.parse(screen.getByTestId(testId).getAttribute('data-graph-data') ?? '[]')
);

const renderedLapTimeTrendData = () => (
    JSON.parse(screen.getByTestId('lap-time-trend-graph').getAttribute('data-graph-data') ?? '[]')
);

const comparableData = (driverGas: number, expertGas: number) => ({
    samples: [{
        driverTimeMs: 0,
        expertTimeMs: 0,
        driverTrackPosition: 0.2,
        expertTrackPosition: 0.2,
        driverGas,
        expertGas,
    }],
});

const replayComparisonData = () => ({
    samples: [{
        driverTimeMs: 0,
        expertTimeMs: 0,
        driverTrackPosition: 0.2,
        expertTrackPosition: 0.2,
        driverGas: 0.2,
        expertGas: 0.3,
    }, {
        driverTimeMs: 2_000,
        expertTimeMs: 2_000,
        driverTrackPosition: 0.3,
        expertTrackPosition: 0.3,
        driverGas: 0.4,
        expertGas: 0.5,
    }],
});

describe('AnalysisResultsChart', () => {
    beforeEach(() => {
        mockOverlayPresentation = { presentationId: 'analysis-overlay-session' };
        mockOverlayComponentDirectory = createOperationComponentRefDirectory();
        mockFloatingChatClosedListener = null;
        mockOverlaySessionListeners.clear();
        (window as any).electronAPI = {
            onFloatingChatClosed: (listener: () => void) => {
                mockFloatingChatClosedListener = listener;
                return () => {
                    if (mockFloatingChatClosedListener === listener) {
                        mockFloatingChatClosedListener = null;
                    }
                };
            },
        };
    });

    afterEach(() => {
        delete (window as any).electronAPI;
        mockOverlayComponentDirectory = null;
        mockOverlayPresentation = null;
        mockFloatingChatClosedListener = null;
        mockOverlaySessionListeners.clear();
    });

    it.each([
        ['analyses', 'QUERY_RESULT_LIMIT_EXCEEDED'],
        ['analyses.elements.id', 'QUERY_RESULT_LIMIT_EXCEEDED'],
        ['$string(analyses)', 'QUERY_RESULT_LIMIT_EXCEEDED'],
        ['$error($string(analyses))', 'QUERY_ERROR_DETAILS_LIMIT_EXCEEDED'],
    ])('bounds component operations and nested workflow failures for %s', async (query, code) => {
        const marker = 'private-analysis-output';
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const elements = Array.from({ length: 51 }, (_, index) => ({
            id: `element-${index}`, labels: [], title: marker.repeat(20),
        }));
        await act(async () => {
            render(<AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="bounded-results"
                data={{ elements }}
            />);
        });
        const directory = createOperationComponentRefDirectory();
        directory.registerComponentRef(chartRef);
        const registry = createAiCommandRegistry({ componentRefs: directory });
        const dispatch = createWorkflowToolDispatcher({ componentRefs: directory });
        const operation = registry.query_lap_analysis_result({ query });
        const termination = new Promise((resolve) => operation.notifyTerminated(resolve));
        const error = await operation.result.catch((failure) => failure);
        expect(error).toBeInstanceOf(analysisResultsQuery.AnalysisResultsQueryError);
        expect(error).toMatchObject({ code });
        expect(error).not.toHaveProperty('data');
        await expect(termination).resolves.toMatchObject({ result: error });
        const frame = buildFormattedToolResultFrame({
            name: 'query_lap_analysis_result', status: 'failed', error: serializeError(normalizeOperationError(error)),
        });
        expect(JSON.stringify(frame)).toContain(code);
        expect(JSON.stringify(frame)).not.toContain(marker);
        expect(Buffer.byteLength(JSON.stringify(error.detail), 'utf8')).toBeLessThanOrEqual(1024);

        const procedure = new ProcedurePlanRunner('procedure-plan', dispatch, undefined, jest.fn());
        const procedureResult = await procedure.createProcedurePlan({ workflow: { name: 'set_procedure_plan',
            goal: 'Query analysis',
            operations: [{ operation: { name: 'query_lap_analysis_result', title: 'Read', arguments: { query } } }],
        } }).result.catch((failure) => failure);
        expect(procedureResult).toBeInstanceOf(Error);
        expect(procedureResult).toMatchObject({
            name: 'ProcedurePlanStepFailedError', cause: { detail: { code } },
        });

        const repeatable = new RepeatablePlanRunner('repeatable-plan', dispatch);
        const repeatableResult = await repeatable.createRepeatablePlan({ workflow: { name: 'create_repeatable_plan',
            goal: 'Query analysis',
            operations: [{ operation: { name: 'query_lap_analysis_result', id: 'read', title: 'Read', arguments: { query } } }],
            stop_when: { tool: { name: 'query_lap_analysis_result', arguments: { query: '1' }  }, operator: 'eq', target: 1 },
        } }).result.catch((failure) => failure);
        expect(repeatableResult).toBeInstanceOf(Error);
        expect(repeatableResult).toMatchObject({ name: 'GoalStepFailedError', cause: { detail: { code } } });
        for (const result of [procedureResult, repeatableResult]) {
            const serialized = JSON.stringify(buildFormattedToolResultFrame({ name: 'workflow', error: serializeError(result) }));
            expect(serialized).toContain(code);
            expect(serialized).not.toContain(marker);
            expect(serialized).not.toContain('"data":');
        }

        // Failure does not remove local data or consume a shared query budget.
        for (let index = 0; index < 2; index += 1) {
            await expect(registry.query_lap_analysis_result({ query: '$count(analyses.elements)' }).result)
                .resolves.toEqual({ status: 'ready', data: 51 });
        }
    });

    it('bounds repeatable plan stop-check errors from real component queries', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        await act(async () => {
            render(<AnalysisResultsChart
                ref={chartRef} name="visualization:analysis-results" id="bounded-stop"
                data={{ elements: [{ id: 'one', labels: [], title: 'private-stop-data'.repeat(1000) }] }}
            />);
        });
        const directory = createOperationComponentRefDirectory();
        directory.registerComponentRef(chartRef);
        const dispatch = createWorkflowToolDispatcher({ componentRefs: directory });
        const runner = new RepeatablePlanRunner('repeatable-plan', dispatch);
        const result = await runner.createRepeatablePlan({ workflow: { name: 'create_repeatable_plan',
            goal: 'Bounded stop',
            operations: [{ operation: { name: 'query_lap_analysis_result', id: 'count', title: 'Count', arguments: { query: '$count(analyses)' } } }],
            stop_when: {
                tool: { name: 'query_lap_analysis_result', arguments: { query: '$error($string(analyses))' }  },
                operator: 'eq', target: 0,
            },
        } }).result.catch((failure) => failure);
        expect(result).toBeInstanceOf(Error);
        expect(result).toMatchObject({
            name: 'GoalStopWhenFailedError',
            message: expect.stringContaining('Query error details exceeded the 1024-byte limit.'),
        });
        const serialized = JSON.stringify(serializeError(result));
        expect(Buffer.byteLength(serialized, 'utf8')).toBeLessThan(2048);
        expect(serialized).not.toContain('private-stop-data');
    });

    it('evaluates JSONata over normalized results and preserves JSON value types', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const view = render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="count-results"
                data={{ elements: [] }}
            />,
        );

        await expect(chartRef.current!.queryLapAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 0,
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: ALL_ANALYSES_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 1,
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: '{"count": $count(analyses.elements)}' }).result).resolves.toEqual({
            status: 'ready',
            data: { count: 0 },
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: '[analyses.elements.id]' }).result).resolves.toEqual({
            status: 'ready',
            data: [],
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: 'analyses.elements[id = "missing"]' }).result).resolves.toEqual({
            status: 'ready',
            data: null,
        });

        view.rerender(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="count-results"
                data={{
                    elements: [
                        { id: 'practice-id', labels: labelRanges('MSP', 'MSP') },
                        { id: 'practice-canonical', labels: labelRanges('Mistake (Practice)') },
                        { id: 'practice-configured', labels: labelRanges('Training Error') },
                        { id: 'racing-id', labels: labelRanges('MSR') },
                        { id: 'racing-canonical', labels: labelRanges('Mistake (Racing)') },
                        { id: 'racing-configured', labels: labelRanges('Race Error') },
                        { id: 'combined', labels: labelRanges('MSP', 'MSR', 'MSP', 'MSR') },
                        { id: 'children-only', labels: labelRanges('MSP1', 'MSR1') },
                        { id: 'unrelated', labels: labelRanges('Expert Adherence') },
                        null,
                        'invalid',
                    ],
                }}
            />,
        );

        await expect(chartRef.current!.queryLapAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 9,
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: MISTAKE_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 7,
        });

        selectView('time-lost-mistakes');

        await expect(chartRef.current!.queryLapAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 9,
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: MISTAKE_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 7,
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: 'result_count' }).result).resolves.toEqual({
            status: 'ready',
            data: null,
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: 'mistake_count' }).result).resolves.toEqual({
            status: 'ready',
            data: null,
        });
    });

    it('queries every retained analysis independently of the active page', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const onSelectPage = jest.fn();
        const pages = [{
            id: 'active-page',
            createdAt: 1,
            baseline: { lap_id: 1, lap_time_ms: 99_000, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'active-mistake', labels: labelRanges('MSP') },
                { id: 'active-unrelated', labels: labelRanges('Telemetry') },
            ],
        }, {
            id: 'latest-page',
            createdAt: 2,
            baseline: { lap_id: 2, lap_time_ms: 98_000, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'latest-one', labels: labelRanges('MSP') },
                { id: 'latest-two', labels: labelRanges('MSR') },
                { id: 'latest-three', labels: labelRanges('Mistake (Practice)') },
            ],
        }];
        const view = render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="active-page-count-results"
                pagination={{
                    pages,
                    activePageId: 'active-page',
                    onSelectPage,
                }}
            />,
        );

        expect(screen.getByRole('button', { name: 'Overall Trends' })).toHaveAttribute('aria-pressed', 'true');
        await expect(chartRef.current!.queryLapAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 5,
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: ALL_ANALYSES_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 2,
        });
        await expect(chartRef.current!.queryLapAnalysisResult({
            query: 'analyses.{"lap_id": baseline.lap_id, "segmentCount": $count(elements)}',
        }).result).resolves.toEqual({
            status: 'ready',
            data: [
                { lap_id: 1, segmentCount: 2 },
                { lap_id: 2, segmentCount: 3 },
            ],
        });
        await expect(chartRef.current!.queryLapAnalysisResult({ query: MISTAKE_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 4,
        });

        view.rerender(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="active-page-count-results"
                pagination={{ pages, activePageId: 'unavailable-page', onSelectPage }}
            />,
        );
        await expect(chartRef.current!.queryLapAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 5,
        });

        view.rerender(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="active-page-count-results"
                pagination={{ pages, activePageId: 'latest-page', onSelectPage }}
            />,
        );
        await expect(chartRef.current!.queryLapAnalysisResult({ query: MISTAKE_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 4,
        });
    });

    it('owns comparison overlay publication, replacement, and replay completion', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="comparison-owner"
                sessionGame="acc"
                pagination={{
                    pages: [{
                        id: 'retained-page',
                        createdAt: 1,
                        baseline: { lap_id: 1, lap_time_ms: 99_000, track: 'Spa', car: 'GT3' },
                        elements: [{
                            id: 'braking-result',
                            labels: labelRanges('MSP', 'MSP1', 'EA', 'EA1', 'RM', 'RM7'),
                            title: 'Late braking',
                            comparison: replayComparisonData(),
                        }],
                    }, {
                        id: 'active-page',
                        createdAt: 2,
                        baseline: { lap_id: 2, lap_time_ms: 98_000, track: 'Spa', car: 'GT3' },
                        elements: [{ id: 'active-result', labels: labelRanges('MSR') }],
                    }],
                    activePageId: 'active-page',
                    onSelectPage: jest.fn(),
                }}
            />,
        );

        expect(() => chartRef.current!.displaySpecificResultInOverlay(
            'active-page',
            'braking-result',
        )).toThrow("Analysis result 'braking-result' was not found");
        const first = chartRef.current!.displaySpecificResultInOverlay(
            'retained-page',
            'braking-result',
        );
        const firstTerminated = new Promise((resolve) => first.notifyTerminated(resolve));
        const firstRef = mockOverlayComponentDirectory.getComponentRefs()[0];
        expect(firstRef.current.getComponentType()).toBe('driver_expert_comparison');
        expect(firstRef.current.getSnapshot()).toEqual({
            title: 'Late braking: Driver vs Expert',
            comparison: replayComparisonData(),
            labelGroups: [
                { category: 'mistakes', subLabels: ['Late turn-in'] },
                { category: 'expert', subLabels: ['Matches expert line'] },
                { category: 'recovery', subLabels: ['Merge back to expert line'] },
            ],
            labelRanges: [
                { category: 'mistakes', label: 'Training Error', startIndex: 0, endIndex: 1 },
                { category: 'mistakes', label: 'Late turn-in', startIndex: 0, endIndex: 1 },
                { category: 'expert', label: 'EA', startIndex: 0, endIndex: 1 },
                { category: 'expert', label: 'Matches expert line', startIndex: 0, endIndex: 1 },
                { category: 'recovery', label: 'RM', startIndex: 0, endIndex: 1 },
                { category: 'recovery', label: 'Merge back to expert line', startIndex: 0, endIndex: 1 },
            ],
            game: 'acc',
        });
        expect(firstRef.current.getOverlayBehavior(firstRef.current.getSnapshot()))
            .toMatchObject({ requestedStatus: 'focus' });

        const voice = { text: 'Late braking.', audioDataUrl: 'data:audio/wav;base64,UklGRg==', durationMs: 8000 };
        (synthesizeTtsPack as jest.Mock).mockResolvedValueOnce([voice]);
        const signal = new AbortController().signal;
        await expect(chartRef.current!.prepareComparisonVoices('retained-page', ['braking-result'], signal))
            .resolves.toEqual({ 'braking-result': 8000 });
        expect(synthesizeTtsPack).toHaveBeenCalledWith([{
            text: 'Late braking. Mistakes: Late turn-in. Expert: Matches expert line. Recovery: Merge back to expert line',
        }], signal);
        expect(mockOverlayComponentDirectory.getComponentRefs()).toHaveLength(1);

        const second = chartRef.current!.displaySpecificResultInOverlay(
            'retained-page',
            'braking-result',
        );
        await expect(first.result).rejects.toThrow('newer graph');
        await expect(firstTerminated).resolves.toMatchObject({ status: 'replaced' });
        expect(mockOverlayComponentDirectory.getComponentRefs()).not.toContain(firstRef);

        const secondRef = mockOverlayComponentDirectory.getComponentRefs()[0];
        expect(JSON.parse(JSON.stringify(secondRef.current.getSnapshot())).voice).toEqual(voice);
        secondRef.current.handleOverlayRendererEvent({
            presentationId: 'analysis-overlay-session',
            componentName: secondRef.current.getComponentName(),
            revision: 1,
            event: 'replay_complete',
        });

        await expect(second.result).resolves.toBe('graph shown');
        expect(mockOverlayComponentDirectory.getComponentRefs()).toHaveLength(0);
    });

    it('owns comparison overlay cancellation when the overlay closes or the task aborts', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="comparison-cancellation"
                data={{
                    elements: [{
                        id: 'corner-result',
                        labels: labelRanges('MSR'),
                        comparison: replayComparisonData(),
                    }],
                }}
            />,
        );

        const closed = chartRef.current!.displaySpecificResultInOverlay(
            'comparison-cancellation',
            'corner-result',
        );
        const closedTermination = new Promise((resolve) => closed.notifyTerminated(resolve));
        mockFloatingChatClosedListener?.();
        await expect(closed.result).rejects.toThrow('floating overlay closed');
        await expect(closedTermination).resolves.toMatchObject({ status: 'cancelled' });

        const abortController = new AbortController();
        const aborted = chartRef.current!.displaySpecificResultInOverlay(
            'comparison-cancellation',
            'corner-result',
            abortController.signal,
        );
        const abortedTermination = new Promise((resolve) => aborted.notifyTerminated(resolve));
        abortController.abort();
        await expect(aborted.result).rejects.toMatchObject({ name: 'AbortError' });
        await expect(abortedTermination).resolves.toMatchObject({ status: 'cancelled' });
        expect(mockOverlayComponentDirectory.getComponentRefs()).toHaveLength(0);

        const directlyAborted = chartRef.current!.displaySpecificResultInOverlay(
            'comparison-cancellation',
            'corner-result',
        );
        const directTermination = new Promise((resolve) => (
            directlyAborted.notifyTerminated(resolve)
        ));
        directlyAborted.abort();
        expect(mockOverlayComponentDirectory.getComponentRefs()).toHaveLength(0);
        await expect(directlyAborted.result).rejects.toMatchObject({ name: 'AbortError' });
        await expect(directTermination).resolves.toMatchObject({ status: 'aborted' });
    });

    it('publishes static backend comparisons and rejects only missing results', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="comparison-validation"
                data={{
                    elements: [{
                        id: 'static-result',
                        labels: labelRanges('MSP'),
                        comparison: comparableData(0.2, 0.3),
                    }, {
                        id: 'replay-result',
                        labels: labelRanges('MSP'),
                        comparison: replayComparisonData(),
                    }],
                }}
            />,
        );

        expect(() => chartRef.current!.displaySpecificResultInOverlay(
            'comparison-validation',
            'missing-result',
        )).toThrow("Analysis result 'missing-result' was not found");
        const staticDisplay = chartRef.current!.displaySpecificResultInOverlay(
            'comparison-validation',
            'static-result',
        );
        const staticRef = mockOverlayComponentDirectory.getComponentRefs()[0];
        expect(staticRef.current.getSnapshot()).toEqual({
            title: 'Driver vs Expert',
            comparison: comparableData(0.2, 0.3),
            labelGroups: [{ category: 'mistakes', subLabels: [] }],
            labelRanges: [{ category: 'mistakes', label: 'Training Error', startIndex: 0, endIndex: 1 }],
        });
        staticRef.current.handleOverlayRendererEvent({
            presentationId: 'analysis-overlay-session',
            componentName: staticRef.current.getComponentName(),
            revision: 1,
            event: 'replay_complete',
        });
        await expect(staticDisplay.result).resolves.toBe('graph shown');

        mockOverlayPresentation = null;
        expect(() => chartRef.current!.displaySpecificResultInOverlay(
            'comparison-validation',
            'replay-result',
        )).toThrow('graph overlay is unavailable');
        expect(mockOverlayComponentDirectory.getComponentRefs()).toHaveLength(0);
    });

    it('treats a zero telemetry lap as retained result 1 and one analyzed lap', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="zero-lap-results"
                showElementId={false}
                pagination={{
                    pages: [{
                        id: 'zero-lap-page',
                        createdAt: 1,
                        baseline: {
                            lap_id: 0,
                            lap_time_ms: 98_000,
                            track: 'Spa',
                            car: 'GT3',
                        },
                        elements: [{ id: 'zero-lap-mistake', labels: labelRanges('MSP'), title: 'Zero lap mistake' }],
                    }],
                    activePageId: 'zero-lap-page',
                    onSelectPage: jest.fn(),
                }}
            />,
        );

        await screen.findByTestId('lap-time-trend-graph');
        expect(screen.queryByTestId('overall-trend-query-error')).not.toBeInTheDocument();
        expect(screen.getByText('1 analyzed lap')).toBeInTheDocument();
        expect(screen.getByTestId('overall-trend-guidance')).toHaveTextContent(
            'Not enough analyzed laps to determine a trend.',
        );
        await expect(chartRef.current!.queryLapAnalysisResult({
            query: ALL_ANALYSES_COUNT_QUERY,
        }).result).resolves.toEqual({ status: 'ready', data: 1 });

        fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
        expect(screen.getByText('Page 1 of 1')).toBeInTheDocument();
        expect(screen.getByText(/Baseline: Spa.*GT3.*Lap 0/)).toBeInTheDocument();
        await waitFor(() => expect(within(screen.getByTestId('analysis-result-zero-lap-mistake'))
            .getByLabelText('Analysis result 1')).toHaveTextContent('1'));
    });

    it.each([
        { requested: undefined },
        { requested: -1 },
        { requested: 0 },
        { requested: 99 },
    ])('applies to the highest retained-array page for fallback request $requested', async ({
        requested,
    }) => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const onSelectPage = jest.fn();
        const pages: AnalysisResultsPaginationPage[] = [{
            id: 'array-page-1',
            createdAt: 999,
            baseline: { lap_id: 1, lap_time_ms: 90_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'first-only', labels: labelRanges('MSP') }],
        }, {
            id: 'array-page-2',
            createdAt: -999,
            baseline: { lap_id: 2, lap_time_ms: 89_000, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'latest-match', labels: labelRanges('MSP') },
                { id: 'latest-other', labels: labelRanges('Telemetry') },
            ],
        }];
        const Harness = () => {
            const [activePageId, setActivePageId] = React.useState('array-page-1');
            return (
                <AnalysisResultsChart
                    ref={chartRef}
                    name="visualization:analysis-results"
                    id="apply-fallback"
                    pagination={{
                        pages,
                        activePageId,
                        onSelectPage: (pageId) => {
                            onSelectPage(pageId);
                            setActivePageId(pageId);
                        },
                    }}
                />
            );
        };
        render(<Harness />);

        const query = 'elements[id = "latest-match"]';
        const operation = chartRef.current!.applyAnalysisResultQuery({
            query,
            ...(requested !== undefined ? { page_number: requested } : {}),
        });
        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'Lap Results' }))
                .toHaveAttribute('aria-pressed', 'true');
            expect(screen.getByText('Page 2 of 2')).toBeInTheDocument();
        });
        let result: unknown;
        await act(async () => {
            result = await operation.result;
        });

        expect(result).toEqual({
            status: 'ready',
        });
        expect(onSelectPage).toHaveBeenCalledWith('array-page-2');
        expect(screen.getByRole('button', { name: 'Lap Results' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByText('Page 2 of 2')).toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Query expression' })).not.toBeInTheDocument();
        await waitFor(() => expect(renderedResultIds()).toEqual(['latest-match']));
    });

    it('applies an explicit displayed page number by retained-array position', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const pages: AnalysisResultsPaginationPage[] = [{
            id: 'displayed-page-1',
            createdAt: 200,
            baseline: { lap_id: 4, lap_time_ms: 90_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'page-one-match', labels: labelRanges('MSP') }],
        }, {
            id: 'displayed-page-2',
            createdAt: 100,
            baseline: { lap_id: 5, lap_time_ms: 89_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'page-two-match', labels: labelRanges('MSP') }],
        }];
        const Harness = () => {
            const [activePageId, setActivePageId] = React.useState('displayed-page-2');
            return (
                <AnalysisResultsChart
                    ref={chartRef}
                    name="visualization:analysis-results"
                    id="apply-explicit"
                    pagination={{ pages, activePageId, onSelectPage: setActivePageId }}
                />
            );
        };
        render(<Harness />);

        const query = 'elements[id = "page-one-match"]';
        const operation = chartRef.current!.applyAnalysisResultQuery({
            query,
            page_number: 1,
        });
        await waitFor(() => expect(screen.getByText('Page 1 of 2')).toBeInTheDocument());
        let result: unknown;
        await act(async () => {
            result = await operation.result;
        });

        expect(result).toEqual({
            status: 'ready',
        });
        expect(screen.getByText('Page 1 of 2')).toBeInTheDocument();
        await waitFor(() => expect(renderedResultIds()).toEqual(['page-one-match']));
    });

    it('treats recorded analysis as one implicit page', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="recorded-apply"
                data={{ elements: [
                    { id: 'recorded-match', labels: labelRanges('MSP') },
                    { id: 'recorded-other', labels: labelRanges('Telemetry') },
                ] }}
            />,
        );

        const query = 'elements[id = "recorded-match"]';
        let result: unknown;
        await act(async () => {
            result = await chartRef.current!.applyAnalysisResultQuery({
                query,
                page_number: 12,
            }).result;
        });

        expect(result).toEqual({
            status: 'ready',
        });
        expect(screen.queryByRole('textbox', { name: 'Query expression' })).not.toBeInTheDocument();
        expect(renderedResultIds()).toEqual(['recorded-match']);
    });

    it('rejects apply when a paginated live chart has no retained pages', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="empty-live-apply"
                pagination={{ pages: [], activePageId: null, onSelectPage: jest.fn() }}
            />,
        );

        await expect(chartRef.current!.applyAnalysisResultQuery({ query: 'elements' }).result)
            .rejects.toMatchObject({
                name: 'VisualizationControlFailedError',
                message: expect.stringContaining('no retained pages'),
            });
    });

    it('keeps successful results after an invalid AI query without exposing query controls', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="invalid-ai-apply"
                data={{ elements: [
                    { id: 'preserved', labels: labelRanges('MSP') },
                    { id: 'excluded', labels: labelRanges('MSP') },
                ] }}
            />,
        );
        await waitFor(() => expect(chartRef.current!.getFilteredSegments().status).toBe('ready'));

        await act(async () => {
            await chartRef.current!.applyAnalysisResultQuery({
                query: 'elements[id = "preserved"]',
                page_number: 1,
            }).result;
        });
        expect(renderedResultIds()).toEqual(['preserved']);

        const invalidQuery = 'elements[';
        let invalidOperation!: ReturnType<AnalysisResultsChartHandle['applyAnalysisResultQuery']>;
        await act(async () => {
            invalidOperation = chartRef.current!.applyAnalysisResultQuery({ query: invalidQuery });
            await expect(invalidOperation.result).rejects.toMatchObject({
                name: 'AnalysisResultsQueryError',
            });
        });

        expect(screen.queryByRole('textbox', { name: 'Query expression' })).not.toBeInTheDocument();
        expect(screen.queryByTestId('active-page-query-error')).not.toBeInTheDocument();
        expect(renderedResultIds()).toEqual(['preserved']);
        expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            committedQuery: 'elements[id = "preserved"]',
            segments: [{ id: 'preserved' }],
        });
    });

    it('rejects a stale page-selection operation when a newer apply replaces it', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const pages: AnalysisResultsPaginationPage[] = [{
            id: 'stale-page-1',
            createdAt: 2,
            baseline: { lap_id: 1, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'newest-wins', labels: labelRanges('MSP') }],
        }, {
            id: 'stale-page-2',
            createdAt: 1,
            baseline: { lap_id: 2, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'stale-result', labels: labelRanges('MSP') }],
        }];
        const onSelectPage = jest.fn();
        const Harness = () => {
            const [activePageId, setActivePageId] = React.useState('stale-page-1');
            return (
                <AnalysisResultsChart
                    ref={chartRef}
                    name="visualization:analysis-results"
                    id="stale-ai-apply"
                    pagination={{
                        pages,
                        activePageId,
                        onSelectPage: (pageId) => {
                            onSelectPage(pageId);
                            if (pageId === 'stale-page-1') setActivePageId(pageId);
                        },
                    }}
                />
            );
        };
        render(<Harness />);

        const stale = chartRef.current!.applyAnalysisResultQuery({
            query: 'elements[id = "stale-result"]',
            page_number: 2,
        });
        const latest = chartRef.current!.applyAnalysisResultQuery({
            query: 'elements[id = "newest-wins"]',
            page_number: 1,
        });

        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'Lap Results' }))
                .toHaveAttribute('aria-pressed', 'true');
            expect(screen.getByText('Page 1 of 2')).toBeInTheDocument();
        });
        await expect(stale.result).rejects.toMatchObject({
            name: 'VisualizationControlFailedError',
            message: expect.stringContaining('newer'),
        });
        let latestResult: unknown;
        await act(async () => { latestResult = await latest.result; });
        expect(latestResult).toEqual({ status: 'ready' });
        expect(renderedResultIds()).toEqual(['newest-wins']);
        expect(onSelectPage).toHaveBeenNthCalledWith(1, 'stale-page-2');
        expect(onSelectPage).toHaveBeenNthCalledWith(2, 'stale-page-1');
    });

    it('keeps page readiness pending until the requested page is committed', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const emptyPagination = {
            pages: [],
            activePageId: null,
            onSelectPage: jest.fn(),
        };
        const view = render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="readiness-results"
                pagination={emptyPagination}
            />,
        );

        let resolved = false;
        const readiness = chartRef.current!.waitForAnalysisResultPage('requested-page');
        void readiness.then(() => { resolved = true; });
        await act(async () => undefined);
        expect(resolved).toBe(false);

        view.rerender(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="readiness-results"
                pagination={{
                    ...emptyPagination,
                    pages: [{
                        id: 'requested-page',
                        createdAt: 1,
                        baseline: {
                            lap_id: 1,
                            lap_time_ms: 100_000,
                            track: 'Spa',
                            car: 'GT3',
                        },
                        elements: [],
                    }],
                }}
            />,
        );

        await act(async () => { await readiness; });
        expect(resolved).toBe(true);
    });

    it('rejects pending page readiness when the chart unmounts', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const view = render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="unmount-readiness-results"
                pagination={{ pages: [], activePageId: null, onSelectPage: jest.fn() }}
            />,
        );
        const readiness = chartRef.current!.waitForAnalysisResultPage('missing-page');
        const rejection = expect(readiness).rejects.toMatchObject({
            name: 'AnalysisResultsVisualizationNotReadyError',
            componentName: 'visualization:analysis-results',
            message: expect.stringContaining('unmounted'),
        });

        view.unmount();
        await rejection;
    });

    it('rejects pending page readiness after the component timeout', async () => {
        jest.useFakeTimers();
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="timeout-readiness-results"
                pagination={{ pages: [], activePageId: null, onSelectPage: jest.fn() }}
            />,
        );
        const readiness = chartRef.current!.waitForAnalysisResultPage('missing-page');
        const rejection = expect(readiness).rejects.toMatchObject({
            name: 'AnalysisResultsVisualizationNotReadyError',
            componentName: 'visualization:analysis-results',
            message: expect.stringContaining('5000ms'),
        });

        act(() => { jest.advanceTimersByTime(5000); });
        await rejection;
        jest.useRealTimers();
    });

    it('keeps Overall Trends separate from retained-array result-page navigation', async () => {
        const chartRef = React.createRef<any>();
        const onUpdate = jest.fn(() => true);
        const onSelectPage = jest.fn();
        const pages = [{
            id: 'page-2',
            createdAt: 2,
            baseline: { lap_id: 7, lap_time_ms: 98_000, track: 'Monza', car: 'GT4' },
            elements: [{ id: 'second-page-result', labels: labelRanges('MSP', 'MSP2'), title: 'Second page mistake' }],
        }, {
            id: 'page-1',
            createdAt: 1,
            baseline: { lap_id: 4, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'first-page-result', labels: labelRanges('MSP', 'MSP1'), title: 'First page mistake' }],
        }];

        const PagingHarness = () => {
            const [activePageId, setActivePageId] = React.useState('page-2');
            const selectPage = (pageId: string) => {
                onSelectPage(pageId);
                setActivePageId(pageId);
            };
            return (
                <AnalysisResultsChart
                    ref={chartRef}
                    name="visualization:analysis-results"
                    id="paged-results"
                    pagination={{ pages, activePageId, onSelectPage: selectPage }}
                    onUpdate={onUpdate}
                />
            );
        };

        const { unmount } = render(<PagingHarness />);

        expect(screen.queryByText(/^Page \d+ of \d+$/)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Overall Trends' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('button', { name: 'Lap Results' })).toHaveAttribute('aria-pressed', 'false');
        expect(screen.getByText('Lap Time Improvement')).toBeInTheDocument();
        expect(screen.getByText('Overall Mistake Trend')).toBeInTheDocument();
        expect(screen.queryByRole('region', { name: 'Edit query' })).not.toBeInTheDocument();
        await waitFor(() => (
            expect(screen.getByText('2 analyzed laps')).toBeInTheDocument()
        ));
        await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 7', totalCount: 1 },
                { analysis: 'Analysis 2 · Lap 4', totalCount: 1 },
            ]));
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toHaveValue('MSP1');
        fireEvent.change(screen.getByRole('combobox', { name: 'Specific mistake' }), {
            target: { value: 'MSP2' },
        });
        expect(screen.queryByRole('button', { name: 'Previous' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Next' })).not.toBeInTheDocument();
        expect(screen.queryByTestId('analysis-result-first-page-result')).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));

        expect(onSelectPage).not.toHaveBeenCalled();
        expect(screen.getByText('Page 1 of 2')).toBeInTheDocument();
        expect(screen.getByText(/Baseline: Monza.*GT4.*Lap 7/)).toBeInTheDocument();
        expect(screen.queryByRole('region', { name: 'Edit query' })).not.toBeInTheDocument();
        await waitFor(() => expect(screen.getByTestId('analysis-result-second-page-result'))
            .toHaveTextContent('Second page mistake'));
        expect(screen.queryByTestId('analysis-result-first-page-result')).not.toBeInTheDocument();
        expect(renderedFrequencyData()).toEqual([{ label: 'Wheel lock', occurrences: 1 }]);
        expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled();

        fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        expect(onSelectPage).toHaveBeenLastCalledWith('page-1');
        expect(screen.getByText('Page 2 of 2')).toBeInTheDocument();
        expect(screen.getByText(/Baseline: Spa.*GT3.*Lap 4/)).toBeInTheDocument();
        await waitFor(() => expect(screen.getByTestId('analysis-result-first-page-result'))
            .toHaveTextContent('First page mistake'));
        expect(renderedFrequencyData()).toEqual([{ label: 'Late turn-in', occurrences: 1 }]);
        expect(screen.getByRole('button', { name: 'Previous' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();

        fireEvent.click(screen.getByRole('button', { name: 'Overall Trends' }));
        expect(screen.queryByText(/^Page \d+ of \d+$/)).not.toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toHaveValue('MSP2');
        expect(onSelectPage).toHaveBeenCalledTimes(1);

        fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
        expect(screen.getByText('Page 2 of 2')).toBeInTheDocument();
        expect(onSelectPage).toHaveBeenCalledTimes(1);
        fireEvent.click(screen.getByRole('button', { name: 'Previous' }));

        expect(onSelectPage).toHaveBeenLastCalledWith('page-2');
        expect(screen.getByText('Page 1 of 2')).toBeInTheDocument();
        expect(screen.getByText(/Baseline: Monza.*GT4.*Lap 7/)).toBeInTheDocument();
        await waitFor(() => expect(screen.getByTestId('analysis-result-second-page-result'))
            .toHaveTextContent('Second page mistake'));
        expect(screen.queryByTestId('analysis-result-first-page-result')).not.toBeInTheDocument();
        expect(renderedFrequencyData()).toEqual([{ label: 'Wheel lock', occurrences: 1 }]);
        expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled();

        act(() => {
            chartRef.current.appendAnalysisResult({ id: 'active-only', labels: labelRanges('MSP') });
        });
        expect(onUpdate).toHaveBeenCalledWith({
            elements: [
                expect.objectContaining({ id: 'second-page-result' }),
                expect.objectContaining({ id: 'active-only' }),
            ],
        });

        unmount();
        render(<PagingHarness />);
        expect(screen.queryByText(/^Page \d+ of \d+$/)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Overall Trends' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByText('Overall Mistake Trend')).toBeInTheDocument();
    });

    it.each(['filtered-first', null, 'removed-page'])('exposes the selected/latest page filter in either view when activePageId is %s', async (initialPageId) => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const pages: AnalysisResultsPaginationPage[] = [{
            id: 'filtered-first',
            createdAt: 1,
            baseline: { lap_id: 1, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
            elements: [{
                id: 'early',
                labels: labelRanges('MSP'),
                normalizedPositionRange: { start: 0.2, end: 0.25 },
                comparison: comparableData(0.2, 0.4),
            }, {
                id: 'late',
                labels: labelRanges('Telemetry'),
                normalizedPositionRange: { start: 0.8, end: 0.85 },
            }],
        }, {
            id: 'filtered-second',
            createdAt: 2,
            baseline: { lap_id: 2, lap_time_ms: 99_000, track: 'Spa', car: 'GT3' },
            elements: [{
                id: 'second-page-only',
                labels: labelRanges('MSR'),
                normalizedPositionRange: { start: 0.5, end: 0.55 },
            }],
        }];
        const Harness = () => {
            const [activePageId, setActivePageId] = React.useState<string | null>(initialPageId);
            return (
                <AnalysisResultsChart
                    ref={chartRef}
                    name="visualization:analysis-results"
                    id="filtered-snapshot"
                    pagination={{ pages, activePageId, onSelectPage: setActivePageId }}
                />
            );
        };
        render(<Harness />);

        const expectedInitialPage = initialPageId === 'filtered-first' ? pages[0] : pages[1];
        expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            status: 'busy',
            activePageId: expectedInitialPage.id,
            segments: [],
        });

        await waitFor(() => expect(chartRef.current!.getFilteredSegments().status).toBe('ready'));
        expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            activePageId: expectedInitialPage.id,
            appliedView: 'mistakes',
            segments: [{ id: expectedInitialPage.elements[0].id }],
        });
        expect(screen.getByRole('button', { name: 'Overall Trends' })).toHaveAttribute('aria-pressed', 'true');

        fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
        selectView('all-results');
        await waitFor(() => expect(chartRef.current!.getFilteredSegments().appliedView)
            .toBe('all-results'));
        const applyOperation = chartRef.current!.applyAnalysisResultQuery({
            query: 'elements^(>normalizedPositionRange.start)',
            page_number: 1,
        });
        await waitFor(() => expect(chartRef.current!.getFilteredSegments().activePageId).toBe('filtered-first'));
        await act(async () => { await applyOperation.result; });
        await waitFor(() => expect(chartRef.current!.getFilteredSegments().appliedView).toBe('custom'));
        const custom = chartRef.current!.getFilteredSegments();
        expect(custom).toMatchObject({
            status: 'ready',
            activePageId: 'filtered-first',
            appliedView: 'custom',
            committedQuery: 'elements^(>normalizedPositionRange.start)',
        });
        expect(custom.segments.map(({ id }) => id)).toEqual(['late', 'early']);
        expect(Object.isFrozen(custom)).toBe(true);
        expect(Object.isFrozen(custom.segments)).toBe(true);
        expect(Object.isFrozen(custom.segments[1].comparison?.samples)).toBe(true);

        fireEvent.click(screen.getByRole('button', { name: 'Overall Trends' }));
        expect(chartRef.current!.getFilteredSegments()).toEqual(custom);
        fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));

        fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        await waitFor(() => expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            status: 'ready',
            activePageId: 'filtered-second',
            appliedView: 'custom',
            segments: [{ id: 'second-page-only' }],
        }));

        await act(async () => {
            await chartRef.current!.applyAnalysisResultQuery({
                query: 'elements[id = "missing"]',
                page_number: 2,
            }).result;
        });
        await waitFor(() => expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            status: 'empty',
            activePageId: 'filtered-second',
            appliedView: 'custom',
            committedQuery: 'elements[id = "missing"]',
            segments: [],
        }));
    });

    it.each([null, 'removed-page', 'older-page'])('queues the selected/latest lap while Overall Trends is open and activePageId is %s', async (activePageId) => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const pages: AnalysisResultsPaginationPage[] = [{
            id: 'older-page',
            createdAt: 0,
            baseline: { lap_id: 1, lap_time_ms: 101_000, track: 'Spa', car: 'GT3' },
            elements: [{
                id: 'older-mistake',
                labels: labelRanges('MSP'),
                normalizedPositionRange: { start: 0.2, end: 0.3 },
                comparison: replayComparisonData(),
            }],
        }, {
            id: 'default-page',
            createdAt: 1,
            baseline: { lap_id: 2, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
            elements: [{
                id: 'default-mistake',
                labels: labelRanges('MSP'),
                normalizedPositionRange: { start: 0.2, end: 0.3 },
                comparison: replayComparisonData(),
            }, {
                id: 'excluded-by-filter',
                labels: labelRanges('Telemetry'),
                normalizedPositionRange: { start: 0.4, end: 0.5 },
                comparison: replayComparisonData(),
            }],
        }];
        render(<AnalysisResultsChart
            ref={chartRef}
            name="visualization:analysis-results"
            id="default-page-queue"
            sessionGame="acc"
            pagination={{ pages, activePageId, onSelectPage: jest.fn() }}
        />);
        await waitFor(() => expect(chartRef.current!.getFilteredSegments().status).toBe('ready'));
        const expectedPage = activePageId === 'older-page' ? pages[0] : pages[1];
        const expectedResultId = expectedPage.elements[0].id;
        const prepareVoices = jest.spyOn(chartRef.current!, 'prepareComparisonVoices')
            .mockResolvedValue({ 'older-mistake': 8_000, 'default-mistake': 8_000 });
        const directory = createOperationComponentRefDirectory();
        const runner = new LiveRangeTodoListRunner('live-range-todo-list');
        directory.registerComponentRef(chartRef);
        directory.registerComponentRef({ current: runner });
        const workflowPanel = {
            getComponentName: () => 'workflow-panel',
            appendLiveRangeTodoList: runner.appendLiveRangeTodoList.bind(runner),
        };
        directory.registerComponentRef({ current: workflowPanel });
        const registry = createAiCommandRegistry({ componentRefs: directory, sessionMode: 'live', sessionGame: 'acc' });

        try {
            await expect(registry.add_analysis_result_to_do_list({}).result)
                .resolves.toMatchObject({
                    status: 'ready',
                    active_page_id: expectedPage.id,
                    applied_view: 'mistakes',
                    matched_count: 1,
                    queued_count: 1,
                });
            expect(prepareVoices).toHaveBeenCalledWith(expectedPage.id, [expectedResultId], expect.any(AbortSignal));
            expect(runner.get().todo_list?.events.map(({ id }) => id)).toEqual([`analysis-comparison:${expectedResultId}`]);
            expect(screen.getByRole('button', { name: 'Overall Trends' })).toHaveAttribute('aria-pressed', 'true');
        } finally {
            prepareVoices.mockRestore();
            runner.dispose();
        }
    });

    it('returns an empty filtered snapshot when no retained pages exist', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        await act(async () => {
            render(
                <AnalysisResultsChart
                    ref={chartRef}
                    name="visualization:analysis-results"
                    id="no-retained-pages"
                    pagination={{ pages: [], activePageId: null, onSelectPage: jest.fn() }}
                />,
            );
        });

        expect(chartRef.current!.getFilteredSegments()).toEqual({
            status: 'empty',
            activePageId: null,
            appliedView: null,
            committedQuery: null,
            segments: [],
        });
    });

    it('reports a busy filtered snapshot while the active filter is evaluating', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="busy-filtered-snapshot"
                data={{ elements: [{ id: 'one', labels: labelRanges('MSP') }] }}
            />,
        );

        expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            status: 'busy',
            segments: [],
        });
        await waitFor(() => expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            status: 'ready',
            activePageId: 'busy-filtered-snapshot',
            segments: [{ id: 'one' }],
        }));
    });

    it.each([
        { name: 'decreasing', values: [5, 4, 3, 2], direction: 'decreasing', slope: -1 },
        { name: 'increasing', values: [1, 2, 4, 5], direction: 'increasing', slope: 1.4 },
        { name: 'flat', values: [3, 3, 3], direction: 'stable', slope: 0 },
        { name: 'noisy but level', values: [2, 5, 2], direction: 'stable', slope: 0 },
        { name: 'single-page', values: [3], direction: 'insufficient', slope: null },
        { name: 'empty', values: [], direction: 'insufficient', slope: null },
    ])('calculates a $name best-fit trend', ({ values, direction, slope }) => {
        expect(getMistakeTrendDirection(values)).toBe(direction);
        if (slope === null) {
            expect(calculateLeastSquaresSlope(values)).toBeNull();
        } else {
            expect(calculateLeastSquaresSlope(values)).toBeCloseTo(slope);
        }
    });

    it('preserves validated lap times and missing-row gaps while fitting the query chronology', async () => {
        const pages = [{
            id: 'latest',
            createdAt: 400,
            baseline: { lap_id: 21, lap_time_ms: 95_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'latest-mistake', labels: labelRanges('MSP', 'MSP1') }],
        }, {
            id: 'first',
            createdAt: 100,
            baseline: { lap_id: 3, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'first-mistake', labels: labelRanges('MSP', 'MSP1') }],
        }, {
            id: 'missing',
            createdAt: 300,
            baseline: { lap_id: 14, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [],
        }, {
            id: 'slower',
            createdAt: 200,
            baseline: { lap_id: 9, lap_time_ms: 105_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'slower-mistake', labels: labelRanges('MSR', 'MSR1') }],
        }];
        const model = buildLapTimeTrendData([
            {
                pageId: 'first',
                label: 'Analysis 1 · Lap 3',
                lap_id: 3,
                lapTimeMs: 100_000,
                totalCount: 0,
                categoryCounts: [],
            },
            {
                pageId: 'slower',
                label: 'Analysis 2 · Lap 9',
                lap_id: 9,
                lapTimeMs: 105_000,
                totalCount: 0,
                categoryCounts: [],
            },
            {
                pageId: 'missing',
                label: 'Analysis 3 · Lap 14',
                lap_id: 14,
                lapTimeMs: null,
                totalCount: 0,
                categoryCounts: [],
            },
            {
                pageId: 'latest',
                label: 'Analysis 4 · Lap 21',
                lap_id: 21,
                lapTimeMs: 95_000,
                totalCount: 0,
                categoryCounts: [],
            },
        ]);

        expect(model.laps.map(({ pageId, lapTimeMs }) => ({ pageId, lapTimeMs }))).toEqual([
            { pageId: 'first', lapTimeMs: 100_000 },
            { pageId: 'slower', lapTimeMs: 105_000 },
            { pageId: 'missing', lapTimeMs: null },
            { pageId: 'latest', lapTimeMs: 95_000 },
        ]);
        expect(model.slopeMsPerAnalysis).toBeCloseTo(-2_142.857, 3);
        expect(model.direction).toBe('improving');
        expect(formatRacingTime(95_001)).toBe('1:35.001');

        render(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="lap-time-trend"
                pagination={{ pages, activePageId: 'latest', onSelectPage: jest.fn() }}
            />,
        );

        await waitFor(() => expect(renderedLapTimeTrendData()).toHaveLength(4));
        const graphData = renderedLapTimeTrendData();
        expect(graphData.map(({ lapTimeSeconds }: any) => lapTimeSeconds)).toEqual([95, 100, null, 105]);
        expect(graphData.every(({ bestFitSeconds }: any) => Number.isFinite(bestFitSeconds))).toBe(true);
        expect(screen.getByText('Lap time by analyzed lap (lower is faster).')).toBeInTheDocument();
        expect(screen.getByTestId('lap-time-trend-status')).toHaveTextContent(
            'Latest lap time: 1:45.000. Versus previous timed lap: 0:05.000 slower. '
            + 'Versus first timed lap: 0:10.000 slower. Overall direction: regressing.',
        );

        const beforeFilterChange = renderedLapTimeTrendData();
        selectTrendParent('MSR');
        await waitFor(() => expect(renderedLapTimeTrendData()).toEqual(beforeFilterChange));
    });

    it('graphs validated totals and category occurrences across zero-filled laps', async () => {
        const pages = [{
            id: 'later-page',
            createdAt: 200,
            baseline: { lap_id: 12, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'b-wheel-1', labels: labelRanges('MSP', 'MSP2') },
                { id: 'b-wheel-2', labels: labelRanges('Mistake (Practice)', 'Wheel lock') },
                { id: 'b-racing', labels: labelRanges('MSR', 'MSR1') },
            ],
        }, {
            id: 'first-page',
            createdAt: 100,
            baseline: { lap_id: 8, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'a-late', labels: labelRanges('MSP', 'MSP1', 'Late turn-in', 'MSP1') },
                { id: 'a-wheel', labels: labelRanges('MSP', 'MSP2') },
                { id: 'a-parent-only', labels: labelRanges('Mistake (Practice)') },
                { id: 'a-child-without-parent', labels: labelRanges('MSP1') },
            ],
        }, {
            id: 'last-page',
            createdAt: 300,
            baseline: { lap_id: 15, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'c-wheel', labels: labelRanges('MSP', 'MSP2', 'Wheel lock') },
                { id: 'c-racing-1', labels: labelRanges('Mistake (Racing)', 'MSR1') },
                { id: 'c-racing-2', labels: labelRanges('MSR', 'Failed overtake attempt', 'MSR1') },
            ],
        }];

        render(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="trend-counts"
                pagination={{ pages, activePageId: 'later-page', onSelectPage: jest.fn() }}
            />,
        );

        await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 12', totalCount: 2 },
                { analysis: 'Analysis 2 · Lap 8', totalCount: 3 },
                { analysis: 'Analysis 3 · Lap 15', totalCount: 1 },
            ]));
        expect(screen.getByTestId('overall-total-trend-status')).toHaveTextContent(
            'Latest: 1 recognized mistake element. Trending downward — fewer mistakes.',
        );
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toHaveValue('MSP2');
        expect(renderedTrendData('specific-mistake-trend-graph')).toEqual([
            { analysis: 'Analysis 1 · Lap 12', specificCount: 2 },
            { analysis: 'Analysis 2 · Lap 8', specificCount: 1 },
            { analysis: 'Analysis 3 · Lap 15', specificCount: 1 },
        ]);
        expect(screen.getByTestId('specific-mistake-trend-status')).toHaveTextContent(
            'Trending downward — fewer mistakes.',
        );

        fireEvent.change(screen.getByRole('combobox', { name: 'Specific mistake' }), {
            target: { value: 'MSP1' },
        });
        expect(renderedTrendData('specific-mistake-trend-graph')).toEqual([
            { analysis: 'Analysis 1 · Lap 12', specificCount: 0 },
            { analysis: 'Analysis 2 · Lap 8', specificCount: 1 },
            { analysis: 'Analysis 3 · Lap 15', specificCount: 0 },
        ]);

        selectTrendParent('MSR');
        await waitFor(() => expect(screen.getByRole('combobox', { name: 'Specific mistake' }))
            .toHaveValue('MSR1'));
        expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 12', totalCount: 1 },
                { analysis: 'Analysis 2 · Lap 8', totalCount: 0 },
                { analysis: 'Analysis 3 · Lap 15', totalCount: 2 },
            ]);
        expect(screen.getByTestId('overall-total-trend-status')).toHaveTextContent(
            'Trending upward — more mistakes.',
        );
    });

    it('re-evaluates Overall Trends after successful taxonomy and retained-page refreshes', async () => {
        const originalGetLabelName = mockGetLabelName;
        let pages: AnalysisResultsPaginationPage[] = [{
            id: 'taxonomy-page',
            createdAt: 1,
            baseline: { lap_id: 1, lap_time_ms: 90_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'fresh-labels', labels: labelRanges('Fresh Training', 'Fresh brake') }],
        }];
        const renderChart = () => (
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="trend-refresh"
                pagination={{ pages, activePageId: pages[0]?.id ?? null, onSelectPage: jest.fn() }}
            />
        );
        const view = render(renderChart());

        try {
            await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 1', totalCount: 0 },
            ]));

            mockGetLabelName = (labelId) => ({
                ...mockLabelNames,
                MSP: 'Fresh Training',
                MSP1: 'Fresh brake',
            }[labelId] ?? originalGetLabelName(labelId));
            view.rerender(renderChart());

            await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 1', totalCount: 1 },
            ]));
            expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toHaveValue('MSP1');

            pages = [...pages, {
                id: 'retained-page',
                createdAt: 2,
                baseline: { lap_id: 2, lap_time_ms: 89_000, track: 'Spa', car: 'GT3' },
                elements: [{ id: 'retained-mistake', labels: labelRanges('MSP', 'MSP1') }],
            }];
            view.rerender(renderChart());

            await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 1', totalCount: 1 },
                { analysis: 'Analysis 2 · Lap 2', totalCount: 1 },
            ]));
            expect(renderedTrendData('specific-mistake-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 1', specificCount: 1 },
                { analysis: 'Analysis 2 · Lap 2', specificCount: 1 },
            ]);
            expect(renderedLapTimeTrendData()).toHaveLength(2);
        } finally {
            view.unmount();
            mockGetLabelName = originalGetLabelName;
        }
    });

    it('ignores a stale trend evaluation after a newer Training selection commits', async () => {
        const pages = [{
            id: 'trend-generation',
            createdAt: 1,
            baseline: { lap_id: 1, lap_time_ms: 90_000, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'training', labels: labelRanges('MSP', 'MSP1') },
                { id: 'racing-one', labels: labelRanges('MSR', 'MSR1') },
                { id: 'racing-two', labels: labelRanges('MSR', 'MSR2') },
            ],
        }];
        render(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="stale-trend-generation"
                pagination={{ pages, activePageId: 'trend-generation', onSelectPage: jest.fn() }}
            />,
        );
        await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toEqual([
            { analysis: 'Analysis 1 · Lap 1', totalCount: 1 },
        ]));

        const realEvaluator = analysisResultsQuery.evaluateAnalysisResultsQuery;
        const trendCompletions: Array<() => Promise<void>> = [];
        const evaluator = jest.spyOn(analysisResultsQuery, 'evaluateAnalysisResultsQuery')
            .mockImplementation((expression, input) => {
                if (!input || typeof input !== 'object' || !('pages' in input)) {
                    return realEvaluator(expression, input);
                }
                return new Promise((resolve, reject) => {
                    trendCompletions.push(async () => {
                        try {
                            resolve(await realEvaluator(expression, input));
                        } catch (error) {
                            reject(error);
                        }
                    });
                });
            });

        try {
            selectTrendParent('MSR');
            await waitFor(() => expect(trendCompletions).toHaveLength(1));
            selectTrendParent('MSP');
            await waitFor(() => expect(trendCompletions).toHaveLength(2));

            await act(async () => trendCompletions[1]());
            await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 1', totalCount: 1 },
            ]));

            await act(async () => trendCompletions[0]());
            expect(screen.getByRole('combobox', { name: 'Showing' })).toHaveValue('MSP');
            expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 1', totalCount: 1 },
            ]);
        } finally {
            evaluator.mockRestore();
        }
    });

    it('keeps the trend parent and active-page View state independent', async () => {
        const pages = [{
            id: 'independent-state',
            createdAt: 1,
            baseline: { lap_id: 1, lap_time_ms: 90_000, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'training', labels: labelRanges('MSP', 'MSP1') },
                { id: 'racing-one', labels: labelRanges('MSR', 'MSR1') },
                { id: 'racing-two', labels: labelRanges('MSR', 'MSR2') },
            ],
        }];
        render(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="independent-query-state"
                pagination={{ pages, activePageId: 'independent-state', onSelectPage: jest.fn() }}
            />,
        );
        await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toHaveLength(1));

        const realEvaluator = analysisResultsQuery.evaluateAnalysisResultsQuery;
        const evaluator = jest.spyOn(analysisResultsQuery, 'evaluateAnalysisResultsQuery')
            .mockImplementation(realEvaluator);

        try {
            fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
            selectView('all-results');
            await waitFor(() => expect(renderedResultIds()).toEqual([
                'training',
                'racing-one',
                'racing-two',
            ]));
            expect(evaluator.mock.calls.filter(([, input]) => (
                input && typeof input === 'object' && 'pages' in input
            ))).toHaveLength(0);

            fireEvent.click(screen.getByRole('button', { name: 'Overall Trends' }));
            selectTrendParent('MSR');
            await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toEqual([
                { analysis: 'Analysis 1 · Lap 1', totalCount: 2 },
            ]));

            fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
            expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('all-results');
        } finally {
            evaluator.mockRestore();
        }
    });

    it('fails closed with one actionable error when a new page generation is invalid', async () => {
        const initialPages = [{
            id: 'old-page',
            createdAt: 1,
            baseline: { lap_id: 1, lap_time_ms: 90_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'old-mistake', labels: labelRanges('MSP', 'MSP1') }],
        }];
        const renderChart = (pages: typeof initialPages) => (
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="invalid-trend-generation"
                pagination={{ pages, activePageId: pages[0]?.id ?? null, onSelectPage: jest.fn() }}
            />
        );
        const view = render(renderChart(initialPages));
        await waitFor(() => expect(renderedTrendData('overall-total-trend-graph')).toHaveLength(1));

        const realEvaluator = analysisResultsQuery.evaluateAnalysisResultsQuery;
        const evaluator = jest.spyOn(analysisResultsQuery, 'evaluateAnalysisResultsQuery')
            .mockImplementation((expression, input) => (
                input && typeof input === 'object' && 'pages' in input
                    ? Promise.resolve({ laps: [], categories: [] })
                    : realEvaluator(expression, input)
            ));

        try {
            view.rerender(renderChart([{
                id: 'new-page',
                createdAt: 2,
                baseline: { lap_id: 2, lap_time_ms: 89_000, track: 'Spa', car: 'GT3' },
                elements: [{ id: 'new-mistake', labels: labelRanges('MSP', 'MSP2') }],
            }]));

            const diagnostic = await screen.findByTestId('overall-trend-query-error');
            expect(diagnostic).toHaveTextContent('INVALID_OVERALL_TREND_QUERY_RESULT');
            expect(screen.getAllByRole('alert')).toEqual([diagnostic]);
            expect(renderedTrendData('overall-total-trend-graph')).toEqual([]);
            expect(renderedTrendData('specific-mistake-trend-graph')).toEqual([]);
            expect(renderedLapTimeTrendData()).toEqual([]);
        } finally {
            evaluator.mockRestore();
        }
    });

    it('shows clear empty and single-page trend guidance', async () => {
        const { rerender } = render(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="empty-trend"
                pagination={{ pages: [], activePageId: null, onSelectPage: jest.fn() }}
            />,
        );

        expect(screen.queryByText(/^Page \d+ of \d+$/)).not.toBeInTheDocument();
        expect(screen.getByTestId('overall-trend-guidance')).toHaveTextContent(
            'No analyzed laps yet. Analyze at least two baseline laps to see a trend.',
        );
        expect(renderedLapTimeTrendData()).toEqual([]);
        expect(screen.getByTestId('lap-time-trend-status')).toHaveTextContent(
            'Latest lap time unavailable',
        );
        expect(screen.getByRole('button', { name: 'Lap Results' })).toBeDisabled();
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toBeDisabled();

        rerender(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="empty-trend"
                pagination={{
                    pages: [{
                        id: 'only-page',
                        createdAt: 1,
                        baseline: { lap_id: 6, lap_time_ms: 98_567, track: 'Spa', car: 'GT3' },
                        elements: [{ id: 'only-result', labels: labelRanges('MSP', 'MSP1') }],
                    }],
                    activePageId: 'only-page',
                    onSelectPage: jest.fn(),
                }}
            />,
        );

        expect(screen.queryByText(/^Page \d+ of \d+$/)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Lap Results' })).toBeEnabled();
        await waitFor(() => expect(screen.getByTestId('overall-trend-guidance')).toHaveTextContent(
            'Not enough analyzed laps to determine a trend.',
        ));
        await waitFor(() => expect(screen.getByTestId('overall-total-trend-status')).toHaveTextContent(
            'Latest: 1 recognized mistake element. Not enough analyzed laps to determine a trend.',
        ));
        expect(renderedLapTimeTrendData()).toEqual([{
            analysis: expect.any(String),
            lapTimeSeconds: 98.567,
            bestFitSeconds: null,
        }]);
        expect(screen.getByTestId('lap-time-trend-status')).toHaveTextContent(
            'Latest lap time: 1:38.567. Versus previous timed lap: unavailable. '
            + 'Versus first timed lap: unchanged. Overall direction: not enough timed laps.',
        );
    });

    it('renders arbitrary labels, context, and metadata safely', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="results"
                data={{
                    elements: [{
                        id: 'future-1',
                        labels: labelRanges('Mistake (Practice)', 'Future category', 'Recovery'),
                        title: 'Generated form result',
                        section: 'Turn 4',
                        normalizedPositionRange: { start: 0.2, end: 0.35 },
                        timeGap: { deltaMs: 125 },
                        metadata: {
                            source: 'hidden-source-value',
                            start_index: 12345,
                            end_index: 67890,
                            nested: { safe: true },
                            score: 0.95,
                        },
                    }],
                }}
            />,
        );

        await waitFor(() => expect(screen.getByText('1 of 1 total')).toBeInTheDocument());
        expect(screen.getByText('Future category')).toBeInTheDocument();
        expect(screen.getByText('Recovery')).toBeInTheDocument();
        fireEvent.click(within(screen.getByTestId('analysis-result-future-1')).getByRole('button'));
        const details = screen.getByRole('region', { name: 'Section details' });
        expect(details).toHaveTextContent('Track position20.0% – 35.0%');
        expect(screen.getByText('nested: {"safe":true}')).toBeInTheDocument();
        expect(screen.getByText('score: 0.95')).toBeInTheDocument();
        expect(screen.queryByText(/source|hidden-source-value/)).not.toBeInTheDocument();
        expect(screen.queryByText(/start_index|12345/)).not.toBeInTheDocument();
        expect(screen.queryByText(/end_index|67890/)).not.toBeInTheDocument();
    });

    it('defaults to Mistakes and exposes preset views without an editable query field', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="default-filter"
                data={{
                    elements: [
                        { id: 'practice-id', labels: labelRanges('MSP') },
                        { id: 'practice-name', labels: labelRanges('Mistake (Practice)') },
                        { id: 'racing-id', labels: labelRanges('MSR') },
                        { id: 'racing-name', labels: labelRanges('Mistake (Racing)') },
                        { id: 'unrelated', labels: labelRanges('Telemetry') },
                        { id: 'unlabeled', labels: [] },
                    ],
                }}
            />,
        );

        const viewSelect = screen.getByRole('combobox', { name: 'View' });
        expect(viewSelect).toHaveValue('mistakes');
        expect(within(viewSelect).getAllByRole('option').map((option) => option.textContent)).toEqual([
            'All results',
            'Mistakes',
            'Most common label in mistakes',
            'Most time lost in mistakes',
            'Custom',
        ]);
        expect(screen.queryByRole('combobox', { name: 'Sort by' })).not.toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: 'Showing' })).not.toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Query expression' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Apply' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Reset' })).not.toBeInTheDocument();
        await waitFor(() => {
            const queryError = screen.queryByTestId('active-page-query-error');
            if (queryError) throw new Error(queryError.textContent ?? 'Query evaluation failed.');
            expect(renderedResultIds()).toEqual([
                'practice-id',
                'practice-name',
                'racing-id',
                'racing-name',
            ]);
        });
        expect(screen.getByText('4 of 6 total')).toBeInTheDocument();

        selectView('all-results');

        await waitFor(() => expect(renderedResultIds()).toEqual([
            'practice-id',
            'practice-name',
            'racing-id',
            'racing-name',
            'unrelated',
            'unlabeled',
        ]));
        expect(screen.getByText('6 of 6 total')).toBeInTheDocument();
    });

    it('shows a query-aware empty state when the selected view has no matches', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="empty"
                data={{ elements: [{ id: 'telemetry', labels: labelRanges('Telemetry') }] }}
            />,
        );

        await waitFor(() => expect(screen.getByText('0 of 1 total')).toBeInTheDocument());
        expect(renderedResultIds()).toEqual([]);
        expect(screen.getByTestId('analysis-results-empty-state')).toHaveTextContent(
            'No results match the Mistakes view.',
        );
    });

    it('applies each selected template as one filtering-and-ordering expression', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="source-order"
                data={{
                    elements: [
                        { id: 'third-fastest', labels: labelRanges('MSP', 'Lockup'), timeGap: { deltaMs: 20 } },
                        { id: 'racing', labels: labelRanges('MSR', 'Wide exit'), timeGap: { deltaMs: 80 } },
                        { id: 'least-time', labels: labelRanges('Mistake (Practice)', 'Lockup'), timeGap: { deltaMs: 5 } },
                    ],
                }}
            />,
        );

        await waitFor(() => expect(renderedResultIds()).toEqual([
            'third-fastest',
            'racing',
            'least-time',
        ]));

        selectView('time-lost-mistakes');

        await waitFor(() => expect(renderedResultIds()).toEqual([
            'racing',
            'third-fastest',
            'least-time',
        ]));
        expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('time-lost-mistakes');
    });

    it('keeps custom queries available to the programmatic API without exposing manual editing', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart ref={chartRef} name="visualization:analysis-results"
                id="dynamic-sort-name"
                data={{
                    elements: [
                        { id: 'practice', labels: labelRanges('MSP', 'MSP1') },
                        { id: 'racing', labels: labelRanges('MSR', 'MSR1') },
                    ],
                }}
            />,
        );

        await waitFor(() => expect(renderedResultIds()).toEqual(['practice', 'racing']));
        expect(screen.queryByRole('textbox', { name: 'Query expression' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Apply' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Reset' })).not.toBeInTheDocument();

        await act(async () => {
            await chartRef.current!.applyAnalysisResultQuery({
                query: 'elements[id = "racing"]',
            }).result;
        });
        await waitFor(() => expect(renderedResultIds()).toEqual(['racing']));
        expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('custom');
    });

    it('preserves the last valid matches after a failed programmatic query', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="invalid-custom-query"
                data={{ elements: [{ id: 'mistake', labels: labelRanges('MSP') }] }}
            />,
        );
        await waitFor(() => expect(renderedResultIds()).toEqual(['mistake']));

        await act(async () => {
            await expect(chartRef.current!.applyAnalysisResultQuery({ query: '5' }).result)
                .rejects.toMatchObject({ name: 'AnalysisResultsQueryError' });
        });

        expect(screen.queryByTestId('active-page-query-error')).not.toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Query expression' })).not.toBeInTheDocument();
        expect(renderedResultIds()).toEqual(['mistake']);
        expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('mistakes');
    });

    it('numbers visible results in exact query order when IDs are hidden', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="numbered-results"
                showElementId={false}
                data={{
                    elements: [
                        { id: 'first', labels: labelRanges('MSP'), title: 'First result', timeGap: { deltaMs: 5 } },
                        { id: 'racing', labels: labelRanges('MSR'), title: 'Filtered result', timeGap: { deltaMs: 50 } },
                        { id: 'third', labels: labelRanges('MSP'), title: 'Third result', timeGap: { deltaMs: 25 } },
                    ],
                }}
            />,
        );

        await waitFor(() => expect(within(screen.getByTestId('analysis-result-first'))
            .getByLabelText('Analysis result 1')).toHaveTextContent('1'));
        expect(within(screen.getByTestId('analysis-result-racing'))
            .getByLabelText('Analysis result 2')).toHaveTextContent('2');
        expect(within(screen.getByTestId('analysis-result-third'))
            .getByLabelText('Analysis result 3')).toHaveTextContent('3');
        expect(screen.queryByText('first')).not.toBeInTheDocument();

        selectView('time-lost-mistakes');

        await waitFor(() => expect(renderedResultIds()).toEqual(['racing', 'third', 'first']));
        expect(within(screen.getByTestId('analysis-result-racing'))
            .getByLabelText('Analysis result 1')).toHaveTextContent('1');
        expect(within(screen.getByTestId('analysis-result-third'))
            .getByLabelText('Analysis result 2')).toHaveTextContent('2');
    });

    it('orders common-label mistakes inside JSONata and aggregates combined taxonomy labels', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="frequency-order"
                data={{
                    elements: [
                        { id: 'unknown-first', labels: labelRanges('MSP', 'Telemetry', 'Telemetry') },
                        { id: 'wheel-duplicate', labels: labelRanges('Mistake (Practice)', 'MSP2', 'Wheel lock', 'MSP2') },
                        { id: 'wheel-name', labels: labelRanges('MSP', 'Wheel lock', 'Telemetry') },
                        { id: 'late-id', labels: labelRanges('MSP', 'MSP1') },
                        { id: 'late-name', labels: labelRanges('Mistake (Practice)', 'Late turn-in') },
                        { id: 'multi', labels: labelRanges('MSP', 'Telemetry', 'MSP2', 'Late turn-in') },
                        { id: 'racing-sub-label-only', labels: labelRanges('MSP', 'MSR1', 'Failed overtake attempt') },
                        { id: 'racing-id', labels: labelRanges('MSR', 'MSR1') },
                        { id: 'unrelated', labels: labelRanges('Telemetry', 'Late turn-in') },
                    ],
                }}
            />,
        );

        selectView('common-label-mistakes');

        await waitFor(() => expect(renderedResultIds()).toEqual([
            'late-id',
            'late-name',
            'multi',
            'wheel-duplicate',
            'wheel-name',
            'racing-sub-label-only',
            'racing-id',
            'unknown-first',
        ]));
        expect(renderedFrequencyData()).toEqual([
            { label: 'Late turn-in', occurrences: 3 },
            { label: 'Wheel lock', occurrences: 3 },
            { label: 'Failed overtake attempt', occurrences: 2 },
        ]);
        expect(screen.getByText('Label frequency — Most common label in mistakes')).toBeInTheDocument();
        expect(screen.getByTestId('label-frequency-graph')).toHaveAttribute(
            'data-graph-orientation',
            'horizontal',
        );
        expect(screen.getByTestId('label-frequency-graph')).toHaveAttribute(
            'data-graph-value-axis-label',
            'Occurrences',
        );
        expect(screen.getByTestId('label-frequency-graph')).toHaveAttribute(
            'data-graph-colors',
            JSON.stringify(['#00e676']),
        );
        expect(screen.getByText('8 of 9 total')).toBeInTheDocument();
    });

    it('counts all exact labels for All results without taxonomy filtering', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="racing-frequency-order"
                data={{
                    elements: [
                        { id: 'unknown-first', labels: labelRanges('Mistake (Racing)', 'Unknown racing label') },
                        { id: 'failed-id', labels: labelRanges('MSR', 'MSR1') },
                        { id: 'practice-sub-label-only', labels: labelRanges('MSR', 'MSP1', 'Late turn-in') },
                        { id: 'failed-name', labels: labelRanges('Mistake (Racing)', 'Failed overtake attempt') },
                        { id: 'contact-duplicate', labels: labelRanges('MSR', 'MSR2', 'Contact', 'MSR2') },
                        { id: 'multi', labels: labelRanges('MSR', 'MSR2', 'Failed overtake attempt') },
                        { id: 'unknown-second', labels: labelRanges('MSR', 'Telemetry') },
                    ],
                }}
            />,
        );

        selectView('all-results');

        await waitFor(() => expect(renderedResultIds()).toEqual([
            'unknown-first',
            'failed-id',
            'practice-sub-label-only',
            'failed-name',
            'contact-duplicate',
            'multi',
            'unknown-second',
        ]));
        expect(renderedFrequencyData()).toEqual(expect.arrayContaining([
            { label: 'MSR', occurrences: 5 },
            { label: 'Unknown racing label', occurrences: 1 },
            { label: 'Late turn-in', occurrences: 1 },
        ]));
        expect(screen.getByText('Label frequency — All results')).toBeInTheDocument();
    });

    it('derives graph data from matched elements while preserving query-result card order', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="independent-graph-order"
                data={{
                    elements: [
                        { id: 'late', labels: labelRanges('MSP', 'MSP1'), timeGap: { deltaMs: 5 } },
                        { id: 'wheel', labels: labelRanges('MSP', 'MSP2'), timeGap: { deltaMs: 50 } },
                        { id: 'both', labels: labelRanges('MSP', 'Late turn-in', 'Wheel lock'), timeGap: { deltaMs: 10 } },
                    ],
                }}
            />,
        );
        await waitFor(() => expect(renderedResultIds()).toEqual(['late', 'wheel', 'both']));
        const graph = screen.getByTestId('label-frequency-graph');
        const initialData = renderedFrequencyData();

        expect(initialData).toEqual([
            { label: 'Late turn-in', occurrences: 2 },
            { label: 'Wheel lock', occurrences: 2 },
        ]);
        expect(graph).toHaveAttribute('data-graph-height', String(160 + (2 * 36)));

        selectView('time-lost-mistakes');

        await waitFor(() => expect(renderedResultIds()).toEqual(['wheel', 'both', 'late']));
        expect(renderedFrequencyData()).toEqual(initialData);
    });

    it('shows the taxonomy-aware graph empty state for mistake templates', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="empty-frequency"
                data={{ elements: [{ id: 'unknown', labels: labelRanges('MSP', 'Unknown mistake') }] }}
            />,
        );

        await waitFor(() => expect(renderedResultIds()).toEqual(['unknown']));
        expect(renderedFrequencyData()).toEqual([]);
        expect(within(screen.getByTestId('label-frequency-graph')).getByRole('status')).toHaveTextContent(
            'No recognized mistake labels in the current query result to graph.',
        );
        expect(screen.getByTestId('label-frequency-graph')).toHaveAttribute(
            'data-graph-height',
            String(160 + 36),
        );
    });

    it('filters and orders numeric time losses in one template evaluation', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="time-order"
                data={{
                    elements: [
                        { id: 'missing', labels: labelRanges('MSP') },
                        { id: 'equal-first', labels: labelRanges('MSP'), timeGap: { deltaMs: 10 } },
                        { id: 'highest', labels: labelRanges('Mistake (Practice)'), timeGap: { deltaMs: 25 } },
                        { id: 'invalid', labels: labelRanges('MSP'), timeGap: { deltaMs: 'not-a-number' } },
                        { id: 'equal-second', labels: labelRanges('MSP'), timeGap: { deltaMs: 10 } },
                        { id: 'negative', labels: labelRanges('MSP'), timeGap: { deltaMs: -5 } },
                        { id: 'racing-highest', labels: labelRanges('MSR'), timeGap: { deltaMs: 1000 } },
                    ],
                }}
            />,
        );

        selectView('time-lost-mistakes');

        await waitFor(() => expect(renderedResultIds()).toEqual([
            'racing-highest',
            'highest',
            'equal-first',
            'equal-second',
            'negative',
            'missing',
            'invalid',
        ]));
    });

    it('re-evaluates the committed custom expression against canonical live data', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const { rerender } = render(
            <AnalysisResultsChart ref={chartRef} name="visualization:analysis-results"
                id="live-ranking"
                data={{
                    elements: [
                        { id: 'one', labels: labelRanges('MSR', 'Unknown racing mistake') },
                        { id: 'two', labels: labelRanges('Mistake (Racing)', 'MSR1') },
                        { id: 'practice', labels: labelRanges('MSP', 'MSP1') },
                    ],
                }}
            />,
        );
        await waitFor(() => expect(renderedResultIds()).toEqual(['one', 'two', 'practice']));
        await act(async () => {
            await chartRef.current!.applyAnalysisResultQuery({
                query: 'elements[labels[label_name in ["MSR", "Mistake (Racing)"]]]',
            }).result;
        });
        await waitFor(() => expect(renderedResultIds()).toEqual(['one', 'two']));

        rerender(
            <AnalysisResultsChart ref={chartRef} name="visualization:analysis-results"
                id="live-ranking"
                data={{
                    elements: [
                        { id: 'one', labels: labelRanges('MSR', 'Unknown racing mistake') },
                        { id: 'two', labels: labelRanges('Mistake (Racing)', 'MSR1'), title: 'Updated canonical result' },
                        { id: 'three', labels: labelRanges('MSR', 'Failed overtake attempt') },
                        { id: 'practice', labels: labelRanges('MSP', 'MSP1') },
                    ],
                }}
            />,
        );

        expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('custom');
        await waitFor(() => expect(renderedResultIds()).toEqual(['one', 'two', 'three']));
        expect(screen.getByTestId('analysis-result-two')).toHaveTextContent('Updated canonical result');
        expect(screen.getByText('3 of 4 total')).toBeInTheDocument();
        expect(renderedFrequencyData()).toEqual(expect.arrayContaining([
            { label: 'MSR', occurrences: 2 },
            { label: 'Failed overtake attempt', occurrences: 1 },
        ]));
    });

    it('re-evaluates the selected template after element append, update, and remove', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const Harness = () => {
            const [currentData, setCurrentData] = React.useState(() => normalizeAnalysisResultsData({
                elements: [{ id: 'initial', labels: labelRanges('MSP', 'MSP1') }],
            }));
            return (
                <AnalysisResultsChart
                    ref={chartRef}
                    name="visualization:analysis-results"
                    id="mutation-query-lifecycle"
                    data={currentData}
                    onUpdate={(nextData) => {
                        setCurrentData(normalizeAnalysisResultsData(nextData));
                        return true;
                    }}
                />
            );
        };
        render(<Harness />);
        await waitFor(() => expect(renderedResultIds()).toEqual(['initial']));

        act(() => {
            chartRef.current!.appendAnalysisResult({ id: 'appended', labels: labelRanges('MSR', 'MSR1') });
        });
        await waitFor(() => expect(renderedResultIds()).toEqual(['initial', 'appended']));
        expect(screen.getByText('2 of 2 total')).toBeInTheDocument();

        act(() => {
            chartRef.current!.updateAnalysisResult('initial', { labels: labelRanges('Telemetry') });
        });
        await waitFor(() => expect(renderedResultIds()).toEqual(['appended']));
        expect(screen.getByText('1 of 2 total')).toBeInTheDocument();

        act(() => {
            chartRef.current!.removeAnalysisResult('appended');
        });
        await waitFor(() => expect(renderedResultIds()).toEqual([]));
        expect(screen.getByText('0 of 1 total')).toBeInTheDocument();
        expect(screen.getByTestId('analysis-results-empty-state')).toBeInTheDocument();
    });

    it('fails closed when automatic evaluation is invalid for a new input generation', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const view = render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="automatic-query-failure"
                data={{ elements: [{ id: 'old', labels: labelRanges('MSP') }] }}
            />,
        );
        await waitFor(() => expect(renderedResultIds()).toEqual(['old']));

        await act(async () => {
            await chartRef.current!.applyAnalysisResultQuery({
                query: '$exists(elements[id = "old"]) ? elements : 5',
            }).result;
        });
        await waitFor(() => expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('custom'));

        view.rerender(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="automatic-query-failure"
                data={{ elements: [{ id: 'new', labels: labelRanges('MSP') }] }}
            />,
        );

        await waitFor(() => expect(renderedResultIds()).toEqual([]));
        expect(screen.queryByTestId('active-page-query-error')).not.toBeInTheDocument();
        expect(screen.getByText('0 of 1 total')).toBeInTheDocument();
    });

    it('suppresses stale programmatic completions after a newer page generation commits', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const view = render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="stale-query"
                data={{ elements: [{ id: 'old', labels: labelRanges('MSP') }] }}
            />,
        );
        await waitFor(() => expect(renderedResultIds()).toEqual(['old']));

        let resolveManual!: (value: analysisResultsQuery.JsonValue) => void;
        let resolvePage!: (value: analysisResultsQuery.JsonValue) => void;
        const manualResult = new Promise<analysisResultsQuery.JsonValue>((resolve) => {
            resolveManual = resolve;
        });
        const pageResult = new Promise<analysisResultsQuery.JsonValue>((resolve) => {
            resolvePage = resolve;
        });
        const evaluator = jest.spyOn(analysisResultsQuery, 'evaluateAnalysisResultsQuery')
            .mockImplementationOnce(() => manualResult)
            .mockImplementationOnce(() => pageResult);

        try {
            const staleOperation = chartRef.current!.applyAnalysisResultQuery({ query: 'elements' });
            const staleResult = staleOperation.result.catch((error) => error);
            await waitFor(() => expect(evaluator).toHaveBeenCalledTimes(1));

            view.rerender(
                <AnalysisResultsChart
                    ref={chartRef}
                    name="visualization:analysis-results"
                    id="stale-query"
                    data={{ elements: [{ id: 'new', labels: labelRanges('MSP') }] }}
                />,
            );
            await act(async () => resolvePage([{ id: 'new' }]));
            await waitFor(() => expect(renderedResultIds()).toEqual(['new']));

            await act(async () => resolveManual([{ id: 'old' }]));
            await expect(staleResult).resolves.toMatchObject({
                name: 'VisualizationControlFailedError',
            });
            expect(renderedResultIds()).toEqual(['new']);
            expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('mistakes');
        } finally {
            evaluator.mockRestore();
        }
    });

    it('regenerates a selected taxonomy template without rewriting Custom', async () => {
        const originalGetLabelName = mockGetLabelName;
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const renderChart = () => (
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="taxonomy-refresh"
                data={{ elements: [{ id: 'fresh', labels: labelRanges('Fresh Training') }] }}
            />
        );
        const view = render(renderChart());

        try {
            await waitFor(() => expect(screen.getByText('0 of 1 total')).toBeInTheDocument());
            mockGetLabelName = (labelId) => (
                labelId === 'MSP' ? 'Fresh Training' : originalGetLabelName(labelId)
            );
            view.rerender(renderChart());
            await waitFor(() => expect(renderedResultIds()).toEqual(['fresh']));

            await act(async () => {
                await chartRef.current!.applyAnalysisResultQuery({ query: 'elements' }).result;
            });
            await waitFor(() => expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('custom'));

            mockGetLabelName = (labelId) => (
                labelId === 'MSP' ? 'Newest Training' : originalGetLabelName(labelId)
            );
            view.rerender(renderChart());
            await waitFor(() => expect(chartRef.current!.getFilteredSegments().committedQuery).toBe('elements'));
            expect(screen.getByRole('combobox', { name: 'View' })).toHaveValue('custom');
        } finally {
            mockGetLabelName = originalGetLabelName;
        }
    });

    it('does not expose the removed most-common-mistakes queue action', () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="analysis-without-queue-action"
                data={{ elements: [{
                    id: 'common-mistake',
                    labels: labelRanges('MSP', 'MSP1'),
                    normalizedPositionRange: { start: 0.25, end: 0.3 },
                    comparison: comparableData(0.2, 0.4),
                }] }}
            />,
        );

        expect(screen.queryByRole('button', { name: 'Send most common mistakes' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Queue filtered comparisons' })).not.toBeInTheDocument();
        expect(screen.queryByText(/Queued:|Skipped:|Live Range To-do List/)).not.toBeInTheDocument();
    });
    it('defaults to collapsed and opens the comparison and section details when clicked', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="comparison-card"
                data={{
                    elements: [{
                        id: 'comparable',
                        labels: labelRanges('MSP', 'MSP1', 'EA', 'EA1', 'RM', 'RM7'),
                        section: 'Turn 4',
                        normalizedPositionRange: { start: 0.2, end: 0.35 },
                        timeGap: { startMs: 250, endMs: 375, deltaMs: 125 },
                        metadata: { note: 'Late turn-in costs exit speed' },
                        comparison: {
                            samples: [{
                                driverTimeMs: 0,
                                expertTimeMs: 0,
                                driverTrackPosition: 0.2,
                                expertTrackPosition: 0.2,
                                driverGas: 0.4,
                                expertGas: 0.5,
                            }],
                        },
                    }],
                }}
            />,
        );

        const card = await screen.findByTestId('analysis-result-comparable');
        expect(card).not.toHaveAttribute('tabindex');
        const toggle = within(card).getByRole('button');
        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(within(card).queryByTestId('driver-expert-comparison')).not.toBeInTheDocument();
        expect(within(card).queryByRole('region', { name: 'Section details' })).not.toBeInTheDocument();
        fireEvent.click(toggle);
        expect(toggle).toHaveAttribute('aria-expanded', 'true');
        const comparison = within(card).getByTestId('driver-expert-comparison');
        expect(comparison).toBeInTheDocument();
        expect(within(card).queryByText(/Hover or focus/)).not.toBeInTheDocument();
        const details = within(card).getByRole('region', { name: 'Section details' });
        expect(details).toHaveTextContent('Lap time difference+0.125 s');
        expect(details).toHaveTextContent('Time lost to Expert in this section');
        expect(details).toHaveTextContent('SectionTurn 4');
        expect(details).toHaveTextContent('Track position20.0% – 35.0%');
        expect(details).toHaveTextContent('Gap at entry+0.250 s');
        expect(details).toHaveTextContent('Gap at exit+0.375 s');
        expect(details).toHaveTextContent('note: Late turn-in costs exit speed');
        expect(screen.getByRole('region', { name: 'Mistakes labels' })).toHaveTextContent('Late turn-in');
        expect(screen.getByRole('region', { name: 'Expert labels' })).toHaveTextContent('Matches expert line');
        expect(screen.getByRole('region', { name: 'Recovery labels' })).toHaveTextContent('Merge back to expert line');
        expect(screen.queryByTestId('driver-telemetry-pod')).not.toBeInTheDocument();
        expect(screen.queryByTestId('expert-telemetry-pod')).not.toBeInTheDocument();
        expect(screen.queryAllByRole('meter')).toHaveLength(0);
        expect(screen.getByTestId('trajectory-unavailable')).toHaveTextContent(
            'Trajectory data unavailable',
        );
        expect(screen.queryByTestId('comparison-graph-gas')).not.toBeInTheDocument();
        fireEvent.mouseEnter(card);
        fireEvent.mouseLeave(card);
        fireEvent.focus(card);
        fireEvent.blur(card);
        expect(within(card).getByTestId('driver-expert-comparison')).toBe(comparison);
        fireEvent.click(toggle);
        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(details).not.toBeVisible();
    });

    it.each([
        [{ deltaMs: -125 }, '-0.125 s', 'Time gained on Expert in this section'],
        [{ deltaMs: 0 }, '0.000 s', 'Gap to Expert unchanged in this section'],
        [{ startMs: -100, endMs: 150 }, '+0.250 s', 'Time lost to Expert in this section'],
        [undefined, 'Unavailable', 'No timing data for this section'],
        [{ deltaMs: 'not-a-number' }, 'Unavailable', 'No timing data for this section'],
        [{ startMs: 100 }, 'Unavailable', 'No timing data for this section'],
    ])('distinguishes time gained, lost, unchanged, and unavailable for %j', async (timeGap, value, description) => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="section-timing"
                data={{ elements: [{ id: 'timing', labels: labelRanges('MSP'), timeGap }] }}
            />,
        );

        const card = await screen.findByTestId('analysis-result-timing');
        fireEvent.click(within(card).getByRole('button'));
        const details = within(card).getByRole('region', { name: 'Section details' });
        expect(details).toHaveTextContent(`Lap time difference${value}`);
        expect(details).toHaveTextContent(description as string);
    });

    it('shows comparison unavailability when the card is expanded', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="unavailable-comparison-card"
                data={{
                    elements: [{
                        id: 'unavailable-comparison',
                        labels: labelRanges('MSP'),
                        comparison: { samples: [{
                            driverTimeMs: 0,
                            expertTimeMs: 0,
                            trackPosition: 0.2,
                            driverGas: 0.4,
                            expertGas: 0.5,
                        }] },
                    }],
                }}
            />,
        );

        const card = await screen.findByTestId('analysis-result-unavailable-comparison');
        expect(card).not.toHaveAttribute('tabindex');
        fireEvent.click(within(card).getByRole('button'));
        expect(within(card).getByText('Expert comparison unavailable')).toBeInTheDocument();
        expect(within(card).getByRole('region', { name: 'Section details' })).toHaveTextContent(
            'No timing data for this section',
        );
        fireEvent.mouseEnter(card);
        expect(screen.queryByTestId('driver-expert-comparison')).not.toBeInTheDocument();
    });

    it.each([
        {
            name: 'an absent payload',
            fields: {},
            expectedCode: 'comparison_data_missing',
        },
        {
            name: 'a null payload',
            fields: { comparison: null },
            expectedCode: 'comparison_data_missing',
        },
        {
            name: 'an absent payload with empty diagnostics',
            fields: { comparisonDiagnostics: [] },
            expectedCode: 'comparison_data_missing',
        },
        {
            name: 'missing Driver records',
            fields: { comparisonDiagnostics: [{
                code: 'driver_records_missing',
                message: 'The recorded lap contains no Driver telemetry rows.',
            }] },
            expectedCode: 'driver_records_missing',
        },
        {
            name: 'a payload without samples',
            fields: { comparison: {} },
            expectedCode: 'comparison_samples_missing',
        },
        {
            name: 'an empty sample list',
            fields: { comparison: { samples: [] } },
            expectedCode: 'comparison_samples_missing',
        },
        {
            name: 'samples with only Driver data',
            fields: { comparison: { samples: [{ driverTimeMs: 0, driverTrackPosition: 0.2 }] } },
            expectedCode: 'comparison_samples_invalid',
        },
    ])('reports the specific comparison reason for $name', async ({ fields, expectedCode }) => {
        const consoleWarn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
        try {
            render(
                <AnalysisResultsChart
                    name="visualization:analysis-results"
                    id="comparison-reason"
                    data={{ elements: [{
                        id: 'segment-reason',
                        labels: labelRanges('MSP'),
                        metadata: { source: 'ai_classifier' },
                        ...fields,
                    }] }}
                />,
            );

            await waitFor(() => expect(consoleWarn).toHaveBeenCalledTimes(1));
            expect(consoleWarn).toHaveBeenCalledWith(
                '[driver-expert-comparison] Expert comparison unavailable.',
                expect.objectContaining({
                    segment_id: 'segment-reason',
                    reason_codes: [expectedCode],
                    reasons: [expect.objectContaining({ code: expectedCode })],
                }),
            );
        } finally {
            consoleWarn.mockRestore();
        }
    });

    it('logs specific classifier comparison failures without adding a missing-data reason', async () => {
        const consoleWarn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
        render(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="comparison-warning"
                sessionGame="acc"
                data={{
                    elements: [{
                        id: 'segment-warning',
                        labels: labelRanges('MSP'),
                        section: 'Turn 5',
                        comparisonDiagnostics: [{
                            code: 'expert_reference_missing',
                            message: 'The analysis segment contains no Expert reference telemetry.',
                        }, {
                            code: 'driver_records_missing',
                            message: 'The recorded lap contains no Driver telemetry rows.',
                        }],
                        metadata: { source: 'ai_classifier' },
                    }],
                }}
            />,
        );

        await waitFor(() => expect(consoleWarn).toHaveBeenCalledTimes(1));
        expect(consoleWarn).toHaveBeenCalledWith(
            '[driver-expert-comparison] Expert comparison unavailable.',
            expect.objectContaining({
                segment_id: 'segment-warning',
                section: 'Turn 5',
                game: 'acc',
                reason_codes: [
                    'expert_reference_missing',
                    'driver_records_missing',
                ],
                reasons: expect.arrayContaining([
                    expect.objectContaining({ code: 'expert_reference_missing' }),
                    expect.objectContaining({ code: 'driver_records_missing' }),
                ]),
            }),
        );
        consoleWarn.mockRestore();
    });
});

describe('analysis results mutations', () => {
    it('throws typed component errors for invalid controls and failed callbacks', () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="analysis-results-test"
                id="analysis-results-test"
                data={{ elements: [] }}
                onUpdate={() => false}
            />,
        );

        expect(() => chartRef.current!.appendAnalysisResult(null)).toThrow(expect.objectContaining({
            name: 'VisualizationControlFailedError',
            componentName: 'analysis-results-test',
            message: 'append_element requires an element object.',
        }));
        expect(() => chartRef.current!.appendAnalysisResult({ id: 'one', labels: [] }))
            .toThrow(VisualizationControlFailedError);
        expect(() => chartRef.current!.replaceAnalysisResults({ elements: [] })).toThrow(expect.objectContaining({
            name: 'VisualizationUpdateFailedError',
            componentName: 'analysis-results-test',
        }));
        expect(() => chartRef.current!.disableAnalysisResults()).toThrow(expect.objectContaining({
            name: 'ComponentDisableFailedError',
            componentName: 'analysis-results-test',
        }));
    });

    it('preserves independent ranges through append, update, and normalization', () => {
        const labels = [
            { label_name: 'MSP1', start_index: 120, end_index: 130 },
            { label_name: 'MSP1', start_index: 150, end_index: 180 },
        ];
        const appended = appendAnalysisResultElement({ elements: [] }, { id: 'section', labels });
        expect(appended.data.elements[0].labels).toEqual(labels);
        const updated = updateAnalysisResultElement(appended.data, 'section', { title: 'Turn 1' });
        expect(updated.data.elements[0].labels).toEqual(labels);
        expect(normalizeAnalysisResultsData(updated.data).elements[0].labels).toEqual(labels);
    });

    it('normalizes aliases and generates IDs for appended elements', () => {
        const mutation = appendAnalysisResultElement({ elements: [] }, {
            labels: labelRanges(' Unknown label '),
            track_section: 'Section A',
            start_position: '0.1',
            end_position: 0.2,
            time_gap: { delta_ms: 50 },
            metadata: { source: 'form' },
        });

        expect(mutation.result.success).toBe(true);
        expect(mutation.result.data).toMatchObject({ count: 1 });
        expect(mutation.data.elements[0]).toMatchObject({
            id: expect.stringMatching(/^analysis-result-/),
            labels: labelRanges('Unknown label'),
            section: 'Section A',
            normalizedPositionRange: { start: 0.1, end: 0.2 },
            timeGap: { deltaMs: 50 },
            metadata: { source: 'form' },
        });
    });

    it('rejects duplicates and invalid or unknown mutation targets', () => {
        const data = normalizeAnalysisResultsData({
            elements: [{ id: 'one', labels: labelRanges('Mistake') }],
        });

        expect(appendAnalysisResultElement(data, { id: 'one', labels: [] }).result).toMatchObject({
            success: false,
            message: expect.stringContaining('already exists'),
        });
        expect(updateAnalysisResultElement(data, '', {}).result.success).toBe(false);
        expect(updateAnalysisResultElement(data, 'missing', {}).result.success).toBe(false);
        expect(updateAnalysisResultElement(data, 'one', { id: 'two' }).result).toMatchObject({
            success: false,
            message: expect.stringContaining('immutable'),
        });
        expect(removeAnalysisResultElement(data, 'missing').result.success).toBe(false);
    });

    it('updates and removes elements while reporting the resulting count', () => {
        const data = normalizeAnalysisResultsData({
            elements: [
                { id: 'one', labels: labelRanges('Mistake') },
                { id: 'two', labels: labelRanges('Adherence') },
            ],
        });
        const updated = updateAnalysisResultElement(data, 'one', {
            labels: labelRanges('Recovery'),
            metadata: { note: 'kept local' },
        });

        expect(updated.result).toMatchObject({
            success: true,
            data: {
                count: 2,
                element: { id: 'one', labels: labelRanges('Recovery') },
            },
        });
        const removed = removeAnalysisResultElement(updated.data, 'two');
        expect(removed.result).toMatchObject({
            success: true,
            data: { id: 'two', count: 1 },
        });
        expect(removed.data.elements.map((element) => element.id)).toEqual(['one']);
    });

    it('preserves compact normalized comparison data through unrelated mutations', () => {
        const data = normalizeAnalysisResultsData({
            elements: [{
                id: 'comparison',
                labels: labelRanges('MSP'),
                comparison: {
                    samples: [{
                        driverTimeMs: 250,
                        expertTimeMs: 500,
                        driverTrackPosition: 0.2,
                        expertTrackPosition: 0.2,
                        driverGas: 0.4,
                        expertGas: 0.5,
                        Physics_gas: 1,
                    }],
                },
                baselineRecords: [{ very: 'large' }],
            }],
        });

        expect(data.elements[0].comparison).toEqual({
            samples: [{
                driverTimeMs: 250,
                expertTimeMs: 500,
                driverTrackPosition: 0.2,
                expertTrackPosition: 0.2,
                driverGas: 0.4,
                expertGas: 0.5,
            }],
        });
        expect(data.elements[0]).not.toHaveProperty('baselineRecords');

        const updated = updateAnalysisResultElement(data, 'comparison', {
            title: 'Updated title',
        });
        expect(updated.data.elements[0].comparison).toEqual(data.elements[0].comparison);
    });
});

function labelRanges(...names: string[]) {
    return names.map((label_name) => ({ label_name, start_index: 0, end_index: 1 }));
}
