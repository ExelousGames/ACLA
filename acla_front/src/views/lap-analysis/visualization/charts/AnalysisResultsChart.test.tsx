import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';

jest.mock('contexts/DesktopGameContext', () => ({
    useDesktopGame: () => ({
        detectedGame: null,
        detectionStatus: 'not-detected',
        error: null,
    }),
}));

jest.mock('@radix-ui/themes', () => {
    const ReactModule = require('react');
    const Component = ({ as: Tag = 'div', children, ...props }: any) => (
        <Tag {...props}>{children}</Tag>
    );
    const HoverCardContext = ReactModule.createContext({
        open: false,
        onOpenChange: (_open: boolean) => undefined,
    });
    const HoverCardRoot = ({ open, onOpenChange, children }: any) => (
        <HoverCardContext.Provider value={{ open, onOpenChange }}>
            {children}
        </HoverCardContext.Provider>
    );
    const HoverCardTrigger = ({ children }: any) => {
        const context = ReactModule.useContext(HoverCardContext);
        return ReactModule.cloneElement(children, {
            onMouseEnter: () => context.onOpenChange(true),
            onMouseLeave: () => context.onOpenChange(false),
            onFocus: () => context.onOpenChange(true),
            onBlur: () => context.onOpenChange(false),
        });
    };
    const HoverCardContent = ({
        children,
        side,
        align,
        avoidCollisions,
        collisionPadding,
        sideOffset,
        ...props
    }: any) => {
        const context = ReactModule.useContext(HoverCardContext);
        return context.open ? (
            <div
                {...props}
                data-testid="comparison-hover-content"
                data-side={side}
                data-align={align}
                data-avoid-collisions={String(avoidCollisions)}
                data-collision-padding={String(collisionPadding)}
                data-side-offset={String(sideOffset)}
            >
                {children}
            </div>
        ) : null;
    };
    return {
        Badge: Component,
        Box: Component,
        Card: Component,
        Flex: Component,
        HoverCard: {
            Root: HoverCardRoot,
            Trigger: HoverCardTrigger,
            Content: HoverCardContent,
        },
        ScrollArea: Component,
        Text: Component,
    };
});

const mockCategoryLabels: Record<string, string[]> = {
    MSP: ['MSP1', 'MSP2'],
    MSR: ['MSR1', 'MSR2'],
};
const mockLabelNames: Record<string, string> = {
    MSP: 'Training Error',
    MSP1: 'Late turn-in',
    MSP2: 'Wheel lock',
    MSR: 'Race Error',
    MSR1: 'Failed overtake attempt',
    MSR2: 'Contact',
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
    VisualizationControlFailedError,
} from 'contexts/AiToolComponentError';
import {
    appendAnalysisResultElement,
    normalizeAnalysisResultsData,
    removeAnalysisResultElement,
    updateAnalysisResultElement,
} from './analysisResultsModel';
import * as analysisResultsQuery from './analysisResultsQuery';

const ALL_ANALYSES_COUNT_QUERY = '$count(analyses)';
const ALL_RESULTS_COUNT_QUERY = '$count(analyses.elements)';
const MISTAKE_COUNT_QUERY = '$count(analyses.elements[labels[$ in ["MSP", "Mistake (Practice)", "Training Error", "MSR", "Mistake (Racing)", "Race Error"]]])';

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

describe('AnalysisResultsChart', () => {
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

        await expect(chartRef.current!.queryAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 0,
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: ALL_ANALYSES_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 1,
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: '{"count": $count(analyses.elements)}' }).result).resolves.toEqual({
            status: 'ready',
            data: { count: 0 },
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: '[analyses.elements.id]' }).result).resolves.toEqual({
            status: 'ready',
            data: [],
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: 'analyses.elements[id = "missing"]' }).result).resolves.toEqual({
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
                        { id: 'practice-id', labels: ['MSP', 'MSP'] },
                        { id: 'practice-canonical', labels: ['Mistake (Practice)'] },
                        { id: 'practice-configured', labels: ['Training Error'] },
                        { id: 'racing-id', labels: ['MSR'] },
                        { id: 'racing-canonical', labels: ['Mistake (Racing)'] },
                        { id: 'racing-configured', labels: ['Race Error'] },
                        { id: 'combined', labels: ['MSP', 'MSR', 'MSP', 'MSR'] },
                        { id: 'children-only', labels: ['MSP1', 'MSR1'] },
                        { id: 'unrelated', labels: ['Expert Adherence'] },
                        null,
                        'invalid',
                    ],
                }}
            />,
        );

        await expect(chartRef.current!.queryAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 9,
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: MISTAKE_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 7,
        });

        selectView('time-lost-mistakes');

        await expect(chartRef.current!.queryAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 9,
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: MISTAKE_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 7,
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: 'result_count' }).result).resolves.toEqual({
            status: 'ready',
            data: null,
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: 'mistake_count' }).result).resolves.toEqual({
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
                { id: 'active-mistake', labels: ['MSP'] },
                { id: 'active-unrelated', labels: ['Telemetry'] },
            ],
        }, {
            id: 'latest-page',
            createdAt: 2,
            baseline: { lap_id: 2, lap_time_ms: 98_000, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'latest-one', labels: ['MSP'] },
                { id: 'latest-two', labels: ['MSR'] },
                { id: 'latest-three', labels: ['Mistake (Practice)'] },
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
        await expect(chartRef.current!.queryAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 5,
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: ALL_ANALYSES_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 2,
        });
        await expect(chartRef.current!.queryAnalysisResult({
            query: 'analyses.{"lap_id": baseline.lap_id, "segmentCount": $count(elements)}',
        }).result).resolves.toEqual({
            status: 'ready',
            data: [
                { lap_id: 1, segmentCount: 2 },
                { lap_id: 2, segmentCount: 3 },
            ],
        });
        await expect(chartRef.current!.queryAnalysisResult({ query: MISTAKE_COUNT_QUERY }).result).resolves.toEqual({
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
        await expect(chartRef.current!.queryAnalysisResult({ query: ALL_RESULTS_COUNT_QUERY }).result).resolves.toEqual({
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
        await expect(chartRef.current!.queryAnalysisResult({ query: MISTAKE_COUNT_QUERY }).result).resolves.toEqual({
            status: 'ready',
            data: 4,
        });
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
                        elements: [{ id: 'zero-lap-mistake', labels: ['MSP'], title: 'Zero lap mistake' }],
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
        await expect(chartRef.current!.queryAnalysisResult({
            query: ALL_ANALYSES_COUNT_QUERY,
        }).result).resolves.toEqual({ status: 'ready', data: 1 });

        fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
        expect(screen.getByText('Page 1 of 1')).toBeInTheDocument();
        expect(screen.getByText(/Baseline: Spa.*GT3.*Lap 0/)).toBeInTheDocument();
        await waitFor(() => expect(within(screen.getByTestId('analysis-result-zero-lap-mistake'))
            .getByLabelText('Analysis result 1')).toHaveTextContent('1'));
    });

    it.each([
        { requested: undefined, requestedResult: null, fallback: true },
        { requested: -1, requestedResult: -1, fallback: true },
        { requested: 0, requestedResult: 0, fallback: true },
        { requested: 99, requestedResult: 99, fallback: true },
    ])('applies to the highest retained-array page for fallback request $requested', async ({
        requested,
        requestedResult,
        fallback,
    }) => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const onSelectPage = jest.fn();
        const pages: AnalysisResultsPaginationPage[] = [{
            id: 'array-page-1',
            createdAt: 999,
            baseline: { lap_id: 1, lap_time_ms: 90_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'first-only', labels: ['MSP'] }],
        }, {
            id: 'array-page-2',
            createdAt: -999,
            baseline: { lap_id: 2, lap_time_ms: 89_000, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'latest-match', labels: ['MSP'] },
                { id: 'latest-other', labels: ['Telemetry'] },
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
            data: 1,
            applied_query: query,
            applied_page_id: 'array-page-2',
            applied_page_number: 2,
            requested_page_number: requestedResult,
            used_most_recent_fallback: fallback,
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
            elements: [{ id: 'page-one-match', labels: ['MSP'] }],
        }, {
            id: 'displayed-page-2',
            createdAt: 100,
            baseline: { lap_id: 5, lap_time_ms: 89_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'page-two-match', labels: ['MSP'] }],
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

        expect(result).toEqual(expect.objectContaining({
            status: 'ready',
            data: 1,
            applied_page_id: 'displayed-page-1',
            applied_page_number: 1,
            requested_page_number: 1,
            used_most_recent_fallback: false,
        }));
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
                    { id: 'recorded-match', labels: ['MSP'] },
                    { id: 'recorded-other', labels: ['Telemetry'] },
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
            data: 1,
            applied_query: query,
            applied_page_id: null,
            applied_page_number: 1,
            requested_page_number: 12,
            used_most_recent_fallback: true,
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
                    { id: 'preserved', labels: ['MSP'] },
                    { id: 'excluded', labels: ['MSP'] },
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
            elements: [{ id: 'newest-wins', labels: ['MSP'] }],
        }, {
            id: 'stale-page-2',
            createdAt: 1,
            baseline: { lap_id: 2, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'stale-result', labels: ['MSP'] }],
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
        expect(latestResult).toEqual(expect.objectContaining({
            applied_page_id: 'stale-page-1',
            applied_page_number: 1,
            data: 1,
        }));
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
            elements: [{ id: 'second-page-result', labels: ['MSP', 'MSP2'], title: 'Second page mistake' }],
        }, {
            id: 'page-1',
            createdAt: 1,
            baseline: { lap_id: 4, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'first-page-result', labels: ['MSP', 'MSP1'], title: 'First page mistake' }],
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
            chartRef.current.appendAnalysisResult({ id: 'active-only', labels: ['MSP'] });
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

    it('exposes immutable committed-filter snapshots for only the displayed concrete page', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const pages: AnalysisResultsPaginationPage[] = [{
            id: 'filtered-first',
            createdAt: 1,
            baseline: { lap_id: 1, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
            elements: [{
                id: 'early',
                labels: ['MSP'],
                normalizedPositionRange: { start: 0.2, end: 0.25 },
                comparison: comparableData(0.2, 0.4),
            }, {
                id: 'late',
                labels: ['Telemetry'],
                normalizedPositionRange: { start: 0.8, end: 0.85 },
            }],
        }, {
            id: 'filtered-second',
            createdAt: 2,
            baseline: { lap_id: 2, lap_time_ms: 99_000, track: 'Spa', car: 'GT3' },
            elements: [{
                id: 'second-page-only',
                labels: ['MSR'],
                normalizedPositionRange: { start: 0.5, end: 0.55 },
            }],
        }];
        const Harness = () => {
            const [activePageId, setActivePageId] = React.useState('filtered-first');
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

        expect(chartRef.current!.getFilteredSegments()).toEqual({
            status: 'empty',
            activePageId: null,
            appliedView: null,
            committedQuery: null,
            segments: [],
        });

        fireEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
        await waitFor(() => expect(chartRef.current!.getFilteredSegments().status).toBe('ready'));
        expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            activePageId: 'filtered-first',
            appliedView: 'mistakes',
            segments: [{ id: 'early' }],
        });

        selectView('all-results');
        await waitFor(() => expect(chartRef.current!.getFilteredSegments().appliedView)
            .toBe('all-results'));
        await act(async () => {
            await chartRef.current!.applyAnalysisResultQuery({
                query: 'elements^(>normalizedPositionRange.start)',
                page_number: 1,
            }).result;
        });
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

    it('reports a busy filtered snapshot while the active filter is evaluating', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="busy-filtered-snapshot"
                data={{ elements: [{ id: 'one', labels: ['MSP'] }] }}
            />,
        );

        expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            status: 'busy',
            segments: [],
        });
        await waitFor(() => expect(chartRef.current!.getFilteredSegments()).toMatchObject({
            status: 'ready',
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
            elements: [{ id: 'latest-mistake', labels: ['MSP', 'MSP1'] }],
        }, {
            id: 'first',
            createdAt: 100,
            baseline: { lap_id: 3, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'first-mistake', labels: ['MSP', 'MSP1'] }],
        }, {
            id: 'missing',
            createdAt: 300,
            baseline: { lap_id: 14, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [],
        }, {
            id: 'slower',
            createdAt: 200,
            baseline: { lap_id: 9, lap_time_ms: 105_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'slower-mistake', labels: ['MSR', 'MSR1'] }],
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
                { id: 'b-wheel-1', labels: ['MSP', 'MSP2'] },
                { id: 'b-wheel-2', labels: ['Mistake (Practice)', 'Wheel lock'] },
                { id: 'b-racing', labels: ['MSR', 'MSR1'] },
            ],
        }, {
            id: 'first-page',
            createdAt: 100,
            baseline: { lap_id: 8, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'a-late', labels: ['MSP', 'MSP1', 'Late turn-in', 'MSP1'] },
                { id: 'a-wheel', labels: ['MSP', 'MSP2'] },
                { id: 'a-parent-only', labels: ['Mistake (Practice)'] },
                { id: 'a-child-without-parent', labels: ['MSP1'] },
            ],
        }, {
            id: 'last-page',
            createdAt: 300,
            baseline: { lap_id: 15, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'c-wheel', labels: ['MSP', 'MSP2', 'Wheel lock'] },
                { id: 'c-racing-1', labels: ['Mistake (Racing)', 'MSR1'] },
                { id: 'c-racing-2', labels: ['MSR', 'Failed overtake attempt', 'MSR1'] },
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
            elements: [{ id: 'fresh-labels', labels: ['Fresh Training', 'Fresh brake'] }],
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
                elements: [{ id: 'retained-mistake', labels: ['MSP', 'MSP1'] }],
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
                { id: 'training', labels: ['MSP', 'MSP1'] },
                { id: 'racing-one', labels: ['MSR', 'MSR1'] },
                { id: 'racing-two', labels: ['MSR', 'MSR2'] },
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
                { id: 'training', labels: ['MSP', 'MSP1'] },
                { id: 'racing-one', labels: ['MSR', 'MSR1'] },
                { id: 'racing-two', labels: ['MSR', 'MSR2'] },
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
            elements: [{ id: 'old-mistake', labels: ['MSP', 'MSP1'] }],
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
                elements: [{ id: 'new-mistake', labels: ['MSP', 'MSP2'] }],
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
                        elements: [{ id: 'only-result', labels: ['MSP', 'MSP1'] }],
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
                        labels: ['Mistake (Practice)', 'Future category', 'Recovery'],
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
        expect(screen.getByText('Position: 20.0% – 35.0%')).toBeInTheDocument();
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
                        { id: 'practice-id', labels: ['MSP'] },
                        { id: 'practice-name', labels: ['Mistake (Practice)'] },
                        { id: 'racing-id', labels: ['MSR'] },
                        { id: 'racing-name', labels: ['Mistake (Racing)'] },
                        { id: 'unrelated', labels: ['Telemetry'] },
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
                data={{ elements: [{ id: 'telemetry', labels: ['Telemetry'] }] }}
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
                        { id: 'third-fastest', labels: ['MSP', 'Lockup'], timeGap: { deltaMs: 20 } },
                        { id: 'racing', labels: ['MSR', 'Wide exit'], timeGap: { deltaMs: 80 } },
                        { id: 'least-time', labels: ['Mistake (Practice)', 'Lockup'], timeGap: { deltaMs: 5 } },
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
                        { id: 'practice', labels: ['MSP', 'MSP1'] },
                        { id: 'racing', labels: ['MSR', 'MSR1'] },
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
                data={{ elements: [{ id: 'mistake', labels: ['MSP'] }] }}
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
                        { id: 'first', labels: ['MSP'], title: 'First result', timeGap: { deltaMs: 5 } },
                        { id: 'racing', labels: ['MSR'], title: 'Filtered result', timeGap: { deltaMs: 50 } },
                        { id: 'third', labels: ['MSP'], title: 'Third result', timeGap: { deltaMs: 25 } },
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
                        { id: 'unknown-first', labels: ['MSP', 'Telemetry', 'Telemetry'] },
                        { id: 'wheel-duplicate', labels: ['Mistake (Practice)', 'MSP2', 'Wheel lock', 'MSP2'] },
                        { id: 'wheel-name', labels: ['MSP', 'Wheel lock', 'Telemetry'] },
                        { id: 'late-id', labels: ['MSP', 'MSP1'] },
                        { id: 'late-name', labels: ['Mistake (Practice)', 'Late turn-in'] },
                        { id: 'multi', labels: ['MSP', 'Telemetry', 'MSP2', 'Late turn-in'] },
                        { id: 'racing-sub-label-only', labels: ['MSP', 'MSR1', 'Failed overtake attempt'] },
                        { id: 'racing-id', labels: ['MSR', 'MSR1'] },
                        { id: 'unrelated', labels: ['Telemetry', 'Late turn-in'] },
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
                        { id: 'unknown-first', labels: ['Mistake (Racing)', 'Unknown racing label'] },
                        { id: 'failed-id', labels: ['MSR', 'MSR1'] },
                        { id: 'practice-sub-label-only', labels: ['MSR', 'MSP1', 'Late turn-in'] },
                        { id: 'failed-name', labels: ['Mistake (Racing)', 'Failed overtake attempt'] },
                        { id: 'contact-duplicate', labels: ['MSR', 'MSR2', 'Contact', 'MSR2'] },
                        { id: 'multi', labels: ['MSR', 'MSR2', 'Failed overtake attempt'] },
                        { id: 'unknown-second', labels: ['MSR', 'Telemetry'] },
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
                        { id: 'late', labels: ['MSP', 'MSP1'], timeGap: { deltaMs: 5 } },
                        { id: 'wheel', labels: ['MSP', 'MSP2'], timeGap: { deltaMs: 50 } },
                        { id: 'both', labels: ['MSP', 'Late turn-in', 'Wheel lock'], timeGap: { deltaMs: 10 } },
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
                data={{ elements: [{ id: 'unknown', labels: ['MSP', 'Unknown mistake'] }] }}
            />,
        );

        await waitFor(() => expect(renderedResultIds()).toEqual(['unknown']));
        expect(renderedFrequencyData()).toEqual([]);
        expect(screen.getByRole('status')).toHaveTextContent(
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
                        { id: 'missing', labels: ['MSP'] },
                        { id: 'equal-first', labels: ['MSP'], timeGap: { deltaMs: 10 } },
                        { id: 'highest', labels: ['Mistake (Practice)'], timeGap: { deltaMs: 25 } },
                        { id: 'invalid', labels: ['MSP'], timeGap: { deltaMs: 'not-a-number' } },
                        { id: 'equal-second', labels: ['MSP'], timeGap: { deltaMs: 10 } },
                        { id: 'negative', labels: ['MSP'], timeGap: { deltaMs: -5 } },
                        { id: 'racing-highest', labels: ['MSR'], timeGap: { deltaMs: 1000 } },
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
                        { id: 'one', labels: ['MSR', 'Unknown racing mistake'] },
                        { id: 'two', labels: ['Mistake (Racing)', 'MSR1'] },
                        { id: 'practice', labels: ['MSP', 'MSP1'] },
                    ],
                }}
            />,
        );
        await waitFor(() => expect(renderedResultIds()).toEqual(['one', 'two', 'practice']));
        await act(async () => {
            await chartRef.current!.applyAnalysisResultQuery({
                query: 'elements[labels[$ in ["MSR", "Mistake (Racing)"]]]',
            }).result;
        });
        await waitFor(() => expect(renderedResultIds()).toEqual(['one', 'two']));

        rerender(
            <AnalysisResultsChart ref={chartRef} name="visualization:analysis-results"
                id="live-ranking"
                data={{
                    elements: [
                        { id: 'one', labels: ['MSR', 'Unknown racing mistake'] },
                        { id: 'two', labels: ['Mistake (Racing)', 'MSR1'], title: 'Updated canonical result' },
                        { id: 'three', labels: ['MSR', 'Failed overtake attempt'] },
                        { id: 'practice', labels: ['MSP', 'MSP1'] },
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
                elements: [{ id: 'initial', labels: ['MSP', 'MSP1'] }],
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
            chartRef.current!.appendAnalysisResult({ id: 'appended', labels: ['MSR', 'MSR1'] });
        });
        await waitFor(() => expect(renderedResultIds()).toEqual(['initial', 'appended']));
        expect(screen.getByText('2 of 2 total')).toBeInTheDocument();

        act(() => {
            chartRef.current!.updateAnalysisResult('initial', { labels: ['Telemetry'] });
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
                data={{ elements: [{ id: 'old', labels: ['MSP'] }] }}
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
                data={{ elements: [{ id: 'new', labels: ['MSP'] }] }}
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
                data={{ elements: [{ id: 'old', labels: ['MSP'] }] }}
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
                    data={{ elements: [{ id: 'new', labels: ['MSP'] }] }}
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
                data={{ elements: [{ id: 'fresh', labels: ['Fresh Training'] }] }}
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
                    labels: ['MSP', 'MSP1'],
                    normalizedPositionRange: { start: 0.25, end: 0.3 },
                    comparison: comparableData(0.2, 0.4),
                }] }}
            />,
        );

        expect(screen.queryByRole('button', { name: 'Send most common mistakes' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Queue filtered comparisons' })).not.toBeInTheDocument();
        expect(screen.queryByText(/Queued:|Skipped:|Live Range To-do List/)).not.toBeInTheDocument();
    });
    it('mounts a collision-aware comparison only while a capable card is hovered or focused', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="comparison-card"
                data={{
                    elements: [{
                        id: 'comparable',
                        labels: ['MSP'],
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
        expect(card).toHaveAttribute('tabindex', '0');
        expect(screen.queryByTestId('driver-expert-comparison')).not.toBeInTheDocument();

        fireEvent.mouseEnter(card);

        expect(screen.getByTestId('driver-expert-comparison')).toBeInTheDocument();
        expect(screen.queryByTestId('driver-telemetry-pod')).not.toBeInTheDocument();
        expect(screen.queryByTestId('expert-telemetry-pod')).not.toBeInTheDocument();
        expect(screen.queryAllByRole('meter')).toHaveLength(0);
        expect(screen.getByTestId('trajectory-unavailable')).toHaveTextContent(
            'Trajectory data unavailable',
        );
        expect(screen.queryByTestId('comparison-graph-gas')).not.toBeInTheDocument();
        expect(screen.getByTestId('comparison-hover-content')).toHaveAttribute('data-side', 'right');
        expect(screen.getByTestId('comparison-hover-content')).toHaveAttribute(
            'data-avoid-collisions',
            'true',
        );

        fireEvent.mouseLeave(card);
        expect(screen.queryByTestId('driver-expert-comparison')).not.toBeInTheDocument();

        fireEvent.focus(card);
        expect(screen.getByTestId('driver-expert-comparison')).toBeInTheDocument();

        fireEvent.blur(card);
        expect(screen.queryByTestId('driver-expert-comparison')).not.toBeInTheDocument();
    });

    it('shows comparison unavailability without making an empty card interactive', async () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="unavailable-comparison-card"
                data={{
                    elements: [{
                        id: 'unavailable-comparison',
                        labels: ['MSP'],
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
        expect(within(card).getByText('Expert comparison unavailable')).toBeInTheDocument();
        fireEvent.mouseEnter(card);
        expect(screen.queryByTestId('comparison-hover-content')).not.toBeInTheDocument();
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

    it('normalizes aliases and generates IDs for appended elements', () => {
        const mutation = appendAnalysisResultElement({ elements: [] }, {
            labels: [' Unknown label '],
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
            labels: ['Unknown label'],
            section: 'Section A',
            normalizedPositionRange: { start: 0.1, end: 0.2 },
            timeGap: { deltaMs: 50 },
            metadata: { source: 'form' },
        });
    });

    it('rejects duplicates and invalid or unknown mutation targets', () => {
        const data = normalizeAnalysisResultsData({
            elements: [{ id: 'one', labels: ['Mistake'] }],
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
                { id: 'one', labels: ['Mistake'] },
                { id: 'two', labels: ['Adherence'] },
            ],
        });
        const updated = updateAnalysisResultElement(data, 'one', {
            labels: ['Recovery'],
            metadata: { note: 'kept local' },
        });

        expect(updated.result).toMatchObject({
            success: true,
            data: {
                count: 2,
                element: { id: 'one', labels: ['Recovery'] },
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
                labels: ['MSP'],
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
