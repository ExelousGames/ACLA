import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';

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

jest.mock('contexts/AiLabelsContext', () => ({
    useAiLabels: () => ({
        getCategoryLabels: (category: string) => ({
            MSP: ['MSP1', 'MSP2'],
            MSR: ['MSR1', 'MSR2'],
        }[category] ?? []),
        getLabelName: (labelId: string) => ({
            MSP: 'Mistake (Practice)',
            MSP1: 'Late turn-in',
            MSP2: 'Wheel lock',
            MSR: 'Mistake (Racing)',
            MSR1: 'Failed overtake attempt',
            MSR2: 'Contact',
        }[labelId]),
    }),
}));

jest.mock('components/data-graphs', () => ({
    DataGraph: ({ spec }: any) => {
        const seriesKey = spec.series?.[0]?.key;
        const testId = spec.type === 'bar'
            ? 'mistake-frequency-graph'
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

const renderedResultIds = (): string[] => (
    screen.queryAllByTestId(/^analysis-result-/).map((element) => (
        element.getAttribute('data-testid')?.replace('analysis-result-', '') ?? ''
    ))
);

const selectSortMode = (value: string): void => {
    fireEvent.change(screen.getByRole('combobox', { name: 'Sort by' }), { target: { value } });
};

const selectMainLabel = (value: string): void => {
    fireEvent.change(screen.getByRole('combobox', { name: 'Showing' }), { target: { value } });
};

const renderedFrequencyData = (): Array<{ label: string; occurrences: number }> => (
    JSON.parse(screen.getByTestId('mistake-frequency-graph').getAttribute('data-graph-data') ?? '[]')
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
    it('reports zero and nonzero normalized analysis result counts', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        const view = render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="count-results"
                data={{ elements: [] }}
            />,
        );

        await expect(chartRef.current!.getAnalysisResultCount().result).resolves.toEqual({
            status: 'ready',
            analysis_result_count: 0,
        });

        view.rerender(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="count-results"
                data={{
                    elements: [
                        { id: 'valid-one', labels: ['MSP'] },
                        null,
                        'invalid',
                        { id: 'valid-two', labels: ['MSR'] },
                    ],
                }}
            />,
        );

        await expect(chartRef.current!.getAnalysisResultCount().result).resolves.toEqual({
            status: 'ready',
            analysis_result_count: 2,
        });
    });

    it('reports the active live page total while Overall Trend is selected', async () => {
        const chartRef = React.createRef<AnalysisResultsChartHandle>();
        render(
            <AnalysisResultsChart
                ref={chartRef}
                name="visualization:analysis-results"
                id="active-page-count-results"
                pagination={{
                    pages: [{
                        id: 'active-page',
                        createdAt: 2,
                        baseline: { lap: 2, lap_time_ms: 98_000, track: 'Spa', car: 'GT3' },
                        elements: [
                            { id: 'active-one', labels: ['MSP'] },
                            { id: 'active-two', labels: ['MSR'] },
                        ],
                    }, {
                        id: 'other-page',
                        createdAt: 1,
                        baseline: { lap: 1, lap_time_ms: 99_000, track: 'Spa', car: 'GT3' },
                        elements: [
                            { id: 'other-one', labels: ['MSP'] },
                            { id: 'other-two', labels: ['MSP'] },
                            { id: 'other-three', labels: ['MSP'] },
                        ],
                    }],
                    activePageId: 'active-page',
                    onSelectPage: jest.fn(),
                }}
            />,
        );

        expect(screen.getByText('Overall Trends')).toBeInTheDocument();
        await expect(chartRef.current!.getAnalysisResultCount().result).resolves.toEqual({
            status: 'ready',
            analysis_result_count: 2,
        });
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
                            lap: 1,
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

    it('opens on Overall Trend and navigates chronologically without synthetic callback IDs', () => {
        const chartRef = React.createRef<any>();
        const onUpdate = jest.fn(() => true);
        const onSelectPage = jest.fn();
        const pages = [{
            id: 'page-2',
            createdAt: 2,
            baseline: { lap: 7, lap_time_ms: 98_000, track: 'Monza', car: 'GT4' },
            elements: [{ id: 'second-page-result', labels: ['MSP', 'MSP2'], title: 'Second page mistake' }],
        }, {
            id: 'page-1',
            createdAt: 1,
            baseline: { lap: 4, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
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

        expect(screen.getByText('Page 1 of 3')).toBeInTheDocument();
        expect(screen.getByText('Overall Trends')).toBeInTheDocument();
        expect(screen.getByText('Lap Time Improvement')).toBeInTheDocument();
        expect(screen.getByText('Overall Mistake Trend')).toBeInTheDocument();
        expect(screen.getByText(/Overall Trend.*2 analyzed laps/)).toBeInTheDocument();
        expect(renderedTrendData('overall-total-trend-graph')).toEqual([
            { analysis: 'Analysis 1 · Lap 4', totalCount: 1 },
            { analysis: 'Analysis 2 · Lap 7', totalCount: 1 },
        ]);
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toHaveValue('MSP1');
        fireEvent.change(screen.getByRole('combobox', { name: 'Specific mistake' }), {
            target: { value: 'MSP2' },
        });
        expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled();
        expect(screen.queryByTestId('analysis-result-first-page-result')).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Next' }));

        expect(onSelectPage).toHaveBeenLastCalledWith('page-1');
        expect(screen.getByText('Page 2 of 3')).toBeInTheDocument();
        expect(screen.getByText(/Baseline: Spa.*GT3.*Lap 4/)).toBeInTheDocument();
        expect(screen.getByTestId('analysis-result-first-page-result')).toHaveTextContent('First page mistake');
        expect(screen.queryByTestId('analysis-result-second-page-result')).not.toBeInTheDocument();
        expect(renderedFrequencyData()).toEqual([{ label: 'Late turn-in', occurrences: 1 }]);
        expect(screen.getByRole('button', { name: 'Previous' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled();

        fireEvent.click(screen.getByRole('button', { name: 'Previous' }));
        expect(screen.getByText('Page 1 of 3')).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toHaveValue('MSP2');
        expect(onSelectPage).toHaveBeenCalledTimes(1);

        fireEvent.click(screen.getByRole('button', { name: 'Next' }));
        fireEvent.click(screen.getByRole('button', { name: 'Next' }));

        expect(onSelectPage).toHaveBeenLastCalledWith('page-2');
        expect(screen.getByText('Page 3 of 3')).toBeInTheDocument();
        expect(screen.getByText(/Baseline: Monza.*GT4.*Lap 7/)).toBeInTheDocument();
        expect(screen.getByTestId('analysis-result-second-page-result')).toHaveTextContent('Second page mistake');
        expect(screen.queryByTestId('analysis-result-first-page-result')).not.toBeInTheDocument();
        expect(renderedFrequencyData()).toEqual([{ label: 'Wheel lock', occurrences: 1 }]);
        expect(screen.getByRole('button', { name: 'Previous' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();

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
        expect(screen.getByText('Page 1 of 3')).toBeInTheDocument();
        expect(screen.getByText('Overall Mistake Trend')).toBeInTheDocument();
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

    it('preserves noisy raw lap times and missing-page gaps while fitting the full chronology', () => {
        const pages = [{
            id: 'latest',
            createdAt: 400,
            baseline: { lap: 21, lap_time_ms: 95_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'latest-mistake', labels: ['MSP', 'MSP1'] }],
        }, {
            id: 'first',
            createdAt: 100,
            baseline: { lap: 3, lap_time_ms: 100_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'first-mistake', labels: ['MSP', 'MSP1'] }],
        }, {
            id: 'missing',
            createdAt: 300,
            baseline: { lap: 14, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [],
        }, {
            id: 'slower',
            createdAt: 200,
            baseline: { lap: 9, lap_time_ms: 105_000, track: 'Spa', car: 'GT3' },
            elements: [{ id: 'slower-mistake', labels: ['MSR', 'MSR1'] }],
        }];
        const model = buildLapTimeTrendData(pages);

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

        const graphData = renderedLapTimeTrendData();
        expect(graphData.map(({ lapTimeSeconds }: any) => lapTimeSeconds)).toEqual([100, 105, null, 95]);
        expect(graphData.every(({ bestFitSeconds }: any) => Number.isFinite(bestFitSeconds))).toBe(true);
        expect(screen.getByText('Lap time by analyzed lap (lower is faster).')).toBeInTheDocument();
        expect(screen.getByTestId('lap-time-trend-status')).toHaveTextContent(
            'Latest lap time: 1:35.000. Versus previous timed lap: 0:10.000 faster. '
            + 'Versus first timed lap: 0:05.000 faster. Overall direction: improving.',
        );

        const beforeFilterChange = renderedLapTimeTrendData();
        selectMainLabel('MSR');
        expect(renderedLapTimeTrendData()).toEqual(beforeFilterChange);
    });

    it('counts totals and deduplicated sub-label occurrences across zero-filled laps', () => {
        const pages = [{
            id: 'later-page',
            createdAt: 200,
            baseline: { lap: 12, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'b-wheel-1', labels: ['MSP', 'MSP2'] },
                { id: 'b-wheel-2', labels: ['Mistake (Practice)', 'Wheel lock'] },
                { id: 'b-racing', labels: ['MSR', 'MSR1'] },
            ],
        }, {
            id: 'first-page',
            createdAt: 100,
            baseline: { lap: 8, lap_time_ms: null, track: 'Spa', car: 'GT3' },
            elements: [
                { id: 'a-late', labels: ['MSP', 'MSP1', 'Late turn-in', 'MSP1'] },
                { id: 'a-wheel', labels: ['MSP', 'MSP2'] },
                { id: 'a-parent-only', labels: ['Mistake (Practice)'] },
                { id: 'a-child-without-parent', labels: ['MSP1'] },
            ],
        }, {
            id: 'last-page',
            createdAt: 300,
            baseline: { lap: 15, lap_time_ms: null, track: 'Spa', car: 'GT3' },
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

        expect(renderedTrendData('overall-total-trend-graph')).toEqual([
            { analysis: 'Analysis 1 · Lap 8', totalCount: 3 },
            { analysis: 'Analysis 2 · Lap 12', totalCount: 2 },
            { analysis: 'Analysis 3 · Lap 15', totalCount: 1 },
        ]);
        expect(screen.getByTestId('overall-total-trend-status')).toHaveTextContent(
            'Latest: 1 recognized mistake element. Trending downward — fewer mistakes.',
        );
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toHaveValue('MSP2');
        expect(renderedTrendData('specific-mistake-trend-graph')).toEqual([
            { analysis: 'Analysis 1 · Lap 8', specificCount: 1 },
            { analysis: 'Analysis 2 · Lap 12', specificCount: 2 },
            { analysis: 'Analysis 3 · Lap 15', specificCount: 1 },
        ]);
        expect(screen.getByTestId('specific-mistake-trend-status')).toHaveTextContent('Trend is stable.');

        fireEvent.change(screen.getByRole('combobox', { name: 'Specific mistake' }), {
            target: { value: 'MSP1' },
        });
        expect(renderedTrendData('specific-mistake-trend-graph')).toEqual([
            { analysis: 'Analysis 1 · Lap 8', specificCount: 1 },
            { analysis: 'Analysis 2 · Lap 12', specificCount: 0 },
            { analysis: 'Analysis 3 · Lap 15', specificCount: 0 },
        ]);

        selectMainLabel('MSR');
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toHaveValue('MSR1');
        expect(renderedTrendData('overall-total-trend-graph')).toEqual([
            { analysis: 'Analysis 1 · Lap 8', totalCount: 0 },
            { analysis: 'Analysis 2 · Lap 12', totalCount: 1 },
            { analysis: 'Analysis 3 · Lap 15', totalCount: 2 },
        ]);
        expect(screen.getByTestId('overall-total-trend-status')).toHaveTextContent(
            'Trending upward — more mistakes.',
        );
    });

    it('shows clear empty and single-page trend guidance', () => {
        const { rerender } = render(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="empty-trend"
                pagination={{ pages: [], activePageId: null, onSelectPage: jest.fn() }}
            />,
        );

        expect(screen.getByText('Page 1 of 1')).toBeInTheDocument();
        expect(screen.getByTestId('overall-trend-guidance')).toHaveTextContent(
            'No analyzed laps yet. Analyze at least two baseline laps to see a trend.',
        );
        expect(renderedLapTimeTrendData()).toEqual([]);
        expect(screen.getByTestId('lap-time-trend-status')).toHaveTextContent(
            'Latest lap time unavailable',
        );
        expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
        expect(screen.getByRole('combobox', { name: 'Specific mistake' })).toBeDisabled();

        rerender(
            <AnalysisResultsChart
                name="visualization:analysis-results"
                id="empty-trend"
                pagination={{
                    pages: [{
                        id: 'only-page',
                        createdAt: 1,
                        baseline: { lap: 6, lap_time_ms: 98_567, track: 'Spa', car: 'GT3' },
                        elements: [{ id: 'only-result', labels: ['MSP', 'MSP1'] }],
                    }],
                    activePageId: 'only-page',
                    onSelectPage: jest.fn(),
                }}
            />,
        );

        expect(screen.getByText('Page 1 of 2')).toBeInTheDocument();
        expect(screen.getByTestId('overall-trend-guidance')).toHaveTextContent(
            'Not enough analyzed laps to determine a trend.',
        );
        expect(screen.getByTestId('overall-total-trend-status')).toHaveTextContent(
            'Latest: 1 recognized mistake element. Not enough analyzed laps to determine a trend.',
        );
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

    it('renders arbitrary labels, context, and metadata safely', () => {
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

        expect(screen.getByText('1 of 1 total')).toBeInTheDocument();
        expect(screen.getByText('Future category')).toBeInTheDocument();
        expect(screen.getByText('Recovery')).toBeInTheDocument();
        expect(screen.getByText('Position: 20.0% – 35.0%')).toBeInTheDocument();
        expect(screen.getByText('nested: {"safe":true}')).toBeInTheDocument();
        expect(screen.getByText('score: 0.95')).toBeInTheDocument();
        expect(screen.queryByText(/source|hidden-source-value/)).not.toBeInTheDocument();
        expect(screen.queryByText(/start_index|12345/)).not.toBeInTheDocument();
        expect(screen.queryByText(/end_index|67890/)).not.toBeInTheDocument();
    });

    it('defaults to Training Mistake, recognizes parent IDs and names, and has no All option', () => {
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

        const mainLabelSelect = screen.getByRole('combobox', { name: 'Showing' });
        expect(mainLabelSelect).toHaveValue('MSP');
        expect(within(mainLabelSelect).getAllByRole('option').map((option) => option.textContent)).toEqual([
            'Training Mistake',
            'Racing Mistake',
        ]);
        expect(renderedResultIds()).toEqual(['practice-id', 'practice-name']);
        expect(screen.getByText('2 of 6 total')).toBeInTheDocument();

        selectMainLabel('MSR');

        expect(renderedResultIds()).toEqual(['racing-id', 'racing-name']);
        expect(screen.getByText('2 of 6 total')).toBeInTheDocument();
    });

    it('shows a category-specific empty state when the selected label has no matches', () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="empty"
                data={{ elements: [{ id: 'practice', labels: ['MSP'] }] }}
            />,
        );

        selectMainLabel('MSR');

        expect(renderedResultIds()).toEqual([]);
        expect(screen.getByTestId('analysis-results-empty-state')).toHaveTextContent(
            'No Racing Mistake results yet.',
        );
        expect(screen.getByText('0 of 1 total')).toBeInTheDocument();
    });

    it('keeps filtered source order by default and exposes all sort modes', () => {
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

        expect(renderedResultIds()).toEqual(['third-fastest', 'least-time']);
        expect(screen.getByRole('combobox', { name: 'Sort by' })).toHaveValue('original');
        expect(within(screen.getByRole('combobox', { name: 'Sort by' }))
            .getAllByRole('option').map((option) => ({
                label: option.textContent,
                value: (option as HTMLOptionElement).value,
            }))).toEqual([
            { label: 'Original order', value: 'original' },
            { label: 'Most common training mistake', value: 'most-frequent-sub-label' },
            { label: 'Most time lost', value: 'most-time-lost' },
        ]);
    });

    it('uses category-specific sort wording without changing the selected sort value', () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="dynamic-sort-name"
                data={{
                    elements: [
                        { id: 'practice', labels: ['MSP', 'MSP1'] },
                        { id: 'racing', labels: ['MSR', 'MSR1'] },
                    ],
                }}
            />,
        );

        const sortSelect = screen.getByRole('combobox', { name: 'Sort by' });
        selectSortMode('most-frequent-sub-label');

        expect(sortSelect).toHaveValue('most-frequent-sub-label');
        expect(within(sortSelect).getByRole('option', { selected: true })).toHaveTextContent(
            'Most common training mistake',
        );

        selectMainLabel('MSR');

        expect(sortSelect).toHaveValue('most-frequent-sub-label');
        expect(within(sortSelect).getByRole('option', { selected: true })).toHaveTextContent(
            'Most common racing mistake',
        );
        expect(within(sortSelect).getAllByRole('option').map((option) => option.textContent))
            .not.toEqual(expect.arrayContaining([expect.stringMatching(/label/i)]));
    });

    it('numbers visible results in display order when IDs are hidden', () => {
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

        expect(within(screen.getByTestId('analysis-result-first'))
            .getByLabelText('Analysis result 1')).toHaveTextContent('1');
        expect(within(screen.getByTestId('analysis-result-third'))
            .getByLabelText('Analysis result 2')).toHaveTextContent('2');
        expect(screen.queryByText('first')).not.toBeInTheDocument();

        selectSortMode('most-time-lost');

        expect(renderedResultIds()).toEqual(['third', 'first']);
        expect(within(screen.getByTestId('analysis-result-third'))
            .getByLabelText('Analysis result 1')).toHaveTextContent('1');
        expect(within(screen.getByTestId('analysis-result-first'))
            .getByLabelText('Analysis result 2')).toHaveTextContent('2');
    });

    it('sorts only recognized training sub-labels with aliases, deduplication, and deterministic ties', () => {
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

        selectSortMode('most-frequent-sub-label');

        expect(renderedResultIds()).toEqual([
            'late-id',
            'late-name',
            'multi',
            'wheel-duplicate',
            'wheel-name',
            'unknown-first',
            'racing-sub-label-only',
        ]);
        expect(renderedFrequencyData()).toEqual([
            { label: 'Late turn-in', occurrences: 3 },
            { label: 'Wheel lock', occurrences: 3 },
        ]);
        expect(screen.getByText('Training mistake frequency')).toBeInTheDocument();
        expect(screen.getByTestId('mistake-frequency-graph')).toHaveAttribute(
            'data-graph-orientation',
            'horizontal',
        );
        expect(screen.getByTestId('mistake-frequency-graph')).toHaveAttribute(
            'data-graph-value-axis-label',
            'Occurrences',
        );
        expect(screen.getByTestId('mistake-frequency-graph')).toHaveAttribute(
            'data-graph-colors',
            JSON.stringify(['#00e676']),
        );
        expect(screen.getByText('7 of 9 total')).toBeInTheDocument();
    });

    it('sorts only recognized racing sub-labels and leaves unranked results in source order', () => {
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

        selectMainLabel('MSR');
        selectSortMode('most-frequent-sub-label');

        expect(renderedResultIds()).toEqual([
            'failed-id',
            'failed-name',
            'multi',
            'contact-duplicate',
            'unknown-first',
            'practice-sub-label-only',
            'unknown-second',
        ]);
        expect(renderedFrequencyData()).toEqual([
            { label: 'Failed overtake attempt', occurrences: 3 },
            { label: 'Contact', occurrences: 2 },
        ]);
        expect(screen.getByText('Racing mistake frequency')).toBeInTheDocument();
    });

    it('keeps aggregation independent from card sorting and sizes the graph by category count', () => {
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
        const graph = screen.getByTestId('mistake-frequency-graph');
        const initialData = renderedFrequencyData();

        expect(initialData).toEqual([
            { label: 'Late turn-in', occurrences: 2 },
            { label: 'Wheel lock', occurrences: 2 },
        ]);
        expect(graph).toHaveAttribute('data-graph-height', String(160 + (2 * 36)));

        selectSortMode('most-time-lost');

        expect(renderedResultIds()).toEqual(['wheel', 'both', 'late']);
        expect(renderedFrequencyData()).toEqual(initialData);
    });

    it('shows the graph empty state when matching cards have no recognized sub-labels', () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="empty-frequency"
                data={{ elements: [{ id: 'unknown', labels: ['MSP', 'Unknown mistake'] }] }}
            />,
        );

        expect(renderedResultIds()).toEqual(['unknown']);
        expect(renderedFrequencyData()).toEqual([]);
        expect(screen.getByRole('status')).toHaveTextContent(
            'No recognized training mistakes to graph.',
        );
        expect(screen.getByTestId('mistake-frequency-graph')).toHaveAttribute(
            'data-graph-height',
            String(160 + 36),
        );
    });

    it('sorts numeric time losses descending and leaves invalid values last in source order', () => {
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

        selectSortMode('most-time-lost');

        expect(renderedResultIds()).toEqual([
            'highest',
            'equal-first',
            'equal-second',
            'negative',
            'missing',
            'invalid',
        ]);
    });

    it('retains the selected filter and recalculates sorting when live data changes', () => {
        const { rerender } = render(
            <AnalysisResultsChart name="visualization:analysis-results"
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
        selectMainLabel('MSR');
        selectSortMode('most-frequent-sub-label');
        expect(renderedResultIds()).toEqual(['two', 'one']);

        rerender(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="live-ranking"
                data={{
                    elements: [
                        { id: 'one', labels: ['MSR', 'Unknown racing mistake'] },
                        { id: 'two', labels: ['Mistake (Racing)', 'MSR1'] },
                        { id: 'three', labels: ['MSR', 'Failed overtake attempt'] },
                        { id: 'practice', labels: ['MSP', 'MSP1'] },
                    ],
                }}
            />,
        );

        expect(screen.getByRole('combobox', { name: 'Showing' })).toHaveValue('MSR');
        expect(screen.getByRole('combobox', { name: 'Sort by' })).toHaveValue('most-frequent-sub-label');
        expect(renderedResultIds()).toEqual(['two', 'three', 'one']);
        expect(screen.getByText('3 of 4 total')).toBeInTheDocument();
        expect(renderedFrequencyData()).toEqual([
            { label: 'Failed overtake attempt', occurrences: 2 },
        ]);
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
        expect(screen.queryByText(/Queued:|Skipped:|Live Range To-do List/)).not.toBeInTheDocument();
    });
    it('mounts a collision-aware comparison only while a capable card is hovered or focused', () => {
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

        const card = screen.getByTestId('analysis-result-comparable');
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

    it('shows comparison unavailability without making an empty card interactive', () => {
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

        const card = screen.getByTestId('analysis-result-unavailable-comparison');
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
