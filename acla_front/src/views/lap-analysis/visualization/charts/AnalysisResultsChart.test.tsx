import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';

const mockRequestVisualization = jest.fn();

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
    DataGraph: ({ spec }: any) => (
        <div
            data-testid="mistake-frequency-graph"
            data-graph-data={JSON.stringify(spec.data)}
            data-graph-height={String(spec.height)}
            data-graph-orientation={spec.orientation}
            data-graph-value-axis-label={spec.xAxisLabel}
            data-graph-colors={JSON.stringify(spec.colors)}
        >
            <span>{spec.title}</span>
            {spec.data.length === 0 && <span role="status">{spec.emptyStateText}</span>}
        </div>
    ),
}));

import AnalysisResultsChart from './AnalysisResultsChart';
import { overlaySessionClient } from 'views/floating-chat/overlay-display-client';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefProvider,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import type { VisualizationManagerHandle } from '../VisualizationPanelManager';
import type {
    LiveRangeTodoEventInput,
    LiveRangeTodoListHandle,
} from 'views/live-session/live-range-todo-list-types';
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

const createQueueHandle = (events: LiveRangeTodoEventInput[]): LiveRangeTodoListHandle => ({
    getComponentName: () => 'live-range-todo-list',
    addEvent: jest.fn((event: LiveRangeTodoEventInput) => {
        events.push(event);
        return { status: 'ready', todo_list: null };
    }),
    replaceEvents: jest.fn(),
    updateEvents: jest.fn(),
    removeEvents: jest.fn(),
    resetEvents: jest.fn(),
    clear: jest.fn(),
    get: jest.fn(),
});

const QueueRegistration = ({ handle }: { handle: LiveRangeTodoListHandle }) => {
    useRegisterAiToolComponentRef(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, handle);
    return null;
};

const ManagerRegistration = () => {
    const handle = React.useMemo<VisualizationManagerHandle>(() => ({
        getComponentName: () => AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER,
        getVisualizationCapabilities: () => ({}),
        getCurrentVisualizations: () => [],
        requestVisualization: (options) => mockRequestVisualization(options),
        updateVisualization: () => ({ success: false, message: 'not used' }),
        closeVisualization: () => ({ success: false, message: 'not used' }),
    }), []);
    useRegisterAiToolComponentRef(AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER, handle);
    return null;
};

const withQueueHandle = (
    chart: React.ReactElement,
    handle: LiveRangeTodoListHandle | null,
) => (
    <AiToolComponentRefProvider>
        <ManagerRegistration />
        {handle && <QueueRegistration handle={handle} />}
        {chart}
    </AiToolComponentRefProvider>
);

describe('AnalysisResultsChart', () => {
    beforeEach(() => {
        mockRequestVisualization.mockReset().mockReturnValue({
            success: true,
            message: 'Opened chart.',
            componentName: AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        });
        localStorage.clear();
    });

    afterEach(() => {
        jest.useRealTimers();
    });

    it('renders arbitrary labels, context, and metadata safely', () => {
        render(withQueueHandle(
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
            null,
        ));

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

    it('appends tied leading occurrences once with fresh IDs, source context, and current-view filtering', () => {
        const existingEvent: LiveRangeTodoEventInput = {
            id: 'existing',
            normalized_position: 0.05,
            content: { title: 'Existing reminder' },
            data: null,
            callback: jest.fn(),
        };
        const queuedEvents = [existingEvent];
        const handle = createQueueHandle(queuedEvents);
        const chart = (
            <AnalysisResultsChart name="visualization:analysis-results"
                id="queue-leading"
                data={{
                    elements: [
                        {
                            id: 'both-leading',
                            labels: ['MSP', 'MSP1', 'MSP2'],
                            title: 'Use a cleaner entry',
                            section: 'Turn 1',
                            normalizedPositionRange: { start: 0.1, end: 0.2 },
                            comparison: comparableData(0.2, 0.4),
                            timeGap: { deltaMs: 5 },
                        },
                        {
                            id: 'late-fallback',
                            labels: ['MSP', 'Late turn-in'],
                            section: 'Turn 2',
                            normalizedPositionRange: { start: 0.2, end: 0.3 },
                            comparison: comparableData(0.3, 0.5),
                            timeGap: { deltaMs: 50 },
                        },
                        {
                            id: 'late-no-comparison',
                            labels: ['MSP', 'MSP1'],
                            normalizedPositionRange: { start: 0.3, end: 0.4 },
                        },
                        {
                            id: 'wheel-no-position',
                            labels: ['MSP', 'MSP2'],
                            comparison: comparableData(0.4, 0.6),
                        },
                        {
                            id: 'wheel-valid',
                            labels: ['MSP', 'Wheel lock'],
                            title: 'Ease off the brake',
                            normalizedPositionRange: { start: 0.5, end: 0.6 },
                            comparison: comparableData(0.5, 0.7),
                            timeGap: { deltaMs: 20 },
                        },
                        {
                            id: 'racing-valid',
                            labels: ['MSR', 'MSR1'],
                            normalizedPositionRange: { start: 0.7, end: 0.8 },
                            comparison: comparableData(0.6, 0.8),
                        },
                    ],
                }}
            />
        );
        render(withQueueHandle(chart, handle));
        selectSortMode('most-time-lost');

        const button = screen.getByRole('button', { name: 'Send most common mistakes' });
        fireEvent.click(button);

        expect(queuedEvents[0]).toBe(existingEvent);
        expect(queuedEvents.slice(1).map((event) => (
            (event.data as any).context.source_result_id
        ))).toEqual(['both-leading', 'late-fallback', 'wheel-valid']);
        expect(queuedEvents.slice(1).every((event) => event.lead_time_seconds === 0)).toBe(true);
        expect(queuedEvents.slice(1).map((event) => event.normalized_position)).toEqual([0.1, 0.2, 0.5]);
        expect(queuedEvents[1].content).toMatchObject({
            title: 'Use a cleaner entry',
            metadata: {
                section: 'Turn 1',
                position: 0.1,
                source_result_id: 'both-leading',
                matched_leading_labels: ['Late turn-in', 'Wheel lock'],
            },
        });
        expect(queuedEvents[2].content.title).toBe('Late turn-in');
        expect(() => JSON.stringify(queuedEvents.slice(1).map((event) => ({
            content: event.content,
            data: event.data,
        })))).not.toThrow();
        expect(screen.getByRole('status')).toHaveTextContent('Queued: 3. Skipped: 2.');
        expect(mockRequestVisualization).not.toHaveBeenCalled();

        const firstClickIds = queuedEvents.slice(1).map((event) => event.id);
        fireEvent.click(button);
        const secondClickIds = queuedEvents.slice(4).map((event) => event.id);
        expect(secondClickIds).toHaveLength(3);
        expect(new Set([...firstClickIds, ...secondClickIds])).toHaveProperty('size', 6);

        selectMainLabel('MSR');
        fireEvent.click(button);
        expect((queuedEvents[7].data as any).context.source_result_id).toBe('racing-valid');
        expect(screen.getByRole('status')).toHaveTextContent('Queued: 1. Skipped: 0.');
    });

    it('disables the action when every leading occurrence lacks a usable position or comparison', () => {
        render(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="queue-disabled"
                data={{
                    elements: [
                        {
                            id: 'invalid-position',
                            labels: ['MSP', 'MSP1'],
                            normalizedPositionRange: { start: -0.1, end: 0.1 },
                            comparison: comparableData(0.2, 0.4),
                        },
                        {
                            id: 'invalid-comparison',
                            labels: ['MSP', 'Late turn-in'],
                            normalizedPositionRange: { start: 0.2, end: 0.3 },
                            comparison: { samples: [{
                                driverTimeMs: 0,
                                expertTimeMs: 0,
                                driverTrackPosition: 0.2,
                                driverGas: 0.2,
                                expertGas: 0.4,
                            }] },
                        },
                    ],
                }}
            />,
        );

        expect(screen.getByRole('button', { name: 'Send most common mistakes' })).toBeDisabled();
    });

    it('opens a closed queue, retains prepared events, and drains them when the handle registers', async () => {
        const queuedEvents: LiveRangeTodoEventInput[] = [];
        const handle = createQueueHandle(queuedEvents);
        const chart = (
            <AnalysisResultsChart name="visualization:analysis-results"
                id="queue-deferred"
                data={{ elements: [{
                    id: 'deferred-result',
                    labels: ['MSP', 'MSP1'],
                    normalizedPositionRange: { start: 0.25, end: 0.3 },
                    comparison: comparableData(0.2, 0.4),
                }] }}
            />
        );
        const view = render(withQueueHandle(chart, null));

        fireEvent.click(screen.getByRole('button', { name: 'Send most common mistakes' }));
        expect(screen.getByRole('button', { name: 'Sending…' })).toBeDisabled();
        await waitFor(() => expect(mockRequestVisualization).toHaveBeenCalledWith({
            name: AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
            type: 'live-range-todo-list',
        }));
        expect(queuedEvents).toHaveLength(0);

        view.rerender(withQueueHandle(chart, handle));
        await waitFor(() => expect(queuedEvents).toHaveLength(1));
        expect((queuedEvents[0].data as any).context.source_result_id).toBe('deferred-result');
        expect(screen.getByRole('status')).toHaveTextContent('Queued: 1. Skipped: 0.');
        expect(screen.getByRole('button', { name: 'Send most common mistakes' })).toBeEnabled();
    });

    it('reports an accessible error after the queue panel mount timeout', async () => {
        jest.useFakeTimers();
        render(withQueueHandle(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="queue-timeout"
                data={{ elements: [{
                    id: 'timeout-result',
                    labels: ['MSP', 'MSP1'],
                    normalizedPositionRange: { start: 0.25, end: 0.3 },
                    comparison: comparableData(0.2, 0.4),
                }] }}
            />,
            null,
        ));

        fireEvent.click(screen.getByRole('button', { name: 'Send most common mistakes' }));
        await act(async () => {
            await Promise.resolve();
            await Promise.resolve();
        });
        expect(mockRequestVisualization).toHaveBeenCalledWith({
            name: AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
            type: 'live-range-todo-list',
        });
        await act(async () => {
            jest.advanceTimersByTime(5000);
            await Promise.resolve();
        });

        expect(screen.getByRole('status')).toHaveTextContent(
            'Unable to open Live Range To-do List. Nothing was queued.',
        );
        expect(screen.getByRole('button', { name: 'Send most common mistakes' })).toBeEnabled();
    });

    it('reports an accessible error when the named manager cannot open the queue', async () => {
        mockRequestVisualization.mockReturnValue({
            success: false,
            message: 'Unable to open chart.',
        });
        render(withQueueHandle(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="queue-open-failure"
                data={{ elements: [{
                    id: 'open-failure-result',
                    labels: ['MSP', 'MSP1'],
                    normalizedPositionRange: { start: 0.25, end: 0.3 },
                    comparison: comparableData(0.2, 0.4),
                }] }}
            />,
            null,
        ));

        fireEvent.click(screen.getByRole('button', { name: 'Send most common mistakes' }));

        await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(
            'Unable to open Live Range To-do List. Nothing was queued.',
        ));
        expect(screen.getByRole('button', { name: 'Send most common mistakes' })).toBeEnabled();
    });

    it('defers an exact comparison snapshot and observes its overlay lifecycle with abort cleanup', async () => {
        jest.useFakeTimers();
        const originalRequestAnimationFrame = window.requestAnimationFrame;
        const originalCancelAnimationFrame = window.cancelAnimationFrame;
        window.requestAnimationFrame = (callback: FrameRequestCallback) => (
            window.setTimeout(() => callback(0), 0)
        );
        window.cancelAnimationFrame = (frameId: number) => window.clearTimeout(frameId);
        let lifecycleListener: ((event: any) => void) | null = null;
        const sendOverlayDisplayRequest = jest.fn(async (request: any) => ({
            presentationId: request.presentationId,
            requestId: request.requestId,
            accepted: true,
            instanceId: 'driver_expert_comparison:multiple:test',
        }));
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: {
                createOverlaySession: jest.fn(async (descriptor: any) => ({
                    success: true,
                    presentation: { ...descriptor, presentationId: 'presentation-analysis' },
                })),
                destroyOverlaySession: jest.fn(async () => ({ success: true, ended: true })),
                setOverlayEnabled: jest.fn(async () => ({ success: true })),
                sendOverlayDisplayRequest,
                onOverlayLifecycle: (listener: (event: any) => void) => {
                    lifecycleListener = listener;
                    return () => { lifecycleListener = null; };
                },
            },
        });
        await overlaySessionClient.create({
            aiSessionId: 'ai-analysis',
            mode: 'recorded',
            displayIdentity: { name: 'Kestrel', agentTags: ['Recorded'] },
        });
        const queuedEvents: LiveRangeTodoEventInput[] = [];
        const comparison = comparableData(0.35, 0.7);
        const handle = createQueueHandle(queuedEvents);
        render(withQueueHandle(
            <AnalysisResultsChart name="visualization:analysis-results"
                id="queue-callback"
                data={{ elements: [{
                    id: 'callback-result',
                    labels: ['MSP', 'MSP1'],
                    title: 'Exact crossing graph',
                    normalizedPositionRange: { start: 0.4, end: 0.5 },
                    comparison,
                }] }}
            />,
            handle,
        ));
        fireEvent.click(screen.getByRole('button', { name: 'Send most common mistakes' }));
        const controller = new AbortController();
        let completed = false;
        const callbackPromise = Promise.resolve(queuedEvents[0].callback({
            signal: controller.signal,
        } as any)).then(() => { completed = true; });

        act(() => jest.advanceTimersByTime(1));
        act(() => jest.advanceTimersByTime(1));
        await act(async () => Promise.resolve());
        expect(sendOverlayDisplayRequest).toHaveBeenCalledWith(expect.objectContaining({
            command: {
                operation: 'upsert',
                type: 'driver_expert_comparison',
                snapshot: {
                title: 'Exact crossing graph',
                comparison,
            },
            },
        }));
        expect(completed).toBe(false);

        act(() => lifecycleListener?.({
            eventId: 'event-1',
            presentationId: 'presentation-analysis',
            instanceId: 'driver_expert_comparison:multiple:test',
            kind: 'exited',
            at: Date.now(),
            reason: 'transient_complete',
        }));
        await act(async () => callbackPromise);
        expect(completed).toBe(true);

        const abortController = new AbortController();
        const abortedPromise = Promise.resolve(queuedEvents[0].callback({
            signal: abortController.signal,
        } as any));
        act(() => jest.advanceTimersByTime(1));
        act(() => jest.advanceTimersByTime(1));
        await act(async () => Promise.resolve());
        act(() => abortController.abort());
        await act(async () => abortedPromise);
        await overlaySessionClient.destroy('presentation-analysis');

        window.requestAnimationFrame = originalRequestAnimationFrame;
        window.cancelAnimationFrame = originalCancelAnimationFrame;
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
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.4');
        expect(screen.getByTestId('trajectory-unavailable')).toHaveTextContent('Track data unavailable');
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
