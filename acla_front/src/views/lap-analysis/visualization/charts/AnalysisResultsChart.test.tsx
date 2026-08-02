import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';

jest.mock('@radix-ui/themes', () => {
    const ReactModule = require('react');
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

import AnalysisResultsChart from './AnalysisResultsChart';
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

describe('AnalysisResultsChart', () => {
    it('renders arbitrary labels, context, and metadata safely', () => {
        render(
            <AnalysisResultsChart
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
            <AnalysisResultsChart
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
            <AnalysisResultsChart
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
            <AnalysisResultsChart
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
            <AnalysisResultsChart
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
            <AnalysisResultsChart
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
            <AnalysisResultsChart
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
        expect(screen.getByText('7 of 9 total')).toBeInTheDocument();
    });

    it('sorts only recognized racing sub-labels and leaves unranked results in source order', () => {
        render(
            <AnalysisResultsChart
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
    });

    it('sorts numeric time losses descending and leaves invalid values last in source order', () => {
        render(
            <AnalysisResultsChart
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
            <AnalysisResultsChart
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
            <AnalysisResultsChart
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
});
