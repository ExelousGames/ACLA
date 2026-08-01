import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';

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
            MSR: ['MSR1'],
        }[category] ?? []),
        getLabelName: (labelId: string) => ({
            MSP: 'Mistake (Practice)',
            MSP1: 'Late turn-in',
            MSP2: 'Wheel lock',
            MSR: 'Mistake (Racing)',
            MSR1: 'Failed overtake attempt',
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
    screen.getAllByTestId(/^analysis-result-/).map((element) => (
        element.getAttribute('data-testid')?.replace('analysis-result-', '') ?? ''
    ))
);

const selectSortMode = (value: string): void => {
    fireEvent.change(screen.getByRole('combobox', { name: 'Sort by' }), { target: { value } });
};

describe('AnalysisResultsChart', () => {
    it('renders arbitrary labels, context, and metadata safely', () => {
        render(
            <AnalysisResultsChart
                id="results"
                data={{
                    elements: [{
                        id: 'future-1',
                        labels: ['Future category', 'Recovery'],
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

        expect(screen.getByText('1 total')).toBeInTheDocument();
        expect(screen.getByText('Future category')).toBeInTheDocument();
        expect(screen.getByText('Recovery')).toBeInTheDocument();
        expect(screen.getByText('Position: 20.0% – 35.0%')).toBeInTheDocument();
        expect(screen.getByText('nested: {"safe":true}')).toBeInTheDocument();
        expect(screen.getByText('score: 0.95')).toBeInTheDocument();
        expect(screen.queryByText(/source|hidden-source-value/)).not.toBeInTheDocument();
        expect(screen.queryByText(/start_index|12345/)).not.toBeInTheDocument();
        expect(screen.queryByText(/end_index|67890/)).not.toBeInTheDocument();
    });

    it('renders an empty state', () => {
        render(<AnalysisResultsChart id="empty" data={{ elements: [] }} />);
        expect(screen.getByTestId('analysis-results-empty-state')).toHaveTextContent('No analysis results yet.');
        expect(screen.getByText('0 total')).toBeInTheDocument();
    });

    it('keeps source order by default and exposes all sort modes', () => {
        render(
            <AnalysisResultsChart
                id="source-order"
                data={{
                    elements: [
                        { id: 'third-fastest', labels: ['Braking', 'Lockup'], timeGap: { deltaMs: 20 } },
                        { id: 'most-time', labels: ['Line', 'Wide exit'], timeGap: { deltaMs: 80 } },
                        { id: 'least-time', labels: ['Braking', 'Lockup'], timeGap: { deltaMs: 5 } },
                    ],
                }}
            />,
        );

        expect(renderedResultIds()).toEqual(['third-fastest', 'most-time', 'least-time']);
        expect(screen.getByRole('combobox', { name: 'Sort by' })).toHaveValue('original');
        expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
            'Original order',
            'Most frequent mistake',
            'Most time lost',
        ]);
    });

    it('sorts only recognized MSP and MSR children by per-card frequency with deterministic ties', () => {
        render(
            <AnalysisResultsChart
                id="frequency-order"
                data={{
                    elements: [
                        { id: 'ignored', labels: ['MSP', 'Mistake (Racing)', 'Telemetry', 'Telemetry'] },
                        { id: 'wheel-duplicate', labels: ['Mistake (Practice)', 'MSP2', 'Wheel lock', 'MSP2'] },
                        { id: 'wheel-name', labels: ['MSP', 'Wheel lock', 'Telemetry'] },
                        { id: 'late-id', labels: ['MSP', 'MSP1'] },
                        { id: 'late-name', labels: ['Mistake (Practice)', 'Late turn-in'] },
                        { id: 'multi', labels: ['Telemetry', 'MSP2', 'Late turn-in'] },
                        { id: 'racing-id', labels: ['MSR', 'MSR1'] },
                        { id: 'racing-name', labels: ['Mistake (Racing)', 'Failed overtake attempt'] },
                    ],
                }}
            />,
        );

        selectSortMode('most-frequent');

        expect(renderedResultIds()).toEqual([
            'late-id',
            'late-name',
            'multi',
            'wheel-duplicate',
            'wheel-name',
            'racing-id',
            'racing-name',
            'ignored',
        ]);
    });

    it('sorts numeric time losses descending and leaves invalid values last in source order', () => {
        render(
            <AnalysisResultsChart
                id="time-order"
                data={{
                    elements: [
                        { id: 'missing', labels: ['Missing'] },
                        { id: 'equal-first', labels: ['Equal first'], timeGap: { deltaMs: 10 } },
                        { id: 'highest', labels: ['Highest'], timeGap: { deltaMs: 25 } },
                        { id: 'invalid', labels: ['Invalid'], timeGap: { deltaMs: 'not-a-number' } },
                        { id: 'equal-second', labels: ['Equal second'], timeGap: { deltaMs: 10 } },
                        { id: 'negative', labels: ['Negative'], timeGap: { deltaMs: -5 } },
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

    it('recalculates the selected ranking when visualization data changes', () => {
        const { rerender } = render(
            <AnalysisResultsChart
                id="live-ranking"
                data={{
                    elements: [
                        { id: 'one', labels: ['MSP2'] },
                        { id: 'two', labels: ['MSP1'] },
                        { id: 'three', labels: ['Late turn-in'] },
                    ],
                }}
            />,
        );
        selectSortMode('most-frequent');
        expect(renderedResultIds()).toEqual(['two', 'three', 'one']);

        rerender(
            <AnalysisResultsChart
                id="live-ranking"
                data={{
                    elements: [
                        { id: 'one', labels: ['MSP2'] },
                        { id: 'two', labels: ['MSP1'] },
                        { id: 'three', labels: ['Wheel lock'] },
                    ],
                }}
            />,
        );

        expect(screen.getByRole('combobox', { name: 'Sort by' })).toHaveValue('most-frequent');
        expect(renderedResultIds()).toEqual(['one', 'three', 'two']);
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
