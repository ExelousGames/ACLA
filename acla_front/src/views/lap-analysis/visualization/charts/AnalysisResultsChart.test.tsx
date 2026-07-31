import React from 'react';
import { render, screen } from '@testing-library/react';

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

import AnalysisResultsChart from './AnalysisResultsChart';
import {
    appendAnalysisResultElement,
    normalizeAnalysisResultsData,
    removeAnalysisResultElement,
    updateAnalysisResultElement,
} from './analysisResultsModel';

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
                        metadata: { nested: { safe: true }, score: 0.95 },
                    }],
                }}
            />,
        );

        expect(screen.getByText('1 total')).toBeInTheDocument();
        expect(screen.getByText('Future category')).toBeInTheDocument();
        expect(screen.getByText('Recovery')).toBeInTheDocument();
        expect(screen.getByText('Position: 20.0% – 35.0%')).toBeInTheDocument();
        expect(screen.getByText('nested: {"safe":true}')).toBeInTheDocument();
    });

    it('renders an empty state', () => {
        render(<AnalysisResultsChart id="empty" data={{ elements: [] }} />);
        expect(screen.getByTestId('analysis-results-empty-state')).toHaveTextContent('No analysis results yet.');
        expect(screen.getByText('0 total')).toBeInTheDocument();
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
