import React from 'react';
import { act, render, screen } from '@testing-library/react';
import { ACLA_DARK_PALETTE } from '../theme';

const mockChart = {
    setOption: jest.fn(),
    resize: jest.fn(),
    dispose: jest.fn(),
};
const mockInit = jest.fn((
    _element?: unknown,
    _theme?: unknown,
    _options?: unknown,
) => mockChart);
const mockUse = jest.fn();

jest.mock('echarts/core', () => ({
    init: (element: unknown, theme?: unknown, options?: unknown) => (
        mockInit(element, theme, options)
    ),
    use: (modules: unknown[]) => mockUse(modules),
}));

jest.mock('echarts/charts', () => ({ BarChart: {}, LineChart: {} }));
jest.mock('echarts/components', () => ({
    AriaComponent: {},
    DatasetComponent: {},
    GridComponent: {},
    TooltipComponent: {},
}));
jest.mock('echarts/renderers', () => ({ CanvasRenderer: {} }));

import type { GraphSpec } from '../types';

const { DataGraph } = require('../DataGraph') as typeof import('../DataGraph');

const registeredModuleCount = mockUse.mock.calls[0]?.[0]?.length;

class MockResizeObserver {
    static instances: MockResizeObserver[] = [];

    callback: ResizeObserverCallback;

    observe = jest.fn();

    disconnect = jest.fn();

    constructor(callback: ResizeObserverCallback) {
        this.callback = callback;
        MockResizeObserver.instances.push(this);
    }

    trigger(): void {
        this.callback([], this as unknown as ResizeObserver);
    }
}

const barSpec = (overrides: Partial<GraphSpec> = {}): GraphSpec => ({
    type: 'bar',
    data: [{ category: 'A', value: 2 }],
    categoryKey: 'category',
    series: [{ key: 'value', label: 'Value' }],
    accessibleLabel: 'Accessible test graph',
    ...overrides,
} as GraphSpec);

describe('DataGraph', () => {
    beforeAll(() => {
        Object.defineProperty(global, 'ResizeObserver', {
            configurable: true,
            writable: true,
            value: MockResizeObserver,
        });
    });

    beforeEach(() => {
        jest.clearAllMocks();
        mockInit.mockReturnValue(mockChart);
        MockResizeObserver.instances = [];
    });

    it('registers only the catalog chart, component, and renderer modules', () => {
        expect(registeredModuleCount).toBe(7);
    });

    it('initializes once, uses canvas, and replaces options on spec updates', () => {
        const { rerender } = render(<DataGraph spec={barSpec()} />);

        expect(mockInit).toHaveBeenCalledTimes(1);
        expect(mockInit).toHaveBeenCalledWith(
            screen.getByTestId('data-graph-canvas'),
            undefined,
            { renderer: 'canvas' },
        );
        expect(mockChart.setOption).toHaveBeenCalledTimes(1);
        expect(mockChart.setOption.mock.calls[0][1]).toEqual({
            notMerge: true,
            lazyUpdate: false,
        });

        rerender(<DataGraph spec={barSpec({ data: [{ category: 'B', value: 4 }] })} />);

        expect(mockInit).toHaveBeenCalledTimes(1);
        expect(mockChart.setOption).toHaveBeenCalledTimes(2);
        expect(mockChart.setOption.mock.calls[1][0].dataset.source).toEqual([
            { category: 'B', value: 4 },
        ]);
    });

    it('applies ARIA descriptions and the ACLA palette by default', () => {
        render(<DataGraph spec={barSpec()} />);

        const option = mockChart.setOption.mock.calls[0][0];
        expect(option.color).toEqual([...ACLA_DARK_PALETTE]);
        expect(option.aria).toEqual({
            enabled: true,
            label: { enabled: true, description: 'Accessible test graph' },
        });
        expect(screen.getByRole('img', { name: 'Accessible test graph' })).toBeInTheDocument();
    });

    it('resizes through ResizeObserver and disconnects and disposes on unmount', () => {
        const { unmount } = render(<DataGraph spec={barSpec()} />);
        const observer = MockResizeObserver.instances[0];

        expect(observer.observe).toHaveBeenCalledWith(screen.getByTestId('data-graph-canvas'));
        act(() => observer.trigger());
        expect(mockChart.resize).toHaveBeenCalledTimes(1);

        unmount();

        expect(observer.disconnect).toHaveBeenCalledTimes(1);
        expect(mockChart.dispose).toHaveBeenCalledTimes(1);
    });

    it('renders configured empty state without initializing ECharts', () => {
        render(<DataGraph spec={barSpec({ data: [], emptyStateText: 'Nothing usable yet.' })} />);

        expect(screen.getByTestId('data-graph-empty-state')).toHaveTextContent('Nothing usable yet.');
        expect(mockInit).not.toHaveBeenCalled();
    });

    it('renders a safe fallback for invalid runtime specifications', () => {
        render(<DataGraph spec={{ type: 'pie' } as unknown as GraphSpec} />);

        expect(screen.getByTestId('data-graph-unsupported-state')).toHaveTextContent(
            'This graph configuration is unsupported.',
        );
        expect(mockInit).not.toHaveBeenCalled();
    });

    it('does not throw when a missing runtime specification reaches the boundary', () => {
        render(<DataGraph spec={null as unknown as GraphSpec} />);

        expect(screen.getByTestId('data-graph-unsupported-state')).toBeInTheDocument();
        expect(mockInit).not.toHaveBeenCalled();
    });

    it('uses the configured height and renders a custom legend when requested', () => {
        render(<DataGraph spec={barSpec({
            height: 420,
            showLegend: true,
            series: [{ key: 'value', label: 'Occurrences', color: '#00e676' }],
        })} />);

        expect(screen.getByTestId('data-graph')).toHaveStyle({ height: '420px' });
        expect(screen.getByLabelText('Graph legend')).toHaveTextContent('Occurrences');
    });
});
