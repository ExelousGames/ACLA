import React from 'react';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = React.forwardRef(({ children, ...props }: any, ref: React.Ref<HTMLDivElement>) => (
        <div ref={ref} {...props}>{children}</div>
    ));

    return {
        Box: Div,
        Button: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>,
        DropdownMenu: {
            Root: ({ children }: any) => <>{children}</>,
            Trigger: ({ children }: any) => <>{children}</>,
            Content: ({ children }: any) => <div role="menu">{children}</div>,
            Item: ({ children, ...props }: any) => <button role="menuitem" {...props}>{children}</button>,
        },
        Flex: Div,
        IconButton: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>,
        Text: ({ children, size, weight, ...props }: any) => <span {...props}>{children}</span>,
    };
});

jest.mock('@radix-ui/react-icons', () => ({
    Cross2Icon: () => <span>Close</span>,
    DragHandleDots2Icon: () => <span>Drag</span>,
    PlusIcon: () => <span>Add</span>,
}));

jest.mock('./charts/TelemetryOverview', () => ({ id, data }: any) => (
    <div data-testid={`recorded-chart-${id}`}>Telemetry chart {data?.label}</div>
));
jest.mock('./charts/MapVisualization', () => ({ id }: any) => <div data-testid={id}>Recorded map</div>);
jest.mock('./charts/ImitationGuidanceChart', () => () => <div>Guidance chart</div>);
jest.mock('./charts/EventLogChart', () => () => <div>Event log chart</div>);
jest.mock('./charts/AnalysisResultsChart', () => () => <div>Analysis results chart</div>);

import DynamicVisualizationManager from './DynamicVisualizationManager';
import { visualizationController } from './VisualizationController';

describe('DynamicVisualizationManager', () => {
    beforeEach(() => {
        visualizationController.setUpdateCallback(() => undefined);
        visualizationController.setCurrentInstances([]);
    });

    afterEach(() => {
        visualizationController.setUpdateCallback(() => undefined);
        visualizationController.setCurrentInstances([]);
    });

    it('uses the recorded registry catalog and always serializes the static map', async () => {
        const onLayoutChange = jest.fn();
        render(<DynamicVisualizationManager onLayoutChange={onLayoutChange} />);

        expect(screen.getByTestId('static-map-visualization')).toHaveTextContent('Recorded map');
        expect(screen.getByRole('menuitem', { name: 'Telemetry Overview' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'AI Driving Guidance' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Event Log' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Track Map' })).not.toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Analysis Results' })).not.toBeInTheDocument();
        expect(visualizationController.getCurrentInstances()).toEqual([
            expect.objectContaining({
                id: 'static-map-visualization',
                type: 'map-visualization',
                position: { x: 0, y: 0, width: '100%', height: '100%' },
            }),
        ]);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Telemetry Overview' }));

        expect(screen.getByTestId(/recorded-chart-telemetry-overview_/)).toHaveTextContent('Telemetry chart');
        expect(visualizationController.getCurrentInstances().map((item) => item.type)).toEqual([
            'map-visualization',
            'telemetry-overview',
        ]);
        expect(onLayoutChange).toHaveBeenLastCalledWith([
            expect.objectContaining({ type: 'telemetry-overview' }),
        ]);
    });

    it('renders controller-supplied recorded chart data and excludes the static map from layout notifications', () => {
        const onLayoutChange = jest.fn();
        render(<DynamicVisualizationManager onLayoutChange={onLayoutChange} />);

        act(() => {
            visualizationController.openVisualization('telemetry-overview', { label: 'from controller' });
        });

        expect(screen.getByText('Telemetry chart from controller')).toBeInTheDocument();
        expect(onLayoutChange).toHaveBeenLastCalledWith([
            expect.objectContaining({ type: 'telemetry-overview', data: { label: 'from controller' } }),
        ]);
        expect(onLayoutChange.mock.calls.at(-1)?.[0]).not.toEqual(expect.arrayContaining([
            expect.objectContaining({ type: 'map-visualization' }),
        ]));
        expect(visualizationController.getCurrentInstances()[0]).toMatchObject({
            id: 'static-map-visualization',
            type: 'map-visualization',
        });
    });
});
