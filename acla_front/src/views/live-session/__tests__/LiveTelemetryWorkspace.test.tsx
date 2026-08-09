import React from 'react';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { VisualizationManagerHandle } from 'views/lap-analysis/visualization/VisualizationPanelManager';
import LiveTelemetryWorkspace from '../LiveTelemetryWorkspace';
import { LiveSessionProvider } from '../LiveSessionContext';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = React.forwardRef(({ children, ...props }: any, ref: React.Ref<HTMLDivElement>) => <div ref={ref} {...props}>{children}</div>);
    return {
        Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>, Box: Div,
        Button: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>, Card: Div,
        DropdownMenu: { Root: ({ children }: any) => <>{children}</>, Trigger: ({ children }: any) => <>{children}</>, Content: ({ children }: any) => <div role="menu">{children}</div>, Item: ({ children, ...props }: any) => <button role="menuitem" {...props}>{children}</button> },
        Flex: Div, IconButton: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>, ScrollArea: Div,
        Text: ({ children, as, ...props }: any) => { const Component = as === 'div' ? 'div' : 'span'; return <Component {...props}>{children}</Component>; },
    };
});
jest.mock('@radix-ui/react-icons', () => ({ Cross2Icon: () => <span>Close</span>, DragHandleDots2Icon: () => <span>Drag</span>, PlusIcon: () => <span>Add</span> }));
jest.mock('contexts/AiLabelsContext', () => ({ useAiLabels: () => ({ getCategoryLabels: () => [], getLabelName: () => undefined }) }));
jest.mock('../LiveTrajectoryMap', () => () => <div>Live trajectory map</div>);
jest.mock('../LiveTelemetryOverview', () => ({ name, telemetry }: any) => <div data-testid={name}>Live telemetry {telemetry?.label}</div>);
jest.mock('../LiveEventLog', () => () => <div>Live event log</div>);

describe('LiveTelemetryWorkspace named manager', () => {
    it('offers Analysis Results manually and supports direct data updates', async () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" />);
        expect(ref.current!.getComponentName()).toBe('live-visualization-manager');
        await userEvent.click(screen.getByRole('menuitem', { name: 'Analysis Results' }));
        expect(screen.getByTestId('analysis-results-empty-state')).toBeInTheDocument();

        act(() => {
            ref.current!.updateVisualization('visualization:analysis-results', {
                elements: [{ id: 'result', title: 'Updated result', labels: ['MSP', 'Late braking'] }],
            });
        });
        expect(screen.getByTestId('analysis-result-result')).toHaveTextContent('Updated result');
    });

    it('owns the already-open todo list and closes it through the same manager', async () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<LiveSessionProvider><LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" /></LiveSessionProvider>);
        await userEvent.click(screen.getByRole('menuitem', { name: 'Live Range To-do List' }));
        expect(screen.getByTestId('live-range-todo-list-empty')).toBeInTheDocument();
        expect(ref.current!.getCurrentVisualizations()).toEqual([
            expect.objectContaining({ name: 'live-range-todo-list', type: 'live-range-todo-list' }),
        ]);
        act(() => { ref.current!.closeVisualization({ name: 'live-range-todo-list' }); });
        expect(screen.queryByTestId('live-range-todo-list-empty')).not.toBeInTheDocument();
    });

    it('supports separate speed and brake telemetry displays and same-name reuse', () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" />);
        act(() => {
            ref.current!.requestVisualization({ name: 'telemetry:speed', type: 'telemetry-overview', data: { label: 'speed' } });
            ref.current!.requestVisualization({ name: 'telemetry:brake', type: 'telemetry-overview', data: { label: 'brake' } });
            expect(ref.current!.requestVisualization({ name: 'telemetry:speed', type: 'telemetry-overview', data: { label: 'speed updated' } })).toMatchObject({ reused: true });
        });
        expect(ref.current!.getCurrentVisualizations().map((item) => item.name)).toEqual(['telemetry:speed', 'telemetry:brake']);
        expect(screen.getByTestId('telemetry:speed')).toHaveTextContent('speed updated');
    });

    it('rejects chart types the live workspace does not implement', () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" />);
        act(() => {
            expect(ref.current!.requestVisualization({ name: 'visualization:imitation-guidance-chart', type: 'imitation-guidance-chart' }))
                .toMatchObject({ success: false });
        });
        expect(ref.current!.getCurrentVisualizations()).toEqual([]);
    });
});
