import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationController';
import LiveTelemetryWorkspace from '../LiveTelemetryWorkspace';
import { LiveSessionProvider } from '../LiveSessionContext';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = React.forwardRef(({ children, ...props }: any, ref: React.Ref<HTMLDivElement>) => (
        <div ref={ref} {...props}>{children}</div>
    ));

    return {
        Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
        Box: Div,
        Button: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>,
        Card: Div,
        DropdownMenu: {
            Root: ({ children }: any) => <>{children}</>,
            Trigger: ({ children }: any) => <>{children}</>,
            Content: ({ children }: any) => <div role="menu">{children}</div>,
            Item: ({ children, ...props }: any) => <button role="menuitem" {...props}>{children}</button>,
        },
        Flex: Div,
        IconButton: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>,
        ScrollArea: Div,
        Text: ({ children, as, ...props }: any) => {
            const Component = as === 'div' ? 'div' : 'span';
            return <Component {...props}>{children}</Component>;
        },
    };
});

jest.mock('@radix-ui/react-icons', () => ({
    Cross2Icon: () => <span>Close</span>,
    DragHandleDots2Icon: () => <span>Drag</span>,
    PlusIcon: () => <span>Add</span>,
}));

jest.mock('contexts/AiLabelsContext', () => ({
    useAiLabels: () => ({
        getCategoryLabels: () => [],
        getLabelName: () => undefined,
    }),
}));

jest.mock('../LiveTrajectoryMap', () => () => <div>Live trajectory map</div>);
jest.mock('../LiveTelemetryOverview', () => () => <div>Live telemetry overview</div>);
jest.mock('../LiveEventLog', () => () => <div>Live event log</div>);

describe('LiveTelemetryWorkspace analysis results', () => {
    beforeEach(() => {
        visualizationController.setUpdateCallback(() => undefined);
        visualizationController.setCurrentInstances([]);
    });

    afterEach(() => {
        visualizationController.setUpdateCallback(() => undefined);
        visualizationController.setCurrentInstances([]);
    });

    it('offers Analysis Results in the manual menu and renders its existing empty state', async () => {
        render(<LiveTelemetryWorkspace />);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Analysis Results' }));

        expect(screen.getByTestId('analysis-results-empty-state')).toHaveTextContent('No Training Mistake results yet.');
        expect(screen.getByRole('button', { name: 'Remove Analysis Results' })).toBeInTheDocument();
    });

    it('renders controller-supplied analysis elements', () => {
        render(<LiveTelemetryWorkspace />);

        act(() => {
            visualizationController.openVisualization('analysis-results', {
                elements: [{ id: 'controller-result', title: 'Controller result', labels: ['MSP', 'Late braking'] }],
            });
        });

        expect(screen.getByTestId('analysis-result-controller-result')).toHaveTextContent('Controller result');
        expect(screen.getByLabelText('Analysis result 1')).toHaveTextContent('1');
        expect(screen.queryByText('controller-result')).not.toBeInTheDocument();
        expect(screen.getByText('Late braking')).toBeInTheDocument();
        expect(screen.getByTestId('live-visualization-content-analysis-results')).toHaveStyle({
            overflowY: 'auto',
            overflowX: 'hidden',
        });
    });

    it('offers one Live Range To-do List panel and supports controller removal', async () => {
        render(<LiveSessionProvider><LiveTelemetryWorkspace /></LiveSessionProvider>);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Live Range To-do List' }));

        expect(screen.getByTestId('live-range-todo-list-empty')).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Live Range To-do List' })).not.toBeInTheDocument();
        const warn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
        expect(visualizationController.openVisualization('live-range-todo-list').success).toBe(false);
        warn.mockRestore();

        const panel = visualizationController.getCurrentInstances()
            .find((item) => item.type === 'live-range-todo-list');
        expect(panel).toBeDefined();
        act(() => {
            visualizationController.closeVisualization({ id: panel!.id });
        });
        expect(screen.queryByTestId('live-range-todo-list-empty')).not.toBeInTheDocument();
    });

    it('replaces controller-updated data on the existing panel', () => {
        render(<LiveTelemetryWorkspace />);

        act(() => {
            visualizationController.openVisualization('analysis-results', {
                elements: [{ id: 'old-result', title: 'Old result', labels: ['MSP', 'Old'] }],
            });
        });
        const instance = visualizationController.getCurrentInstances()
            .find((item) => item.type === 'analysis-results');
        expect(instance).toBeDefined();

        act(() => {
            visualizationController.executeCommand({
                action: 'update',
                id: instance!.id,
                data: { elements: [{ id: 'new-result', title: 'New result', labels: ['MSP', 'New'] }] },
            });
        });

        expect(screen.queryByTestId('analysis-result-old-result')).not.toBeInTheDocument();
        expect(screen.getByTestId('analysis-result-new-result')).toHaveTextContent('New result');
        expect(screen.getAllByRole('button', { name: 'Remove Analysis Results' })).toHaveLength(1);
    });

    it('filters controller instances that are unsupported in the live workspace', () => {
        render(<LiveTelemetryWorkspace />);
        let controllerResult: ReturnType<typeof visualizationController.openVisualization> | undefined;

        act(() => {
            controllerResult = visualizationController.openVisualization(
                'imitation-guidance-chart',
                { label: 'recorded only' },
            );
        });

        expect(controllerResult).toEqual(expect.objectContaining({ success: true, chartId: expect.any(String) }));
        expect(screen.queryByText('AI Driving Guidance')).not.toBeInTheDocument();
        expect(visualizationController.getCurrentInstances()).toEqual([]);
    });

    it('keeps existing live panel duplicate prevention, ordering, resizing, and removal behavior', async () => {
        class MockPointerEvent extends MouseEvent {
            pointerId: number;

            constructor(type: string, init: PointerEventInit = {}) {
                super(type, init);
                this.pointerId = init.pointerId ?? 0;
            }
        }
        (window as any).PointerEvent = MockPointerEvent;
        (HTMLElement.prototype as any).setPointerCapture = jest.fn();
        (HTMLElement.prototype as any).hasPointerCapture = jest.fn(() => true);
        (HTMLElement.prototype as any).releasePointerCapture = jest.fn();
        render(<LiveTelemetryWorkspace />);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Live Telemetry Overview' }));
        expect(screen.queryByRole('menuitem', { name: 'Live Telemetry Overview' })).not.toBeInTheDocument();
        await userEvent.click(screen.getByRole('menuitem', { name: 'Live Event Log' }));
        expect(visualizationController.getCurrentInstances().map((item) => item.type)).toEqual([
            'telemetry-overview',
            'event-log',
        ]);

        // These drag and resize affordances have no semantic role, so the lifecycle test targets their CSS hooks.
        // eslint-disable-next-line testing-library/no-node-access
        const containers = Array.from(document.querySelectorAll<HTMLElement>('.visualization-container'));
        // eslint-disable-next-line testing-library/no-node-access
        const sourceHeader = containers[0].querySelector<HTMLElement>('.visualization-header')!;
        let draggedId = '';
        const dataTransfer = {
            setData: (_type: string, value: string) => { draggedId = value; },
            getData: () => draggedId,
        };
        fireEvent.dragStart(sourceHeader, { dataTransfer });
        fireEvent.drop(containers[1], { dataTransfer });
        expect(visualizationController.getCurrentInstances().map((item) => item.type)).toEqual([
            'event-log',
            'telemetry-overview',
        ]);

        // eslint-disable-next-line testing-library/no-node-access
        const resizeHandle = document.querySelector<HTMLElement>('.visualization-resize-handle')!;
        fireEvent.pointerDown(resizeHandle, { button: 0, pointerId: 1, clientY: 100 });
        fireEvent.pointerMove(resizeHandle, { pointerId: 1, clientY: 260 });
        fireEvent.pointerUp(resizeHandle, { pointerId: 1, clientY: 260 });
        expect(visualizationController.getCurrentInstances()[0].position?.height).toBe(440);

        await userEvent.click(screen.getByRole('button', { name: 'Remove Live Event Log' }));
        expect(screen.queryByText('Live event log')).not.toBeInTheDocument();
        expect(visualizationController.getCurrentInstances().map((item) => item.type)).toEqual([
            'telemetry-overview',
        ]);
    });
});
