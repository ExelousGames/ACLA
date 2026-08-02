import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
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

jest.mock('./charts/TelemetryOverview', () => () => null);
jest.mock('./charts/MapVisualization', () => () => null);
jest.mock('./charts/ImitationGuidanceChart', () => () => null);
jest.mock('./charts/EventLogChart', () => () => null);
jest.mock('./charts/AnalysisResultsChart', () => () => null);

import { visualizationController } from './VisualizationController';
import type { VisualizationInstance } from './VisualizationRegistry';
import VisualizationPanelManager from './VisualizationPanelManager';

interface TestInstance {
    id: string;
    type: string;
    height: number;
}

let nextId = 0;

class TestVisualizationPanelManager extends VisualizationPanelManager<{}, TestInstance> {
    protected getManagerTitle() { return 'Test Visualizations'; }
    protected getStaticMapTitle() { return 'Test Map'; }
    protected getPanelTypes() { return ['alpha', 'beta']; }
    protected getPanelName(type: string) { return { alpha: 'Alpha', beta: 'Beta' }[type]; }
    protected createPanelInstance(type: string) {
        return { id: `${type}-${++nextId}`, type, height: 280 };
    }
    protected getPanelHeight(instance: TestInstance) { return instance.height; }
    protected setPanelHeight(instance: TestInstance, height: number) { return { ...instance, height }; }
    protected deserializeControllerInstances(instances: VisualizationInstance[]) {
        return instances
            .filter((instance) => ['alpha', 'beta'].includes(instance.type))
            .map((instance) => ({
                id: instance.id,
                type: instance.type,
                height: typeof instance.position?.height === 'number' ? instance.position.height : 280,
            }));
    }
    protected serializeControllerInstances(instances: TestInstance[]) {
        return instances.map((instance) => ({
            id: instance.id,
            type: instance.type,
            position: { x: 0, y: 0, width: '100%', height: instance.height },
        }));
    }
    protected renderStaticMap() { return <div>Static map content</div>; }
    protected renderPanelContent(instance: TestInstance) { return <div>{instance.type} content</div>; }
    protected getRemoveButtonAriaLabel(instance: TestInstance) { return `Remove ${instance.type}`; }
}

class MockPointerEvent extends MouseEvent {
    pointerId: number;

    constructor(type: string, init: PointerEventInit = {}) {
        super(type, init);
        this.pointerId = init.pointerId ?? 0;
    }
}

describe('VisualizationPanelManager', () => {
    beforeEach(() => {
        nextId = 0;
        visualizationController.setUpdateCallback(() => undefined);
        visualizationController.setCurrentInstances([]);
        (window as any).PointerEvent = MockPointerEvent;
        (HTMLElement.prototype as any).setPointerCapture = jest.fn();
        (HTMLElement.prototype as any).hasPointerCapture = jest.fn(() => true);
        (HTMLElement.prototype as any).releasePointerCapture = jest.fn();
    });

    afterEach(() => {
        visualizationController.setUpdateCallback(() => undefined);
        visualizationController.setCurrentInstances([]);
        jest.restoreAllMocks();
    });

    it('adds unique panels, removes them, and synchronizes controller instances', async () => {
        const managerRef = React.createRef<TestVisualizationPanelManager>();
        render(<TestVisualizationPanelManager ref={managerRef} />);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Alpha' }));
        expect(screen.getByText('alpha content')).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Alpha' })).not.toBeInTheDocument();

        act(() => {
            (managerRef.current as any).addVisualization('alpha');
        });
        expect(visualizationController.getCurrentInstances().map((item) => item.type)).toEqual(['alpha']);

        await userEvent.click(screen.getByRole('button', { name: 'Remove alpha' }));
        expect(screen.queryByText('alpha content')).not.toBeInTheDocument();
        expect(visualizationController.getCurrentInstances()).toEqual([]);
    });

    it('reorders panels through drag and drop', async () => {
        render(<TestVisualizationPanelManager />);
        await userEvent.click(screen.getByRole('menuitem', { name: 'Alpha' }));
        await userEvent.click(screen.getByRole('menuitem', { name: 'Beta' }));

        // These affordances intentionally expose CSS hooks rather than semantic roles.
        // eslint-disable-next-line testing-library/no-node-access
        const containers = Array.from(document.querySelectorAll<HTMLElement>('.visualization-container'));
        // eslint-disable-next-line testing-library/no-node-access
        const sourceHeader = containers[0].querySelector<HTMLElement>('.visualization-header')!;
        let draggedId = '';
        const dataTransfer = {
            effectAllowed: 'none',
            setData: (_type: string, value: string) => { draggedId = value; },
            getData: () => draggedId,
        };

        fireEvent.dragStart(sourceHeader, { dataTransfer });
        fireEvent.drop(containers[1], { dataTransfer });

        expect(visualizationController.getCurrentInstances().map((item) => item.type)).toEqual(['beta', 'alpha']);
    });

    it('resizes with pointer capture and clamps heights to 180-900px', async () => {
        render(<TestVisualizationPanelManager />);
        await userEvent.click(screen.getByRole('menuitem', { name: 'Alpha' }));

        // eslint-disable-next-line testing-library/no-node-access
        const resizeHandle = document.querySelector<HTMLElement>('.visualization-resize-handle')!;
        fireEvent.pointerDown(resizeHandle, { button: 0, pointerId: 1, clientY: 100 });
        fireEvent.pointerMove(resizeHandle, { pointerId: 1, clientY: 2000 });
        fireEvent.pointerUp(resizeHandle, { pointerId: 1, clientY: 2000 });
        expect(visualizationController.getCurrentInstances()[0].position?.height).toBe(900);

        fireEvent.pointerDown(resizeHandle, { button: 0, pointerId: 2, clientY: 100 });
        fireEvent.pointerMove(resizeHandle, { pointerId: 2, clientY: -2000 });
        fireEvent.pointerUp(resizeHandle, { pointerId: 2, clientY: -2000 });
        expect(visualizationController.getCurrentInstances()[0].position?.height).toBe(180);
        expect(HTMLElement.prototype.setPointerCapture).toHaveBeenCalledTimes(2);
        expect(HTMLElement.prototype.releasePointerCapture).toHaveBeenCalledTimes(2);
    });

    it('clears the singleton controller callback on unmount', () => {
        const callbackSpy = jest.spyOn(visualizationController, 'setUpdateCallback');
        const { unmount } = render(<TestVisualizationPanelManager />);

        unmount();

        expect(callbackSpy).toHaveBeenCalledTimes(2);
        expect(callbackSpy.mock.calls[0][0]).not.toBe(callbackSpy.mock.calls[1][0]);
    });
});
