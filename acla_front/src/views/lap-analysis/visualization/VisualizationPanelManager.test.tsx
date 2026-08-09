import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import VisualizationPanelManager, { ManagedVisualizationInstance } from './VisualizationPanelManager';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = React.forwardRef(({ children, ...props }: any, ref: React.Ref<HTMLDivElement>) => <div ref={ref} {...props}>{children}</div>);
    return {
        Box: Div, Button: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>,
        DropdownMenu: { Root: ({ children }: any) => <>{children}</>, Trigger: ({ children }: any) => <>{children}</>, Content: ({ children }: any) => <div role="menu">{children}</div>, Item: ({ children, ...props }: any) => <button role="menuitem" {...props}>{children}</button> },
        Flex: Div, IconButton: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>, Text: ({ children, size, weight, ...props }: any) => <span {...props}>{children}</span>,
    };
});
jest.mock('@radix-ui/react-icons', () => ({ Cross2Icon: () => <span>Close</span>, DragHandleDots2Icon: () => <span>Drag</span>, PlusIcon: () => <span>Add</span> }));

interface TestInstance extends ManagedVisualizationInstance { height: number; }
let nextId = 0;

class TestVisualizationPanelManager extends VisualizationPanelManager<{ name: string }, TestInstance> {
    protected getManagerTitle() { return 'Test Visualizations'; }
    protected getStaticMapTitle() { return 'Test Map'; }
    protected getPanelTypes() { return ['alpha', 'beta']; }
    protected getPanelName(type: string) { return ({ alpha: 'Alpha', beta: 'Beta' } as Record<string, string>)[type]; }
    protected createPanelInstance(type: string, name: string, data?: any, config?: any) { return { id: `${type}-${++nextId}`, name, type, data, config, height: 280 }; }
    protected getDefaultComponentName(type: string) { return `visualization:${type}`; }
    protected getPanelHeight(instance: TestInstance) { return instance.height; }
    protected setPanelHeight(instance: TestInstance, height: number) { return { ...instance, height }; }
    protected renderStaticMap() { return <div>Static map content</div>; }
    protected renderPanelContent(instance: TestInstance) { return <div>{instance.type} content</div>; }
    protected getRemoveButtonAriaLabel(instance: TestInstance) { return `Remove ${instance.type}`; }
}

class MockPointerEvent extends MouseEvent {
    pointerId: number;
    constructor(type: string, init: PointerEventInit = {}) { super(type, init); this.pointerId = init.pointerId ?? 0; }
}

describe('VisualizationPanelManager concrete ref state', () => {
    beforeEach(() => {
        nextId = 0;
        (window as any).PointerEvent = MockPointerEvent;
        (HTMLElement.prototype as any).setPointerCapture = jest.fn();
        (HTMLElement.prototype as any).hasPointerCapture = jest.fn(() => true);
        (HTMLElement.prototype as any).releasePointerCapture = jest.fn();
    });

    it('adds singleton panels, exposes them through its handle, and removes them', async () => {
        const ref = React.createRef<TestVisualizationPanelManager>();
        render(<TestVisualizationPanelManager ref={ref} name="test-manager" />);
        expect(ref.current!.getComponentName()).toBe('test-manager');
        await userEvent.click(screen.getByRole('menuitem', { name: 'Alpha' }));
        expect(ref.current!.getCurrentVisualizations()).toEqual([expect.objectContaining({ name: 'visualization:alpha', type: 'alpha' })]);
        await userEvent.click(screen.getByRole('button', { name: 'Remove alpha' }));
        expect(ref.current!.getCurrentVisualizations()).toEqual([]);
    });

    it('reorders and resizes fresh concrete state', async () => {
        const ref = React.createRef<TestVisualizationPanelManager>();
        render(<TestVisualizationPanelManager ref={ref} name="test-manager" />);
        await userEvent.click(screen.getByRole('menuitem', { name: 'Alpha' }));
        await userEvent.click(screen.getByRole('menuitem', { name: 'Beta' }));
        const containers = Array.from(document.querySelectorAll<HTMLElement>('.visualization-container'));
        const sourceHeader = containers[0].querySelector<HTMLElement>('.visualization-header')!;
        let draggedId = '';
        const dataTransfer = { effectAllowed: 'none', setData: (_type: string, value: string) => { draggedId = value; }, getData: () => draggedId };
        fireEvent.dragStart(sourceHeader, { dataTransfer });
        fireEvent.drop(containers[1], { dataTransfer });
        expect(ref.current!.getCurrentVisualizations().map((item) => item.type)).toEqual(['beta', 'alpha']);

        const resizeHandle = document.querySelector<HTMLElement>('.visualization-resize-handle')!;
        fireEvent.pointerDown(resizeHandle, { button: 0, pointerId: 1, clientY: 100 });
        fireEvent.pointerMove(resizeHandle, { pointerId: 1, clientY: 2000 });
        fireEvent.pointerUp(resizeHandle, { pointerId: 1, clientY: 2000 });
        expect((ref.current!.getCurrentVisualizations()[0] as TestInstance).height).toBe(900);
    });

    it('reuses non-telemetry types but permits distinct telemetry names', () => {
        const ref = React.createRef<TestVisualizationPanelManager>();
        render(<TestVisualizationPanelManager ref={ref} name="test-manager" />);
        act(() => {
            ref.current!.requestVisualization({ name: 'visualization:alpha', type: 'alpha' });
            expect(ref.current!.requestVisualization({ name: 'alternate-alpha', type: 'alpha' })).toMatchObject({ reused: true, componentName: 'visualization:alpha' });
            ref.current!.requestVisualization({ name: 'telemetry:speed', type: 'telemetry-overview' });
            ref.current!.requestVisualization({ name: 'telemetry:brake', type: 'telemetry-overview' });
        });
        expect(ref.current!.getCurrentVisualizations().map((item) => item.name)).toEqual(['visualization:alpha', 'telemetry:speed', 'telemetry:brake']);
    });
});
