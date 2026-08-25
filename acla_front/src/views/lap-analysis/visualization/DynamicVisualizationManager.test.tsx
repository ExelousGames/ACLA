import React from 'react';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
    AiToolComponentRefDirectory,
    AiToolComponentRefProvider,
    useAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import type { VisualizationManagerHandle } from './VisualizationPanelManager';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = React.forwardRef(({ children, ...props }: any, ref: React.Ref<HTMLDivElement>) => <div ref={ref} {...props}>{children}</div>);
    return {
        Box: Div,
        Button: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>,
        DropdownMenu: {
            Root: ({ children }: any) => <>{children}</>, Trigger: ({ children }: any) => <>{children}</>,
            Content: ({ children }: any) => <div role="menu">{children}</div>,
            Item: ({ children, ...props }: any) => <button role="menuitem" {...props}>{children}</button>,
        },
        Flex: Div,
        IconButton: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>,
        Text: ({ children, size, weight, ...props }: any) => <span {...props}>{children}</span>,
    };
});
jest.mock('@radix-ui/react-icons', () => ({
    Cross2Icon: () => <span>Close</span>, PlusIcon: () => <span>Add</span>,
}));
jest.mock('./charts/TelemetryOverview', () => ({ id, data }: any) => <div data-testid={`recorded-chart-${id}`}>Telemetry chart {data?.label}</div>);
jest.mock('./charts/MapVisualization', () => {
    const React = require('react');
    return React.forwardRef(({ id, name }: any, ref: React.Ref<any>) => {
        React.useImperativeHandle(ref, () => ({
            getComponentName: () => name,
            updateMap: () => true,
            disableMap: () => true,
        }), [name]);
        return <div data-testid={id}>Recorded map</div>;
    });
});
jest.mock('./charts/ImitationGuidanceChart', () => () => <div>Guidance chart</div>);
jest.mock('./charts/EventLogChart', () => () => <div>Event log chart</div>);
jest.mock('./charts/AnalysisResultsChart', () => () => <div>Analysis results chart</div>);

import DynamicVisualizationManager from './DynamicVisualizationManager';

let componentDirectory: AiToolComponentRefDirectory | null = null;

const ComponentDirectoryObserver = () => {
    componentDirectory = useAiToolComponentRefDirectory();
    return null;
};

describe('DynamicVisualizationManager named ref', () => {
    it('uses the recorded catalog and reports manual layout changes', async () => {
        const onLayoutChange = jest.fn();
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<DynamicVisualizationManager ref={ref} name="recorded-visualization-manager" onLayoutChange={onLayoutChange} />);

        expect(ref.current!.getComponentName()).toBe('recorded-visualization-manager');
        expect(screen.getByTestId('static-map-visualization')).toHaveTextContent('Recorded map');
        expect(screen.getByRole('menuitem', { name: 'Telemetry Overview' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Analysis Results' })).not.toBeInTheDocument();

        await userEvent.click(screen.getByRole('menuitem', { name: 'Telemetry Overview' }));
        expect(ref.current!.getCurrentVisualizations()).toEqual([
            expect.objectContaining({ name: 'telemetry:general', type: 'telemetry-overview' }),
        ]);
        expect(onLayoutChange).toHaveBeenLastCalledWith([
            expect.objectContaining({ name: 'telemetry:general', type: 'telemetry-overview' }),
        ]);
    });

    it('creates and updates an exact semantic child through its concrete handle', () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<DynamicVisualizationManager ref={ref} name="recorded-visualization-manager" />);

        act(() => {
            expect(ref.current!.requestVisualization({
                name: 'telemetry:speed', type: 'telemetry-overview', data: { label: 'speed' },
            })).toMatchObject({ success: true, reused: false, componentName: 'telemetry:speed' });
        });
        expect(screen.getByText('Telemetry chart speed')).toBeInTheDocument();

        act(() => {
            expect(ref.current!.requestVisualization({
                name: 'telemetry:speed', type: 'telemetry-overview', data: { label: 'updated' },
            })).toMatchObject({ success: true, reused: true, componentName: 'telemetry:speed' });
        });
        expect(screen.getByText('Telemetry chart updated')).toBeInTheDocument();
        expect(ref.current!.getCurrentVisualizations()).toHaveLength(1);
    });

    it('registers the permanent map from its parent manager', () => {
        render(
            <AiToolComponentRefProvider>
                <ComponentDirectoryObserver />
                <DynamicVisualizationManager name="recorded-visualization-manager" />
            </AiToolComponentRefProvider>,
        );

        const map = componentDirectory!
            .findComponentRef<any>('visualization:map-visualization')!
            .current;
        expect(map.getComponentName()).toBe('visualization:map-visualization');
        expect(map.updateMap({ selected: true })).toBe(true);
    });
});
