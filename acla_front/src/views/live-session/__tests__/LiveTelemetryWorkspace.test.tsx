import React, { useContext } from 'react';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { VisualizationManagerHandle } from 'views/lap-analysis/visualization/VisualizationPanelManager';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefProvider,
    useAiToolComponentRefDirectory,
    type AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import LiveTelemetryWorkspace from '../LiveTelemetryWorkspace';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';
import { VisualizationRequestFailedError } from 'contexts/AiToolComponentError';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Div = React.forwardRef(({ children, ...props }: any, ref: React.Ref<HTMLDivElement>) => <div ref={ref} {...props}>{children}</div>);
    return {
        Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>, Box: Div,
        Button: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>, Card: Div,
        DropdownMenu: { Root: ({ children }: any) => <>{children}</>, Trigger: ({ children }: any) => <>{children}</>, Content: ({ children }: any) => <div role="menu">{children}</div>, Item: ({ children, ...props }: any) => <button role="menuitem" {...props}>{children}</button> },
        Flex: Div, HoverCard: { Root: ({ children }: any) => <>{children}</>, Trigger: ({ children }: any) => children, Content: () => null }, IconButton: ({ children, size, variant, ...props }: any) => <button {...props}>{children}</button>, ScrollArea: Div,
        Text: ({ children, as, ...props }: any) => { const Component = as === 'div' ? 'div' : 'span'; return <Component {...props}>{children}</Component>; },
    };
});
jest.mock('@radix-ui/react-icons', () => ({ Cross2Icon: () => <span>Close</span>, PlusIcon: () => <span>Add</span> }));
jest.mock('contexts/AiLabelsContext', () => {
    const getCategoryLabels = () => [];
    const getLabelName = () => undefined;
    return { useAiLabels: () => ({ getCategoryLabels, getLabelName }) };
});
jest.mock('components/data-graphs', () => ({
    DataGraph: ({ spec }: any) => <div data-testid="data-graph">{spec.title}</div>,
}));
jest.mock('../LiveTrajectoryMap', () => () => <div>Live trajectory map</div>);
jest.mock('../LiveTelemetryOverview', () => ({ name, telemetry }: any) => <div data-testid={name}>Live telemetry {telemetry?.label}</div>);
jest.mock('../LiveEventLog', () => () => <div>Live event log</div>);

let directory: AiToolComponentRefDirectory | null = null;
const DirectoryObserver = () => {
    directory = useAiToolComponentRefDirectory();
    return null;
};

describe('LiveTelemetryWorkspace named manager', () => {
    it('adds and removes the live 2D telemetry trajectory', async () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" />);

        expect(screen.queryByText('Live trajectory map')).not.toBeInTheDocument();
        expect(ref.current!.getVisualizationCapabilities().availableCharts).toContainEqual(
            expect.objectContaining({
                type: 'live-trajectory-map',
                name: 'Live 2D Telemetry Trajectory',
                openCount: 0,
                canOpen: true,
            }),
        );

        await userEvent.click(screen.getByRole('menuitem', { name: 'Live 2D Telemetry Trajectory' }));
        expect(screen.getByText('Live trajectory map')).toBeInTheDocument();
        expect(ref.current!.getCurrentVisualizations()).toEqual([
            expect.objectContaining({
                name: 'visualization:live-trajectory-map',
                type: 'live-trajectory-map',
            }),
        ]);

        await userEvent.click(screen.getByRole('button', { name: 'Remove Live 2D Telemetry Trajectory' }));
        expect(screen.queryByText('Live trajectory map')).not.toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Live 2D Telemetry Trajectory' })).toBeInTheDocument();
        expect(ref.current!.getCurrentVisualizations()).toEqual([]);
    });

    it('offers Analysis Results manually with empty-history trend guidance', async () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" />);
        expect(ref.current!.getComponentName()).toBe('live-visualization-manager');
        await userEvent.click(screen.getByRole('menuitem', { name: 'Analysis Results' }));
        expect(screen.getByText('Overall Mistake Trend')).toBeInTheDocument();
        expect(screen.getByTestId('overall-trend-guidance')).toHaveTextContent('No analyzed laps yet.');
        expect(screen.queryByText(/^Page \d+ of \d+$/)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Lap Results' })).toBeDisabled();
    });

    it('returns to Overall Trend after the Analysis Results panel is reopened', async () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        let runtime!: React.ContextType<typeof LiveSessionContext>;
        const Harness = () => {
            runtime = useContext(LiveSessionContext);
            return <LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" />;
        };

        render(<LiveSessionProvider><Harness /></LiveSessionProvider>);
        await userEvent.click(screen.getByRole('menuitem', { name: 'Analysis Results' }));
        act(() => {
            runtime.appendAnalysisResultPage({
                baseline: {
                    id: 'baseline-1',
                    lap_id: 3,
                    lap_time_ms: null,
                    captured_at: 1,
                    track: 'Spa',
                    car: 'GT3',
                    sample_count: 2,
                },
                elements: [{ id: 'retained-result', title: 'Retained result', labels: ['MSP'] }],
            });
        });

        expect(screen.queryByText(/^Page \d+ of \d+$/)).not.toBeInTheDocument();
        expect(screen.getByText('Overall Mistake Trend')).toBeInTheDocument();
        expect(screen.queryByTestId('analysis-result-retained-result')).not.toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
        expect(screen.getByText('Page 1 of 1')).toBeInTheDocument();
        expect(await screen.findByTestId('analysis-result-retained-result')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: 'Remove Analysis Results' }));
        expect(screen.queryByTestId('analysis-result-retained-result')).not.toBeInTheDocument();

        await userEvent.click(screen.getByRole('menuitem', { name: 'Analysis Results' }));
        expect(screen.queryByText(/^Page \d+ of \d+$/)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Overall Trends' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByText('Overall Mistake Trend')).toBeInTheDocument();
        expect(screen.queryByTestId('analysis-result-retained-result')).not.toBeInTheDocument();
    });

    it('does not expose or open the live range to-do runtime as a visualization', () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(<LiveSessionProvider><LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" /></LiveSessionProvider>);
        expect(screen.queryByRole('menuitem', { name: 'Live Range To-do List' })).not.toBeInTheDocument();
        expect(ref.current!.getVisualizationCapabilities().availableCharts).not.toContainEqual(
            expect.objectContaining({ type: 'live-range-todo-list' }),
        );
        expect(() => ref.current!.requestVisualization({
            name: AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
            type: 'live-range-todo-list',
        })).toThrow(expect.objectContaining({
            name: 'VisualizationRequestFailedError',
            componentName: 'live-visualization-manager',
        }));
        expect(ref.current!.getCurrentVisualizations()).toEqual([]);
    });

    it('uses the game captured by the live session for analysis comparisons', async () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        let runtime!: React.ContextType<typeof LiveSessionContext>;
        const Harness = () => {
            const liveSession = useContext(LiveSessionContext);
            runtime = liveSession;
            return (
                <>
                    <button onClick={() => liveSession.startLiveSession('acc')}>Capture ACC</button>
                    <LiveTelemetryWorkspace ref={ref} name="live-visualization-manager" />
                </>
            );
        };
        render(<LiveSessionProvider><Harness /></LiveSessionProvider>);
        await userEvent.click(screen.getByRole('button', { name: 'Capture ACC' }));
        await userEvent.click(screen.getByRole('menuitem', { name: 'Analysis Results' }));

        act(() => {
            runtime.appendAnalysisResultPage({
                baseline: {
                    id: 'acc-baseline',
                    lap_id: 4,
                    lap_time_ms: 98_000,
                    captured_at: 1,
                    track: 'Spa',
                    car: 'GT3',
                    sample_count: 1,
                },
                elements: [{
                    id: 'acc-result',
                    labels: ['MSP'],
                    comparison: {
                        samples: [{
                            driverTimeMs: 0,
                            expertTimeMs: 0,
                            driverTrackPosition: 0.2,
                            expertTrackPosition: 0.2,
                            driverTrajectory: { x: 1, z: 2 },
                            expertTrajectory: { x: 1, y: 2 },
                        }],
                    },
                }],
            });
        });

        await userEvent.click(screen.getByRole('button', { name: 'Lap Results' }));
        expect(await screen.findByTestId('analysis-result-acc-result')).toHaveAttribute('tabindex', '0');
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
        expect(() => ref.current!.requestVisualization({
            name: 'visualization:imitation-guidance-chart',
            type: 'imitation-guidance-chart',
        })).toThrow(VisualizationRequestFailedError);
        expect(() => ref.current!.requestVisualization({
            name: 'visualization:imitation-guidance-chart',
            type: 'imitation-guidance-chart',
        })).toThrow(expect.objectContaining({
            name: 'VisualizationRequestFailedError',
            componentName: 'live-visualization-manager',
        }));
        expect(ref.current!.getCurrentVisualizations()).toEqual([]);
    });

    it('keeps Baseline Collection singleton and unregisters it when closed', () => {
        const ref = React.createRef<VisualizationManagerHandle>();
        render(
            <AiToolComponentRefProvider>
                <DirectoryObserver />
                <LiveSessionProvider>
                    <LiveTelemetryWorkspace ref={ref} name={AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER} />
                </LiveSessionProvider>
            </AiToolComponentRefProvider>,
        );

        act(() => {
            expect(ref.current!.requestVisualization({ name: 'first-name', type: 'baseline-collection' }))
                .toMatchObject({ success: true, reused: false, componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION });
            expect(ref.current!.requestVisualization({ name: 'second-name', type: 'baseline-collection' }))
                .toMatchObject({ success: true, reused: true, componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION });
        });

        expect(ref.current!.getCurrentVisualizations()).toEqual([
            expect.objectContaining({
                name: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                type: 'baseline-collection',
            }),
        ]);
        expect(directory!.findComponentRef(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)?.current).not.toBeNull();

        act(() => { ref.current!.closeVisualization({ type: 'baseline-collection' }); });
        expect(directory!.findComponentRef(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)).toBeNull();
    });
});
