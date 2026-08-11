import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefProvider,
    useAiToolComponentRefDirectory,
    type AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import { getToolEnvelopeUiOutput } from 'views/lap-analysis/ai-chat/ai-tool-base';
import BaselineCollection, {
    getCompletedBaselineLapTimeMs,
    type BaselineCollectionHandle,
} from '../BaselineCollection';
import { LiveSessionContext } from '../LiveSessionContext';

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: jest.fn() },
}));

const mockPost = apiService.post as jest.Mock;

const analysisResult = {
    status: 'success',
    session_id: 'baseline-analysis-1',
    samples_analyzed: 3,
    parent_segment_count: 2,
    segments: [{
        id: 'segment-1',
        labels: ['MSP'],
        track_section: 'turn-1',
        start_index: 0,
        end_index: 1,
        expert_reference_data: [{
            expert_optimal_time: 4,
            Graphics_normalized_car_position: 0.001,
        }, {
            expert_optimal_time: 39_000,
            Graphics_normalized_car_position: 0.4,
        }],
    }, {
        id: 'segment-2',
        labels: ['EA'],
        track_section: 'turn-2',
        start_index: 1,
        end_index: 2,
        expert_reference_data: [],
    }],
};

const makeSample = (lap: number, position: number, currentTime: number, lastTime?: unknown) => ({
    Static_track: 'brands_hatch',
    Static_car_model: 'Ferrari 296',
    Static_num_cars: 1,
    Graphics_completed_laps: lap,
    Graphics_normalized_car_position: position,
    Graphics_current_time: currentTime,
    ...(lastTime === undefined ? {} : { Graphics_last_time: lastTime }),
});

let directory: AiToolComponentRefDirectory | null = null;
let appendedPages: any[] = [];
const appendAnalysisResultPage = jest.fn();

const DirectoryObserver = () => {
    directory = useAiToolComponentRefDirectory();
    return null;
};

const Harness = ({
    telemetry,
    show = true,
}: {
    telemetry: Record<string, any>;
    show?: boolean;
}) => (
    <AiToolComponentRefProvider>
        <DirectoryObserver />
        <LiveSessionContext.Provider value={{ currentTelemetry: telemetry, appendAnalysisResultPage } as any}>
            {show && <BaselineCollection name={AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION} />}
        </LiveSessionContext.Provider>
    </AiToolComponentRefProvider>
);

const getHandle = () => directory!
    .findComponentRef<BaselineCollectionHandle>(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)!
    .current!;

const reserve = (name: string, value: Record<string, any>) => {
    directory!.reserveComponentRef(name, Symbol(name), {
        getComponentName: () => name,
        ...value,
    } as any);
};

const installAnalysisComponents = ({ existingChart = false } = {}) => {
    const chartName = 'visualization:analysis-results';
    const visualizations: any[] = existingChart
        ? [{ id: 'analysis-chart-1', name: chartName, type: 'analysis-results' }]
        : [];

    const manager = {
        getCurrentVisualizations: jest.fn(() => visualizations),
        requestVisualization: jest.fn((options: any) => {
            const instance = { id: 'analysis-chart-1', ...options };
            visualizations.push(instance);
            reserve(chartName, {});
            return {
                success: true,
                message: 'Opened chart.',
                componentName: chartName,
                chartId: instance.id,
                chartType: instance.type,
                reused: false,
            };
        }),
    };
    act(() => {
        reserve(AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, {
            getLabelName: jest.fn((id: string) => ({
                MSP: 'Mistake (Practice)',
                EA: 'Expert Adherence',
                'turn-1': 'Turn One',
                'turn-2': 'Turn Two',
            }[id] || id)),
        });
        if (existingChart) reserve(chartName, {});
        reserve(AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER, manager);
    });
    return { manager, visualizations };
};

const completeBaselineLap = (
    view: ReturnType<typeof render>,
    lap = 5,
    start = true,
    lastTime: unknown = 98_765,
) => {
    const handle = getHandle();
    if (start) act(() => { handle.startCollection(); });
    view.rerender(<Harness telemetry={makeSample(lap - 1, 0.9, 90_000)} />);
    view.rerender(<Harness telemetry={makeSample(lap, 0.001, 5)} />);
    view.rerender(<Harness telemetry={makeSample(lap, 0.4, 40_000)} />);
    view.rerender(<Harness telemetry={makeSample(lap, 0.98, 98_000)} />);
    view.rerender(<Harness telemetry={makeSample(lap + 1, 0.001, 5, lastTime)} />);
    return handle;
};

describe('BaselineCollection visualization', () => {
    beforeEach(() => {
        directory = null;
        appendedPages = [];
        appendAnalysisResultPage.mockReset().mockImplementation((input: any) => {
            const result = {
                pageId: `baseline-analysis-page-${appendedPages.length + 1}`,
                pageCount: appendedPages.length + 1,
            };
            appendedPages.push({ ...input, ...result });
            return result;
        });
        mockPost.mockReset().mockResolvedValue({ data: analysisResult });
    });

    it('prefers exact last-lap timing, falls back to the highest valid sample time, and rejects invalid timing', () => {
        const rows = [
            { Graphics_current_time: 5 },
            { Graphics_current_time: 98_000 },
            { Graphics_current_time: -1 },
            { Graphics_current_time: Number.POSITIVE_INFINITY },
        ];

        expect(getCompletedBaselineLapTimeMs({ Graphics_last_time: 98_765 }, rows)).toBe(98_765);
        expect(getCompletedBaselineLapTimeMs({ Graphics_last_time: 0 }, rows)).toBe(98_000);
        expect(getCompletedBaselineLapTimeMs(
            { Graphics_last_time: -10 },
            [{ Graphics_current_time: 0 }, { Graphics_current_time: Number.NaN }],
        )).toBeNull();
    });

    it('starts collection from the visualization', () => {
        render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = getHandle();

        expect(handle.getTag()).toBeNull();
        act(() => {
            screen.getByRole('button', { name: 'Start Baseline Collection' }).click();
        });

        expect(handle.getTag()).toMatchObject({
            status: 'waiting_for_start',
            progress_percent: 0,
        });
        expect(screen.queryByRole('button', { name: 'Start Baseline Collection' }))
            .not.toBeInTheDocument();
        expect(screen.getByLabelText('Baseline collection progress')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Request Analysis' }))
            .not.toBeInTheDocument();
    });

    it('waits for the next boundary, records one lap, reports progress, and emits completion once', () => {
        const outputs: any[] = [];
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = getHandle();
        handle.subscribeToolOutput((output) => outputs.push(output));

        act(() => {
            expect(handle.startCollection('goal-baseline-run')).toMatchObject({
                status: 'waiting_for_start',
                progress_percent: 0,
            });
        });
        expect(handle.getLapRecord()).toBeNull();

        view.rerender(<Harness telemetry={makeSample(4, 0.9, 90_000)} />);
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start', progress_percent: 0 });

        view.rerender(<Harness telemetry={makeSample(5, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(5, 0.4, 40_000)} />);
        expect(handle.getTag()).toMatchObject({ status: 'collecting', progress_percent: 40, baseline_lap: 5 });

        view.rerender(<Harness telemetry={makeSample(5, 0.98, 98_000)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.2, 20_000)} />);

        expect(handle.getLapRecord()).toMatchObject({
            lap: 5,
            lap_time_ms: 98_000,
            track: 'brands_hatch',
            car: 'Ferrari 296',
            sample_count: 3,
        });
        expect(handle.getLapRecord()?.records.map((row) => row.Graphics_normalized_car_position))
            .toEqual([0.001, 0.4, 0.98]);
        expect(handle.getTag()).toMatchObject({ status: 'complete', progress_percent: 100 });
        expect(screen.getByRole('button', { name: 'Request Analysis' })).toBeEnabled();
        expect(outputs).toHaveLength(1);
        expect(outputs[0]).toMatchObject({
            tool_name: 'collect_live_baseline',
            run_id: 'goal-baseline-run',
            final: true,
        });
        expect(getToolEnvelopeUiOutput(outputs[0])).toEqual({
            progress_percent: 100,
            status: 'complete',
            car: 'Ferrari 296',
            track: 'brands_hatch',
            message: 'Baseline complete. Cached lap record is ready.',
        });
        expect(handle.getToolOutput()).toBe(outputs[0]);
    });

    it('completes when position wraps before the lap counter advances', () => {
        const view = render(<Harness telemetry={{}} />);
        const handle = getHandle();
        act(() => { handle.startCollection(); });

        view.rerender(<Harness telemetry={makeSample(3, 0.001, 10)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.5, 50_000)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.99, 99_000)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.002, 20)} />);

        expect(handle.getLapRecord()).toMatchObject({ lap: 3, sample_count: 3 });
        expect(handle.getTag()).toMatchObject({ status: 'complete', progress_percent: 100 });
    });

    it('restarts the mounted collector and waits for a fresh lap start', () => {
        const view = render(<Harness telemetry={{}} />);
        const handle = getHandle();
        act(() => { handle.startCollection(); });
        view.rerender(<Harness telemetry={makeSample(0, 0.001, 10)} />);
        view.rerender(<Harness telemetry={makeSample(0, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.001, 5)} />);
        expect(handle.getLapRecord()).not.toBeNull();

        act(() => { handle.restartCollection(); });
        expect(handle.getLapRecord()).toBeNull();
        expect(handle.getToolOutput()).toBeNull();
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start', baseline_lap: null });
        expect(screen.queryByRole('button', { name: /Analysis/ })).not.toBeInTheDocument();

        view.rerender(<Harness telemetry={makeSample(1, 0.002, 20)} />);
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start' });
        view.rerender(<Harness telemetry={makeSample(1, 0.4, 40_000)} />);
        view.rerender(<Harness telemetry={makeSample(2, 0.001, 5)} />);
        expect(handle.getTag()).toMatchObject({ status: 'collecting', baseline_lap: 2 });
    });

    it('unregisters and discards partial or completed state when closed, then reopens fresh', () => {
        const view = render(<Harness telemetry={{}} />);
        const firstHandle = getHandle();
        act(() => { firstHandle.startCollection(); });
        view.rerender(<Harness telemetry={makeSample(0, 0.001, 10)} />);
        view.rerender(<Harness telemetry={makeSample(0, 0.5, 50_000)} />);
        expect(firstHandle.getTag()).toMatchObject({ status: 'collecting' });

        view.rerender(<Harness telemetry={makeSample(0, 0.5, 50_000)} show={false} />);
        expect(directory!.findComponentRef(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)).toBeNull();
        expect(firstHandle.getTag()).toBeNull();
        expect(firstHandle.getLapRecord()).toBeNull();
        expect(firstHandle.getToolOutput()).toBeNull();

        view.rerender(<Harness telemetry={makeSample(0, 0.5, 50_000)} />);
        const secondHandle = getHandle();
        expect(secondHandle).not.toBe(firstHandle);
        expect(secondHandle.getTag()).toBeNull();
        act(() => { secondHandle.startCollection(); });
        expect(secondHandle.getTag()).toMatchObject({ status: 'waiting_for_start', baseline_lap: null });
        expect(screen.getByLabelText('Baseline collection progress')).toBeInTheDocument();

        view.rerender(<Harness telemetry={makeSample(0, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(2, 0.001, 5)} />);
        expect(secondHandle.getLapRecord()).not.toBeNull();

        view.rerender(<Harness telemetry={makeSample(2, 0.001, 5)} show={false} />);
        expect(secondHandle.getTag()).toBeNull();
        expect(secondHandle.getLapRecord()).toBeNull();
        expect(secondHandle.getToolOutput()).toBeNull();
    });

    it('owns the exact cached-lap request, normalization, labels, comparison, and result payload', async () => {
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        const { manager } = installAnalysisComponents();

        let payload: Awaited<ReturnType<BaselineCollectionHandle['requestAnalysis']>> | undefined;
        await act(async () => {
            payload = await handle.requestAnalysis({ limit: 1 });
        });

        const record = handle.getLapRecord()!;
        expect(mockPost).toHaveBeenCalledWith(
            '/racing-session/analyze-live-recorded-analysis',
            {
                track: record.track,
                car: record.car,
                baseline_lap: record.lap,
                records: record.records,
            },
            { timeout: 120000 },
        );
        expect(manager.requestVisualization).toHaveBeenCalledWith({
            name: 'visualization:analysis-results',
            type: 'analysis-results',
        });
        expect(appendAnalysisResultPage).toHaveBeenCalledWith({
            baseline: {
                id: record.id,
                lap: record.lap,
                lap_time_ms: 98_765,
                captured_at: record.captured_at,
                track: 'brands_hatch',
                car: 'Ferrari 296',
                sample_count: 3,
            },
            elements: [
                expect.objectContaining({
                    id: 'segment-1',
                    labels: ['Mistake (Practice)'],
                    section: 'Turn One',
                    comparison: {
                        samples: [
                            expect.objectContaining({
                                driverTimeMs: 5,
                                expertTimeMs: 4,
                                driverTrackPosition: 0.001,
                            }),
                            expect.objectContaining({
                                driverTimeMs: 40_000,
                                expertTimeMs: 39_000,
                                driverTrackPosition: 0.4,
                            }),
                        ],
                    },
                }),
                expect.objectContaining({ id: 'segment-2', labels: ['Expert Adherence'] }),
            ],
        });
        expect(payload).toMatchObject({
            status: 'ready',
            message: 'Telemetry analysis is ready.',
            analysis: {
                status: 'success',
                session_id: 'baseline-analysis-1',
                samples_analyzed: 3,
                segments: [{
                    id: 'segment-1',
                    labels: ['Mistake (Practice)'],
                    track_section: 'Turn One',
                }],
            },
            source: 'baseline_lap_record',
            baseline: {
                id: record.id,
                lap: record.lap,
                lap_time_ms: 98_765,
                track: 'brands_hatch',
                car: 'Ferrari 296',
                sample_count: 3,
                captured_at: record.captured_at,
            },
            chartId: 'analysis-chart-1',
            component_name: 'visualization:analysis-results',
            pageId: 'baseline-analysis-page-1',
            pageCount: 1,
        });
        expect(screen.getByRole('button', { name: 'Analysis Complete' })).toBeDisabled();
    });

    it('locks out duplicate in-flight requests and reuses the same promise', async () => {
        let resolveRequest!: (value: any) => void;
        mockPost.mockReturnValue(new Promise((resolve) => { resolveRequest = resolve; }));
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        installAnalysisComponents();

        let first!: ReturnType<BaselineCollectionHandle['requestAnalysis']>;
        let duplicate!: ReturnType<BaselineCollectionHandle['requestAnalysis']>;
        act(() => {
            first = handle.requestAnalysis();
            duplicate = handle.requestAnalysis();
        });

        expect(first).toBe(duplicate);
        expect(mockPost).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: 'Analyzing Baseline…' })).toBeDisabled();

        await act(async () => {
            resolveRequest({ data: analysisResult });
            await first;
        });
        expect(screen.getByRole('button', { name: 'Analysis Complete' })).toBeDisabled();
    });

    it('creates an empty page for a successful analysis with no classified segments', async () => {
        mockPost.mockResolvedValueOnce({
            data: { ...analysisResult, segments: [], parent_segment_count: 0 },
        });
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        installAnalysisComponents();

        let payload: any;
        await act(async () => { payload = await handle.requestAnalysis(); });

        expect(appendAnalysisResultPage).toHaveBeenCalledWith(expect.objectContaining({ elements: [] }));
        expect(payload).toMatchObject({
            status: 'empty',
            pageId: 'baseline-analysis-page-1',
            pageCount: 1,
        });
    });

    it('reuses the mounted Analysis Results chart and accumulates pages on later requests', async () => {
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        const { manager, visualizations } = installAnalysisComponents();

        await act(async () => { await handle.requestAnalysis(); });
        mockPost.mockResolvedValueOnce({
            data: {
                ...analysisResult,
                segments: [{ ...analysisResult.segments[0], id: 'segment-later' }],
            },
        });
        let payload: any;
        await act(async () => { payload = await handle.requestAnalysis(); });

        expect(manager.requestVisualization).toHaveBeenCalledTimes(1);
        expect(visualizations).toHaveLength(1);
        expect(appendAnalysisResultPage).toHaveBeenCalledTimes(2);
        expect(appendAnalysisResultPage).toHaveBeenLastCalledWith({
            baseline: expect.objectContaining({ id: handle.getLapRecord()!.id }),
            elements: [expect.objectContaining({ id: 'segment-later' })],
        });
        expect(payload).toMatchObject({
            chartId: 'analysis-chart-1',
            component_name: 'visualization:analysis-results',
            pageId: 'baseline-analysis-page-2',
            pageCount: 2,
        });
    });

    it('shows errors, allows retry, and resets the analysis state on restart', async () => {
        mockPost.mockRejectedValueOnce({ data: { message: 'classifier unavailable' } });
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        installAnalysisComponents();

        act(() => {
            screen.getByRole('button', { name: 'Request Analysis' }).click();
        });
        await waitFor(() => expect(
            screen.getByRole('button', { name: 'Retry Analysis' }),
        ).toBeEnabled());
        expect(appendAnalysisResultPage).not.toHaveBeenCalled();

        mockPost.mockResolvedValueOnce({ data: analysisResult });
        act(() => {
            screen.getByRole('button', { name: 'Retry Analysis' }).click();
        });
        await waitFor(() => expect(
            screen.getByRole('button', { name: 'Analysis Complete' }),
        ).toBeDisabled());

        act(() => { handle.restartCollection(); });
        expect(screen.queryByRole('button', { name: 'Analysis Complete' })).not.toBeInTheDocument();

        view.rerender(<Harness telemetry={makeSample(6, 0.4, 40_000)} />);
        view.rerender(<Harness telemetry={makeSample(7, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(7, 0.4, 40_000)} />);
        view.rerender(<Harness telemetry={makeSample(7, 0.98, 98_000)} />);
        view.rerender(<Harness telemetry={makeSample(8, 0.001, 5)} />);
        expect(screen.getByRole('button', { name: 'Request Analysis' })).toBeEnabled();
    });

    it('keeps panel-creation failures retriable', async () => {
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        const { manager } = installAnalysisComponents();
        manager.requestVisualization.mockImplementationOnce(() => ({
            success: false,
            message: 'Analysis Results is unavailable.',
            componentName: 'visualization:analysis-results',
            chartId: null,
            chartType: 'analysis-results',
            reused: false,
        }));

        let failure: any;
        await act(async () => { failure = await handle.requestAnalysis(); });

        expect(failure).toEqual({
            status: 'error',
            error: 'analysis_results_visualization_unavailable',
            message: 'Analysis Results is unavailable.',
        });
        expect(appendAnalysisResultPage).not.toHaveBeenCalled();
        expect(screen.getByRole('button', { name: 'Retry Analysis' })).toBeEnabled();

        act(() => {
            screen.getByRole('button', { name: 'Retry Analysis' }).click();
        });
        await waitFor(() => expect(
            screen.getByRole('button', { name: 'Analysis Complete' }),
        ).toBeDisabled());
        expect(manager.requestVisualization).toHaveBeenCalledTimes(2);
    });

    it('returns the existing incomplete-baseline failure from requestAnalysis', async () => {
        render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);

        await expect(getHandle().requestAnalysis()).resolves.toEqual({
            status: 'error',
            error: 'baseline_lap_record_required',
            message: 'Live recorded analysis requires a recorded baseline lap before it can run.',
        });
        expect(mockPost).not.toHaveBeenCalled();
    });
});
