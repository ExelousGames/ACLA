import React, { useLayoutEffect } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefProvider,
    useAiToolComponentRefDirectory,
    type AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import BaselineCollection, {
    getCompletedBaselineLapTimeMs,
    type BaselineAnalysisPayload,
    type BaselineCollectionHandle,
} from '../BaselineCollection';
import { LiveSessionContext } from '../LiveSessionContext';
import { liveTelemetryStore } from '../live-telemetry-store';
import {
    AnalysisResultsVisualizationNotReadyError,
    AnalysisResultsVisualizationUnavailableError,
    BaselineCollectionAlreadyStartedError,
    BaselineCollectionNotStartedError,
    RecordedAnalysisFailedError,
    VisualizationRequestFailedError,
} from 'contexts/AiToolComponentError';

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
    Graphics_completed_lap: lap,
    Graphics_normalized_car_position: position,
    Graphics_current_time: currentTime,
    ...(lastTime === undefined ? {} : { Graphics_last_time: lastTime }),
});

let directory: AiToolComponentRefDirectory | null = null;
let appendedPages: any[] = [];
const appendAnalysisResultPage = jest.fn();
let telemetrySequence = 0;

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
}) => {
    useLayoutEffect(() => {
        if (Object.keys(telemetry).length === 0) return;
        telemetrySequence += 1;
        liveTelemetryStore.publishFrame({
            type: 'frame',
            game: 'acc',
            sample: telemetry,
            sequence: telemetrySequence,
            committedSequence: telemetrySequence,
            committedCount: telemetrySequence,
        }, telemetry);
    }, [telemetry]);
    return (
        <AiToolComponentRefProvider>
            <DirectoryObserver />
            <LiveSessionContext.Provider value={{ appendAnalysisResultPage } as any}>
                {show && <BaselineCollection name={AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION} />}
            </LiveSessionContext.Provider>
        </AiToolComponentRefProvider>
    );
};

const getHandle = () => directory!
    .findComponentRef<BaselineCollectionHandle>(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)!
    .current!;

const reserve = (name: string, value: Record<string, any>) => {
    directory!.registerComponentRef({ current: {
        getComponentName: () => name,
        ...value,
    } as any });
};

const installAnalysisComponents = ({
    existingChart = false,
    waitForAnalysisResultPage = jest.fn().mockResolvedValue(undefined),
}: {
    existingChart?: boolean;
    waitForAnalysisResultPage?: jest.Mock;
} = {}) => {
    const chartName = 'visualization:analysis-results';
    const chartHandle = { waitForAnalysisResultPage };
    const visualizations: any[] = existingChart
        ? [{ id: 'analysis-chart-1', name: chartName, type: 'analysis-results' }]
        : [];

    const manager = {
        getCurrentVisualizations: jest.fn(() => visualizations),
        requestVisualization: jest.fn((options: any) => {
            const instance = { id: 'analysis-chart-1', ...options };
            visualizations.push(instance);
            reserve(chartName, chartHandle);
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
        if (existingChart) reserve(chartName, chartHandle);
        reserve(AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER, manager);
    });
    return { chartHandle, manager, visualizations };
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
        liveTelemetryStore.resetSession();
        telemetrySequence = 0;
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

    it('rejects restart when collection is not in progress', async () => {
        render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = getHandle();

        const restart = handle.restartCollection();

        await expect(restart.result).rejects.toMatchObject({
            name: 'BaselineCollectionNotStartedError',
            componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            message: 'Baseline collection is not in progress. Start a new collection instead.',
        });
        await expect(restart.result).rejects.toBeInstanceOf(BaselineCollectionNotStartedError);
        expect(handle.getTag()).toBeNull();
        expect(handle.getLapRecord()).toBeNull();
        expect(screen.getByRole('button', { name: 'Start Baseline Collection' })).toBeEnabled();
    });

    it('rejects duplicate starts while waiting and allows restart to clear the operation', async () => {
        render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = getHandle();
        let original!: ReturnType<BaselineCollectionHandle['startCollection']>;
        let duplicate!: ReturnType<BaselineCollectionHandle['startCollection']>;

        act(() => {
            original = handle.startCollection();
            duplicate = handle.startCollection();
        });

        expect(duplicate.statuses).toHaveLength(0);
        await expect(duplicate.result).rejects.toMatchObject({
            name: 'BaselineCollectionAlreadyStartedError',
            componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            message: 'Baseline collection is already in progress.',
        });
        await expect(duplicate.result).rejects.toBeInstanceOf(BaselineCollectionAlreadyStartedError);
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start', progress_percent: 0 });
        expect(handle.getLapRecord()).toBeNull();

        let restart!: ReturnType<BaselineCollectionHandle['restartCollection']>;
        act(() => { restart = handle.restartCollection(); });
        await expect(original.result).rejects.toMatchObject({ name: 'BaselineAnalysisCancelledError' });
        await expect(restart.result).resolves.toMatchObject({ status: 'waiting_for_start' });
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start', baseline_lap_id: null });
    });

    it('rejects duplicate starts during collection without disrupting the original operation', async () => {
        const view = render(<Harness telemetry={{}} />);
        const handle = getHandle();
        let original!: ReturnType<BaselineCollectionHandle['startCollection']>;
        act(() => { original = handle.startCollection(); });
        view.rerender(<Harness telemetry={makeSample(0, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(0, 0.4, 40_000)} />);
        const tagBeforeDuplicate = handle.getTag();

        const duplicate = handle.startCollection();
        expect(duplicate.statuses).toHaveLength(0);
        await expect(duplicate.result).rejects.toMatchObject({
            name: 'BaselineCollectionAlreadyStartedError',
            componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            message: 'Baseline collection is already in progress.',
        });
        expect(handle.getTag()).toEqual(tagBeforeDuplicate);
        expect(handle.getLapRecord()).toBeNull();

        view.rerender(<Harness telemetry={makeSample(0, 0.98, 98_000)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.001, 5)} />);

        await expect(original.result).resolves.toMatchObject({ status: 'complete' });
        expect(handle.getLapRecord()).toMatchObject({
            lap_id: 0,
            sample_count: 3,
        });
        expect(handle.getLapRecord()?.records.map((row) => row.Graphics_normalized_car_position))
            .toEqual([0.001, 0.4, 0.98]);
    });

    it('continues from current-lap telemetry cached directly from LiveSessionContext', async () => {
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        const cachedBaseline = handle.getLapRecord();
        const cachedCurrentLap = [
            makeSample(6, 0.001, 5),
            makeSample(6, 0.2, 20_000),
            makeSample(6, 0.45, 45_000),
        ];
        cachedCurrentLap.forEach((telemetry) => {
            view.rerender(<Harness telemetry={telemetry} />);
        });

        let continued!: ReturnType<BaselineCollectionHandle['startCollection']>;
        act(() => {
            continued = handle.startCollection();
        });

        expect(handle.getLapRecord()).toBeNull();
        expect(handle.getTag()).toMatchObject({
            status: 'collecting',
            progress_percent: 45,
            baseline_lap_id: 6,
        });

        view.rerender(<Harness telemetry={makeSample(6, 0.45, 45_000)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.7, 70_000)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.98, 98_000)} />);
        view.rerender(<Harness telemetry={makeSample(7, 0.001, 5, 98_765)} />);

        await expect(continued.result).resolves.toMatchObject({ status: 'complete' });
        expect(handle.getLapRecord()).not.toBe(cachedBaseline);
        expect(handle.getLapRecord()).toMatchObject({ lap_id: 6, sample_count: 5 });
        expect(handle.getLapRecord()?.records.map((row) => row.Graphics_normalized_car_position))
            .toEqual([0.001, 0.2, 0.45, 0.7, 0.98]);
    });

    it('seeds only post-wrap telemetry while the lap counter catches up', async () => {
        const view = render(<Harness telemetry={{}} />);
        const handle = getHandle();
        act(() => { handle.startCollection(); });

        const sendCachedSample = (sample: Record<string, any>) => {
            view.rerender(<Harness telemetry={sample} />);
        };
        sendCachedSample(makeSample(3, 0.001, 10));
        sendCachedSample(makeSample(3, 0.5, 50_000));
        sendCachedSample(makeSample(3, 0.99, 99_000));
        sendCachedSample(makeSample(3, 0.002, 20));
        expect(handle.getLapRecord()).toMatchObject({ lap_id: 3, sample_count: 3 });

        sendCachedSample(makeSample(3, 0.2, 20_000));
        sendCachedSample(makeSample(3, 0.4, 40_000));

        let continued!: ReturnType<BaselineCollectionHandle['startCollection']>;
        act(() => { continued = handle.startCollection(); });
        expect(handle.getTag()).toMatchObject({
            status: 'collecting',
            progress_percent: 40,
            baseline_lap_id: 3,
        });

        view.rerender(<Harness telemetry={makeSample(3, 0.4, 40_000)} />);
        view.rerender(<Harness telemetry={makeSample(4, 0.41, 41_000)} />);
        expect(handle.getLapRecord()).toBeNull();
        expect(handle.getTag()).toMatchObject({ status: 'collecting', baseline_lap_id: 4 });

        view.rerender(<Harness telemetry={makeSample(4, 0.8, 80_000)} />);
        view.rerender(<Harness telemetry={makeSample(4, 0.99, 99_000)} />);
        view.rerender(<Harness telemetry={makeSample(5, 0.002, 20, 99_500)} />);

        await expect(continued.result).resolves.toMatchObject({ status: 'complete' });
        expect(handle.getLapRecord()).toMatchObject({ lap_id: 4, sample_count: 6 });
        expect(handle.getLapRecord()?.records.map((row) => row.Graphics_normalized_car_position))
            .toEqual([0.002, 0.2, 0.4, 0.41, 0.8, 0.99]);
    });

    it('continues from the next-lap boundary sample that completed the prior baseline', async () => {
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        const cachedBaseline = handle.getLapRecord();

        let restarted!: ReturnType<BaselineCollectionHandle['startCollection']>;
        act(() => {
            restarted = handle.startCollection();
        });

        expect(restarted.statuses).toHaveLength(6);
        expect(handle.getLapRecord()).toBeNull();
        expect(handle.getTag()).toMatchObject({
            status: 'collecting',
            progress_percent: 1,
            baseline_lap_id: 6,
        });

        view.rerender(<Harness telemetry={makeSample(6, 0.4, 40_000)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.98, 98_000)} />);
        view.rerender(<Harness telemetry={makeSample(7, 0.001, 5, 98_765)} />);

        await expect(restarted.result).resolves.toMatchObject({ status: 'complete' });
        expect(handle.getLapRecord()).not.toBe(cachedBaseline);
        expect(handle.getLapRecord()).toMatchObject({ lap_id: 6, sample_count: 3 });
    });

    it('waits for the next boundary, records one lap, reports progress, and resolves the operation', async () => {
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = getHandle();

        let operation!: ReturnType<BaselineCollectionHandle['startCollection']>;
        act(() => {
            operation = handle.startCollection();
            expect(handle.getTag()).toMatchObject({
                status: 'waiting_for_start',
                progress_percent: 0,
            });
        });
        expect(handle.getLapRecord()).toBeNull();

        view.rerender(<Harness telemetry={makeSample(4, 0.9, 90_000)} />);
        expect(handle.getTag()).toMatchObject({ status: 'waiting_for_start', progress_percent: 0 });

        view.rerender(<Harness telemetry={makeSample(5, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(5, 0.4, 40_000)} />);
        expect(handle.getTag()).toMatchObject({ status: 'collecting', progress_percent: 40, baseline_lap_id: 5 });

        view.rerender(<Harness telemetry={makeSample(5, 0.98, 98_000)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(6, 0.2, 20_000)} />);

        expect(handle.getLapRecord()).toMatchObject({
            lap_id: 5,
            lap_time_ms: 98_000,
            track: 'brands_hatch',
            car: 'Ferrari 296',
            sample_count: 3,
        });
        expect(handle.getLapRecord()?.records.map((row) => row.Graphics_normalized_car_position))
            .toEqual([0.001, 0.4, 0.98]);
        expect(handle.getTag()).toMatchObject({ status: 'complete', progress_percent: 100 });
        expect(screen.queryByLabelText('Baseline collection progress')).not.toBeInTheDocument();
        expect(handle.getOverlayBehavior(handle.getTag())).toMatchObject({ remove: true });
        expect(screen.getByRole('button', { name: 'Request Analysis' })).toBeEnabled();
        await expect(operation.result).resolves.toEqual({
            progress_percent: 100,
            status: 'complete',
            car: 'Ferrari 296',
            track: 'brands_hatch',
            message: 'Baseline complete. Cached lap record is ready.',
        });
        await expect(Promise.all(operation.statuses)).resolves.toEqual(expect.arrayContaining([
            expect.objectContaining({ event: 'baseline_progress', milestone: 100 }),
        ]));
    });

    it('completes when position wraps before the lap counter advances', () => {
        const view = render(<Harness telemetry={{}} />);
        const handle = getHandle();
        act(() => { handle.startCollection(); });

        view.rerender(<Harness telemetry={makeSample(3, 0.001, 10)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.5, 50_000)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.99, 99_000)} />);
        view.rerender(<Harness telemetry={makeSample(3, 0.002, 20)} />);

        expect(handle.getLapRecord()).toMatchObject({ lap_id: 3, sample_count: 3 });
        expect(handle.getTag()).toMatchObject({ status: 'complete', progress_percent: 100 });
    });

    it('rejects restart after completion and preserves the recorded baseline', async () => {
        const view = render(<Harness telemetry={{}} />);
        const handle = getHandle();
        act(() => { handle.startCollection(); });
        view.rerender(<Harness telemetry={makeSample(0, 0.001, 10)} />);
        view.rerender(<Harness telemetry={makeSample(0, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.001, 5)} />);
        const completedBaseline = handle.getLapRecord();
        expect(completedBaseline).not.toBeNull();

        const restart = handle.restartCollection();

        await expect(restart.result).rejects.toMatchObject({
            name: 'BaselineCollectionNotStartedError',
            message: 'Baseline collection is not in progress. Start a new collection instead.',
        });
        expect(handle.getLapRecord()).toBe(completedBaseline);
        expect(handle.getTag()).toMatchObject({ status: 'complete', baseline_lap_id: 0 });
        expect(screen.getByRole('button', { name: 'Request Analysis' })).toBeEnabled();
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

        view.rerender(<Harness telemetry={makeSample(0, 0.5, 50_000)} />);
        const secondHandle = getHandle();
        expect(secondHandle).not.toBe(firstHandle);
        expect(secondHandle.getTag()).toBeNull();
        act(() => { secondHandle.startCollection(); });
        expect(secondHandle.getTag()).toMatchObject({ status: 'waiting_for_start', baseline_lap_id: null });
        expect(screen.getByLabelText('Baseline collection progress')).toBeInTheDocument();

        view.rerender(<Harness telemetry={makeSample(0, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.001, 5)} />);
        view.rerender(<Harness telemetry={makeSample(1, 0.9, 90_000)} />);
        view.rerender(<Harness telemetry={makeSample(2, 0.001, 5)} />);
        expect(secondHandle.getLapRecord()).not.toBeNull();

        view.rerender(<Harness telemetry={makeSample(2, 0.001, 5)} show={false} />);
        expect(secondHandle.getTag()).toBeNull();
        expect(secondHandle.getLapRecord()).toBeNull();
    });

    it('owns the exact cached-lap request, normalization, labels, comparison, and result payload', async () => {
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        const { chartHandle, manager } = installAnalysisComponents();

        let payload: BaselineAnalysisPayload | Error | undefined;
        await act(async () => {
            payload = await handle.requestAnalysis({ limit: 1 }).result;
        });

        const record = handle.getLapRecord()!;
        expect(mockPost).toHaveBeenCalledWith(
            '/racing-session/analyze-live-recorded-analysis',
            {
                track: record.track,
                car: record.car,
                baseline_lap_id: record.lap_id,
                records: record.records,
            },
            { timeout: 120000 },
        );
        expect(manager.requestVisualization).toHaveBeenCalledWith({
            name: 'visualization:analysis-results',
            type: 'analysis-results',
        });
        expect(chartHandle.waitForAnalysisResultPage)
            .toHaveBeenCalledWith('baseline-analysis-page-1');
        expect(appendAnalysisResultPage).toHaveBeenCalledWith({
            baseline: {
                id: record.id,
                lap_id: record.lap_id,
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
                lap_id: record.lap_id,
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

        expect(first).not.toBe(duplicate);
        expect(mockPost).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: 'Analyzing Baseline…' })).toBeDisabled();

        await act(async () => {
            resolveRequest({ data: analysisResult });
            await first.result;
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
        await act(async () => { payload = await handle.requestAnalysis().result; });

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
        const { chartHandle, manager, visualizations } = installAnalysisComponents();

        await act(async () => { await handle.requestAnalysis().result; });
        mockPost.mockResolvedValueOnce({
            data: {
                ...analysisResult,
                segments: [{ ...analysisResult.segments[0], id: 'segment-later' }],
            },
        });
        let payload: any;
        await act(async () => { payload = await handle.requestAnalysis().result; });

        expect(manager.requestVisualization).toHaveBeenCalledTimes(1);
        expect(visualizations).toHaveLength(1);
        expect(appendAnalysisResultPage).toHaveBeenCalledTimes(2);
        expect(chartHandle.waitForAnalysisResultPage).toHaveBeenNthCalledWith(
            1,
            'baseline-analysis-page-1',
        );
        expect(chartHandle.waitForAnalysisResultPage).toHaveBeenNthCalledWith(
            2,
            'baseline-analysis-page-2',
        );
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

    it.each([
        ['newly opened', false],
        ['reused', true],
    ])('does not finish for a %s Analysis Results panel until its page is committed', async (
        _panelState,
        existingChart,
    ) => {
        let resolvePageCommit!: () => void;
        const pageCommit = new Promise<void>((resolve) => { resolvePageCommit = resolve; });
        const waitForAnalysisResultPage = jest.fn(() => pageCommit);
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        const { manager } = installAnalysisComponents({
            existingChart,
            waitForAnalysisResultPage,
        });

        let completed = false;
        let request!: ReturnType<BaselineCollectionHandle['requestAnalysis']>;
        act(() => {
            request = handle.requestAnalysis();
            void request.result.then(() => { completed = true; });
        });

        await waitFor(() => expect(waitForAnalysisResultPage)
            .toHaveBeenCalledWith('baseline-analysis-page-1'));
        expect(completed).toBe(false);
        expect(screen.getByRole('button', { name: /Analyzing Baseline/ })).toBeDisabled();
        expect(manager.requestVisualization).toHaveBeenCalledTimes(existingChart ? 0 : 1);

        await act(async () => {
            resolvePageCommit();
            await request.result;
        });
        expect(completed).toBe(true);
        expect(screen.getByRole('button', { name: 'Analysis Complete' })).toBeDisabled();
    });

    it('preserves the stable visualization-readiness failure from the chart', async () => {
        const readinessFailure = new AnalysisResultsVisualizationNotReadyError(
            'visualization:analysis-results',
            "Analysis Results unmounted before page 'baseline-analysis-page-1' was committed.",
        );
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        installAnalysisComponents({
            waitForAnalysisResultPage: jest.fn().mockRejectedValue(readinessFailure),
        });

        await act(async () => {
            await expect(handle.requestAnalysis().result).rejects.toBe(readinessFailure);
        });
        expect(screen.getByRole('button', { name: 'Retry Analysis' })).toBeEnabled();
    });

    it('shows errors, allows retry, and resets the analysis state on a new collection', async () => {
        mockPost.mockRejectedValueOnce({ data: { message: 'classifier unavailable' } });
        const view = render(<Harness telemetry={makeSample(4, 0.45, 45_000)} />);
        const handle = completeBaselineLap(view);
        installAnalysisComponents();

        let failure: unknown;
        await act(async () => {
            try {
                await handle.requestAnalysis().result;
            } catch (error) {
                failure = error;
            }
        });
        expect(failure).toBeInstanceOf(RecordedAnalysisFailedError);
        expect(failure).toMatchObject({
            name: 'RecordedAnalysisFailedError',
            componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            message: 'classifier unavailable',
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

        act(() => { handle.startCollection(); });
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
        manager.requestVisualization.mockImplementationOnce(() => {
            throw new VisualizationRequestFailedError(
                AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER,
                'Analysis Results is unavailable.',
            );
        });

        let failure: any;
        await act(async () => {
            try {
                await handle.requestAnalysis().result;
            } catch (error) {
                failure = error;
            }
        });

        expect(failure).toBeInstanceOf(AnalysisResultsVisualizationUnavailableError);
        expect(failure).toMatchObject({
            name: 'AnalysisResultsVisualizationUnavailableError',
            componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            message: 'Analysis Results is unavailable.',
            cause: expect.any(VisualizationRequestFailedError),
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

        await act(async () => {
            await expect(getHandle().requestAnalysis().result).rejects.toMatchObject({
                name: 'BaselineLapRecordRequiredError',
                componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                message: 'Live recorded analysis requires a recorded baseline lap before it can run.',
            });
        });
        expect(mockPost).not.toHaveBeenCalled();
    });
});
