import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import type { DesktopGame, DesktopGameContextValue } from 'contexts/DesktopGameContext';
import { RecordingState } from 'views/lap-analysis/recording-state';
import { LiveSessionContext } from '../LiveSessionContext';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    AiToolComponentRefProvider,
    useAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import type { LiveSessionHandle } from '../LiveSessionView';
import {
    BaselineCollectionAlreadyStartedError,
    BaselineCollectionNotStartedError,
} from 'contexts/AiToolComponentError';
import { createAiToolOperation, createAiToolOperationFrom } from 'components/ai-engineering-tools';
import { liveTelemetryStore } from '../live-telemetry-store';

jest.mock('contexts/DesktopGameContext', () => ({
    useDesktopGame: jest.fn(),
}));

jest.mock('../LiveTelemetryWorkspace', () => () => <div>Live workspace</div>);
jest.mock('views/lap-analysis/LiveSessionDetectionManager', () => () => (
    <div data-testid="live-session-detection-manager" />
));
jest.mock('views/lap-analysis/liveAnalysisSessionRecording', () => ({ recorderHostId }: { recorderHostId?: string }) => (
    <div data-testid="live-analysis-session-recording" data-host={recorderHostId} />
));

import { useDesktopGame } from 'contexts/DesktopGameContext';
import { LiveSessionContent } from '../LiveSessionView';

const mockedUseDesktopGame = useDesktopGame as jest.Mock;

const detectionCases: Array<{
    name: string;
    detection: DesktopGameContextValue;
    title: string;
    visualState: string;
    canStart: boolean;
}> = [
    {
        name: 'checking',
        detection: { detectedGame: null, detectionStatus: 'checking', error: null },
        title: 'Scanning for simulator...',
        visualState: 'checking',
        canStart: false,
    },
    {
        name: 'ACC',
        detection: { detectedGame: 'acc', detectionStatus: 'detected', error: null },
        title: 'Assetto Corsa Competizione detected',
        visualState: 'ready',
        canStart: true,
    },
    {
        name: 'Assetto Corsa',
        detection: { detectedGame: 'ac', detectionStatus: 'detected', error: null },
        title: 'Assetto Corsa detected',
        visualState: 'limited',
        canStart: true,
    },
    {
        name: 'iRacing',
        detection: { detectedGame: 'iracing', detectionStatus: 'detected', error: null },
        title: 'iRacing detected',
        visualState: 'limited',
        canStart: true,
    },
    {
        name: 'not detected',
        detection: { detectedGame: null, detectionStatus: 'not-detected', error: null },
        title: 'No simulator detected.',
        visualState: 'idle',
        canStart: false,
    },
    {
        name: 'unsupported',
        detection: { detectedGame: null, detectionStatus: 'unsupported', error: null },
        title: 'Simulator detection unavailable',
        visualState: 'unsupported',
        canStart: false,
    },
    {
        name: 'error',
        detection: { detectedGame: null, detectionStatus: 'error', error: 'tasklist failed' },
        title: 'Simulator detection failed',
        visualState: 'error',
        canStart: false,
    },
];

const createRuntime = (sessionGame: DesktopGame | null = null) => ({
    sessionGame,
    staticData: {},
    recordingState: RecordingState.CHECKING,
    recordingMetadata: null,
    recordingFileKey: null,
    restorationStatus: 'idle',
    restorationError: null,
    recordingFileValidation: null,
    analysisResultPages: [],
    activeAnalysisResultPageId: null,
    getNextCorner: jest.fn((): ReturnType<LiveSessionHandle['getNextCorner']> => null),
    getLiveSessionSnapshot: jest.fn(() => ({
        status: 'empty' as const,
        track: '',
        car: '',
        current_lap: 0,
        completed_laps: 0,
        normalized_position: 0,
        sample_count: 0,
        live_session_type: 'unknown' as const,
        completed_lap_count: 0,
    })),
    recorderControl: { openUploadFlow: jest.fn() },
    startLiveSession: jest.fn(),
    endLiveSession: jest.fn(),
    setRecordingMetadata: jest.fn(),
    transitionRecordingState: jest.fn(),
    streamRecordedTelemetry: jest.fn(async () => ({ rowCount: 0, totalBytes: 0 })),
    clearRecordingSession: jest.fn(),
    clearPersistedDraft: jest.fn(),
    registerRecorderControl: jest.fn(),
});

const renderView = (runtime = createRuntime()) => render(
    <LiveSessionContext.Provider value={runtime as any}>
        <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
    </LiveSessionContext.Provider>,
);

let componentDirectory: AiToolComponentRefDirectory | null = null;
const RegistrationObserver = () => {
    componentDirectory = useAiToolComponentRefDirectory();
    return null;
};

const renderRegisteredView = (runtime: any) => {
    render(
        <AiToolComponentRefProvider>
            <LiveSessionContext.Provider value={runtime as any}>
                <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
            </LiveSessionContext.Provider>
            <RegistrationObserver />
        </AiToolComponentRefProvider>,
    );
    return componentDirectory!
        .findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;
};

const createTelemetryRuntime = () => {
    const rows = [{
        Static_track: 'brands_hatch',
        Graphics_completed_lap: 1,
        Graphics_normalized_car_position: 0.1,
        Physics_timestamp: 0,
        Physics_speed_kmh: 100,
        Physics_wheel_pressure_front_left: 27,
        Physics_wheel_pressure_front_right: 28,
        Physics_wheel_pressure_rear_left: 29,
        Physics_wheel_pressure_rear_right: 30,
        status: 5,
        message: 6,
    }, {
        Static_track: 'brands_hatch',
        Graphics_completed_lap: 1,
        Graphics_normalized_car_position: 0.2,
        Physics_timestamp: 100,
        Physics_speed_kmh: 120,
        Physics_wheel_pressure_front_left: 28,
        Physics_wheel_pressure_front_right: 29,
        Physics_wheel_pressure_rear_left: 30,
        Physics_wheel_pressure_rear_right: 31,
        status: 7,
        message: 8,
    }];
    liveTelemetryStore.publishFrame({
        type: 'frame',
        game: 'acc',
        sample: rows[rows.length - 1],
        sequence: 1,
        committedSequence: 1,
        committedCount: rows.length,
    }, { Static_track: 'brands_hatch' });
    return {
        ...createRuntime('acc'),
        streamRecordedTelemetry: jest.fn(async (onChunk: (rows: Record<string, any>[]) => void | Promise<void>) => {
            await onChunk(rows.slice(0, 1));
            await onChunk(rows.slice(1));
            return { rowCount: rows.length, totalBytes: 100 };
        }),
        rows,
    };
};

describe('LiveSessionView', () => {
    beforeEach(() => {
        liveTelemetryStore.resetSession();
        mockedUseDesktopGame.mockReset();
        componentDirectory = null;
    });

    it('registers current live operations under the exact component name', () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const runtime: any = createRuntime('acc');
        runtime.recordingState = RecordingState.RECORDING;
        runtime.recordingMetadata = {
            sessionName: 'Monza Run',
            mapName: 'Monza',
            carName: 'BMW M4 GT3',
            gameRecordedFrom: 'acc',
        };
        liveTelemetryStore.publishFrame({
            type: 'frame',
            game: 'acc',
            sample: { Physics_speed_kmh: 210 },
            sequence: 1,
            committedSequence: 1,
            committedCount: 1,
        });
        runtime.getLiveSessionSnapshot = jest.fn(() => ({
            status: 'ready',
            track: 'Monza',
            car: 'BMW M4 GT3',
            current_lap: 4,
            completed_laps: 3,
            normalized_position: 0.42,
            sample_count: 1250,
            live_session_type: 'solo_practice',
            completed_lap_count: 3,
        }));

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={runtime as any}>
                    <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </LiveSessionContext.Provider>
                <RegistrationObserver />
            </AiToolComponentRefProvider>,
        );

        const handle = componentDirectory!.findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;
        expect(handle.getComponentName()).toBe(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION);
        expect(handle.getLiveSessionSnapshot()).toMatchObject({
            track: 'Monza',
            car: 'BMW M4 GT3',
            current_lap: 4,
            sample_count: 1250,
        });
        expect(handle.getRecordingState()).toBe(RecordingState.RECORDING);
        expect(handle.getCurrentTelemetry()).toEqual({ Physics_speed_kmh: 210 });
        expect(handle.getAssistantSnapshot()).toBe(runtime);
    });

    it('keeps assistant snapshots isolated while lossless subscribers receive every frame', () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const runtime = createRuntime('acc');
        const handle = renderRegisteredView(runtime);
        const assistantListener = jest.fn();
        const frameSequences: number[] = [];
        handle.subscribeAssistantSnapshot(assistantListener);
        liveTelemetryStore.subscribeEvents((event) => {
            if (event.type === 'frame') frameSequences.push(event.update.sequence);
        });

        act(() => {
            for (let sequence = 1; sequence <= 120; sequence += 1) {
                liveTelemetryStore.publishFrame({
                    type: 'frame',
                    game: 'acc',
                    sample: {
                        Graphics_status: 2,
                        Graphics_sequence: sequence,
                        Physics_speed_kmh: sequence,
                    },
                    sequence,
                    committedSequence: sequence,
                    committedCount: sequence,
                });
            }
        });

        expect(assistantListener).not.toHaveBeenCalled();
        expect(frameSequences).toEqual(Array.from({ length: 120 }, (_, index) => index + 1));
        expect(handle.getAssistantSnapshot()).not.toHaveProperty('currentTelemetry');
        expect(handle.getAssistantSnapshot()).not.toHaveProperty('telemetryStatus');
        expect(handle.getAssistantSnapshot()).not.toHaveProperty('recordedSampleCount');
    });

    it.each([
        ['avg', 110],
        ['min', 100],
        ['max', 120],
        ['stats', { avg: 110, min: 100, max: 120, stddev: 10 }],
    ] as const)('returns a ready telemetry envelope for %s reduction', async (reduce, expected) => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const handle = renderRegisteredView(createTelemetryRuntime());

        const operation = handle.queryTelemetryMetricForAi({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce,
        });

        await expect(operation.result).resolves.toEqual({
            status: 'ready',
            data: { Physics_speed_kmh: expected },
        });
    });

    it('keeps telemetry status and message fields nested inside data', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const handle = renderRegisteredView(createTelemetryRuntime());

        await expect(handle.queryTelemetryMetricForAi({
            fields: ['status', 'message'],
            scope: { type: 'now' },
            reduce: 'avg',
        }).result).resolves.toEqual({
            status: 'ready',
            data: { status: 7, message: 8 },
        });
    });

    it('wraps alias and group expansion under raw telemetry field keys', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const handle = renderRegisteredView(createTelemetryRuntime());

        await expect(handle.queryTelemetryMetricForAi({
            fields: ['tyre_pressure', 'tire_pressure', 'Physics_wheel_pressure_front_left'],
            scope: { type: 'now' },
            reduce: 'avg',
        }).result).resolves.toEqual({
            status: 'ready',
            data: {
                Physics_wheel_pressure_front_left: 28,
                Physics_wheel_pressure_front_right: 29,
                Physics_wheel_pressure_rear_left: 30,
                Physics_wheel_pressure_rear_right: 31,
            },
        });
    });

    it.each([
        ['JSON-string fields', { fields: '["speed"]', scope: { type: 'now' }, reduce: 'avg' }],
        ['comma-delimited fields', { fields: 'speed,brake', scope: { type: 'now' }, reduce: 'avg' }],
        ['alternate field name', { field: ['speed'], scope: { type: 'now' }, reduce: 'avg' }],
        ['missing fields', { scope: { type: 'now' }, reduce: 'avg' }],
        ['empty fields', { fields: [], scope: { type: 'now' }, reduce: 'avg' }],
        ['blank field entry', { fields: [''], scope: { type: 'now' }, reduce: 'avg' }],
        ['trimmed field repair', { fields: [' speed '], scope: { type: 'now' }, reduce: 'avg' }],
        ['JSON-string scope', { fields: ['speed'], scope: '{"type":"now"}', reduce: 'avg' }],
        ['missing scope data', { fields: ['speed'], scope: { type: 'last_seconds' }, reduce: 'avg' }],
        ['alternate scope property', { fields: ['speed'], scope: { type: 'event', event_type: 'CORNER', which: 'last' }, reduce: 'avg' }],
        ['malformed lap scope', { fields: ['speed'], scope: { type: 'lap', lap: '1' }, reduce: 'avg' }],
        ['malformed range scope', { fields: ['speed'], scope: { type: 'range', start: 0, end: '10' }, reduce: 'avg' }],
        ['missing reduction', { fields: ['speed'], scope: { type: 'now' } }],
        ['raw reduction', { fields: ['speed'], scope: { type: 'now' }, reduce: 'raw' }],
        ['invalid reduction', { fields: ['speed'], scope: { type: 'now' }, reduce: 'average' }],
        ['extra property', { fields: ['speed'], scope: { type: 'now' }, reduce: 'avg', reducer: 'avg' }],
    ])('rejects malformed telemetry arguments: %s', async (_name, args) => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const handle = renderRegisteredView(createTelemetryRuntime());

        const operation = handle.queryTelemetryMetricForAi(args as any);

        await expect(operation.result).rejects.toMatchObject({
            name: 'InvalidToolCallError',
        });
    });

    it('propagates a mounted collector duplicate-start failure without progress statuses', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const failure = new BaselineCollectionAlreadyStartedError(
            AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            'Baseline collection is already in progress.',
        );
        const startCollection = jest.fn(() => createAiToolOperationFrom(() => { throw failure; }, 'failed'));

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={createRuntime('acc') as any}>
                    <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </LiveSessionContext.Provider>
                <RegistrationObserver />
            </AiToolComponentRefProvider>,
        );
        act(() => {
            componentDirectory!.registerComponentRef({ current: {
                    getComponentName: () => AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                    startCollection,
                } as any });
        });
        const handle = componentDirectory!
            .findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;

        const operation = handle.collectLiveBaselineForAi({
            query: { preset: 'full_lap' },
            timeout_seconds: 12,
        });

        expect(startCollection).toHaveBeenCalledWith({
            query: { preset: 'full_lap' },
            timeoutMs: 12_000,
        });
        expect(operation.statuses).toHaveLength(0);
        await expect(operation.result).rejects.toBe(failure);
    });

    it.each([
        ['missing query', {}],
        ['unknown preset', { query: { preset: 'sector' } }],
        ['preset mixed with a start query', {
            query: {
                preset: 'full_lap',
                start_query: { field: 'Physics_speed_kmh', operator: 'gte', value: 100 },
            },
        }],
        ['custom query missing its end condition', {
            query: {
                start_query: { field: 'Physics_speed_kmh', operator: 'gte', value: 100 },
            },
        }],
    ])('rejects malformed baseline collection arguments: %s', async (_name, args) => {
        mockedUseDesktopGame.mockReturnValue({
            detectedGame: 'acc',
            detectionStatus: 'detected',
            error: null,
        });
        const view = renderRegisteredView(createRuntime('acc'));

        await expect(view.collectLiveBaselineForAi(args).result).rejects.toMatchObject({
            name: 'InvalidToolCallError',
        });
    });

    it('resolves telemetry scopes from the recorded writer file', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const runtime = createTelemetryRuntime();
        const handle = renderRegisteredView(runtime);

        await expect(handle.getTelemetryForScope({ type: 'lap', lap: 'current' }))
            .resolves.toEqual(runtime.rows);
        expect(runtime.streamRecordedTelemetry).toHaveBeenCalledTimes(1);
    });

    it('uses recorded row indexes for resolved event scopes', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const runtime = createTelemetryRuntime();
        const handle = renderRegisteredView(runtime);
        act(() => {
            componentDirectory!.registerComponentRef({ current: {
                getComponentName: () => 'visualization:event-log',
                findEvents: jest.fn(() => [{
                    id: 'corner-1',
                    type: 'CORNER',
                    startSampleIdx: 0,
                    endSampleIdx: 0,
                    lap: 1,
                    trackPosition: 0.1,
                    timestamp: 0,
                }]),
            } as any });
        });

        await expect(handle.getTelemetryForScope({
            type: 'event',
            eventType: 'CORNER',
            which: 'last',
        })).resolves.toEqual([runtime.rows[0]]);
    });

    it('returns an empty historical scope before a writer file exists', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const handle = renderRegisteredView(createRuntime('acc'));

        await expect(handle.getTelemetryForScope({ type: 'now' })).resolves.toEqual([]);
        await expect(handle.queryTelemetryMetric({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'avg',
        })).resolves.toEqual({ Physics_speed_kmh: 0 });
    });

    it('uses the next corner owned by LiveSessionContext', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const runtime = createTelemetryRuntime();
        runtime.getNextCorner.mockReturnValue({
            name: 'T2 Seconda Variante',
            trackPosition: 0.18,
            distanceAhead: 0.08,
        });
        const handle = renderRegisteredView(runtime);

        const corner = handle.getNextCorner();
        expect(corner).toMatchObject({
            name: 'T2 Seconda Variante',
            trackPosition: 0.18,
        });
        expect(corner?.distanceAhead).toBeCloseTo(0.08);

        const aiResult = await handle.getNextCornerForAi().result;
        if (aiResult instanceof Error) throw aiResult;
        expect(aiResult).toMatchObject({
            status: 'complete',
            corner: {
                name: 'T2 Seconda Variante',
                track_position: 0.18,
            },
        });
        expect(aiResult.corner.distance_ahead).toBeCloseTo(0.08);
        expect(runtime.getNextCorner).toHaveBeenCalledTimes(2);
    });

    it('owns the live-session detection and recording runtimes', () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });

        renderView(createRuntime('acc'));

        expect(screen.getByTestId('live-session-detection-manager')).toBeInTheDocument();
        expect(screen.getByTestId('live-analysis-session-recording')).toHaveAttribute(
            'data-host',
            'live-session-recorder-host',
        );
        expect(screen.getByTestId('live-session-recorder-host')).toBeInTheDocument();
    });

    it('exposes only the terminal baseline result without progress statuses', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const completed = {
            progress_percent: 100,
            status: 'complete' as const,
            car: 'Ferrari 296',
            track: 'brands_hatch',
            message: 'Baseline complete. Cached baseline record is ready.',
        };
        const startCollection = jest.fn(() => createAiToolOperation(
            Promise.resolve(completed),
            [Promise.resolve({
                ...completed,
                progress_percent: 50,
                status: 'collecting' as const,
                event: 'baseline_progress' as const,
                milestone: 50,
            })],
            'collector-complete',
        ));

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={createRuntime('acc') as any}>
                    <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </LiveSessionContext.Provider>
                <RegistrationObserver />
            </AiToolComponentRefProvider>,
        );
        act(() => {
            componentDirectory!.registerComponentRef({ current: {
                    getComponentName: () => AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                    startCollection,
                } as any });
        });
        const handle = componentDirectory!
            .findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;

        const operation = handle.collectLiveBaselineForAi({
            query: {
                start_query: { field: 'Physics_speed_kmh', operator: 'gte', value: 100 },
                end_query: { field: 'Physics_brake', operator: 'gte', value: 0.8 },
            },
            timeout_seconds: 12,
        });
        const termination = new Promise((resolve) => operation.notifyTerminated(resolve));

        expect(startCollection).toHaveBeenCalledWith({
            query: {
                start_query: { field: 'Physics_speed_kmh', operator: 'gte', value: 100 },
                end_query: { field: 'Physics_brake', operator: 'gte', value: 0.8 },
            },
            timeoutMs: 12_000,
        });
        expect(operation.statuses).toEqual([]);
        await expect(operation.result).resolves.toEqual(completed);
        await expect(termination).resolves.toEqual({
            status: 'collector-complete',
            result: completed,
        });
    });

    it('rejects restart without mounting a collector when collection is not in progress', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={createRuntime('acc') as any}>
                    <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </LiveSessionContext.Provider>
                <RegistrationObserver />
            </AiToolComponentRefProvider>,
        );
        const handle = componentDirectory!
            .findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;
        expect(componentDirectory!.findComponentRef(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)).toBeNull();

        const operation = handle.restartLiveBaselineForAi();

        await expect(operation.result).rejects.toMatchObject({
            name: 'BaselineCollectionNotStartedError',
            componentName: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            message: 'Baseline collection is not in progress. Start a new collection instead.',
        });
        await expect(operation.result).rejects.toBeInstanceOf(BaselineCollectionNotStartedError);
        expect(componentDirectory!.findComponentRef(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)).toBeNull();
    });

    it('returns only the analysis status to AI', async () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const requestAnalysis = jest.fn(() => createAiToolOperationFrom(() => ({
            status: 'ready' as const,
            message: 'Telemetry analysis is ready.',
            analysis: {
                status: 'success',
                session_id: 'baseline-analysis-1',
                samples_analyzed: 3,
                segments: [{ id: 'segment-1' }],
            },
            source: 'baseline_lap_record' as const,
            baseline: {
                id: 'baseline-1',
                lap_id: 2,
                lap_time_ms: 98_765,
                captured_at: 1,
                track: 'brands_hatch',
                car: 'Ferrari 296',
                sample_count: 3,
            },
            chartId: 'analysis-chart-1',
            component_name: 'visualization:analysis-results',
            pageId: 'baseline-analysis-page-1',
            pageCount: 1,
        }), 'ready'));

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={createRuntime('acc') as any}>
                    <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </LiveSessionContext.Provider>
                <RegistrationObserver />
            </AiToolComponentRefProvider>,
        );
        act(() => {
            componentDirectory!.registerComponentRef({ current: {
                    getComponentName: () => AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                    requestAnalysis,
                } as any });
        });
        const handle = componentDirectory!
            .findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;

        await expect(handle.analyzeLiveRecordedAnalysisForAi({ limit: 1 }).result)
            .resolves.toEqual({ status: 'ready' });
        expect(requestAnalysis).toHaveBeenCalledWith({ limit: 1 });
    });

    it('reads the most recently appended analysis page regardless of the displayed page', () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const runtime: any = createRuntime('acc');
        runtime.analysisResultPages = [
            { id: 'older-page', elements: [], baseline: { lap_id: 1 } },
            { id: 'latest-page', elements: [], baseline: { lap_id: 2 } },
        ];
        runtime.activeAnalysisResultPageId = 'older-page';

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={runtime}>
                    <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </LiveSessionContext.Provider>
                <RegistrationObserver />
            </AiToolComponentRefProvider>,
        );

        const handle = componentDirectory!
            .findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;
        expect(handle.getLatestAnalysisResultPage()).toMatchObject({ id: 'latest-page' });
    });

    it.each(detectionCases)('shows the $name detector state at the session gate', ({
        detection,
        title,
        visualState,
        canStart,
    }) => {
        mockedUseDesktopGame.mockReturnValue(detection);

        renderView();

        expect(screen.getByRole('status')).toHaveTextContent(title);
        expect(screen.getByRole('status').parentElement).toHaveAttribute('data-state', visualState);
        expect(screen.getByTestId('live-session-gate')).toBeInTheDocument();
        expect(screen.queryByTestId('live-session-recorder-host')).not.toBeInTheDocument();
        expect(screen.queryByText('Live workspace')).not.toBeInTheDocument();
        const startButton = screen.getByRole('button', { name: 'Start New Session' });
        if (canStart) {
            expect(startButton).toBeEnabled();
        } else {
            expect(startButton).toBeDisabled();
        }
    });

    it.each([
        ['acc', 'Assetto Corsa Competizione'],
        ['ac', 'Assetto Corsa'],
        ['iracing', 'iRacing'],
    ] as const)('captures detected %s when Start New Session is selected', (game, label) => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: game, detectionStatus: 'detected', error: null });
        const runtime = createRuntime();
        renderView(runtime);

        fireEvent.click(screen.getByRole('button', { name: 'Start New Session' }));

        expect(runtime.startLiveSession).toHaveBeenCalledWith(game);
        expect(screen.getByRole('status')).toHaveTextContent(`${label} detected`);
    });

    it('keeps the captured ACC workspace when detector polling changes or errors', () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });
        const runtime = createRuntime('acc');
        const view = renderView(runtime);

        expect(screen.getByRole('status')).toHaveTextContent('Assetto Corsa Competizione session');
        expect(screen.getByText('Live workspace')).toBeInTheDocument();

        mockedUseDesktopGame.mockReturnValue({ detectedGame: null, detectionStatus: 'error', error: 'tasklist failed' });
        view.rerender(
            <LiveSessionContext.Provider value={runtime as any}>
                <LiveSessionContent name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
            </LiveSessionContext.Provider>,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Assetto Corsa Competizione session');
        expect(screen.getByRole('status')).not.toHaveTextContent('tasklist failed');
        expect(screen.getByText('Live workspace')).toBeInTheDocument();
    });

    it.each([
        ['ac', 'Assetto Corsa'],
        ['iracing', 'iRacing'],
    ] as const)('renders a limited workspace without ACC controls for %s', (game, label) => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: 'acc', detectionStatus: 'detected', error: null });

        renderView(createRuntime(game));

        expect(screen.getByTestId('limited-live-workspace')).toHaveAccessibleName(`${label} limited live workspace`);
        expect(screen.queryByText('Live workspace')).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Start Recording' })).not.toBeInTheDocument();
        expect(screen.getByTestId('live-session-recorder-host')).toBeInTheDocument();
    });

    it('keeps New Session enabled during an active recording and opens the recorder flow', () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: null, detectionStatus: 'not-detected', error: null });
        const runtime = {
            ...createRuntime('acc'),
            recordingState: RecordingState.RECORDING,
        };
        renderView(runtime);

        const newSessionButton = screen.getByRole('button', { name: 'New Session' });
        expect(newSessionButton).toBeEnabled();
        fireEvent.click(newSessionButton);
        expect(runtime.recorderControl.openUploadFlow).toHaveBeenCalledTimes(1);
    });

    it('shows a restored local-recording error without hiding the session workspace', () => {
        mockedUseDesktopGame.mockReturnValue({ detectedGame: null, detectionStatus: 'not-detected', error: null });
        const runtime = {
            ...createRuntime('acc'),
            recordingState: RecordingState.UPLOAD_READY,
            restorationStatus: 'error',
            restorationError: 'The local recording file is missing or unreadable. Upload is unavailable; discard this draft to clear it.',
        };

        renderView(runtime as any);

        expect(screen.getByRole('alert')).toHaveTextContent('local recording file is missing or unreadable');
        expect(screen.getByText('Live workspace')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'New Session' })).toBeEnabled();
    });

    it('exposes detector updates as an atomic polite live status', () => {
        mockedUseDesktopGame.mockReturnValue(detectionCases[0].detection);
        renderView();

        expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite');
        expect(screen.getByRole('status')).toHaveAttribute('aria-atomic', 'true');
    });
});
