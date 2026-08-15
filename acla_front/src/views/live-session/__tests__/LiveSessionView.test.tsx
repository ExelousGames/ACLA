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
import { BaselineCollectionAlreadyStartedError } from 'contexts/AiToolComponentError';
import { createAiToolOperationFrom } from 'components/ai-engineering-tools';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';

jest.mock('contexts/DesktopGameContext', () => ({
    useDesktopGame: jest.fn(),
}));

jest.mock('../LiveTelemetryWorkspace', () => () => <div>Live workspace</div>);

import { useDesktopGame } from 'contexts/DesktopGameContext';
import LiveSessionView from '../LiveSessionView';

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
    currentTelemetry: {},
    telemetryStatus: null,
    staticData: {},
    recordingState: RecordingState.CHECKING,
    recordingMetadata: null,
    recordingFileKey: null,
    recordedSampleCount: 0,
    restorationStatus: 'idle',
    restorationError: null,
    recordingFileValidation: null,
    sessionIntelligence: {},
    analysisResultPages: [],
    activeAnalysisResultPageId: null,
    recorderControl: { openUploadFlow: jest.fn() },
    startLiveSession: jest.fn(),
    endLiveSession: jest.fn(),
    setCurrentTelemetry: jest.fn(),
    setStaticData: jest.fn(),
    setRecordingMetadata: jest.fn(),
    transitionRecordingState: jest.fn(),
    appendTelemetrySample: jest.fn(),
    readRecordedTelemetry: jest.fn(),
    finalizeRecordingWrites: jest.fn(),
    clearRecordingSession: jest.fn(),
    clearPersistedDraft: jest.fn(),
    registerRecorderControl: jest.fn(),
});

const renderView = (runtime = createRuntime()) => render(
    <LiveSessionContext.Provider value={runtime as any}>
        <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
    </LiveSessionContext.Provider>,
);

let componentDirectory: AiToolComponentRefDirectory | null = null;
const RegistrationObserver = () => {
    componentDirectory = useAiToolComponentRefDirectory();
    return null;
};

const renderRegisteredView = (runtime: ReturnType<typeof createRuntime>) => {
    render(
        <AiToolComponentRefProvider>
            <LiveSessionContext.Provider value={runtime as any}>
                <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
            </LiveSessionContext.Provider>
            <RegistrationObserver />
        </AiToolComponentRefProvider>,
    );
    return componentDirectory!
        .findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;
};

const createTelemetryRuntime = () => {
    const sessionIntelligence = new SessionIntelligence();
    sessionIntelligence.tick({
        Static_track: 'brands_hatch',
        Graphics_completed_laps: 1,
        Graphics_normalized_car_position: 0.1,
        Physics_timestamp: 0,
        Physics_speed_kmh: 100,
        Physics_wheel_pressure_front_left: 27,
        Physics_wheel_pressure_front_right: 28,
        Physics_wheel_pressure_rear_left: 29,
        Physics_wheel_pressure_rear_right: 30,
        status: 5,
        message: 6,
    });
    sessionIntelligence.tick({
        Static_track: 'brands_hatch',
        Graphics_completed_laps: 1,
        Graphics_normalized_car_position: 0.2,
        Physics_timestamp: 100,
        Physics_speed_kmh: 120,
        Physics_wheel_pressure_front_left: 28,
        Physics_wheel_pressure_front_right: 29,
        Physics_wheel_pressure_rear_left: 30,
        Physics_wheel_pressure_rear_right: 31,
        status: 7,
        message: 8,
    });
    return {
        ...createRuntime('acc'),
        sessionIntelligence,
    };
};

describe('LiveSessionView', () => {
    beforeEach(() => {
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
        runtime.currentTelemetry = { Physics_speed_kmh: 210 };
        runtime.sessionIntelligence = {
            getLiveSessionSnapshot: () => ({
                status: 'ready',
                track: 'Monza',
                car: 'BMW M4 GT3',
                current_lap: 4,
                completed_laps: 3,
                normalized_position: 0.42,
                sample_count: 1250,
                live_session_type: 'practice',
                baseline_ready: true,
                baseline_collection_started: true,
                baseline_progress_percent: 100,
                baseline_lap: 3,
                completed_lap_count: 3,
            }),
        };

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={runtime as any}>
                    <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
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
        const startCollection = jest.fn(() => createAiToolOperationFrom(() => { throw failure; }));

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={createRuntime('acc') as any}>
                    <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </LiveSessionContext.Provider>
                <RegistrationObserver />
            </AiToolComponentRefProvider>,
        );
        act(() => {
            componentDirectory!.reserveComponentRef(
                AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                Symbol('baseline-collection-test'),
                {
                    getComponentName: () => AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                    startCollection,
                } as any,
            );
        });
        const handle = componentDirectory!
            .findComponentRef<LiveSessionHandle>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current!;

        const operation = handle.collectLiveBaselineForAi({ timeout_seconds: 12 });

        expect(startCollection).toHaveBeenCalledWith({ timeoutMs: 12_000 });
        expect(operation.statuses).toHaveLength(0);
        await expect(operation.result).rejects.toBe(failure);
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
                lap: 2,
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
        })));

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={createRuntime('acc') as any}>
                    <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
                </LiveSessionContext.Provider>
                <RegistrationObserver />
            </AiToolComponentRefProvider>,
        );
        act(() => {
            componentDirectory!.reserveComponentRef(
                AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                Symbol('baseline-analysis-test'),
                {
                    getComponentName: () => AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                    requestAnalysis,
                } as any,
            );
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
            { id: 'older-page', elements: [], baseline: { lap: 1 } },
            { id: 'latest-page', elements: [], baseline: { lap: 2 } },
        ];
        runtime.activeAnalysisResultPageId = 'older-page';

        render(
            <AiToolComponentRefProvider>
                <LiveSessionContext.Provider value={runtime}>
                    <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
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
                <LiveSessionView name={AI_TOOL_COMPONENT_NAMES.LIVE_SESSION} />
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
