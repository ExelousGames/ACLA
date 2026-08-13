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
                section_count: 6,
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
