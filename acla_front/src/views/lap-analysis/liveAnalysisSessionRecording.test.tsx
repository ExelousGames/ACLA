import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import { RecordingState } from './recording-state';
import LiveAnalysisSessionRecording from './liveAnalysisSessionRecording';

const mockApiPost = jest.fn();

jest.mock('hooks/AuthProvider', () => ({
    useAuth: () => ({
        userEmail: 'driver@example.com',
        userProfile: { id: 'user-1' },
    }),
}));

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: (...args: any[]) => mockApiPost(...args) },
}));

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const container = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
    return {
        AlertDialog: {
            Root: ({ open, children }: { open?: boolean; children?: React.ReactNode }) => open ? children : null,
            Content: container,
            Title: container,
            Description: container,
        },
        Box: container,
        Card: container,
        Flex: container,
        Grid: container,
        Heading: container,
        Spinner: () => React.createElement('span'),
        Text: container,
        Button: ({ children, onClick, disabled }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
            React.createElement('button', { onClick, disabled }, children)
        ),
    };
});

const createRuntime = (overrides: Record<string, unknown> = {}) => ({
    sessionGame: 'acc' as const,
    currentTelemetry: { Static_track: 'Monza', Static_car_model: 'GT3' },
    telemetryStatus: null,
        staticData: { Static_track: 'Monza', Static_car_model: 'GT3' },
    recordingState: RecordingState.READY,
    recordingMetadata: null,
    recordingFileKey: null,
    recordingActive: false,
    recordingGame: null,
    recordedSampleCount: 0,
    restorationStatus: 'idle',
    restorationError: null,
    recordingFileValidation: null,
    recorderControl: null,
    analysisResultPages: [],
    activeAnalysisResultPageId: null,
    startLiveSession: jest.fn(),
    endLiveSession: jest.fn().mockResolvedValue(undefined),
    setRecordingMetadata: jest.fn(),
    transitionRecordingState: jest.fn(),
    startRecordingSession: jest.fn().mockResolvedValue({
        ok: true,
        game: 'acc',
        filePath: 'C:\\recordings\\acc.jsonl',
        startedAt: 1,
    }),
    stopRecordingSession: jest.fn().mockResolvedValue({
        game: 'acc',
        filePath: 'C:\\recordings\\acc.jsonl',
        writtenSamples: 5,
    }),
    streamRecordedTelemetry: jest.fn().mockResolvedValue({ rowCount: 0, totalBytes: 0 }),
    clearRecordingSession: jest.fn(),
    clearPersistedDraft: jest.fn(),
    registerRecorderControl: jest.fn(),
    appendAnalysisResultPage: jest.fn(),
    selectAnalysisResultPage: jest.fn(),
    updateActiveAnalysisResultPage: jest.fn(),
    ...overrides,
});

const renderRecorder = (runtime: ReturnType<typeof createRuntime>) => {
    const host = document.createElement('div');
    host.id = `recorder-host-${Math.random()}`;
    document.body.appendChild(host);
    const view = render(
        <LiveSessionContext.Provider value={runtime as any}>
            <LiveAnalysisSessionRecording recorderHostId={host.id} />
        </LiveSessionContext.Provider>,
    );
    return { ...view, host };
};

describe('recording session controls', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: { deleteTempFile: jest.fn().mockResolvedValue({ success: true }) },
        });
    });

    it('starts the shared recording pipeline for the active DesktopGame', async () => {
        const runtime = createRuntime();
        const view = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Start Recording' }));

        await waitFor(() => expect(runtime.startRecordingSession).toHaveBeenCalledWith('acc'));
        expect(runtime.setRecordingMetadata).toHaveBeenCalledWith(expect.objectContaining({
            mapName: 'Monza',
            carName: 'GT3',
            gameRecordedFrom: 'acc',
        }));
        view.unmount();
        view.host.remove();
    });

    it('drives coming-soon behavior from the discriminated unsupported result', async () => {
        const runtime = createRuntime({
            sessionGame: 'iracing',
            startRecordingSession: jest.fn().mockResolvedValue({
                ok: false,
                error: { type: 'unsupported-recording-game', message: 'Reader missing.' },
            }),
        });
        const view = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Start Recording' }));

        expect(await screen.findByText('Live recording for this simulator is coming soon.')).toBeInTheDocument();
        expect(runtime.startRecordingSession).toHaveBeenCalledWith('iracing');
        view.unmount();
        view.host.remove();
    });

    it('uses the application-owned stop boundary without a recording id', async () => {
        const runtime = createRuntime({
            recordingState: RecordingState.RECORDING,
            recordingActive: true,
            recordingGame: 'acc',
        });
        const view = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Stop Recording' }));

        await waitFor(() => expect(runtime.stopRecordingSession).toHaveBeenCalledWith('manual'));
        view.unmount();
        view.host.remove();
    });

    it('streams finalized chunks into upload without rebuilding one telemetry array', async () => {
        mockApiPost.mockImplementation((url: string) => Promise.resolve({
            data: url === '/racing-session/upload/init' ? { uploadId: 'upload-1' } : {},
        }));
        const streamRecordedTelemetry = jest.fn(async (onChunk) => {
            onChunk([{ Physics_speed_kmh: 100 }]);
            onChunk([{ Physics_speed_kmh: 101 }]);
            return { rowCount: 2, totalBytes: 100 };
        });
        const runtime = createRuntime({
            recordingState: RecordingState.UPLOAD_READY,
            recordingFileKey: 'C:\\recordings\\iracing.jsonl',
            recordedSampleCount: 2,
            sessionGame: 'iracing',
            recordingMetadata: {
                sessionName: 'Race',
                mapName: 'Track',
                carName: 'Car',
                gameRecordedFrom: 'iracing',
            },
            streamRecordedTelemetry,
        });
        const view = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        const buttons = await screen.findAllByRole('button', { name: 'Upload Session' });
        fireEvent.click(buttons[buttons.length - 1]);

        await waitFor(() => expect(mockApiPost).toHaveBeenCalledWith(
            '/racing-session/upload/init',
            expect.objectContaining({ game_recorded_from: 'iracing' }),
        ));
        await waitFor(() => expect(mockApiPost.mock.calls.filter(([url]) => String(url).includes('/chunk?'))).toHaveLength(2));
        expect(streamRecordedTelemetry).toHaveBeenCalledTimes(1);
        view.unmount();
        view.host.remove();
    });
});
