import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import { RecordingState } from '../recording-state';
import LiveAnalysisSessionRecording from './liveAnalysisSessionRecording';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import { SessionScreenCaptureProvider } from './SessionScreenCapture';

const mockApiPost = jest.fn();

jest.mock('hooks/AuthProvider', () => ({
    useAuth: () => ({
        userEmail: 'driver@example.com',
        userProfile: { id: 'user-1' },
    }),
}));

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: (...args: any[]) => mockApiPost(...args), delete: jest.fn().mockResolvedValue({}) },
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
    staticData: { Static_track: 'Monza', Static_car_model: 'GT3' },
    recordingState: RecordingState.READY,
    recordingMetadata: null,
    recordingFileKey: null as string | null,
    recordingActive: false,
    recordingGame: null,
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

const renderRecorder = (runtime: ReturnType<typeof createRuntime>, committedCount = 0) => {
    liveTelemetryStore.restoreCommittedSampleCount(committedCount);
    const host = document.createElement('div');
    host.id = `recorder-host-${Math.random()}`;
    document.body.appendChild(host);
    const recorder = (value: ReturnType<typeof createRuntime>) => (
        <LiveSessionContext.Provider value={value as any}>
            <SessionScreenCaptureProvider>
                <LiveAnalysisSessionRecording recorderHostId={host.id} />
            </SessionScreenCaptureProvider>
        </LiveSessionContext.Provider>
    );
    const view = render(recorder(runtime));
    return { ...view, host, rerenderRuntime: (value: ReturnType<typeof createRuntime>) => view.rerender(recorder(value)) };
};

describe('recording session controls', () => {
    afterEach(() => { delete window.sessionVideo; delete window.screenCapture; });
    beforeEach(() => {
        liveTelemetryStore.resetSession();
        jest.clearAllMocks();
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: {
                deleteTempFile: jest.fn().mockResolvedValue({ success: true }),
                importLocalIRacingTelemetry: jest.fn().mockRejectedValue(new Error('No .ibt file available')),
            },
        });
    });

    it('selects and controls the shared video from the recording bar without a Track Vision panel', async () => {
        const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');
        const track = { stop: jest.fn(), onended: null };
        const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
        const getDisplayMedia = jest.fn().mockResolvedValue(stream);
        Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getDisplayMedia } });
        window.screenCapture = {
            listSources: jest.fn().mockResolvedValue([{ id: 'window:42', name: 'Simulator' }]),
            selectSource: jest.fn().mockResolvedValue(undefined),
        };
        const runtime = createRuntime();
        const view = renderRecorder(runtime);
        try {
            const bar = within(view.host);
            expect(bar.getByRole('button', { name: 'Share game screen' })).toBeDisabled();
            fireEvent.click(bar.getByRole('button', { name: 'Refresh sources' }));
            await bar.findByRole('option', { name: 'Simulator' });
            fireEvent.change(bar.getByRole('combobox', { name: 'Game window or screen' }), { target: { value: 'window:42' } });
            fireEvent.click(bar.getByRole('button', { name: 'Share game screen' }));
            await bar.findByRole('button', { name: 'Stop capture' });
            expect(window.screenCapture.selectSource).toHaveBeenCalledWith('window:42');
            expect(getDisplayMedia).toHaveBeenCalledWith(expect.objectContaining({ audio: false }));
            expect(bar.getByRole('combobox', { name: 'Game window or screen' })).toBeDisabled();
            fireEvent.click(bar.getByRole('button', { name: 'Start Recording' }));
            await waitFor(() => expect(runtime.startRecordingSession).toHaveBeenCalledWith('acc'));
            view.rerenderRuntime({ ...runtime, recordingState: RecordingState.RECORDING, recordingActive: true,
                recordingFileKey: 'C:\\recordings\\acc.jsonl' });
            expect(track.stop).not.toHaveBeenCalled();
            fireEvent.click(bar.getByRole('button', { name: 'Stop capture' }));
            expect(track.stop).toHaveBeenCalledTimes(1);
            expect(runtime.stopRecordingSession).not.toHaveBeenCalled();
            expect(bar.getByRole('combobox', { name: 'Game window or screen' })).toBeDisabled();
            expect(bar.getByRole('button', { name: 'Share game screen' })).toBeDisabled();
        } finally {
            view.unmount();
            view.host.remove();
            delete window.screenCapture;
            if (originalMediaDevices) Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices);
            else Reflect.deleteProperty(navigator, 'mediaDevices');
        }
    });

    it('locks screen sharing while telemetry starts and unlocks it if startup fails', async () => {
        window.screenCapture = {
            listSources: jest.fn().mockResolvedValue([{ id: 'window:42', name: 'Simulator' }]),
            selectSource: jest.fn(),
        };
        let completeStart!: (result: unknown) => void;
        const runtime = createRuntime({ startRecordingSession: jest.fn(() => new Promise(resolve => { completeStart = resolve; })) });
        const view = renderRecorder(runtime);
        try {
            const bar = within(view.host);
            fireEvent.click(bar.getByRole('button', { name: 'Refresh sources' }));
            await bar.findByRole('option', { name: 'Simulator' });
            fireEvent.change(bar.getByRole('combobox'), { target: { value: 'window:42' } });
            expect(bar.getByRole('button', { name: 'Share game screen' })).toBeEnabled();
            fireEvent.click(bar.getByRole('button', { name: 'Start Recording' }));
            expect(bar.getByRole('button', { name: 'Share game screen' })).toBeDisabled();
            expect(bar.getByRole('button', { name: 'Start Recording' })).toBeDisabled();
            expect(bar.getByRole('combobox')).toBeDisabled();
            await act(async () => completeStart({ ok: false, error: { type: 'unsupported-recording-game' } }));
            expect(bar.getByRole('button', { name: 'Share game screen' })).toBeEnabled();
            expect(bar.getByRole('button', { name: 'Start Recording' })).toBeEnabled();
        } finally { view.unmount(); view.host.remove(); }
    });

    it('waits for pending screen sharing before allowing telemetry recording', async () => {
        const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');
        let finishCapture!: (stream: unknown) => void;
        const track = { stop: jest.fn(), onended: null };
        const getDisplayMedia = jest.fn(() => new Promise(resolve => { finishCapture = resolve; }));
        Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getDisplayMedia } });
        window.screenCapture = {
            listSources: jest.fn().mockResolvedValue([{ id: 'window:42', name: 'Simulator' }]),
            selectSource: jest.fn().mockResolvedValue(undefined),
        };
        const runtime = createRuntime();
        const view = renderRecorder(runtime);
        try {
            const bar = within(view.host);
            fireEvent.click(bar.getByRole('button', { name: 'Refresh sources' }));
            await bar.findByRole('option', { name: 'Simulator' });
            fireEvent.change(bar.getByRole('combobox'), { target: { value: 'window:42' } });
            fireEvent.click(bar.getByRole('button', { name: 'Share game screen' }));
            expect(bar.getByRole('button', { name: 'Start Recording' })).toBeDisabled();
            fireEvent.click(bar.getByRole('button', { name: 'Start Recording' }));
            expect(runtime.startRecordingSession).not.toHaveBeenCalled();
            await waitFor(() => expect(getDisplayMedia).toHaveBeenCalled());
            await act(async () => finishCapture({ getTracks: () => [track], getVideoTracks: () => [track] }));
            expect(bar.getByRole('button', { name: 'Start Recording' })).toBeEnabled();
        } finally {
            view.unmount(); view.host.remove();
            if (originalMediaDevices) Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices);
            else Reflect.deleteProperty(navigator, 'mediaDevices');
        }
    });

    it.each([RecordingState.RECORDING, RecordingState.HOLDING, RecordingState.RESUME_READY, RecordingState.UPLOAD_READY])(
        'keeps video start locked for an existing session in %s', (recordingState) => {
            window.screenCapture = { listSources: jest.fn(), selectSource: jest.fn() };
            const view = renderRecorder(createRuntime({ recordingState, recordingFileKey: 'C:\\recordings\\acc.jsonl' }));
            try {
                const bar = within(view.host);
                expect(bar.getByRole('combobox')).toBeDisabled();
                expect(bar.getByRole('button', { name: 'Refresh sources' })).toBeDisabled();
                expect(bar.getByRole('button', { name: 'Share game screen' })).toBeDisabled();
                expect(bar.getByText('Video can only start with a new session recording.')).toBeInTheDocument();
            } finally { view.unmount(); view.host.remove(); }
        },
    );

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

    it.each([true, false])('restores local video and uploads it when Include recorded video is %s', async (includeVideo) => {
        const clip = { filePath: 'C:\\recordings\\video.mp4', name: 'video.mp4', size: 4, mimeType: 'video/mp4', startedAt: 1 };
        window.sessionVideo = { create: jest.fn(), append: jest.fn(), finish: jest.fn(),
            list: jest.fn().mockResolvedValue([clip]), read: jest.fn().mockResolvedValue({ data: 'AQIDBA==', bytesRead: 4 }),
            delete: jest.fn(), reveal: jest.fn().mockResolvedValue(undefined) };
        mockApiPost.mockResolvedValue({ data: { uploadId: 'upload-1' } });
        const runtime = createRuntime({
            recordingState: RecordingState.UPLOAD_READY, recordingFileKey: 'C:\\recordings\\acc.jsonl',
            recordingMetadata: { sessionName: 'Race', mapName: 'Track', carName: 'Car', gameRecordedFrom: 'acc' },
            streamRecordedTelemetry: jest.fn(async (onChunk) => {
                await onChunk([{ Physics_speed_kmh: 100 }]);
                return { rowCount: 1, totalBytes: 50 };
            }),
        });
        const view = renderRecorder(runtime, 1);
        fireEvent.click(await screen.findByRole('button', { name: 'Show local video' }));
        expect(window.sessionVideo.reveal).toHaveBeenCalledWith(clip.filePath);
        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        expect(screen.getByLabelText('Include recorded video')).toBeChecked();
        if (!includeVideo) fireEvent.click(screen.getByLabelText('Include recorded video'));
        const buttons = screen.getAllByRole('button', { name: 'Upload Session' });
        fireEvent.click(buttons[buttons.length - 1]);
        await waitFor(() => expect(runtime.clearPersistedDraft).toHaveBeenCalledTimes(1));
        expect(mockApiPost.mock.calls.filter(([url]) => url.includes('/video-chunk'))).toHaveLength(includeVideo ? 1 : 0);
        expect(window.sessionVideo.delete).not.toHaveBeenCalled();
        view.unmount(); view.host.remove();
    });

    it('gets the floating bar map from Static_track in the live session context', () => {
        const runtime = createRuntime({
            recordingMetadata: {
                sessionName: 'Race',
                mapName: 'Stale Track',
                carName: 'GT3',
                gameRecordedFrom: 'acc',
            },
        });
        const view = renderRecorder(runtime);

        expect(view.host).toHaveTextContent('MAP');
        expect(view.host).toHaveTextContent('Monza');
        expect(view.host).not.toHaveTextContent('Stale Track');
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

    it('reads all recorded fields on demand for a restored session without uploading', async () => {
        const runtime = createRuntime({
            recordingState: RecordingState.UPLOAD_READY,
            recordingFileKey: 'C:\\recordings\\restored.jsonl',
            recordingFileValidation: { exists: true, readable: true, hasData: true, size: 100 },
            streamRecordedTelemetry: jest.fn(async (onChunk) => {
                onChunk([{ Physics_speed_kmh: 100 }]);
                onChunk([{ Graphics_car_coordinates: [{ x: 1, y: 2, z: 3 }], Static_track: 'Monza' }]);
                return { rowCount: 2, totalBytes: 100 };
            }),
        });
        const view = renderRecorder(runtime);
        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        expect(runtime.streamRecordedTelemetry).not.toHaveBeenCalled();

        fireEvent.click(screen.getByText('Telemetry data structure'));
        expect(await screen.findByText(/Structure from 2 records/)).toBeInTheDocument();
        fireEvent.click(screen.getByText('Record', { selector: 'code' }));
        expect(await screen.findByText('Physics_speed_kmh')).toBeInTheDocument();
        expect(screen.getByText('Graphics_car_coordinates')).toBeInTheDocument();
        expect(screen.getByText('Static_track')).toBeInTheDocument();
        expect(runtime.streamRecordedTelemetry).toHaveBeenCalledWith(expect.any(Function), undefined, runtime.recordingFileKey);
        expect(mockApiPost).not.toHaveBeenCalled();
        expect(runtime.stopRecordingSession).not.toHaveBeenCalled();
        expect(window.electronAPI.deleteTempFile).not.toHaveBeenCalled();

        fireEvent.click(screen.getByRole('button', { name: 'Keep Session' }));
        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        expect(screen.queryByText('Record', { selector: 'code' })).not.toBeInTheDocument();
        view.unmount();
        view.host.remove();
    });

    it('previews only the app recording for iRacing without requiring an IBT import', async () => {
        const runtime = createRuntime({
            sessionGame: 'iracing', recordingState: RecordingState.UPLOAD_READY, recordingFileKey: 'C:\\recordings\\iracing.jsonl',
            streamRecordedTelemetry: jest.fn(async (onChunk) => {
                onChunk([{ Physics_speed_kmh: 100 }]);
                return { rowCount: 1, totalBytes: 50 };
            }),
        });
        const view = renderRecorder(runtime, 1);
        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        expect(screen.queryByText(/\.ibt|exit the car/i)).not.toBeInTheDocument();
        fireEvent.click(screen.getByText('Telemetry data structure'));
        expect(await screen.findByText('iracing_live')).toBeInTheDocument();
        expect(screen.queryByText('iracing_recorded')).not.toBeInTheDocument();
        expect(runtime.streamRecordedTelemetry).not.toHaveBeenCalled();

        fireEvent.click(screen.getByText('iracing_live'));
        fireEvent.click(await screen.findByText('Record', { selector: 'code' }));
        expect(await screen.findByText('Physics_speed_kmh')).toBeInTheDocument();
        expect(runtime.streamRecordedTelemetry).toHaveBeenCalledTimes(1);
        expect(runtime.streamRecordedTelemetry).toHaveBeenCalledWith(expect.any(Function), undefined, runtime.recordingFileKey);
        expect(window.electronAPI.importLocalIRacingTelemetry).not.toHaveBeenCalled();
        expect(mockApiPost).not.toHaveBeenCalled();
        expect(runtime.stopRecordingSession).not.toHaveBeenCalled();
        view.unmount(); view.host.remove();
    });

    it('discards only the app recording for iRacing', async () => {
        const runtime = createRuntime({
            sessionGame: 'iracing', recordingState: RecordingState.UPLOAD_READY, recordingFileKey: 'C:\\recordings\\iracing.jsonl',
        });
        const view = renderRecorder(runtime, 1);
        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        fireEvent.click(screen.getByRole('button', { name: 'Discard Session' }));
        await waitFor(() => expect(runtime.endLiveSession).toHaveBeenCalledTimes(1));
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledTimes(1);
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(runtime.recordingFileKey);
        expect(window.electronAPI.importLocalIRacingTelemetry).not.toHaveBeenCalled();
        expect(runtime.clearPersistedDraft).toHaveBeenCalledTimes(1);
        expect(runtime.streamRecordedTelemetry).not.toHaveBeenCalled();
        view.unmount(); view.host.remove();
    });

    it.each([RecordingState.UPLOAD_READY, RecordingState.RECORDING])('uploads only live iRacing samples from %s without an IBT file', async (recordingState) => {
        mockApiPost.mockImplementation((url: string) => Promise.resolve({
            data: url === '/racing-session/upload/init' ? { uploadId: 'upload-1' } : {},
        }));
        const streamRecordedTelemetry = jest.fn(async (onChunk, _onProgress, _filePath) => {
            expect(window.electronAPI.deleteTempFile).not.toHaveBeenCalled();
            if (recordingState === RecordingState.RECORDING) {
                expect(runtime.stopRecordingSession).toHaveBeenCalledWith('manual');
            }
            await onChunk([{ Physics_speed_kmh: 100 }]);
            await onChunk([{ Physics_speed_kmh: 101 }]);
            return { rowCount: 2, totalBytes: 100 };
        });
        const runtime = createRuntime({
            recordingState,
            recordingFileKey: 'C:\\recordings\\iracing.jsonl',
            sessionGame: 'iracing',
            recordingMetadata: {
                sessionName: 'Race', mapName: 'Track', carName: 'Car', gameRecordedFrom: 'iracing',
            },
            streamRecordedTelemetry,
        });
        const view = renderRecorder(runtime, 2);
        if (recordingState === RecordingState.RECORDING) {
            const control = runtime.registerRecorderControl.mock.calls[0][0];
            act(() => control.openUploadFlow());
        } else {
            fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        }
        const buttons = await screen.findAllByRole('button', { name: 'Upload Session' });
        fireEvent.click(buttons[buttons.length - 1]);

        await waitFor(() => expect(runtime.clearPersistedDraft).toHaveBeenCalledTimes(1));
        const initializations = mockApiPost.mock.calls.filter(([url]) => url === '/racing-session/upload/init');
        expect(initializations).toEqual([['/racing-session/upload/init', {
            sessionName: 'Race', game_recorded_from: 'iracing_live', mapName: 'Track', carName: 'Car', userId: 'user-1',
        }]]);
        expect(streamRecordedTelemetry).toHaveBeenCalledTimes(1);
        expect(streamRecordedTelemetry.mock.calls[0][2]).toBe(runtime.recordingFileKey);
        const chunks = mockApiPost.mock.calls.filter(([url]) => String(url).includes('/chunk?'));
        expect(chunks.map(([, body]) => body)).toEqual([
            { chunkIndex: 0, chunk: [{ Physics_speed_kmh: 100 }] },
            { chunkIndex: 1, chunk: [{ Physics_speed_kmh: 101 }] },
        ]);
        expect(mockApiPost.mock.calls.filter(([url]) => String(url).includes('/complete?'))).toHaveLength(1);
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledTimes(1);
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(runtime.recordingFileKey);
        expect(window.electronAPI.importLocalIRacingTelemetry).not.toHaveBeenCalled();
        view.unmount(); view.host.remove();
    });

    it('keeps the app recording after a failed upload and cleans up after a successful retry', async () => {
        let failCompletion = true;
        mockApiPost.mockImplementation((url: string) => {
            if (String(url).includes('/complete?') && failCompletion) throw new Error('Network unavailable');
            return Promise.resolve({ data: { uploadId: 'upload-1' } });
        });
        const runtime = createRuntime({
            sessionGame: 'iracing', recordingState: RecordingState.UPLOAD_READY, recordingFileKey: 'C:\\recordings\\iracing.jsonl',
            recordingMetadata: { sessionName: 'Race', mapName: 'Track', carName: 'Car', gameRecordedFrom: 'iracing' },
            streamRecordedTelemetry: jest.fn(async (onChunk) => {
                await onChunk([{ Physics_speed_kmh: 100 }]);
                return { rowCount: 1, totalBytes: 50 };
            }),
        });
        const view = renderRecorder(runtime, 1);
        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        const buttons = await screen.findAllByRole('button', { name: 'Upload Session' });
        fireEvent.click(buttons[buttons.length - 1]);
        expect(await screen.findByText('Network unavailable')).toBeInTheDocument();
        expect(runtime.clearPersistedDraft).not.toHaveBeenCalled();
        expect(window.electronAPI.deleteTempFile).not.toHaveBeenCalled();
        failCompletion = false;
        fireEvent.click(screen.getByRole('button', { name: 'Retry Upload' }));
        await waitFor(() => expect(runtime.clearPersistedDraft).toHaveBeenCalledTimes(1));
        const initializations = mockApiPost.mock.calls.filter(([url]) => url === '/racing-session/upload/init');
        expect(initializations).toHaveLength(2);
        expect(initializations.every(([, body]) => body.game_recorded_from === 'iracing_live')).toBe(true);
        expect(runtime.streamRecordedTelemetry).toHaveBeenCalledTimes(2);
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledTimes(1);
        expect(window.electronAPI.importLocalIRacingTelemetry).not.toHaveBeenCalled();
        view.unmount(); view.host.remove();
    });

    it.each(['acc', 'ac'])('continues uploading one app recording for %s', async (game) => {
        mockApiPost.mockResolvedValue({ data: { uploadId: 'upload-1' } });
        const runtime = createRuntime({
            sessionGame: game, recordingState: RecordingState.UPLOAD_READY, recordingFileKey: `C:\\recordings\\${game}.jsonl`,
            recordingMetadata: { sessionName: 'Race', mapName: 'Track', carName: 'Car', gameRecordedFrom: game },
            streamRecordedTelemetry: jest.fn(async (onChunk) => {
                await onChunk([{ Physics_speed_kmh: 100 }]);
                return { rowCount: 1, totalBytes: 50 };
            }),
        });
        const view = renderRecorder(runtime, 1);
        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        const buttons = await screen.findAllByRole('button', { name: 'Upload Session' });
        fireEvent.click(buttons[buttons.length - 1]);
        await waitFor(() => expect(runtime.clearPersistedDraft).toHaveBeenCalledTimes(1));
        expect(mockApiPost).toHaveBeenCalledWith('/racing-session/upload/init', expect.objectContaining({ sessionName: 'Race', game_recorded_from: game }));
        expect(runtime.streamRecordedTelemetry).toHaveBeenCalledTimes(1);
        expect(window.electronAPI.importLocalIRacingTelemetry).not.toHaveBeenCalled();
        view.unmount(); view.host.remove();
    });
});
