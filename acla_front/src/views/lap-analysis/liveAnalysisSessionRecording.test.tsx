import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import { RecordingState } from './recording-state';
import LiveAnalysisSessionRecording from './liveAnalysisSessionRecording';

const mockApiPost = jest.fn();
let pythonMessageHandler: ((incomingId: number, message: string) => void) | null = null;
let registeredRecorderControl: { openUploadFlow: () => void } | null = null;

jest.mock('hooks/AuthProvider', () => ({
    useAuth: () => ({
        userEmail: 'driver@example.com',
        userProfile: { id: 'user-1' },
    }),
}));

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: {
        post: (...args: any[]) => mockApiPost(...args),
    },
}));

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const container = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
    const text = ({ children }: { children?: React.ReactNode }) => React.createElement('span', null, children);

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
        Text: text,
        Button: ({ children, onClick, disabled }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
            React.createElement('button', { onClick, disabled }, children)
        ),
    };
});

const createRuntime = (overrides: Record<string, unknown> = {}) => ({
    sessionGame: 'acc' as const,
    currentTelemetry: {
        Static_track: 'Monza',
        Static_car_model: 'GT3',
    },
    telemetryStatus: ACC_STATUS.ACC_LIVE,
    staticData: { track: 'Monza', car_model: 'GT3' },
    recordingState: RecordingState.READY,
    recordingMetadata: null,
    recordingFileKey: null,
    recordedSampleCount: 0,
    restorationStatus: 'idle',
    restorationError: null,
    recordingFileValidation: null,
    sessionIntelligence: {},
    recorderControl: null,
    startLiveSession: jest.fn(),
    endLiveSession: jest.fn(),
    setCurrentTelemetry: jest.fn(),
    setStaticData: jest.fn(),
    setRecordingMetadata: jest.fn(),
    transitionRecordingState: jest.fn(),
    appendTelemetrySample: jest.fn().mockResolvedValue(undefined),
    readRecordedTelemetry: jest.fn().mockResolvedValue([]),
    finalizeRecordingWrites: jest.fn().mockResolvedValue(undefined),
    clearRecordingSession: jest.fn(),
    clearPersistedDraft: jest.fn(),
    registerRecorderControl: jest.fn(),
    ...overrides,
});

const renderRecorder = (runtime: ReturnType<typeof createRuntime>) => {
    const host = document.createElement('div');
    host.id = 'recorder-host';
    document.body.appendChild(host);
    runtime.registerRecorderControl.mockImplementation((control) => {
        registeredRecorderControl = control;
    });

    const view = render(
        <LiveSessionContext.Provider value={runtime as any}>
            <LiveAnalysisSessionRecording recorderHostId="recorder-host" />
        </LiveSessionContext.Provider>,
    );

    return { ...view, host };
};

describe('ACC racing session recording metadata', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        pythonMessageHandler = null;
        registeredRecorderControl = null;
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: {
                runPythonScript: jest.fn().mockResolvedValue({ shellId: 7 }),
                onPythonMessage: jest.fn((handler) => {
                    pythonMessageHandler = handler;
                    return jest.fn();
                }),
                onPythonEnd: jest.fn().mockReturnValue(jest.fn()),
                stopPythonScript: jest.fn().mockResolvedValue({ success: true }),
            },
        });
    });

    const startRecordingAndEmit = async (
        runtime: ReturnType<typeof createRuntime>,
        samples: Record<string, unknown>[],
    ) => {
        const { host } = renderRecorder(runtime);
        fireEvent.click(await screen.findByRole('button', { name: 'Start Recording' }));
        await waitFor(() => expect(pythonMessageHandler).not.toBeNull());

        await act(async () => {
            for (const sample of samples) {
                pythonMessageHandler?.(7, JSON.stringify(sample));
            }
            await Promise.resolve();
        });

        return host;
    };

    const baselineSample = {
        Static_track: 'Monza',
        Graphics_session_time_left: 900,
        Static_car_model: 'GT3',
        Graphics_completed_lap: 3,
        Graphics_current_time: 45_000,
        Graphics_distance_traveled: 12_000,
        Graphics_used_fuel: 18,
        Physics_packed_id: 500,
    };

    it('captures ACC as the game when recording starts', async () => {
        const runtime = createRuntime();
        const { host } = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Start Recording' }));

        await waitFor(() => {
            expect(runtime.setRecordingMetadata).toHaveBeenCalledWith(expect.objectContaining({
                mapName: 'Monza',
                carName: 'GT3',
                gameRecordedFrom: 'acc',
            }));
        });
        host.remove();
    });

    it('sends the locked game under the snake_case upload field', async () => {
        mockApiPost.mockImplementation((url: string) => Promise.resolve({
            data: url === '/racing-session/upload/init' ? { uploadId: 'upload-1' } : {},
        }));
        const runtime = createRuntime({
            recordingState: RecordingState.UPLOAD_READY,
            recordingMetadata: {
                sessionName: 'Race 1',
                mapName: 'Monza',
                carName: 'GT3',
                gameRecordedFrom: 'iracing',
            },
            recordingFileKey: '../session_recording/temp/race.jsonl',
            recordedSampleCount: 1,
            readRecordedTelemetry: jest.fn().mockResolvedValue([{ speed: 120 }]),
        });
        const { host } = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        const uploadButtons = await screen.findAllByRole('button', { name: 'Upload Session' });
        fireEvent.click(uploadButtons[uploadButtons.length - 1]);

        await waitFor(() => {
            const initCall = mockApiPost.mock.calls.find(([url]) => url === '/racing-session/upload/init');
            expect(initCall?.[1]).toEqual(expect.objectContaining({
                userId: 'user-1',
                game_recorded_from: 'acc',
            }));
            expect(initCall?.[1]).not.toHaveProperty('gameRecordedFrom');
        });
        host.remove();
    });

    it('stops and finalizes an active recording before reading data for upload', async () => {
        const lifecycle: string[] = [];
        mockApiPost.mockImplementation((url: string) => Promise.resolve({
            data: url === '/racing-session/upload/init' ? { uploadId: 'upload-1' } : {},
        }));
        const runtime = createRuntime();
        const view = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Start Recording' }));
        await waitFor(() => expect(pythonMessageHandler).not.toBeNull());
        (window.electronAPI.stopPythonScript as jest.Mock).mockImplementation(async () => {
            lifecycle.push('stop');
            return { success: true };
        });

        const activeRuntime = {
            ...runtime,
            recordingState: RecordingState.RECORDING,
            recordingMetadata: {
                sessionName: 'Active Race',
                mapName: 'Monza',
                carName: 'GT3',
                gameRecordedFrom: 'acc' as const,
            },
            recordingFileKey: '../session_recording/temp/active.jsonl',
            recordedSampleCount: 2,
            finalizeRecordingWrites: jest.fn(async () => { lifecycle.push('finalize'); }),
            readRecordedTelemetry: jest.fn(async () => {
                lifecycle.push('read');
                return [{ speed: 120 }];
            }),
        };
        view.rerender(
            <LiveSessionContext.Provider value={activeRuntime as any}>
                <LiveAnalysisSessionRecording recorderHostId="recorder-host" />
            </LiveSessionContext.Provider>,
        );
        await waitFor(() => expect(registeredRecorderControl).not.toBeNull());

        act(() => registeredRecorderControl?.openUploadFlow());
        fireEvent.click(screen.getByRole('button', { name: 'Upload Session' }));

        await waitFor(() => expect(activeRuntime.readRecordedTelemetry).toHaveBeenCalledTimes(1));
        expect(lifecycle.slice(0, 3)).toEqual(['stop', 'finalize', 'read']);
        expect(activeRuntime.finalizeRecordingWrites).toHaveBeenCalledTimes(1);
        view.host.remove();
    });

    it('discards an active recording only after stopping it and returns to the detector gate', async () => {
        const runtime = createRuntime();
        const view = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Start Recording' }));
        await waitFor(() => expect(pythonMessageHandler).not.toBeNull());

        const activeRuntime = {
            ...runtime,
            recordingState: RecordingState.RECORDING,
            recordingMetadata: {
                sessionName: 'Discard Race',
                mapName: 'Monza',
                carName: 'GT3',
                gameRecordedFrom: 'acc' as const,
            },
            recordingFileKey: '../session_recording/temp/discard.jsonl',
            recordedSampleCount: 1,
            finalizeRecordingWrites: jest.fn().mockResolvedValue(undefined),
            endLiveSession: jest.fn(),
        };
        view.rerender(
            <LiveSessionContext.Provider value={activeRuntime as any}>
                <LiveAnalysisSessionRecording recorderHostId="recorder-host" />
            </LiveSessionContext.Provider>,
        );
        await waitFor(() => expect(registeredRecorderControl).not.toBeNull());

        act(() => registeredRecorderControl?.openUploadFlow());
        fireEvent.click(screen.getByRole('button', { name: 'Discard Session' }));

        await waitFor(() => expect(activeRuntime.endLiveSession).toHaveBeenCalledTimes(1));
        expect(activeRuntime.clearPersistedDraft).toHaveBeenCalledTimes(1);
        expect(window.electronAPI.stopPythonScript).toHaveBeenCalledWith(7);
        expect(activeRuntime.finalizeRecordingWrites).toHaveBeenCalledTimes(1);
        expect(window.electronAPI.runPythonScript).toHaveBeenCalledWith(
            'delete_telemetry_file.py',
            expect.objectContaining({ args: ['../session_recording/temp/discard.jsonl'] }),
        );
        view.host.remove();
    });

    it('stops a recording process that finishes launching after the session was discarded', async () => {
        let resolveLaunch!: (value: { shellId: number }) => void;
        const launchPromise = new Promise<{ shellId: number }>((resolve) => {
            resolveLaunch = resolve;
        });
        (window.electronAPI.runPythonScript as jest.Mock).mockReturnValueOnce(launchPromise);
        const runtime = createRuntime();
        const view = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Start Recording' }));
        const activeRuntime = {
            ...runtime,
            recordingState: RecordingState.RECORDING,
            endLiveSession: jest.fn(),
        };
        view.rerender(
            <LiveSessionContext.Provider value={activeRuntime as any}>
                <LiveAnalysisSessionRecording recorderHostId="recorder-host" />
            </LiveSessionContext.Provider>,
        );
        await waitFor(() => expect(registeredRecorderControl).not.toBeNull());

        act(() => registeredRecorderControl?.openUploadFlow());
        fireEvent.click(screen.getByRole('button', { name: 'Discard Session' }));
        await waitFor(() => expect(activeRuntime.endLiveSession).toHaveBeenCalledTimes(1));
        expect(window.electronAPI.stopPythonScript).not.toHaveBeenCalled();

        await act(async () => {
            resolveLaunch({ shellId: 7 });
            await launchPromise;
        });

        await waitFor(() => expect(window.electronAPI.stopPythonScript).toHaveBeenCalledWith(7));
        view.host.remove();
    });

    it('keeps the captured session when the decision flow is closed', async () => {
        const runtime = createRuntime({
            recordingState: RecordingState.RECORDING,
            recordingFileKey: '../session_recording/temp/keep.jsonl',
            recordedSampleCount: 1,
        });
        const { host } = renderRecorder(runtime);
        await waitFor(() => expect(registeredRecorderControl).not.toBeNull());

        act(() => registeredRecorderControl?.openUploadFlow());
        fireEvent.click(screen.getByRole('button', { name: 'Keep Session' }));

        expect(runtime.endLiveSession).not.toHaveBeenCalled();
        expect(runtime.clearPersistedDraft).not.toHaveBeenCalled();
        expect(window.electronAPI.stopPythonScript).not.toHaveBeenCalled();
        expect(screen.queryByText('Finish Live Session')).not.toBeInTheDocument();
        host.remove();
    });

    it('keeps the captured session locked when upload fails', async () => {
        mockApiPost.mockRejectedValue(new Error('network unavailable'));
        const runtime = createRuntime({
            recordingState: RecordingState.UPLOAD_READY,
            recordingMetadata: {
                sessionName: 'Failed Race',
                mapName: 'Monza',
                carName: 'GT3',
                gameRecordedFrom: 'acc',
            },
            recordingFileKey: '../session_recording/temp/failure.jsonl',
            recordedSampleCount: 1,
            readRecordedTelemetry: jest.fn().mockResolvedValue([{ speed: 120 }]),
        });
        const { host } = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        const uploadButtons = await screen.findAllByRole('button', { name: 'Upload Session' });
        fireEvent.click(uploadButtons[uploadButtons.length - 1]);

        expect(await screen.findByText('network unavailable')).toBeInTheDocument();
        expect(runtime.endLiveSession).not.toHaveBeenCalled();
        expect(runtime.clearPersistedDraft).not.toHaveBeenCalled();
        host.remove();
    });

    it('shows a broken local recording, disables upload, and still allows discard', async () => {
        const runtime = createRuntime({
            recordingState: RecordingState.UPLOAD_READY,
            recordingMetadata: {
                sessionName: 'Broken Race',
                mapName: 'Monza',
                carName: 'GT3',
                gameRecordedFrom: 'acc',
            },
            recordingFileKey: 'C:\\recordings\\telemetry_broken.jsonl',
            recordedSampleCount: 10,
            restorationStatus: 'error',
            restorationError: 'The local recording file is missing or unreadable. Upload is unavailable; discard this draft to clear it.',
            recordingFileValidation: {
                exists: false,
                readable: false,
                hasData: false,
                size: 0,
            },
        });
        const { host } = renderRecorder(runtime);

        await waitFor(() => expect(registeredRecorderControl).not.toBeNull());
        act(() => registeredRecorderControl?.openUploadFlow());

        expect(screen.getByText(/local recording file is missing or unreadable/i)).toBeInTheDocument();
        expect(screen.getAllByRole('button', { name: 'Upload Session' }).every((button) => button.hasAttribute('disabled'))).toBe(true);
        fireEvent.click(screen.getByRole('button', { name: 'Discard Session' }));

        await waitFor(() => expect(runtime.endLiveSession).toHaveBeenCalledTimes(1));
        expect(runtime.clearPersistedDraft).toHaveBeenCalledTimes(1);
        host.remove();
    });

    it('returns to the detector gate after a successful upload', async () => {
        mockApiPost.mockImplementation((url: string) => Promise.resolve({
            data: url === '/racing-session/upload/init' ? { uploadId: 'upload-1' } : {},
        }));
        const runtime = createRuntime({
            recordingState: RecordingState.UPLOAD_READY,
            recordingMetadata: {
                sessionName: 'Successful Race',
                mapName: 'Monza',
                carName: 'GT3',
                gameRecordedFrom: 'acc',
            },
            recordingFileKey: '../session_recording/temp/success.jsonl',
            recordedSampleCount: 1,
            readRecordedTelemetry: jest.fn().mockResolvedValue([{ speed: 120 }]),
        });
        const { host } = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Upload Session' }));
        jest.useFakeTimers();
        const uploadButtons = screen.getAllByRole('button', { name: 'Upload Session' });
        fireEvent.click(uploadButtons[uploadButtons.length - 1]);

        await act(async () => {
            await Promise.resolve();
            await Promise.resolve();
            await Promise.resolve();
            await Promise.resolve();
        });
        expect(screen.getByText('Upload completed successfully!')).toBeInTheDocument();
        act(() => jest.advanceTimersByTime(2500));

        expect(runtime.endLiveSession).toHaveBeenCalledTimes(1);
        expect(runtime.clearPersistedDraft).toHaveBeenCalledTimes(1);
        jest.useRealTimers();
        host.remove();
    });

    it('keeps the floating bar Discard action and returns to the detector gate', async () => {
        const runtime = createRuntime({
            recordingState: RecordingState.UPLOAD_READY,
            recordingMetadata: {
                sessionName: 'Discarded Race',
                mapName: 'Monza',
                carName: 'GT3',
                gameRecordedFrom: 'acc',
            },
            recordingFileKey: '../session_recording/temp/floating-discard.jsonl',
            recordedSampleCount: 1,
        });
        const { host } = renderRecorder(runtime);

        fireEvent.click(await screen.findByRole('button', { name: 'Discard' }));

        await waitFor(() => expect(runtime.endLiveSession).toHaveBeenCalledTimes(1));
        expect(runtime.clearPersistedDraft).toHaveBeenCalledTimes(1);
        expect(window.electronAPI.runPythonScript).toHaveBeenCalledWith(
            'delete_telemetry_file.py',
            expect.objectContaining({ args: ['../session_recording/temp/floating-discard.jsonl'] }),
        );
        host.remove();
    });

    it('does not run reconnect detection without a preceding unavailable packet', async () => {
        const runtime = createRuntime();
        const resetSample = {
            ...baselineSample,
            Graphics_session_time_left: 1_800,
            Graphics_completed_lap: 0,
            Graphics_current_time: 0,
            Graphics_distance_traveled: 0,
            Graphics_used_fuel: 0,
        };

        const host = await startRecordingAndEmit(runtime, [baselineSample, resetSample]);

        expect(runtime.appendTelemetrySample).toHaveBeenCalledTimes(2);
        expect(runtime.transitionRecordingState).not.toHaveBeenCalledWith({
            type: 'recordingStopped',
            reason: 'complete',
        });
        host.remove();
    });

    it('continues after an outage when all seven continuity fields remain valid', async () => {
        const runtime = createRuntime();
        const continuingSample = {
            ...baselineSample,
            Graphics_session_time_left: 899,
            Graphics_current_time: 45_016,
            Graphics_distance_traveled: 12_003,
            Graphics_used_fuel: 18.01,
        };

        const host = await startRecordingAndEmit(runtime, [
            baselineSample,
            { available: false },
            continuingSample,
        ]);

        expect(runtime.appendTelemetrySample).toHaveBeenCalledTimes(2);
        expect(runtime.transitionRecordingState).not.toHaveBeenCalledWith({
            type: 'recordingStopped',
            reason: 'complete',
        });
        host.remove();
    });

    it('completes the old recording and does not append a new-session sample', async () => {
        const runtime = createRuntime();
        const newSessionSample = {
            ...baselineSample,
            Graphics_session_time_left: 1_800,
            Graphics_completed_lap: 0,
            Graphics_current_time: 0,
            Graphics_distance_traveled: 0,
            Graphics_used_fuel: 0,
        };

        const host = await startRecordingAndEmit(runtime, [
            baselineSample,
            { available: false },
            newSessionSample,
        ]);

        expect(runtime.appendTelemetrySample).toHaveBeenCalledTimes(1);
        expect(runtime.transitionRecordingState).toHaveBeenCalledWith({
            type: 'recordingStopped',
            reason: 'complete',
        });
        expect(runtime.finalizeRecordingWrites).toHaveBeenCalledTimes(1);
        host.remove();
    });
});
