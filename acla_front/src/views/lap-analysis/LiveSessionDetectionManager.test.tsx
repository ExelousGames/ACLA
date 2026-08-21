import React from 'react';
import { act, render } from '@testing-library/react';
import { RecordingState } from './recording-state';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { PythonStreamEvent } from 'services/pythonStreaming';

let mockDetectedGame: DesktopGame | null = null;
let mockDetectionStatus = 'not-detected';

jest.mock('contexts/DesktopGameContext', () => ({
    useDesktopGame: () => ({
        detectedGame: mockDetectedGame,
        detectionStatus: mockDetectionStatus,
        error: mockDetectionStatus === 'error' ? 'tasklist failed' : null,
    }),
}));

jest.mock('services/pythonStreaming', () => ({
    createPythonStreamSession: jest.fn(),
}));

import { createPythonStreamSession } from 'services/pythonStreaming';
import LiveSessionDetectionManager from './LiveSessionDetectionManager';

const mockedCreatePythonStreamSession = createPythonStreamSession as jest.Mock;

const flushPromises = async () => {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
};

const createRuntime = () => ({
    sessionGame: mockDetectedGame,
    currentTelemetry: {},
    telemetryStatus: null,
    staticData: {},
    recordingState: RecordingState.CHECKING,
    recordingMetadata: null,
    recordingFileKey: null,
    recordedSampleCount: 0,
    recorderControl: null,
    startLiveSession: jest.fn(),
    endLiveSession: jest.fn(),
    setRecordingMetadata: jest.fn(),
    transitionRecordingState: jest.fn(),
    clearRecordingSession: jest.fn(),
    registerRecorderControl: jest.fn(),
});

describe('LiveSessionDetectionManager desktop game gating', () => {
    let stream: {
        dispose: jest.Mock;
        onMessage: jest.Mock;
        waitUntilReady: jest.Mock;
    };
    let removeMessageListener: jest.Mock;
    let processStreamUpdate: ((event: PythonStreamEvent<Record<string, unknown>>) => void) | null;

    beforeEach(() => {
        mockDetectedGame = null;
        mockDetectionStatus = 'not-detected';
        removeMessageListener = jest.fn();
        processStreamUpdate = null;
        stream = {
            dispose: jest.fn().mockResolvedValue(undefined),
            onMessage: jest.fn().mockImplementation((callback) => {
                processStreamUpdate = callback;
                return removeMessageListener;
            }),
            waitUntilReady: jest.fn().mockResolvedValue(undefined),
        };
        mockedCreatePythonStreamSession.mockReset();
        mockedCreatePythonStreamSession.mockResolvedValue(stream);
    });

    const managerTree = (runtime: ReturnType<typeof createRuntime>) => (
        <LiveSessionContext.Provider value={runtime as any}>
            <LiveSessionDetectionManager />
        </LiveSessionContext.Provider>
    );

    const renderManager = (runtime: ReturnType<typeof createRuntime>) => render(
        managerTree(runtime),
    );

    it('starts the ACC shared-memory checker exactly once after ACC is captured', async () => {
        mockDetectedGame = 'acc';
        const runtime = createRuntime();
        const view = render(<React.StrictMode>{managerTree(runtime)}</React.StrictMode>);
        await flushPromises();

        expect(mockedCreatePythonStreamSession).toHaveBeenCalledTimes(1);
        expect(mockedCreatePythonStreamSession).toHaveBeenCalledWith(expect.objectContaining({
            scriptName: 'ACCCheckAvailableSession.py',
        }));

        view.rerender(<React.StrictMode>{managerTree(runtime)}</React.StrictMode>);
        await flushPromises();
        expect(mockedCreatePythonStreamSession).toHaveBeenCalledTimes(1);
    });

    it('uses checker updates only to detect ACC session availability', async () => {
        mockDetectedGame = 'acc';
        const runtime = createRuntime();
        renderManager(runtime);
        await flushPromises();

        act(() => {
            processStreamUpdate?.({
                status: 'update',
                data: {
                    Graphics_status: ACC_STATUS.ACC_LIVE,
                    Static_track: 'Monza',
                    Static_car_model: 'Ferrari 296 GT3',
                },
            });
        });

        expect(runtime.transitionRecordingState).toHaveBeenCalledWith({ type: 'sessionAvailable' });
        expect(runtime.transitionRecordingState).toHaveBeenCalledTimes(1);
    });

    it('treats checker control messages as availability state, not telemetry', async () => {
        mockDetectedGame = 'acc';
        const runtime = createRuntime();
        renderManager(runtime);
        await flushPromises();

        act(() => {
            processStreamUpdate?.({
                status: 'update',
                data: { available: false, checking: true },
            });
        });

        expect(runtime.transitionRecordingState).toHaveBeenCalledWith({ type: 'sessionUnavailable' });
        expect(runtime.transitionRecordingState).toHaveBeenCalledTimes(1);
    });

    it.each([
        ['Assetto Corsa', 'ac', 'detected'],
        ['iRacing', 'iracing', 'detected'],
        ['no detected game', null, 'not-detected'],
        ['a detector error', null, 'error'],
    ] as const)('does not start the ACC session checker for %s', async (_label, detectedGame, detectionStatus) => {
        mockDetectedGame = detectedGame;
        mockDetectionStatus = detectionStatus;
        const runtime = createRuntime();
        renderManager(runtime);
        await flushPromises();

        expect(mockedCreatePythonStreamSession).not.toHaveBeenCalled();
        expect(runtime.transitionRecordingState).toHaveBeenCalledWith({ type: 'sessionUnavailable' });
    });

    it('retains the checker when later detector polling reports a different game', async () => {
        mockDetectedGame = 'acc';
        const runtime = createRuntime();
        const view = renderManager(runtime);
        await flushPromises();
        expect(mockedCreatePythonStreamSession).toHaveBeenCalledTimes(1);

        mockDetectedGame = 'ac';
        view.rerender(
            <LiveSessionContext.Provider value={runtime as any}>
                <LiveSessionDetectionManager />
            </LiveSessionContext.Provider>,
        );
        await flushPromises();

        expect(mockedCreatePythonStreamSession).toHaveBeenCalledTimes(1);
        expect(removeMessageListener).not.toHaveBeenCalled();
        expect(stream.dispose).not.toHaveBeenCalled();
    });

    it('stops the checker when the captured session ends', async () => {
        mockDetectedGame = 'acc';
        const runtime = createRuntime();
        const view = renderManager(runtime);
        await flushPromises();
        expect(mockedCreatePythonStreamSession).toHaveBeenCalledTimes(1);

        const endedRuntime = { ...runtime, sessionGame: null };
        view.rerender(
            <LiveSessionContext.Provider value={endedRuntime as any}>
                <LiveSessionDetectionManager />
            </LiveSessionContext.Provider>,
        );
        await flushPromises();

        expect(removeMessageListener).toHaveBeenCalledTimes(1);
        expect(stream.dispose).toHaveBeenCalled();
        expect(runtime.transitionRecordingState).toHaveBeenCalledWith({ type: 'sessionUnavailable' });
    });
});
