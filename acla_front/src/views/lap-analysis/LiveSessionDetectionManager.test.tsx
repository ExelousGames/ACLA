import React from 'react';
import { act, render } from '@testing-library/react';
import { RecordingState } from './recording-state';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';

let mockDetectedGame: 'ac' | 'acc' | 'iracing' | null = null;
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
    currentTelemetry: {},
    telemetryStatus: null,
    staticData: {},
    recordingState: RecordingState.CHECKING,
    recordingMetadata: null,
    recordingFileKey: null,
    recordedSampleCount: 0,
    sessionIntelligence: {},
    liveRangeTodoListHandle: null,
    liveRangeTodoListSnapshot: null,
    setCurrentTelemetry: jest.fn(),
    setStaticData: jest.fn(),
    setRecordingMetadata: jest.fn(),
    transitionRecordingState: jest.fn(),
    appendTelemetrySample: jest.fn(),
    readRecordedTelemetry: jest.fn(),
    finalizeRecordingWrites: jest.fn(),
    clearRecordingSession: jest.fn(),
    registerLiveRangeTodoListHandle: jest.fn(),
    publishLiveRangeTodoListSnapshot: jest.fn(),
});

describe('LiveSessionDetectionManager desktop game gating', () => {
    let stream: {
        dispose: jest.Mock;
        onMessage: jest.Mock;
        waitUntilReady: jest.Mock;
    };
    let removeMessageListener: jest.Mock;

    beforeEach(() => {
        mockDetectedGame = null;
        mockDetectionStatus = 'not-detected';
        removeMessageListener = jest.fn();
        stream = {
            dispose: jest.fn().mockResolvedValue(undefined),
            onMessage: jest.fn().mockReturnValue(removeMessageListener),
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

    it('starts the ACC shared-memory checker exactly once while ACC remains selected', async () => {
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

    it.each([
        ['Assetto Corsa', 'ac', 'detected'],
        ['iRacing', 'iracing', 'detected'],
        ['no detected game', null, 'not-detected'],
        ['a detector error', null, 'error'],
    ] as const)('does not start ACC telemetry for %s', async (_label, detectedGame, detectionStatus) => {
        mockDetectedGame = detectedGame;
        mockDetectionStatus = detectionStatus;
        const runtime = createRuntime();
        renderManager(runtime);
        await flushPromises();

        expect(mockedCreatePythonStreamSession).not.toHaveBeenCalled();
        expect(runtime.transitionRecordingState).toHaveBeenCalledWith({ type: 'sessionUnavailable' });
    });

    it('stops the checker and marks the session unavailable when ACC is lost', async () => {
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

        expect(removeMessageListener).toHaveBeenCalledTimes(1);
        expect(stream.dispose).toHaveBeenCalled();
        expect(runtime.transitionRecordingState).toHaveBeenCalledWith({ type: 'sessionUnavailable' });
    });
});
