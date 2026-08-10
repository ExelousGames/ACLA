import React, { useContext } from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { createPythonStreamSession } from 'services/pythonStreaming';
import { RecordingState } from 'views/lap-analysis/recording-state';
import {
    getPersistedLiveSessionDraft,
    LIVE_SESSION_DRAFT_STORAGE_KEY,
    savePersistedLiveSessionDraft,
} from '../live-session-draft-storage';
import { PERSISTED_LIVE_SESSION_DRAFT_VERSION } from '../live-session-types';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';

jest.mock('services/pythonStreaming', () => ({
    createPythonStreamSession: jest.fn(),
}));

const mockedCreatePythonStreamSession = createPythonStreamSession as jest.Mock;
const telemetryPath = 'C:\\Users\\driver\\AppData\\Roaming\\Kestrel\\acla-temp\\telemetry_live_1.jsonl';

const saveDraft = (ownerEmail: string, lastRuntimeState: RecordingState, sampleCount = 42) => {
    savePersistedLiveSessionDraft({
        version: PERSISTED_LIVE_SESSION_DRAFT_VERSION,
        ownerEmail,
        sessionGame: 'acc',
        recordingMetadata: {
            sessionName: 'Friday Practice',
            mapName: 'Monza',
            carName: 'GT3',
            gameRecordedFrom: 'acc',
        },
        telemetryFilePath: telemetryPath,
        recordedSampleCount: sampleCount,
        lastRuntimeState,
        updatedAt: '2026-08-07T12:00:00.000Z',
    });
};

const RuntimeProbe = () => {
    const runtime = useContext(LiveSessionContext);
    return (
        <>
            <output data-testid="game">{runtime.sessionGame || 'none'}</output>
            <output data-testid="state">{runtime.recordingState}</output>
            <output data-testid="name">{runtime.recordingMetadata?.sessionName || 'none'}</output>
            <output data-testid="samples">{runtime.recordedSampleCount}</output>
            <output data-testid="file">{runtime.recordingFileKey || 'none'}</output>
            <output data-testid="restoration">{runtime.restorationStatus}</output>
            <output data-testid="error">{runtime.restorationError || 'none'}</output>
            <output data-testid="has-data">{String(runtime.recordingFileValidation?.hasData)}</output>
        </>
    );
};

const RecordingHarness = () => {
    const runtime = useContext(LiveSessionContext);
    return (
        <>
            <button type="button" onClick={() => runtime.startLiveSession('acc')}>Start</button>
            <button type="button" onClick={() => runtime.setRecordingMetadata({
                sessionName: 'New Run',
                mapName: 'Spa',
                carName: 'GT3',
                gameRecordedFrom: 'acc',
            })}>Metadata</button>
            <button type="button" onClick={() => runtime.transitionRecordingState({ type: 'recordingStarted' })}>Record</button>
            <button type="button" onClick={() => { void runtime.appendTelemetrySample({ speed: 120 }); }}>Sample</button>
        </>
    );
};

const ClearDraftHarness = () => {
    const runtime = useContext(LiveSessionContext);
    return (
        <>
            <output data-testid="clear-restoration">{runtime.restorationStatus}</output>
            <output data-testid="clear-page-count">{runtime.analysisResultPages.length}</output>
            <button type="button" onClick={() => runtime.appendAnalysisResultPage({
                baseline: {
                    id: 'baseline-before-discard',
                    lap: 1,
                    lap_time_ms: null,
                    captured_at: 1,
                    track: 'Monza',
                    car: 'GT3',
                    sample_count: 1,
                },
                elements: [{ id: 'result-before-discard', labels: ['MSP'] }],
            })}>Add page</button>
            <button type="button" onClick={runtime.clearPersistedDraft}>Clear draft</button>
        </>
    );
};

describe('live session draft persistence', () => {
    beforeEach(() => {
        window.localStorage.clear();
        mockedCreatePythonStreamSession.mockReset();
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: {
                validateTelemetryFile: jest.fn().mockResolvedValue({
                    exists: true,
                    readable: true,
                    hasData: true,
                    size: 1024,
                }),
                writeTempFile: jest.fn().mockResolvedValue({ success: true, path: telemetryPath }),
            },
        });
    });

    it.each([
        RecordingState.RECORDING,
        RecordingState.HOLDING,
        RecordingState.RESUME_READY,
        RecordingState.UPLOAD_READY,
    ])('restores a %s draft as upload-ready without restarting live processes', async (lastRuntimeState) => {
        saveDraft('Driver@Example.com', lastRuntimeState);

        render(<LiveSessionProvider ownerEmail=" driver@example.com "><RuntimeProbe /></LiveSessionProvider>);

        await waitFor(() => expect(screen.getByTestId('restoration')).toHaveTextContent('restored'));
        expect(screen.getByTestId('game')).toHaveTextContent('acc');
        expect(screen.getByTestId('state')).toHaveTextContent('UPLOAD_READY');
        expect(screen.getByTestId('name')).toHaveTextContent('Friday Practice');
        expect(screen.getByTestId('samples')).toHaveTextContent('42');
        expect(screen.getByTestId('file')).toHaveTextContent(telemetryPath);
        expect(mockedCreatePythonStreamSession).not.toHaveBeenCalled();
        expect(window.electronAPI.writeTempFile).not.toHaveBeenCalled();
    });

    it('ignores another account draft without deleting it', async () => {
        saveDraft('first@example.com', RecordingState.RECORDING);

        render(<LiveSessionProvider ownerEmail="second@example.com"><RuntimeProbe /></LiveSessionProvider>);

        await waitFor(() => expect(screen.getByTestId('restoration')).toHaveTextContent('not-found'));
        expect(screen.getByTestId('game')).toHaveTextContent('none');
        expect(getPersistedLiveSessionDraft('first@example.com')).not.toBeNull();
        expect(JSON.parse(window.localStorage.getItem(LIVE_SESSION_DRAFT_STORAGE_KEY) || '{}').drafts['first@example.com'])
            .toBeDefined();
    });

    it('keeps a missing recording visible as a broken upload-ready draft', async () => {
        saveDraft('driver@example.com', RecordingState.HOLDING);
        (window.electronAPI.validateTelemetryFile as jest.Mock).mockResolvedValue({
            exists: false,
            readable: false,
            hasData: false,
            size: 0,
            error: 'ENOENT',
        });

        render(<LiveSessionProvider ownerEmail="driver@example.com"><RuntimeProbe /></LiveSessionProvider>);

        await waitFor(() => expect(screen.getByTestId('restoration')).toHaveTextContent('error'));
        expect(screen.getByTestId('game')).toHaveTextContent('acc');
        expect(screen.getByTestId('state')).toHaveTextContent('UPLOAD_READY');
        expect(screen.getByTestId('error')).toHaveTextContent('local recording file is missing or unreadable');
        expect(getPersistedLiveSessionDraft('driver@example.com')).not.toBeNull();
    });

    it('restores a readable empty file but marks it as having no uploadable data', async () => {
        saveDraft('driver@example.com', RecordingState.RECORDING, 0);
        (window.electronAPI.validateTelemetryFile as jest.Mock).mockResolvedValue({
            exists: true,
            readable: true,
            hasData: false,
            size: 0,
        });

        render(<LiveSessionProvider ownerEmail="driver@example.com"><RuntimeProbe /></LiveSessionProvider>);

        await waitFor(() => expect(screen.getByTestId('restoration')).toHaveTextContent('restored'));
        expect(screen.getByTestId('has-data')).toHaveTextContent('false');
        expect(screen.getByTestId('error')).toHaveTextContent('none');
    });

    it('does not recreate a terminally cleared manifest when the provider unmounts', async () => {
        saveDraft('driver@example.com', RecordingState.UPLOAD_READY);
        const view = render(
            <LiveSessionProvider ownerEmail="driver@example.com">
                <ClearDraftHarness />
            </LiveSessionProvider>,
        );
        await waitFor(() => expect(screen.getByTestId('clear-restoration')).toHaveTextContent('restored'));

        fireEvent.click(screen.getByRole('button', { name: 'Add page' }));
        expect(screen.getByTestId('clear-page-count')).toHaveTextContent('1');
        fireEvent.click(screen.getByRole('button', { name: 'Clear draft' }));
        expect(screen.getByTestId('clear-page-count')).toHaveTextContent('0');
        view.unmount();

        expect(getPersistedLiveSessionDraft('driver@example.com')).toBeNull();
    });

    it('creates a new recording at the absolute persistent Electron path and saves it for the account', async () => {
        let onWriterMessage: ((event: Record<string, unknown>) => void) | null = null;
        mockedCreatePythonStreamSession.mockImplementation(async () => ({
            waitUntilReady: jest.fn().mockResolvedValue(undefined),
            onMessage: jest.fn((handler) => {
                onWriterMessage = handler;
                return jest.fn();
            }),
            send: jest.fn(async (_action, _payload, requestId) => {
                onWriterMessage?.({ status: 'ok', request_id: requestId });
            }),
            dispose: jest.fn().mockResolvedValue(undefined),
        }));

        render(
            <LiveSessionProvider ownerEmail="Driver@Example.com">
                <RecordingHarness />
            </LiveSessionProvider>,
        );
        await waitFor(() => expect(screen.getByRole('button', { name: 'Start' })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: 'Start' }));
        fireEvent.click(screen.getByRole('button', { name: 'Metadata' }));
        fireEvent.click(screen.getByRole('button', { name: 'Record' }));
        fireEvent.click(screen.getByRole('button', { name: 'Sample' }));

        await waitFor(() => expect(getPersistedLiveSessionDraft('driver@example.com')?.recordedSampleCount).toBe(1));
        expect(window.electronAPI.writeTempFile).toHaveBeenCalledWith({
            content: '',
            prefix: 'telemetry_live',
            extension: '.jsonl',
        });
        expect(mockedCreatePythonStreamSession).toHaveBeenCalledWith(expect.objectContaining({
            pythonOptions: expect.objectContaining({ args: [telemetryPath] }),
        }));
    });
});
