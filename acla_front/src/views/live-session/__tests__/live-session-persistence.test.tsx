import React, { useContext } from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { RecordingState } from 'views/lap-analysis/recording-state';
import {
    getPersistedLiveSessionDraft,
    LIVE_SESSION_DRAFT_STORAGE_KEY,
    savePersistedLiveSessionDraft,
} from '../live-session-draft-storage';
import { PERSISTED_LIVE_SESSION_DRAFT_VERSION, RecordingStartResult } from '../live-session-types';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';
import {
    liveTelemetryStore,
    useCommittedSampleCount,
    useTelemetrySampleIndex,
} from '../live-telemetry-store';

const telemetryPath = 'C:\\Users\\driver\\AppData\\Roaming\\Kestrel\\acla-temp\\telemetry_live_1.jsonl';
let recordedFileHandler: ((event: any) => void) | null = null;
let recordingViewHandler: ((event: any) => void) | null = null;

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
    const recordedSampleCount = useCommittedSampleCount();
    return (
        <>
            <output data-testid="game">{runtime.sessionGame || 'none'}</output>
            <output data-testid="state">{runtime.recordingState}</output>
            <output data-testid="name">{runtime.recordingMetadata?.sessionName || 'none'}</output>
            <output data-testid="samples">{recordedSampleCount}</output>
            <output data-testid="file">{runtime.recordingFileKey || 'none'}</output>
            <output data-testid="restoration">{runtime.restorationStatus}</output>
            <output data-testid="error">{runtime.restorationError || 'none'}</output>
            <output data-testid="has-data">{String(runtime.recordingFileValidation?.hasData)}</output>
        </>
    );
};

const RecordingHarness = () => {
    const runtime = useContext(LiveSessionContext);
    const sampleIndex = useTelemetrySampleIndex();
    return (
        <>
            <output data-testid="harness-game">{runtime.sessionGame || 'none'}</output>
            <output data-testid="harness-active">{String(runtime.recordingActive)}</output>
            <output data-testid="harness-sample-index">{sampleIndex}</output>
            <button type="button" onClick={() => runtime.startLiveSession('acc')}>Start</button>
            <button type="button" onClick={runtime.endLiveSession}>End</button>
            <button type="button" onClick={() => runtime.setRecordingMetadata({
                sessionName: 'New Run',
                mapName: 'Spa',
                carName: 'GT3',
                gameRecordedFrom: 'acc',
            })}>Metadata</button>
            <button type="button" onClick={() => { void runtime.startRecordingSession('acc'); }}>Record</button>
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
                    lap_id: 1,
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
        liveTelemetryStore.resetSession();
        window.localStorage.clear();
        recordedFileHandler = null;
        recordingViewHandler = null;
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: {
                onRecordingViewUpdate: jest.fn((handler) => {
                    recordingViewHandler = handler;
                    return jest.fn();
                }),
                onRecordingSessionEnded: jest.fn().mockReturnValue(jest.fn()),
                startRecordingSession: jest.fn().mockResolvedValue({
                    ok: true,
                    game: 'acc',
                    filePath: telemetryPath,
                    startedAt: 1,
                }),
                stopRecordingSession: jest.fn().mockResolvedValue({
                    game: 'acc',
                    filePath: telemetryPath,
                    writtenSamples: 1,
                }),
                onRecordedFileReadEvent: jest.fn((handler) => {
                    recordedFileHandler = handler;
                    return jest.fn();
                }),
                startRecordedFileRead: jest.fn().mockImplementation(async () => {
                    recordedFileHandler?.({
                        type: 'complete',
                        readId: 'read-1',
                        format: 'standard-flat',
                        game: 'acc',
                        rowCount: 42,
                        totalBytes: 1024,
                    });
                    return { readId: 'read-1' };
                }),
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
        expect(window.electronAPI.startRecordingSession).not.toHaveBeenCalled();
        expect(window.electronAPI.startRecordedFileRead).toHaveBeenCalledWith({
            filePath: telemetryPath,
            game: 'acc',
            purpose: 'validate',
        });
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
        (window.electronAPI.startRecordedFileRead as jest.Mock).mockRejectedValue(new Error('ENOENT'));

        render(<LiveSessionProvider ownerEmail="driver@example.com"><RuntimeProbe /></LiveSessionProvider>);

        await waitFor(() => expect(screen.getByTestId('restoration')).toHaveTextContent('error'));
        expect(screen.getByTestId('game')).toHaveTextContent('acc');
        expect(screen.getByTestId('state')).toHaveTextContent('UPLOAD_READY');
        expect(screen.getByTestId('error')).toHaveTextContent('local recording file is missing or unreadable');
        expect(getPersistedLiveSessionDraft('driver@example.com')).not.toBeNull();
    });

    it('restores a readable empty file but marks it as having no uploadable data', async () => {
        saveDraft('driver@example.com', RecordingState.RECORDING, 0);
        (window.electronAPI.startRecordedFileRead as jest.Mock).mockImplementation(async () => {
            recordedFileHandler?.({
                type: 'complete',
                readId: 'read-empty',
                format: 'standard-flat',
                game: 'acc',
                rowCount: 0,
                totalBytes: 0,
            });
            return { readId: 'read-empty' };
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
        render(
            <LiveSessionProvider ownerEmail="Driver@Example.com">
                <RecordingHarness />
            </LiveSessionProvider>,
        );
        await waitFor(() => expect(screen.getByRole('button', { name: 'Start' })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: 'Start' }));
        await waitFor(() => expect(screen.getByTestId('harness-game')).toHaveTextContent('acc'));
        fireEvent.click(screen.getByRole('button', { name: 'Metadata' }));
        fireEvent.click(screen.getByRole('button', { name: 'Record' }));
        await waitFor(() => expect(window.electronAPI.startRecordingSession).toHaveBeenCalledWith({ game: 'acc' }));
        act(() => {
            recordingViewHandler?.({
                type: 'frame',
                game: 'acc',
                sample: { Physics_speed_kmh: 120 },
                sequence: 1,
                committedSequence: 1,
                committedCount: 1,
            });
        });

        await waitFor(() => expect(getPersistedLiveSessionDraft('driver@example.com')?.recordedSampleCount).toBe(1));
        expect(getPersistedLiveSessionDraft('driver@example.com')?.telemetryFilePath).toBe(telemetryPath);
    });

    it('aligns recording view updates to writer row indexes', async () => {
        render(
            <LiveSessionProvider ownerEmail="driver@example.com">
                <RecordingHarness />
            </LiveSessionProvider>,
        );
        fireEvent.click(screen.getByRole('button', { name: 'Start' }));
        await waitFor(() => expect(screen.getByTestId('harness-game')).toHaveTextContent('acc'));

        fireEvent.click(screen.getByRole('button', { name: 'Record' }));
        await waitFor(() => expect(screen.getByTestId('harness-sample-index')).toHaveTextContent('-1'));
        act(() => {
            recordingViewHandler?.({
                type: 'frame',
                game: 'acc',
                sample: { Physics_speed_kmh: 120 },
                sequence: 1,
                committedSequence: 0,
                committedCount: 0,
            });
        });

        expect(screen.getByTestId('harness-sample-index')).toHaveTextContent('0');
    });

    it('waits for in-flight recording startup before stopping and resetting the session', async () => {
        let resolveStart!: (result: RecordingStartResult) => void;
        const startPromise = new Promise<RecordingStartResult>((resolve) => {
            resolveStart = resolve;
        });
        (window.electronAPI.startRecordingSession as jest.Mock).mockReturnValue(startPromise);

        render(
            <LiveSessionProvider ownerEmail="driver@example.com">
                <RecordingHarness />
            </LiveSessionProvider>,
        );
        fireEvent.click(screen.getByRole('button', { name: 'Start' }));
        await waitFor(() => expect(screen.getByTestId('harness-game')).toHaveTextContent('acc'));

        fireEvent.click(screen.getByRole('button', { name: 'Record' }));
        await waitFor(() => expect(window.electronAPI.startRecordingSession).toHaveBeenCalledWith({ game: 'acc' }));
        expect(screen.getByTestId('harness-active')).toHaveTextContent('true');

        fireEvent.click(screen.getByRole('button', { name: 'End' }));
        expect(window.electronAPI.stopRecordingSession).not.toHaveBeenCalled();

        resolveStart({
            ok: true,
            game: 'acc',
            filePath: telemetryPath,
            startedAt: 1,
        });

        await waitFor(() => expect(window.electronAPI.stopRecordingSession).toHaveBeenCalledTimes(1));
        await waitFor(() => expect(screen.getByTestId('harness-game')).toHaveTextContent('none'));
        expect(screen.getByTestId('harness-active')).toHaveTextContent('false');
    });
});
