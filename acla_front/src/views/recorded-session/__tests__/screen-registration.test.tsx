import React, { useContext } from 'react';
import { act, render, waitFor } from '@testing-library/react';
import apiService from 'services/api.service';
import {
    OPERATION_COMPONENT_NAMES,
    OperationComponentRefDirectory,
    OperationComponentRefProvider,
    useOperationComponentRefDirectory,
} from 'contexts/OperationComponentRefContext';
import { AnalysisContext, AnalysisContextType } from '../analysis-context';
import { RecordedSessionData, RecordedSessionDataProvider, useRecordedSessionData } from '../data/RecordedSessionDataProvider';
import {
    createEmptyRecordedPlaybackSummary,
    createIdleRecordedAiAnalysis,
} from '../recorded-session-analysis';
import type { SessionAnalysisHandle } from '../session-analysis';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Element = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
    return { Box: Element, Text: Element, Tabs: { Root: Element, List: Element, Trigger: Element, Content: Element } };
});
jest.mock('../map-list/map-list', () => () => <div>Maps</div>);
jest.mock('../session-list/session-list', () => () => <div>Sessions</div>);
jest.mock('../sessionAnalysis/session-analysis-split', () => () => <div>Recorded workspace</div>);
jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: jest.fn() },
}));

import { SessionAnalysisContent, SessionAnalysisProvider } from '../session-analysis';
import {
    ExpertLineGuidanceFailedError,
    LapComparisonFailedError,
    PerformanceInsightsFailedError,
    RecordedAnalysisFailedError,
    SessionAnalysisFailedError,
    TelemetryDataFailedError,
} from 'contexts/OperationComponentError';

const mockPost = apiService.post as jest.Mock;

let directory: OperationComponentRefDirectory | null = null;
const DirectoryObserver = () => {
    directory = useOperationComponentRefDirectory();
    return null;
};

const createAnalysisContext = (overrides: Partial<AnalysisContextType> = {}): AnalysisContextType => ({
    activeTab: 'mapLists',
    mapSelected: null,
    sessionSelected: null,
    activeVisualizations: [],
    latestGuidanceMessage: null,
    recordedAiAnalysis: createIdleRecordedAiAnalysis(),
    recordedPlaybackSummary: createEmptyRecordedPlaybackSummary(),
    setMap: jest.fn(),
    setSession: jest.fn(),
    setRecordedPlaybackSummary: jest.fn(),
    runRecordedAiAnalysis: jest.fn(),
    setActiveTab: jest.fn(),
    setActiveVisualizations: jest.fn(),
    sendGuidanceToChat: jest.fn(),
    ...overrides,
});

const Harness = ({ value }: { value: AnalysisContextType }) => (
    <OperationComponentRefProvider>
        <AnalysisContext.Provider value={value}>
            <SessionAnalysisContent name={OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS} />
        </AnalysisContext.Provider>
        <DirectoryObserver />
    </OperationComponentRefProvider>
);

describe('SessionAnalysis named component handle', () => {
    beforeEach(() => {
        directory = null;
        mockPost.mockReset();
    });

    it('keeps its exact name and exposes fresh recorded-session operations', () => {
        const view = render(<Harness value={createAnalysisContext({ activeTab: 'sessionLists', mapSelected: 'Monza' })} />);
        const ref = directory!.findComponentRef<SessionAnalysisHandle>(OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS)!;
        expect(ref.current!.getComponentName()).toBe(OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS);
        expect(ref.current!.getMapSelected()).toBe('Monza');
        expect(ref.current!.getSelectedSession()).toBeNull();
        expect(ref.current!.getAssistantSnapshot().mapSelected).toBe('Monza');
        const onSnapshotChange = jest.fn();
        const unsubscribe = ref.current!.subscribeAssistantSnapshot(onSnapshotChange);

        view.rerender(<Harness value={createAnalysisContext({
            activeTab: 'session',
            mapSelected: 'Monza',
            sessionSelected: { SessionId: 'session-17', session_name: 'Sunday Race', map: 'Monza', car: 'BMW M4 GT3' } as any,
            recordedPlaybackSummary: {
                sessionId: 'session-17', sampleCount: 800, durationSeconds: 92,
                playbackIndex: 80, playbackTimeSeconds: 12.5, activeSegment: null,
            },
        })} />);

        expect(directory!.findComponentRef(OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS)).toBe(ref);
        expect(ref.current!.getSelectedSession()).toMatchObject({
            SessionId: 'session-17',
            session_name: 'Sunday Race',
            map: 'Monza',
            car: 'BMW M4 GT3',
        });
        expect(ref.current!.getRecordedPlaybackSummary()).toMatchObject({
            sampleCount: 800,
            playbackTimeSeconds: 12.5,
        });
        expect(ref.current!.getAssistantSnapshot().sessionSelected?.SessionId).toBe('session-17');
        expect(onSnapshotChange).toHaveBeenCalled();
        unsubscribe();
    });

    it('exposes the current shared table through its named handle', () => {
        const rows = [{ Physics_speed_kmh: '180.00', original: null }];
        const selected = {
            SessionId: 'local-ibt:1', storage: 'local' as const, session_name: 'Spa',
            map: 'Spa', car: 'GT3', user_id: '', points: [], data: rows,
        };
        const view = render(<OperationComponentRefProvider>
            <RecordedSessionDataProvider session={selected} map="Spa">
                <SessionAnalysisContent name={OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS} />
            </RecordedSessionDataProvider>
            <DirectoryObserver />
        </OperationComponentRefProvider>);
        const handle = directory!.findComponentRef<SessionAnalysisHandle>(OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS)!.current!;
        expect(handle.getRecordedSessionData()).toMatchObject({ sessionId: selected.SessionId, status: 'ready' });
        expect(handle.getRecordedSessionData().table).toBe(rows);
        view.rerender(<OperationComponentRefProvider>
            <RecordedSessionDataProvider session={null} map={null}>
                <SessionAnalysisContent name={OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS} />
            </RecordedSessionDataProvider>
            <DirectoryObserver />
        </OperationComponentRefProvider>);
        expect(handle.getRecordedSessionData()).toMatchObject({ status: 'idle', table: [] });
    });

    it.each([
        ['requestSessionAnalysis', ['session-17'], SessionAnalysisFailedError],
        ['requestPerformanceInsights', ['session-17'], PerformanceInsightsFailedError],
        ['requestLapComparison', [['session-17', 'session-18']], LapComparisonFailedError],
        ['requestExpertLineGuidance', ['session-17'], ExpertLineGuidanceFailedError],
        ['requestTelemetryData', ['session-17'], TelemetryDataFailedError],
    ] as const)('wraps %s transport failures with its concrete exception', async (method, args, ErrorType) => {
        render(<Harness value={createAnalysisContext()} />);
        const handle = directory!.findComponentRef<SessionAnalysisHandle>(
            OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
        )!.current!;
        const cause = { response: { data: { message: 'Transport unavailable.' } } };
        mockPost.mockRejectedValueOnce(cause);

        let thrown: unknown;
        try {
            await (handle[method] as (...values: any[]) => Promise<any>)(...args);
        } catch (error) {
            thrown = error;
        }

        expect(thrown).toBeInstanceOf(ErrorType);
        expect(thrown).toMatchObject({
            name: ErrorType.name,
            componentName: OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
            message: 'Transport unavailable.',
            cause,
        });
    });
});

describe('SessionAnalysisProvider shared telemetry analysis', () => {
    let providerContext: AnalysisContextType;
    let recordedData: RecordedSessionData;
    const ContextObserver = () => {
        providerContext = useContext(AnalysisContext);
        recordedData = useRecordedSessionData();
        return null;
    };
    const rows = [{ Physics_speed_kmh: '180.00', original: null }, {}, { Physics_speed_kmh: 190 }];
    const selectedSession = (storage: 'local' | 'cloud') => ({
        SessionId: storage === 'local' ? 'local-ibt:1' : 'session-17',
        session_name: 'Sunday Race', storage, map: 'Spa', car: 'GT3', user_id: '', points: [],
        data: storage === 'local' ? rows : [],
    });
    const analysisResponse = {
        data: {
            status: 'success', session_id: 'live-baseline', samples_analyzed: rows.length,
            segments: [{ start_index: 1, end_index: 3, labels: ['EA'], expert_reference_data: [] }],
        },
    };

    beforeEach(() => {
        mockPost.mockReset();
    });

    it.each(['local', 'cloud'] as const)('analyzes the unchanged shared %s table and retains its session identity', async (storage) => {
        const selected = selectedSession(storage);
        mockPost.mockImplementation(async (url) => {
            if (url === '/racing-session/download/init') {
                return { data: { downloadId: 'download-1', sessionMetadata: [{ sessionId: selected.SessionId, chunkCount: 1 }] } };
            }
            if (url === '/racing-session/download/chunk') return { data: rows };
            if (url === '/racing-session/analyze-live-recorded-analysis') return analysisResponse;
            throw new Error(`Unexpected request: ${url}`);
        });
        render(<SessionAnalysisProvider><ContextObserver /></SessionAnalysisProvider>);
        act(() => providerContext.setSession(selected));
        await waitFor(() => expect(recordedData.status).toBe('ready'));
        const table = recordedData.table;
        mockPost.mockClear();

        await act(async () => { await providerContext.runRecordedAiAnalysis(); });

        expect(mockPost).toHaveBeenCalledTimes(1);
        expect(mockPost).toHaveBeenCalledWith('/racing-session/analyze-live-recorded-analysis', {
            track: 'Spa', car: 'GT3', records: rows,
        }, { timeout: 120000, signal: expect.any(AbortSignal) });
        expect(mockPost.mock.calls[0][1].records).toEqual(table);
        expect(mockPost.mock.calls[0][1].records[0]).toBe(table[0]);
        expect(recordedData.table).toBe(table);
        expect(providerContext.recordedAiAnalysis).toMatchObject({
            sessionId: selected.SessionId, status: 'ready',
            result: { session_id: selected.SessionId, samples_analyzed: rows.length, segments: [{ start_index: 1, end_index: 3 }] },
        });

        await act(async () => { await providerContext.runRecordedAiAnalysis(); });
        expect(mockPost).toHaveBeenCalledTimes(1);
        await act(async () => { await providerContext.runRecordedAiAnalysis({ force: true }); });
        expect(mockPost).toHaveBeenCalledTimes(2);
    });

    it('rejects analysis while the shared table is loading, then allows it when ready', async () => {
        let finishDownload!: (response: unknown) => void;
        mockPost.mockResolvedValueOnce({
            data: { downloadId: 'download-1', sessionMetadata: [{ sessionId: 'session-17', chunkCount: 1 }] },
        }).mockImplementationOnce(() => new Promise((resolve) => { finishDownload = resolve; }));
        render(<SessionAnalysisProvider><ContextObserver /></SessionAnalysisProvider>);
        act(() => providerContext.setSession(selectedSession('cloud')));
        await waitFor(() => expect(mockPost).toHaveBeenCalledTimes(2));
        expect(recordedData.status).toBe('loading');
        await act(async () => {
            await expect(providerContext.runRecordedAiAnalysis()).rejects.toThrow(/loading/i);
        });
        expect(mockPost).toHaveBeenCalledTimes(2);
        await act(async () => { finishDownload({ data: rows }); });
        mockPost.mockResolvedValueOnce(analysisResponse);
        await act(async () => { await providerContext.runRecordedAiAnalysis(); });
        expect(providerContext.recordedAiAnalysis.status).toBe('ready');
    });

    it('rejects an empty shared table without calling the classifier', async () => {
        render(<SessionAnalysisProvider><ContextObserver /></SessionAnalysisProvider>);
        act(() => providerContext.setSession({ ...selectedSession('local'), data: [] }));
        await act(async () => {
            await expect(providerContext.runRecordedAiAnalysis()).rejects.toThrow('No telemetry samples');
        });
        expect(mockPost).not.toHaveBeenCalled();
        expect(providerContext.recordedAiAnalysis.status).toBe('error');
    });

    it('publishes the recorded error state before rejecting', async () => {
        render(
            <SessionAnalysisProvider>
                <ContextObserver />
            </SessionAnalysisProvider>,
        );
        act(() => providerContext.setSession(selectedSession('local')));
        const cause = new Error('Classifier unavailable.');
        mockPost.mockRejectedValueOnce(cause);

        let thrown: unknown;
        await act(async () => {
            try {
                await providerContext.runRecordedAiAnalysis();
            } catch (error) {
                thrown = error;
            }
        });

        expect(thrown).toBeInstanceOf(RecordedAnalysisFailedError);
        expect(thrown).toMatchObject({
            name: 'RecordedAnalysisFailedError',
            componentName: OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
            message: 'Classifier unavailable.',
            cause,
        });
        expect(providerContext.recordedAiAnalysis).toMatchObject({
            sessionId: 'local-ibt:1',
            status: 'error',
            message: 'Classifier unavailable.',
        });
    });

    it('shares an in-flight analysis between callers', async () => {
        let finish!: (response: unknown) => void;
        mockPost.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
        render(<SessionAnalysisProvider><ContextObserver /></SessionAnalysisProvider>);
        act(() => providerContext.setSession(selectedSession('local')));
        let first!: Promise<unknown>;
        let second!: Promise<unknown>;
        await act(async () => {
            first = providerContext.runRecordedAiAnalysis();
            second = providerContext.runRecordedAiAnalysis();
        });
        expect(mockPost).toHaveBeenCalledTimes(1);
        expect(providerContext.recordedAiAnalysis.message).toContain('0 / 3');
        await act(async () => {
            finish(analysisResponse);
            const [firstResult, secondResult] = await Promise.all([first, second]);
            expect(firstResult).toBe(secondResult);
        });
        expect(providerContext.recordedAiAnalysis.status).toBe('ready');
    });

    it('aborts old requests on selection changes and ignores their late results', async () => {
        let finish!: (response: unknown) => void;
        mockPost.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
        render(<SessionAnalysisProvider><ContextObserver /></SessionAnalysisProvider>);
        act(() => providerContext.setSession(selectedSession('local')));
        let pending!: Promise<unknown>;
        await act(async () => {
            pending = providerContext.runRecordedAiAnalysis().catch((error) => error);
        });
        const signal = mockPost.mock.calls[0][2].signal as AbortSignal;
        act(() => providerContext.setSession({ ...selectedSession('local'), SessionId: 'local-ibt:2' }));
        expect(signal.aborted).toBe(true);
        await act(async () => {
            finish(analysisResponse);
            expect(await pending).toBeInstanceOf(RecordedAnalysisFailedError);
        });
        expect(providerContext.recordedAiAnalysis).toMatchObject({ sessionId: 'local-ibt:2', status: 'idle', result: null });
        act(() => providerContext.setSession(selectedSession('local')));
        expect(providerContext.recordedAiAnalysis).toMatchObject({ status: 'idle', result: null });
    });

    it('a forced run replaces the previous request without stale error or result updates', async () => {
        let finishOld!: (response: unknown) => void;
        mockPost.mockImplementationOnce(() => new Promise((resolve) => { finishOld = resolve; }))
            .mockResolvedValueOnce(analysisResponse);
        render(<SessionAnalysisProvider><ContextObserver /></SessionAnalysisProvider>);
        act(() => providerContext.setSession(selectedSession('local')));
        let pending!: Promise<unknown>;
        await act(async () => {
            pending = providerContext.runRecordedAiAnalysis().catch((error) => error);
        });
        const signal = mockPost.mock.calls[0][2].signal as AbortSignal;
        await act(async () => { await providerContext.runRecordedAiAnalysis({ force: true }); });
        expect(signal.aborted).toBe(true);
        const completed = providerContext.recordedAiAnalysis;
        await act(async () => {
            finishOld(analysisResponse);
            await pending;
        });
        expect(providerContext.recordedAiAnalysis).toBe(completed);
        expect(completed.status).toBe('ready');
    });

    it('aborts the active request when the provider unmounts', async () => {
        let finish!: (response: unknown) => void;
        mockPost.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
        const view = render(<SessionAnalysisProvider><ContextObserver /></SessionAnalysisProvider>);
        act(() => providerContext.setSession(selectedSession('local')));
        let pending!: Promise<unknown>;
        await act(async () => {
            pending = providerContext.runRecordedAiAnalysis().catch((error) => error);
        });
        const signal = mockPost.mock.calls[0][2].signal as AbortSignal;
        view.unmount();
        expect(signal.aborted).toBe(true);
        await act(async () => {
            finish(analysisResponse);
            expect(await pending).toBeInstanceOf(RecordedAnalysisFailedError);
        });
    });

});
