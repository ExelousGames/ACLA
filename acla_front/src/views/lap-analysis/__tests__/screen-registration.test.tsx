import React, { useContext } from 'react';
import { act, render } from '@testing-library/react';
import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    AiToolComponentRefProvider,
    useAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import { AnalysisContext, AnalysisContextType } from '../analysis-context';
import {
    createEmptyRecordedPlaybackSummary,
    createIdleRecordedAiAnalysis,
} from '../recorded-session-analysis';
import type { SessionAnalysisHandle } from '../session-analysis';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Element = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
    return { Box: Element, Tabs: { Root: Element, List: Element, Trigger: Element, Content: Element } };
});
jest.mock('../map-list/map-list', () => () => <div>Maps</div>);
jest.mock('../session-list/session-list', () => () => <div>Sessions</div>);
jest.mock('../sessionAnalysis/session-analysis-split', () => () => <div>Recorded workspace</div>);
jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: jest.fn() },
}));

import SessionAnalysis, { SessionAnalysisProvider } from '../session-analysis';
import {
    ExpertLineGuidanceFailedError,
    LapComparisonFailedError,
    PerformanceInsightsFailedError,
    RecordedAnalysisFailedError,
    SessionAnalysisFailedError,
    TelemetryDataFailedError,
} from 'contexts/AiToolComponentError';

const mockPost = apiService.post as jest.Mock;

let directory: AiToolComponentRefDirectory | null = null;
const DirectoryObserver = () => {
    directory = useAiToolComponentRefDirectory();
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
    <AiToolComponentRefProvider>
        <AnalysisContext.Provider value={value}>
            <SessionAnalysis name={AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS} />
        </AnalysisContext.Provider>
        <DirectoryObserver />
    </AiToolComponentRefProvider>
);

describe('SessionAnalysis named component handle', () => {
    beforeEach(() => {
        directory = null;
        mockPost.mockReset();
    });

    it('keeps its exact name and exposes fresh recorded-session operations', () => {
        const view = render(<Harness value={createAnalysisContext({ activeTab: 'sessionLists', mapSelected: 'Monza' })} />);
        const ref = directory!.findComponentRef<SessionAnalysisHandle>(AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS)!;
        expect(ref.current!.getComponentName()).toBe(AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS);
        expect(ref.current!.getMapSelected()).toBe('Monza');
        expect(ref.current!.getSelectedSession()).toBeNull();

        view.rerender(<Harness value={createAnalysisContext({
            activeTab: 'session',
            mapSelected: 'Monza',
            sessionSelected: { SessionId: 'session-17', session_name: 'Sunday Race', map: 'Monza', car: 'BMW M4 GT3' } as any,
            recordedPlaybackSummary: {
                sessionId: 'session-17', sampleCount: 800, durationSeconds: 92,
                playbackIndex: 80, playbackTimeSeconds: 12.5, activeSegment: null,
            },
        })} />);

        expect(directory!.findComponentRef(AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS)).toBe(ref);
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
            AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
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
            componentName: AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
            message: 'Transport unavailable.',
            cause,
        });
    });
});

describe('SessionAnalysisProvider recorded analysis failures', () => {
    let providerContext: AnalysisContextType;
    const ContextObserver = () => {
        providerContext = useContext(AnalysisContext);
        return null;
    };

    beforeEach(() => {
        mockPost.mockReset();
    });

    it('publishes the recorded error state before rejecting', async () => {
        render(
            <SessionAnalysisProvider>
                <ContextObserver />
            </SessionAnalysisProvider>,
        );
        act(() => providerContext.setSession({
            SessionId: 'session-17',
            session_name: 'Sunday Race',
        } as any));
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
            componentName: AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
            message: 'Classifier unavailable.',
            cause,
        });
        expect(providerContext.recordedAiAnalysis).toMatchObject({
            sessionId: 'session-17',
            status: 'error',
            message: 'Classifier unavailable.',
        });
    });
});
