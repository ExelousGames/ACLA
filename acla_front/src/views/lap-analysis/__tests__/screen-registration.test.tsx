import React from 'react';
import { render } from '@testing-library/react';
import {
    AiChatScreenProvider,
    AiChatScreenRegistration,
    useAiChatScreen,
} from 'contexts/AiChatScreenContext';
import { AnalysisContext, AnalysisContextType } from '../analysis-context';
import {
    createEmptyRecordedPlaybackSummary,
    createIdleRecordedAiAnalysis,
} from '../recorded-session-analysis';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Element = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
    return {
        Box: Element,
        Tabs: {
            Root: Element,
            List: Element,
            Trigger: Element,
            Content: Element,
        },
    };
});
jest.mock('../map-list/map-list', () => () => <div>Maps</div>);
jest.mock('../session-list/session-list', () => () => <div>Sessions</div>);
jest.mock('../sessionAnalysis/session-analysis-split', () => () => <div>Recorded workspace</div>);

import SessionAnalysis from '../session-analysis';

let observedScreen: AiChatScreenRegistration | null = null;
const RegistrationObserver = () => {
    observedScreen = useAiChatScreen().activeScreen;
    return null;
};

const createAnalysisContext = (
    overrides: Partial<AnalysisContextType> = {},
): AnalysisContextType => ({
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
    <AiChatScreenProvider>
        <AnalysisContext.Provider value={value}>
            <SessionAnalysis />
        </AnalysisContext.Provider>
        <RegistrationObserver />
    </AiChatScreenProvider>
);

describe('analysis screen registrations', () => {
    beforeEach(() => {
        observedScreen = null;
    });

    it('switches a stable handle from Front Desk to the selected recorded session', () => {
        const frontDeskContext = createAnalysisContext({
            activeTab: 'sessionLists',
            mapSelected: 'Monza',
        });
        const view = render(<Harness value={frontDeskContext} />);
        const componentRef = observedScreen!.componentRef;

        expect(observedScreen).toMatchObject({
            screenId: 'front-desk',
            assistantMode: 'front_desk',
            pillLabel: 'Front Desk',
        });
        expect(componentRef.current!.getAiContext()).toMatchObject({
            active_analysis_area: 'sessionLists',
            selected_map_id: 'Monza',
        });
        expect(componentRef.current!.getToolHandlers()).toEqual({});

        const recordedContext = createAnalysisContext({
            activeTab: 'session',
            mapSelected: 'Monza',
            sessionSelected: {
                SessionId: 'session-17',
                session_name: 'Sunday Race',
                map: 'Monza',
                car: 'BMW M4 GT3',
            } as any,
            recordedPlaybackSummary: {
                sessionId: 'session-17',
                sampleCount: 800,
                durationSeconds: 92,
                playbackIndex: 80,
                playbackTimeSeconds: 12.5,
                activeSegment: null,
            },
        });
        view.rerender(<Harness value={recordedContext} />);

        expect(observedScreen).toMatchObject({
            screenId: 'recorded-session',
            assistantMode: 'recorded',
            pillLabel: 'Sunday Race',
            recordedSessionId: 'session-17',
        });
        expect(observedScreen!.componentRef).toBe(componentRef);
        expect(componentRef.current!.getAiContext()).toMatchObject({
            selected_session: {
                id: 'session-17',
                name: 'Sunday Race',
                map: 'Monza',
                car: 'BMW M4 GT3',
            },
            recorded_session: {
                playback: {
                    sampleCount: 800,
                    playbackTimeSeconds: 12.5,
                },
            },
        });
        expect(componentRef.current!.getToolHandlers()).toEqual(expect.objectContaining({
            run_recorded_ai_analysis: expect.any(Function),
            get_recorded_session_context: expect.any(Function),
            invoke_visualization_control: expect.any(Function),
        }));
    });
});
