import React from 'react';
import { render } from '@testing-library/react';
import {
    AiChatScreenProvider,
    AiChatScreenRegistration,
    useAiChatScreen,
} from 'contexts/AiChatScreenContext';

const mockUseUserSummary = jest.fn();
const mockUseAiLabels = jest.fn();

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Element = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
    return {
        Box: Element,
        Card: Element,
        Flex: Element,
        Heading: Element,
        Separator: () => React.createElement('hr'),
        Text: Element,
        TextArea: (props: Record<string, unknown>) => React.createElement('textarea', props),
    };
});
jest.mock('contexts/UserSummaryContext', () => ({
    useUserSummary: () => mockUseUserSummary(),
}));
jest.mock('contexts/AiLabelsContext', () => ({
    useAiLabels: () => mockUseAiLabels(),
}));
jest.mock('../AnalyzeAllSessionsControl', () => () => <button type="button">Analyze</button>);

import UserSummary from '../user-summary';

let observedScreen: AiChatScreenRegistration | null = null;
const RegistrationObserver = () => {
    observedScreen = useAiChatScreen().activeScreen;
    return null;
};

const summary = {
    sessionAnalysis: {
        practice: {
            tracks: {
                monza: {
                    trackName: 'Monza',
                    analyzedSessionCount: 3,
                    sections: {},
                },
            },
        },
    },
};

const Harness = () => (
    <AiChatScreenProvider>
        <UserSummary />
        <RegistrationObserver />
    </AiChatScreenProvider>
);

describe('User Summary screen registration', () => {
    beforeEach(() => {
        observedScreen = null;
        mockUseUserSummary.mockReturnValue({
            userSummary: summary,
            userSummaryLoading: false,
            userSummaryError: '',
            loadUserSummary: jest.fn(),
        });
        mockUseAiLabels.mockReturnValue({
            getLabelName: (id: string) => id,
            getCategoryLabels: () => [],
            loading: false,
            error: '',
        });
    });

    it('registers normalized summary context and summary query tools', () => {
        render(<Harness />);

        expect(observedScreen).toMatchObject({
            screenId: 'user-summary',
            assistantMode: 'user_summary',
            pillLabel: 'User Summary',
        });
        expect(observedScreen!.getPillInfo()).toMatchObject({
            status: { label: 'Ready', tone: 'success' },
            facts: expect.arrayContaining([{ label: 'Tracks', value: '1' }]),
        });
        expect(observedScreen!.componentRef.current!.getAiContext()).toMatchObject({
            summary_state: 'ready',
            track_count: 1,
            normalized_summary: [expect.objectContaining({ id: 'monza', name: 'Monza' })],
        });
        expect(observedScreen!.componentRef.current!.getToolHandlers()).toEqual(expect.objectContaining({
            get_user_summary_map_level: expect.any(Function),
            get_available_user_summary_maps: expect.any(Function),
            search_user_summary_map_level: expect.any(Function),
        }));
    });

    it('publishes loading state changes through the existing handle', () => {
        const view = render(<Harness />);
        const componentRef = observedScreen!.componentRef;
        mockUseUserSummary.mockReturnValue({
            userSummary: summary,
            userSummaryLoading: true,
            userSummaryError: '',
            loadUserSummary: jest.fn(),
        });

        view.rerender(<Harness />);

        expect(observedScreen!.componentRef).toBe(componentRef);
        expect(componentRef.current!.getAiContext()).toMatchObject({
            summary_state: 'loading',
            loading: true,
        });
        expect(observedScreen!.getPillInfo().status).toEqual({ label: 'Loading', tone: 'info' });
    });
});
