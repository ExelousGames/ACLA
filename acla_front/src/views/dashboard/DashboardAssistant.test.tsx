import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefProvider,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import type { AnalysisContextType } from 'views/lap-analysis/analysis-context';
import { DASHBOARD_TABS } from './dashboard-navigation';
import DashboardAssistant from './DashboardAssistant';

jest.mock('views/lap-analysis/ai-chat/ai-chat', () => (props: Record<string, unknown>) => (
    <div
        data-testid="dashboard-ai-chat"
        data-component-name={props.name}
        data-screen-component-name={(props.activeScreen as any)?.componentName}
        data-session-id={(props.activeScreen as any)?.recordedSessionId}
        data-session-mode={(props.activeScreen as any)?.assistantMode}
        data-screen-label={(props.activeScreen as any)?.label}
    />
));

const RecordedScreenReference = ({ snapshot }: { snapshot: AnalysisContextType }) => {
    const snapshotRef = React.useRef(snapshot);
    snapshotRef.current = snapshot;
    const listenersRef = React.useRef(new Set<() => void>());
    const componentRef = React.useRef<any>(null);
    if (componentRef.current === null) {
        componentRef.current = {
            getComponentName: () => AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
            getAssistantSnapshot: () => snapshotRef.current,
            subscribeAssistantSnapshot: (listener: () => void) => {
                listenersRef.current.add(listener);
                return () => listenersRef.current.delete(listener);
            },
        };
    }
    useRegisterAiToolComponentRef(componentRef);
    return null;
};

describe('DashboardAssistant', () => {
    it.each([
        [DASHBOARD_TABS.LIVE_SESSION, 'live', 'live-session', 'Live Session'],
        [DASHBOARD_TABS.USER_SUMMARY, 'user_summary', 'user-summary', 'User Summary'],
        [DASHBOARD_TABS.CIRCUIT_MAPS, 'front_desk', null, 'Front Desk'],
    ])('tracks the %s dashboard screen', (activeDashboardTab, mode, componentName, label) => {
        render(<DashboardAssistant activeDashboardTab={activeDashboardTab} />);

        const chat = screen.getByTestId('dashboard-ai-chat');
        expect(chat).toHaveAttribute('data-session-mode', mode);
        if (componentName === null) {
            expect(chat).not.toHaveAttribute('data-screen-component-name');
        } else {
            expect(chat).toHaveAttribute('data-screen-component-name', componentName);
        }
        expect(chat).toHaveAttribute('data-screen-label', label);
    });

    it('tracks the recorded screen within the Analysis dashboard tab', () => {
        render(
            <AiToolComponentRefProvider>
                <RecordedScreenReference snapshot={{
                    activeTab: 'session',
                    sessionSelected: {
                        SessionId: 'session-12',
                        session_name: 'Race 12',
                    },
                } as any} />
                <DashboardAssistant activeDashboardTab={DASHBOARD_TABS.ANALYSIS} />
            </AiToolComponentRefProvider>,
        );

        expect(screen.getByTestId('dashboard-ai-chat')).toHaveAttribute('data-session-mode', 'recorded');
        expect(screen.getByTestId('dashboard-ai-chat')).toHaveAttribute('data-screen-component-name', 'session-analysis');
        expect(screen.getByTestId('dashboard-ai-chat')).toHaveAttribute('data-session-id', 'session-12');
        expect(screen.getByTestId('dashboard-ai-chat')).toHaveAttribute('data-screen-label', 'Race 12');
    });

    it('keeps fold state in the dashboard component while the active screen changes', () => {
        const view = render(<DashboardAssistant activeDashboardTab={DASHBOARD_TABS.LIVE_SESSION} />);
        const chat = screen.getByTestId('dashboard-ai-chat');

        fireEvent.click(screen.getByRole('button', { name: 'Open AI Assistant' }));
        expect(screen.getByLabelText('AI Assistant')).toHaveClass('main-dashboard-assistant--open');

        view.rerender(<DashboardAssistant activeDashboardTab={DASHBOARD_TABS.USER_SUMMARY} />);

        expect(screen.getByLabelText('AI Assistant')).toHaveClass('main-dashboard-assistant--open');
        expect(screen.getByTestId('dashboard-ai-chat')).toBe(chat);
        expect(screen.getByTestId('dashboard-ai-chat')).toHaveAttribute('data-session-mode', 'user_summary');

        fireEvent.click(screen.getByRole('button', { name: 'Fold AI Assistant' }));
        expect(screen.getByTestId('dashboard-ai-chat')).toBe(chat);
    });
});
