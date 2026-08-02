import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';

const mockUseEnvironment = jest.fn(() => 'electron');

jest.mock('contexts/EnvironmentContext', () => ({ useEnvironment: () => mockUseEnvironment() }));
jest.mock('views/header-menu/header-menu', () => () => <header>Header</header>);
jest.mock('views/side-main-menu/side-main-menu', () => ({
    activeTab,
    onTabChange,
}: {
    activeTab: string;
    onTabChange: (tab: string) => void;
}) => (
    <main>
        <span data-testid="active-dashboard-tab">{activeTab}</span>
        <button type="button" onClick={() => onTabChange('analysis')}>Analysis</button>
        <button type="button" onClick={() => onTabChange('userSummary')}>User Summary</button>
        <button type="button" onClick={() => onTabChange('liveSession')}>Live Session</button>
    </main>
));
jest.mock('views/lap-analysis/session-analysis', () => ({
    SessionAnalysisProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    SessionAnalysisAssistant: ({ assistantModeOverride }: { assistantModeOverride?: string }) => (
        <aside aria-label="AI Assistant" data-mode={assistantModeOverride}>Assistant</aside>
    ),
}));
jest.mock('views/live-session/LiveSessionContext', () => ({
    LiveSessionProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
jest.mock('views/lap-analysis/LiveSessionDetectionManager', () => () => null);
jest.mock('views/lap-analysis/liveAnalysisSessionRecording', () => ({ recorderHostId }: { recorderHostId?: string }) => (
    <div data-testid="recorder-runtime" data-host={recorderHostId} />
));
jest.mock('views/live-session/LiveSessionView', () => ({
    LIVE_SESSION_RECORDER_HOST_ID: 'live-session-recorder-host',
}));

import MainDashboard from './MainDashboard';

describe('MainDashboard desktop composition', () => {
    beforeEach(() => mockUseEnvironment.mockReturnValue('electron'));

    it('selects Live Session by default and keeps one dashboard-level assistant and recorder runtime', () => {
        render(<MainDashboard onTaskCreated={jest.fn()} />);

        expect(screen.getByTestId('active-dashboard-tab')).toHaveTextContent('liveSession');
        expect(screen.getAllByLabelText('AI Assistant')).toHaveLength(1);
        expect(screen.getByLabelText('AI Assistant')).toHaveAttribute('data-mode', 'live');
        expect(screen.getByTestId('recorder-runtime')).toHaveAttribute('data-host', 'live-session-recorder-host');
    });

    it('maps Analysis, User Summary, and Live Session tabs to assistant overrides', () => {
        render(<MainDashboard onTaskCreated={jest.fn()} />);

        fireEvent.click(screen.getByRole('button', { name: 'Analysis' }));
        expect(screen.getByLabelText('AI Assistant')).not.toHaveAttribute('data-mode');

        fireEvent.click(screen.getByRole('button', { name: 'User Summary' }));
        expect(screen.getByLabelText('AI Assistant')).toHaveAttribute('data-mode', 'user_summary');

        fireEvent.click(screen.getByRole('button', { name: 'Live Session' }));
        expect(screen.getByLabelText('AI Assistant')).toHaveAttribute('data-mode', 'live');
    });
});
