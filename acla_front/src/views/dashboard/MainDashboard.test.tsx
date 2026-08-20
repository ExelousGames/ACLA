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
}));
jest.mock('./DashboardAssistant', () => ({ activeDashboardTab }: { activeDashboardTab: string }) => (
        <aside aria-label="AI Assistant" data-active-tab={activeDashboardTab}>Assistant</aside>
));
jest.mock('views/live-session/LiveSessionContext', () => ({
    LiveSessionProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

import MainDashboard from './MainDashboard';

describe('MainDashboard desktop composition', () => {
    beforeEach(() => mockUseEnvironment.mockReturnValue('electron'));

    it('selects Live Session by default and keeps one dashboard-level assistant', () => {
        render(<MainDashboard onTaskCreated={jest.fn()} />);

        expect(screen.getByTestId('active-dashboard-tab')).toHaveTextContent('liveSession');
        expect(screen.getAllByLabelText('AI Assistant')).toHaveLength(1);
        expect(screen.getByLabelText('AI Assistant')).toHaveAttribute('data-active-tab', 'liveSession');
    });

    it('tracks the active named owner without remounting the dashboard assistant', () => {
        render(<MainDashboard onTaskCreated={jest.fn()} />);
        const assistant = screen.getByLabelText('AI Assistant');

        fireEvent.click(screen.getByRole('button', { name: 'Analysis' }));
        expect(screen.getByLabelText('AI Assistant')).toBe(assistant);
        expect(assistant).toHaveAttribute('data-active-tab', 'analysis');

        fireEvent.click(screen.getByRole('button', { name: 'User Summary' }));
        expect(screen.getByLabelText('AI Assistant')).toBe(assistant);
        expect(assistant).toHaveAttribute('data-active-tab', 'userSummary');

        fireEvent.click(screen.getByRole('button', { name: 'Live Session' }));
        expect(screen.getByLabelText('AI Assistant')).toBe(assistant);
        expect(assistant).toHaveAttribute('data-active-tab', 'liveSession');
    });
});
