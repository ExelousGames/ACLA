import React from 'react';
import { render, screen } from '@testing-library/react';
import SideMainMenu from '../side-main-menu';

const mockUseEnvironment = jest.fn(() => 'web');

jest.mock('contexts/EnvironmentContext', () => ({
    useEnvironment: () => mockUseEnvironment(),
}));

jest.mock('@radix-ui/themes', () => ({
    Box: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Text: ({ children, ...props }: any) => <span {...props}>{children}</span>,
}));

jest.mock('radix-ui', () => ({
    ScrollArea: {
        Root: ({ children, ...props }: any) => <div {...props}>{children}</div>,
        Viewport: ({ children, ...props }: any) => <div {...props}>{children}</div>,
        Scrollbar: ({ children, ...props }: any) => <div {...props}>{children}</div>,
        Thumb: (props: any) => <div {...props} />,
        Corner: (props: any) => <div {...props} />,
    },
    Tabs: {
        Root: ({ children, ...props }: any) => <div {...props}>{children}</div>,
        List: ({ children, ...props }: any) => <div role="tablist" {...props}>{children}</div>,
        Trigger: ({ children, ...props }: any) => <button role="tab" {...props}>{children}</button>,
        Content: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    },
}));

jest.mock('views/lap-analysis/session-analysis', () => () => <div>Analysis Content</div>);
jest.mock('views/user-summary/user-summary', () => () => <div>User Summary Content</div>);
jest.mock('views/circuit-maps/circuit-maps', () => () => <div>Circuit Maps Content</div>);
jest.mock('views/live-session/LiveSessionView', () => () => <div>Live Session Content</div>);
jest.mock('components/ProtectedComponent', () => ({ children }: any) => <>{children}</>);

describe('SideMainMenu', () => {
    beforeEach(() => mockUseEnvironment.mockReturnValue('web'));

    it('renders Circuit Maps as the third main tab', () => {
        render(<SideMainMenu />);

        const tabs = screen.getAllByRole('tab');
        expect(tabs.map((tab) => tab.textContent)).toEqual([
            'Analysis',
            'User Summary',
            'Circuit Maps'
        ]);
    });

    it('renders Live Session first only in Electron', () => {
        mockUseEnvironment.mockReturnValue('electron');
        render(<SideMainMenu />);

        expect(screen.getAllByRole('tab').map((tab) => tab.textContent)).toEqual([
            'Live Session',
            'Analysis',
            'User Summary',
            'Circuit Maps',
        ]);
    });
});
