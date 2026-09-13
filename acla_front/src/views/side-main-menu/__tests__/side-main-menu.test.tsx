import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
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
        Root: ({ children, asChild: _asChild, ...props }: any) => <div {...props}>{children}</div>,
        Viewport: ({ children, ...props }: any) => <div {...props}>{children}</div>,
        Scrollbar: ({ children, ...props }: any) => <div {...props}>{children}</div>,
        Thumb: (props: any) => <div {...props} />,
        Corner: (props: any) => <div {...props} />,
    },
    Tabs: {
        Root: ({ children, onValueChange: _onValueChange, ...props }: any) => <div {...props}>{children}</div>,
        List: ({ children, ...props }: any) => <div role="tablist" {...props}>{children}</div>,
        Trigger: ({ children, ...props }: any) => <button role="tab" {...props}>{children}</button>,
        Content: ({ children, forceMount: _forceMount, ...props }: any) => <div {...props}>{children}</div>,
    },
    Tooltip: {
        Provider: ({ children }: any) => <>{children}</>,
        Root: ({ children }: any) => <>{children}</>,
        Trigger: ({ children }: any) => <>{children}</>,
        Portal: ({ children }: any) => <>{children}</>,
        Content: ({ children, side: _side, sideOffset: _sideOffset, ...props }: any) => (
            <div role="tooltip" {...props}>{children}</div>
        ),
        Arrow: (props: any) => <span {...props} />,
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

    it('renders a distinct icon and tooltip for every menu option', () => {
        render(<SideMainMenu />);

        const tabs = screen.getAllByRole('tab');
        expect(tabs.every((tab) => Boolean(tab.querySelector('svg')))).toBe(true);
        expect(screen.getAllByRole('tooltip').map((tooltip) => tooltip.textContent)).toEqual(
            expect.arrayContaining(['Analysis', 'User Summary', 'Circuit Maps']),
        );
    });

    it('collapses and expands the menu from the toggle', () => {
        const { container } = render(<SideMainMenu />);
        const root = container.querySelector('.TabsRoot');
        const collapseButton = screen.getByRole('button', { name: 'Collapse main menu' });

        expect(root).toHaveAttribute('data-menu-collapsed', 'false');
        expect(collapseButton).toHaveAttribute('aria-expanded', 'true');

        fireEvent.click(collapseButton);

        expect(root).toHaveAttribute('data-menu-collapsed', 'true');
        expect(screen.getByRole('button', { name: 'Expand main menu' })).toHaveAttribute('aria-expanded', 'false');
    });
});
