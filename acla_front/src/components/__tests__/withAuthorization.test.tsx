import React from 'react';
import { render, screen } from '@testing-library/react';
import withAuthorization from 'components/withAuthorization';

// Mock useAuth
const mockHasPermission = jest.fn();
const mockHasRole = jest.fn();

jest.mock('hooks/AuthProvider', () => ({
    useAuth: () => ({
        hasPermission: mockHasPermission,
        hasRole: mockHasRole,
    }),
}));

const TestComponent = () => <div data-testid="wrapped">Wrapped Content</div>;
const FallbackComponent = () => <div data-testid="fallback">Fallback</div>;

describe('withAuthorization', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        mockHasPermission.mockReturnValue(true);
        mockHasRole.mockReturnValue(true);
    });

    it('should render wrapped component when user has the required role', () => {
        const Protected = withAuthorization(TestComponent, { requiredRole: 'admin' });

        render(<Protected />);

        expect(screen.getByTestId('wrapped')).toBeInTheDocument();
    });

    it('should show "Access Denied" when user lacks the required role', () => {
        mockHasRole.mockReturnValue(false);
        const Protected = withAuthorization(TestComponent, { requiredRole: 'admin' });

        render(<Protected />);

        expect(screen.queryByTestId('wrapped')).not.toBeInTheDocument();
        expect(screen.getByText('Access Denied')).toBeInTheDocument();
    });

    it('should render fallback component when user lacks role and fallback is provided', () => {
        mockHasRole.mockReturnValue(false);
        const Protected = withAuthorization(TestComponent, {
            requiredRole: 'admin',
            fallback: FallbackComponent,
        });

        render(<Protected />);

        expect(screen.getByTestId('fallback')).toBeInTheDocument();
        expect(screen.queryByTestId('wrapped')).not.toBeInTheDocument();
    });

    it('should render wrapped component when user has required permission', () => {
        const Protected = withAuthorization(TestComponent, {
            requiredPermission: { action: 'read', resource: 'user' },
        });

        render(<Protected />);

        expect(screen.getByTestId('wrapped')).toBeInTheDocument();
    });

    it('should show "Access Denied" when user lacks the required permission', () => {
        mockHasPermission.mockReturnValue(false);
        const Protected = withAuthorization(TestComponent, {
            requiredPermission: { action: 'delete', resource: 'user' },
        });

        render(<Protected />);

        expect(screen.queryByTestId('wrapped')).not.toBeInTheDocument();
        expect(screen.getByText('Access Denied')).toBeInTheDocument();
    });

    it('should pass props through to the wrapped component', () => {
        const PropsComponent = ({ title }: { title: string }) => (
            <div data-testid="with-props">{title}</div>
        );
        const Protected = withAuthorization(PropsComponent, {});

        render(<Protected title="Hello" />);

        expect(screen.getByText('Hello')).toBeInTheDocument();
    });
});
