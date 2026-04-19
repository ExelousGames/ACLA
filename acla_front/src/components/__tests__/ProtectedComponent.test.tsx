import React from 'react';
import { render, screen } from '@testing-library/react';
import ProtectedComponent from 'components/ProtectedComponent';

// Mock useAuth and useNavigate
const mockHasPermission = jest.fn();
const mockHasRole = jest.fn();
const mockNavigate = jest.fn();

jest.mock('hooks/AuthProvider', () => ({
    useAuth: () => ({
        hasPermission: mockHasPermission,
        hasRole: mockHasRole,
    }),
}));

jest.mock('react-router-dom', () => ({
    useNavigate: () => mockNavigate,
}));

describe('ProtectedComponent', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        mockHasPermission.mockReturnValue(true);
        mockHasRole.mockReturnValue(true);
    });

    it('should render children when no permissions or roles are required', () => {
        render(
            <ProtectedComponent>
                <div data-testid="child">Content</div>
            </ProtectedComponent>
        );
        expect(screen.getByTestId('child')).toBeInTheDocument();
    });

    it('should render children when user has the required role', () => {
        mockHasRole.mockReturnValue(true);

        render(
            <ProtectedComponent requiredRole="admin">
                <div data-testid="child">Content</div>
            </ProtectedComponent>
        );

        expect(screen.getByTestId('child')).toBeInTheDocument();
    });

    it('should show fallback when user lacks the required role', () => {
        mockHasRole.mockReturnValue(false);

        render(
            <ProtectedComponent requiredRole="admin" fallback="Access Denied">
                <div data-testid="child">Content</div>
            </ProtectedComponent>
        );

        expect(screen.queryByTestId('child')).not.toBeInTheDocument();
        expect(screen.getByText('Access Denied')).toBeInTheDocument();
    });

    it('should render children when user has the required permission', () => {
        mockHasPermission.mockReturnValue(true);

        render(
            <ProtectedComponent requiredPermission={{ action: 'read', resource: 'user' }}>
                <div data-testid="child">Content</div>
            </ProtectedComponent>
        );

        expect(screen.getByTestId('child')).toBeInTheDocument();
    });

    it('should show fallback when user lacks the required permission', () => {
        mockHasPermission.mockReturnValue(false);

        render(
            <ProtectedComponent
                requiredPermission={{ action: 'delete', resource: 'user' }}
                fallback="No Access"
            >
                <div data-testid="child">Content</div>
            </ProtectedComponent>
        );

        expect(screen.queryByTestId('child')).not.toBeInTheDocument();
        expect(screen.getByText('No Access')).toBeInTheDocument();
    });

    it('should redirect when user lacks role and redirectTo is set', () => {
        mockHasRole.mockReturnValue(false);

        render(
            <ProtectedComponent requiredRole="admin" redirectTo="/login">
                <div data-testid="child">Content</div>
            </ProtectedComponent>
        );

        expect(mockNavigate).toHaveBeenCalledWith('/login');
        expect(screen.queryByTestId('child')).not.toBeInTheDocument();
    });

    it('should redirect when user lacks permission and redirectTo is set', () => {
        mockHasPermission.mockReturnValue(false);

        render(
            <ProtectedComponent
                requiredPermission={{ action: 'manage', resource: 'all' }}
                redirectTo="/dashboard"
            >
                <div data-testid="child">Content</div>
            </ProtectedComponent>
        );

        expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });
});
