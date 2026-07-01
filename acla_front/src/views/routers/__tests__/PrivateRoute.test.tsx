import React from 'react';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import PrivateRoute from '../PrivateRoute';

// Mock the useAuth hook
const mockUseAuth = jest.fn();
jest.mock('hooks/AuthProvider', () => ({
    useAuth: () => mockUseAuth(),
}));

// A simple child component that PrivateRoute should render when authenticated
const TestChild = () => <div data-testid="protected-content">Protected</div>;

const renderWithRouter = (initialEntries: string[] = ['/']) => {
    const { Routes, Route } = require('react-router-dom');
    return render(
        <MemoryRouter initialEntries={initialEntries}>
            <Routes>
                <Route element={<PrivateRoute />}>
                    <Route path="/" element={<TestChild />} />
                </Route>
                <Route path="/login" element={<div data-testid="login-page">Login</div>} />
            </Routes>
        </MemoryRouter>
    );
};

describe('PrivateRoute', () => {
    it('should render child route when user has a token', () => {
        mockUseAuth.mockReturnValue({ token: 'valid-token' });

        renderWithRouter();

        expect(screen.getByTestId('protected-content')).toBeInTheDocument();
    });

    it('should redirect to /login when user has no token', () => {
        mockUseAuth.mockReturnValue({ token: '' });

        renderWithRouter();

        expect(screen.getByTestId('login-page')).toBeInTheDocument();
        expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument();
    });

    it('should redirect to /login when token is null', () => {
        mockUseAuth.mockReturnValue({ token: null });

        renderWithRouter();

        expect(screen.getByTestId('login-page')).toBeInTheDocument();
    });
});
