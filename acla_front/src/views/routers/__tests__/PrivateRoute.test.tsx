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

const renderWithRouter = (initialEntries: string[] = ['/dashboard']) => {
    const { Routes, Route } = require('react-router-dom');
    return render(
        <MemoryRouter initialEntries={initialEntries}>
            <Routes>
                <Route element={<PrivateRoute />}>
                    <Route path="/dashboard" element={<TestChild />} />
                </Route>
                <Route path="/" element={<div data-testid="landing-page">Landing</div>} />
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

    it('should redirect to the public landing page when user has no token', () => {
        mockUseAuth.mockReturnValue({ token: '' });

        renderWithRouter();

        expect(screen.getByTestId('landing-page')).toBeInTheDocument();
        expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument();
    });

    it('should redirect to the public landing page when token is null', () => {
        mockUseAuth.mockReturnValue({ token: null });

        renderWithRouter();

        expect(screen.getByTestId('landing-page')).toBeInTheDocument();
    });
});
