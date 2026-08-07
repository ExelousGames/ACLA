import { act, render, screen, waitFor } from '@testing-library/react';

jest.mock('contexts/DesktopGameContext', () => ({
    useDesktopGame: () => ({
        detectedGame: null,
        detectionStatus: 'not-detected',
        error: null,
    }),
}));

import FloatingChat from './FloatingChat';

describe('FloatingChat', () => {
    it('renders compact appended UI payloads from storage events', async () => {
        const resizeFloatingChat = jest.fn();
        Object.defineProperty(window, 'electronAPI', {
            value: { resizeFloatingChat },
            configurable: true,
        });

        render(<FloatingChat />);

        act(() => {
            window.dispatchEvent(new StorageEvent('storage', {
                key: 'acla-pill-msg',
                newValue: JSON.stringify({
                    kind: 'baseline',
                    text: 'Lap 1 baseline',
                    ts: Date.now() + 1,
                    data: {
                        status: 'collecting',
                        progress_percent: 64,
                        detail: 'Lap 1 baseline',
                        track: 'brands_hatch',
                        car: 'Ferrari 296',
                        current_lap: 0,
                        baseline_lap: 0,
                    },
                }),
            }));
        });

        await waitFor(() => expect(screen.getAllByText('Lap 1 baseline').length).toBeGreaterThan(0));
        expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '64');
        await waitFor(() => expect(resizeFloatingChat).toHaveBeenCalledWith(420, 136));
    });

    it('shows a plan update without replaying the previous assistant line', async () => {
        const { container } = render(<FloatingChat />);

        const now = Date.now();
        act(() => {
            window.dispatchEvent(new StorageEvent('storage', {
                key: 'acla-pill-msg',
                newValue: JSON.stringify({
                    kind: 'message',
                    text: 'I will collect a baseline lap before comparing sectors.',
                    ts: now + 1,
                }),
            }));
        });

        act(() => {
            window.dispatchEvent(new StorageEvent('storage', {
                key: 'acla-pill-msg',
                newValue: JSON.stringify({
                    kind: 'plan',
                    text: 'Collect Baseline Lap',
                    ts: now + 2,
                    data: {
                        goal: 'Live Performance Analysis',
                        currentStep: 0,
                        requests: [
                            { type: 'tool_call', title: 'Collect Baseline Lap', status: 'running' },
                            { type: 'tool_call', title: 'Compare Live Lap', status: 'pending' },
                        ],
                    },
                }),
            }));
        });

        await waitFor(() => {
            expect(container.querySelector('.pill .msg-inner')).toHaveTextContent(
                'Collect Baseline Lap',
            );
        });
        expect(screen.getByText('Live Performance Analysis')).toBeInTheDocument();
        expect(screen.getAllByText('Collect Baseline Lap').length).toBeGreaterThan(0);
    });

    it('renders due live range to-do lifecycle content', async () => {
        render(<FloatingChat />);

        act(() => {
            window.dispatchEvent(new StorageEvent('storage', {
                key: 'acla-pill-msg',
                newValue: JSON.stringify({
                    kind: 'live_range_todo_list',
                    text: 'Brake reminder: running',
                    ts: Date.now() + 10,
                    data: {
                        events: [{
                            id: 'brake',
                            normalized_position: 0.25,
                            lead_time_seconds: 2,
                            content: { title: 'Brake reminder' },
                            data: { source: 'ai' },
                            status: 'running',
                            eta_seconds: 1.4,
                            created_at: 1,
                            updated_at: 2,
                        }],
                        current_position: 0.2,
                        rolling_rate: 0.04,
                        created_at: 1,
                        updated_at: 2,
                    },
                }),
            }));
        });

        await waitFor(() => expect(screen.getByText('Brake reminder')).toBeInTheDocument());
        expect(screen.getByText('running')).toBeInTheDocument();
        expect(screen.getByText('ETA 1.4s')).toBeInTheDocument();
    });

    it('renders the pedal replay HUD inside the 760 by 500 comparison panel', async () => {
        const resizeFloatingChat = jest.fn();
        Object.defineProperty(window, 'electronAPI', {
            value: { resizeFloatingChat },
            configurable: true,
        });
        render(<FloatingChat />);

        act(() => {
            window.dispatchEvent(new StorageEvent('storage', {
                key: 'acla-pill-msg',
                newValue: JSON.stringify({
                    kind: 'driver_expert_comparison',
                    text: 'Turn 6 replay',
                    ts: Date.now() + 20,
                    data: {
                        title: 'Turn 6 replay',
                        comparison: {
                            samples: [{
                                driverTimeMs: 1_000,
                                expertTimeMs: 2_000,
                                driverTrajectory: { x: 10, y: 20, z: 30 },
                                expertTrajectory: { x: 11, y: 21, z: 31 },
                                driverGas: 0.7,
                                expertGas: 0.8,
                                driverBrake: 0.2,
                                expertBrake: 0.1,
                                driverGear: 4,
                                expertGear: 5,
                            }],
                        },
                    },
                }),
            }));
        });

        await waitFor(() => expect(screen.getByTestId('driver-expert-comparison')).toBeInTheDocument());
        expect(screen.getByTestId('comparison-track-map')).toBeInTheDocument();
        expect(screen.getByTestId('driver-throttle-gauge')).toHaveAttribute('data-value', '0.7');
        expect(screen.getByTestId('expert-gear')).toHaveTextContent('5');
        expect(screen.queryByTestId('comparison-graph-gas')).not.toBeInTheDocument();
        await waitFor(() => expect(resizeFloatingChat).toHaveBeenCalledWith(760, 500));
    });

    it('keeps a long comparison open through the slower replay plus its completion pause', () => {
        jest.useFakeTimers();
        const { container, unmount } = render(<FloatingChat />);

        act(() => {
            window.dispatchEvent(new StorageEvent('storage', {
                key: 'acla-pill-msg',
                newValue: JSON.stringify({
                    kind: 'driver_expert_comparison',
                    text: 'Long replay',
                    ts: Date.now() + 30,
                    data: {
                        title: 'Long replay',
                        comparison: {
                            samples: [{
                                driverTimeMs: 1_000,
                                expertTimeMs: 9_000,
                                driverGas: 0,
                                expertGas: 0,
                            }, {
                                driverTimeMs: 6_000,
                                expertTimeMs: 11_000,
                                driverGas: 1,
                                expertGas: 1,
                            }],
                        },
                    },
                }),
            }));
        });

        expect(container.querySelector('.pill')).toHaveClass('open');
        act(() => jest.advanceTimersByTime(5_799));
        expect(container.querySelector('.pill')).toHaveClass('open');
        act(() => jest.advanceTimersByTime(1));
        expect(container.querySelector('.pill')).not.toHaveClass('open');

        unmount();
        jest.useRealTimers();
    });
});
