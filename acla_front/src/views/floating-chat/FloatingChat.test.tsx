import { act, render, screen, waitFor } from '@testing-library/react';
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
});
