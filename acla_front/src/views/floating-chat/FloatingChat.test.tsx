import { act, render, screen } from '@testing-library/react';
import type {
    OverlayDisplayAcknowledgement,
    OverlayDisplayRequest,
    OverlayLifecycleEvent,
    OverlayPresentationChange,
} from './overlay-display-types';

jest.mock('contexts/DesktopGameContext', () => ({
    useDesktopGame: () => ({
        detectedGame: null,
        detectionStatus: 'not-detected',
        error: null,
    }),
}));

import FloatingChat from './FloatingChat';

describe('FloatingChat overlay stack', () => {
    const presentation = {
        presentationId: 'presentation-live',
        aiSessionId: 'ai-live',
        mode: 'live' as const,
        displayIdentity: { name: 'Kestrel', emotion: 'idle', agentTags: ['Live'] },
    };
    let commandListener: ((request: OverlayDisplayRequest) => void) | null;
    let enabledListener: ((enabled: boolean) => void) | null;
    let presentationListener: ((change: OverlayPresentationChange) => void) | null;
    let acknowledgements: OverlayDisplayAcknowledgement[];
    let lifecycleEvents: OverlayLifecycleEvent[];
    let requestSequence: number;
    const resizeFloatingChat = jest.fn();

    const send = (
        command: OverlayDisplayRequest['command'],
        presentationId = presentation.presentationId,
    ) => {
        const request: OverlayDisplayRequest = {
            presentationId,
            requestId: `test-request-${++requestSequence}`,
            command,
        };
        act(() => commandListener?.(request));
        return acknowledgements[acknowledgements.length - 1];
    };

    beforeEach(() => {
        commandListener = null;
        enabledListener = null;
        presentationListener = null;
        acknowledgements = [];
        lifecycleEvents = [];
        requestSequence = 0;
        resizeFloatingChat.mockReset();
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: {
                onOverlayDisplayCommand: (listener: typeof commandListener) => {
                    commandListener = listener;
                    return () => { commandListener = null; };
                },
                acknowledgeOverlayDisplayRequest: (ack: OverlayDisplayAcknowledgement) => {
                    acknowledgements.push(ack);
                },
                emitOverlayLifecycle: (event: OverlayLifecycleEvent) => {
                    lifecycleEvents.push(event);
                },
                reportOverlayReady: () => presentationListener?.({ kind: 'started', presentation }),
                onOverlayEnabledChange: (listener: typeof enabledListener) => {
                    enabledListener = listener;
                    return () => { enabledListener = null; };
                },
                onOverlayPresentationChange: (listener: typeof presentationListener) => {
                    presentationListener = listener;
                    return () => { presentationListener = null; };
                },
                resizeFloatingChat,
            },
        });
    });

    afterEach(() => {
        jest.useRealTimers();
    });

    it('renders the compact idle shell even when the presentation has no content', () => {
        const { container } = render(<FloatingChat />);
        expect(container.querySelectorAll('.overlay-shell')).toHaveLength(1);
        expect(container.querySelector('.overlay-shell')).toHaveClass('overlay-shell--idle');
        expect(screen.getByText('Overlay ready')).toBeInTheDocument();
        expect(resizeFloatingChat).toHaveBeenCalled();
    });

    it('renders concurrent singleton displays and orders the pinned card first', () => {
        const { container } = render(<FloatingChat />);
        act(() => enabledListener?.(true));

        send({
            operation: 'upsert',
            type: 'procedure_plan',
            snapshot: {
                goal: 'Live Performance Analysis',
                currentStep: 0,
                requests: [{ type: 'tool_call', title: 'Collect Baseline Lap', status: 'running' }],
            },
        });
        send({
            operation: 'upsert',
            type: 'baseline_progress',
            snapshot: {
                status: 'collecting',
                progress_percent: 64,
                detail: 'Lap 1 baseline',
                track: 'brands_hatch',
                car: 'Ferrari 296',
                current_lap: 0,
                baseline_lap: 0,
            },
        });
        send({
            operation: 'upsert',
            type: 'live_range_todo',
            snapshot: {
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
        });

        const cards = Array.from(container.querySelectorAll<HTMLElement>('.overlay-list-item'));
        expect(cards).toHaveLength(3);
        expect(cards[0]).toHaveAttribute('data-display-type', 'live_range_todo');
        expect(screen.getByText('Live Performance Analysis')).toBeInTheDocument();
        expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '64');
        expect(screen.getByText('Brake reminder')).toBeInTheDocument();
        expect(lifecycleEvents.some((event) => event.kind === 'shown')).toBe(true);
    });

    it('replaces a singleton with a complete snapshot instead of merging it', () => {
        render(<FloatingChat />);
        const first = {
            status: 'collecting' as const,
            progress_percent: 12,
            detail: 'Old detail',
            track: 'old',
            car: 'old',
            current_lap: 1,
            baseline_lap: null,
        };
        send({ operation: 'upsert', type: 'baseline_progress', snapshot: first });
        const acknowledgement = send({
            operation: 'upsert',
            type: 'baseline_progress',
            snapshot: { ...first, progress_percent: 80, detail: 'New detail', track: null },
        });

        expect(acknowledgement.instanceId).toBe('baseline_progress:singleton');
        expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '80');
        expect(screen.queryByText('Old detail')).not.toBeInTheDocument();
        expect(screen.getByText('New detail')).toBeInTheDocument();
    });

    it('folds baseline updates after their pulse while retaining the instance', () => {
        jest.useFakeTimers();
        const { container } = render(<FloatingChat />);
        send({
            operation: 'upsert',
            type: 'baseline_progress',
            snapshot: {
                status: 'collecting', progress_percent: 50, detail: 'Halfway',
                track: null, car: null, current_lap: null, baseline_lap: null,
            },
        });

        act(() => jest.advanceTimersByTime(3_799));
        expect(container.querySelector('.overlay-list-item')).not.toHaveClass('overlay-list-item--folded');
        act(() => jest.advanceTimersByTime(1));
        expect(container.querySelector('.overlay-list-item')).toHaveClass('overlay-list-item--folded');
        expect(lifecycleEvents.some((event) => event.kind === 'folded')).toBe(true);
    });

    it('holds completed AI text for 3.8 seconds without restarting the typewriter', () => {
        jest.useFakeTimers();
        const { container } = render(<FloatingChat />);
        const text = 'Done';
        send({ operation: 'upsert', type: 'ai_message', snapshot: { text } });

        act(() => jest.advanceTimersByTime((text.length * 28) - 1));
        expect(screen.getByTestId('overlay-ai-message')).toHaveTextContent('Don');
        expect(container.querySelector('.overlay-card__caret')).toBeInTheDocument();

        act(() => jest.advanceTimersByTime(1));
        expect(screen.getByTestId('overlay-ai-message')).toHaveTextContent('Done');
        expect(container.querySelector('.overlay-card__caret')).not.toBeInTheDocument();

        act(() => jest.advanceTimersByTime(3_799));
        expect(screen.getByTestId('overlay-ai-message')).toHaveTextContent('Done');
        expect(container.querySelector('.overlay-card__caret')).not.toBeInTheDocument();

        act(() => jest.advanceTimersByTime(1));
        expect(container.querySelector('[data-display-type="ai_message"]')).not.toBeInTheDocument();
        expect(container.querySelector('.overlay-shell')).toHaveClass('overlay-shell--idle');
        expect(lifecycleEvents.filter((event) => (
            event.kind === 'exited' && event.reason === 'transient_complete'
        ))).toHaveLength(1);
    });

    it('restarts an in-progress typewriter only for the replacement revision', () => {
        jest.useFakeTimers();
        const { container } = render(<FloatingChat />);
        const first = send({ operation: 'upsert', type: 'ai_message', snapshot: { text: 'First' } });
        act(() => jest.advanceTimersByTime(2 * 28));
        expect(screen.getByTestId('overlay-ai-message')).toHaveTextContent('Fi');

        const second = send({ operation: 'upsert', type: 'ai_message', snapshot: { text: 'Second' } });

        expect(second.instanceId).toBe(first.instanceId);
        expect(container.querySelectorAll('[data-display-type="ai_message"]')).toHaveLength(1);
        expect(screen.getByTestId('overlay-ai-message')).toHaveTextContent('');

        act(() => jest.advanceTimersByTime(28));
        expect(screen.getByTestId('overlay-ai-message')).toHaveTextContent('S');
        expect(screen.queryByText('First')).not.toBeInTheDocument();

        act(() => jest.advanceTimersByTime(5 * 28));
        expect(screen.getByTestId('overlay-ai-message')).toHaveTextContent('Second');
        expect(container.querySelector('.overlay-card__caret')).not.toBeInTheDocument();
    });

    it('updates keyed tool status in place and renders map and comparison siblings', () => {
        const { container } = render(<FloatingChat />);
        const started = send({
            operation: 'upsert',
            type: 'tool_status',
            snapshot: { runId: 'run-7', name: 'analyze', title: 'Analyzing', status: 'started' },
            options: { key: 'run-7' },
        });
        const completed = send({
            operation: 'upsert',
            type: 'tool_status',
            snapshot: { runId: 'run-7', name: 'analyze', title: 'Analysis complete', status: 'completed', ok: true },
            options: { key: 'run-7' },
        });
        send({
            operation: 'upsert',
            type: 'map',
            snapshot: { status: 'unavailable', title: 'Track map', reason: 'No circuit selected' },
        });
        send({
            operation: 'upsert',
            type: 'driver_expert_comparison',
            snapshot: {
                title: 'Turn 6 replay',
                comparison: { samples: [{
                    driverTimeMs: 1_000,
                    expertTimeMs: 2_000,
                    driverTrackPosition: 0.2,
                    expertTrackPosition: 0.2,
                    driverGas: 0.7,
                    expertGas: 0.8,
                }] },
            },
        });

        expect(started.instanceId).toBe(completed.instanceId);
        expect(container.querySelectorAll('[data-display-type="tool_status"]')).toHaveLength(1);
        expect(screen.getByText('Analysis complete')).toBeInTheDocument();
        expect(screen.getByText('Map is not available')).toBeInTheDocument();
        expect(screen.getByTestId('driver-expert-comparison')).toBeInTheDocument();
        expect(container.querySelector('.overlay-shell')).toHaveStyle({ width: '760px' });
    });

    it('fills the shell with the newest full-size requester while retaining headers and mounted siblings', () => {
        const { container } = render(<FloatingChat />);
        send({ operation: 'upsert', type: 'ai_message', snapshot: { text: 'Compare these laps' } });
        const baselineAck = send({
            operation: 'upsert',
            type: 'baseline_progress',
            snapshot: {
                status: 'collecting', progress_percent: 40, detail: 'Still collecting',
                track: null, car: null, current_lap: null, baseline_lap: null,
            },
        });
        const first = send({
            operation: 'upsert',
            type: 'driver_expert_comparison',
            snapshot: {
                title: 'First replay',
                comparison: { samples: [{
                    driverTimeMs: 0,
                    expertTimeMs: 0,
                    driverTrackPosition: 0.2,
                    expertTrackPosition: 0.2,
                }] },
            },
        });
        const firstGraph = screen.getByTestId('driver-expert-comparison');

        expect(send({
            operation: 'request_full_size', target: { instanceId: first.instanceId! },
        }).accepted).toBe(true);

        const shell = container.querySelector('.overlay-shell');
        const baselineCard = container.querySelector(`[data-instance-id="${baselineAck.instanceId}"]`);
        const firstCard = container.querySelector(`[data-instance-id="${first.instanceId}"]`);
        expect(shell).toHaveClass('overlay-shell--full-size');
        expect(shell).toHaveStyle({ width: '760px', height: '500px' });
        expect(shell).toHaveAttribute('data-full-size-instance-id', first.instanceId);
        expect(resizeFloatingChat).toHaveBeenLastCalledWith(760, 500);
        expect(screen.getByText('Kestrel')).toBeInTheDocument();
        expect(screen.getByTestId('overlay-ai-message')).toBeInTheDocument();
        expect(screen.getByTestId('driver-expert-comparison')).toBe(firstGraph);
        expect(firstGraph).toHaveClass('floating-pill-comparison--full-size');
        expect(firstCard).toHaveClass('overlay-list-item--full-size-active');
        expect(baselineCard).toHaveClass('overlay-list-item--full-size-hidden');
        expect(baselineCard?.querySelector('[role="progressbar"]')).toBeInTheDocument();

        const second = send({
            operation: 'upsert',
            type: 'driver_expert_comparison',
            snapshot: {
                title: 'Second replay',
                comparison: { samples: [{
                    driverTimeMs: 0,
                    expertTimeMs: 0,
                    driverTrackPosition: 0.4,
                    expertTrackPosition: 0.4,
                }] },
            },
        });
        send({ operation: 'request_full_size', target: { instanceId: second.instanceId! } });
        expect(shell).toHaveAttribute('data-full-size-instance-id', second.instanceId);
        expect(firstCard).toHaveClass('overlay-list-item--full-size-hidden');

        send({ operation: 'exit', target: { instanceId: second.instanceId! }, reason: 'producer_exit' });
        expect(shell).toHaveAttribute('data-full-size-instance-id', first.instanceId);
        expect(firstCard).toHaveClass('overlay-list-item--full-size-active');
    });

    it('rejects malformed comparisons and does not render user item controls', () => {
        const { container } = render(<FloatingChat />);
        const rejected = send({
            operation: 'upsert',
            type: 'driver_expert_comparison',
            snapshot: {
                title: 'Unavailable replay',
                comparison: { samples: [{
                    driverTimeMs: 1_000,
                    expertTimeMs: 2_000,
                    trackPosition: 0.2,
                }] },
            },
        });
        expect(rejected.accepted).toBe(false);
        expect(container.querySelector('[data-display-type="driver_expert_comparison"]')).not.toBeInTheDocument();

        send({ operation: 'upsert', type: 'ai_message', snapshot: { text: 'Dismiss me' } });
        send({
            operation: 'upsert',
            type: 'baseline_progress',
            snapshot: {
                status: 'collecting', progress_percent: 50, detail: 'Halfway',
                track: null, car: null, current_lap: null, baseline_lap: null,
            },
        });

        expect(container.querySelector('[data-display-type="ai_message"]')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Dismiss overlay item' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Fold overlay item' })).not.toBeInTheDocument();
        expect(container.querySelector('.overlay-item__actions')).not.toBeInTheDocument();
    });

    it('clears replaced content and ignores stale requests and stale cleanup', () => {
        const { container } = render(<FloatingChat />);
        send({ operation: 'upsert', type: 'ai_message', snapshot: { text: 'Old speech' } });
        const replacement = {
            presentationId: 'presentation-agent',
            aiSessionId: 'ai-agent',
            mode: 'agent' as const,
            displayIdentity: { name: 'Track Guide', agentTags: ['Agent'] },
        };

        act(() => presentationListener?.({ kind: 'started', presentation: replacement }));
        expect(container.querySelector('[data-display-type="ai_message"]')).not.toBeInTheDocument();
        expect(screen.getByText('Track Guide')).toBeInTheDocument();
        expect(container.querySelector('.overlay-shell')).toHaveClass('overlay-shell--idle');

        const stale = send(
            { operation: 'upsert', type: 'ai_message', snapshot: { text: 'Too late' } },
            presentation.presentationId,
        );
        expect(stale.accepted).toBe(false);
        expect(screen.queryByText('Too late')).not.toBeInTheDocument();

        send(
            { operation: 'upsert', type: 'ai_message', snapshot: { text: 'New speech' } },
            replacement.presentationId,
        );
        act(() => presentationListener?.({ kind: 'ended', presentationId: presentation.presentationId }));
        expect(container.querySelector('[data-display-type="ai_message"]')).toBeInTheDocument();

        act(() => presentationListener?.({ kind: 'ended', presentationId: replacement.presentationId }));
        expect(container.querySelector('[data-display-type="ai_message"]')).not.toBeInTheDocument();
        expect(container.querySelector('.overlay-shell')).toHaveClass('overlay-shell--idle');
    });
});
