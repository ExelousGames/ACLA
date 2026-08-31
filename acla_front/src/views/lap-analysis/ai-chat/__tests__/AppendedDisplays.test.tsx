import { fireEvent, render, screen } from '@testing-library/react';
import {
    LiveRangeTodoListDisplay,
    ProcedurePlan,
} from 'components/ai-engineering-tools';
import BaselineProgressDisplay from 'views/live-session/BaselineProgressDisplay';
import ToolMessageDisplay from '../ToolMessageDisplay';

describe('appended AI display components', () => {
    it('renders only the active baseline step on the floating pill surface', () => {
        render(
            <BaselineProgressDisplay
                surface="pill"
                tag={{
                    status: 'collecting',
                    progress_percent: 42,
                    detail: 'Lap 1 baseline',
                    track: 'brands_hatch',
                    car: 'Ferrari 296',
                    current_lap: 0,
                    baseline_lap_id: 0,
                }}
            />,
        );

        expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '42');
        expect(screen.getByText('Lap 1 baseline')).toBeInTheDocument();
        expect(screen.getByText('Recording baseline')).toBeInTheDocument();
        expect(screen.queryByText('Start collection')).not.toBeInTheDocument();
        expect(screen.queryByText('Finish collection')).not.toBeInTheDocument();
    });

    it('keeps every baseline step visible on the panel surface', () => {
        render(<BaselineProgressDisplay tag={null} />);

        expect(screen.getByText('Start collection')).toBeInTheDocument();
        expect(screen.getByText('Record baseline')).toBeInTheDocument();
        expect(screen.getByText('Finish collection')).toBeInTheDocument();
    });

    it('renders a procedure plan in chat without a close button', () => {
        render(
            <ProcedurePlan
                surface="chat"
                plan={{
                    goal: 'Collect a clean baseline',
                    currentStep: 1,
                    requests: [
                        { type: 'tool_call', title: 'Start baseline', status: 'complete' },
                        { type: 'tool_call', title: 'Analyze baseline', status: 'running' },
                        { type: 'driver_action', title: 'Run next lap', status: 'pending' },
                    ],
                }}
            />,
        );

        expect(screen.getByText('Collect a clean baseline')).toBeInTheDocument();
        expect(screen.getByText('Analyze baseline')).toBeInTheDocument();
        expect(screen.queryByText(/tool_call/i)).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Dismiss the visible plan')).not.toBeInTheDocument();
    });

    it('renders tool status without debug output in pill mode', () => {
        render(
            <ToolMessageDisplay
                surface="pill"
                debugMode
                tool={{
                    name: 'collect_live_baseline',
                    title: 'Baseline complete',
                    status: 'completed',
                    ok: true,
                    result: { hidden: true },
                }}
            />,
        );

        expect(screen.getByText('Baseline complete')).toBeInTheDocument();
        expect(screen.queryByText('collect_live_baseline')).not.toBeInTheDocument();
    });

    it('renders a compact live range to-do list snapshot', () => {
        render(
            <LiveRangeTodoListDisplay
                surface="pill"
                snapshot={{
                    created_at: 1,
                    updated_at: 2,
                    current_position: 0.1,
                    rolling_rate: 0.05,
                    events: [
                        {
                            id: 'range-1',
                            normalized_position: 0.2,
                            lead_time_seconds: 2,
                            content: { title: 'Turn exit', description: 'Use all the road' },
                            status: 'running',
                            eta_seconds: 2,
                            created_at: 1,
                            updated_at: 2,
                        },
                    ],
                }}
            />,
        );

        expect(screen.getByText('Turn exit')).toBeInTheDocument();
        expect(screen.getByText('Use all the road')).toBeInTheDocument();
        expect(screen.getByText('running')).toBeInTheDocument();
    });

    it('collapses long live range to-do lists in chat', () => {
        render(
            <LiveRangeTodoListDisplay
                surface="chat"
                snapshot={{
                    created_at: 1,
                    updated_at: 2,
                    current_position: 0.1,
                    rolling_rate: 0.05,
                    events: Array.from({ length: 5 }, (_, index) => ({
                        id: `range-${index + 1}`,
                        normalized_position: (index + 1) / 10,
                        lead_time_seconds: 2,
                        content: { title: `Event ${index + 1}` },
                        status: 'pending' as const,
                        eta_seconds: index + 1,
                        created_at: 1,
                        updated_at: 2,
                    })),
                }}
            />,
        );

        expect(screen.getByText('Event 3')).toBeInTheDocument();
        expect(screen.queryByText('Event 4')).not.toBeInTheDocument();
        const showMore = screen.getByRole('button', { name: 'Show 2 more' });
        expect(showMore).toHaveAttribute('aria-expanded', 'false');

        fireEvent.click(showMore);

        expect(screen.getByText('Event 5')).toBeInTheDocument();
        const showLess = screen.getByRole('button', { name: 'Show less' });
        expect(showLess).toHaveAttribute('aria-expanded', 'true');

        fireEvent.click(showLess);

        expect(screen.queryByText('Event 4')).not.toBeInTheDocument();
    });
});
