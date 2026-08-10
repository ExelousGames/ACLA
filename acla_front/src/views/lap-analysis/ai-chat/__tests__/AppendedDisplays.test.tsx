import { render, screen } from '@testing-library/react';
import {
    LiveRangeTodoListDisplay,
    ProcedurePlan,
} from 'components/ai-engineering-tools';
import BaselineProgressDisplay from 'views/live-session/BaselineProgressDisplay';
import ToolMessageDisplay from '../ToolMessageDisplay';

describe('appended AI display components', () => {
    it('renders baseline progress on the floating pill surface', () => {
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
                    baseline_lap: 0,
                }}
            />,
        );

        expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '42');
        expect(screen.getByText('Lap 1 baseline')).toBeInTheDocument();
    });

    it('renders a compact procedure plan without the chat clear button', () => {
        render(
            <ProcedurePlan
                surface="pill"
                plan={{
                    goal: 'Collect a clean baseline',
                    currentStep: 1,
                    requests: [
                        { type: 'tool_call', title: 'Start baseline', status: 'complete' },
                        { type: 'tool_call', title: 'Analyze baseline', status: 'running' },
                        { type: 'driver_action', title: 'Run next lap', status: 'pending' },
                    ],
                }}
                onClear={jest.fn()}
            />,
        );

        expect(screen.getByText('Collect a clean baseline')).toBeInTheDocument();
        expect(screen.getByText('Analyze baseline')).toBeInTheDocument();
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
                            content: { title: 'Turn exit' },
                            data: {},
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
        expect(screen.getByText('running')).toBeInTheDocument();
    });
});
