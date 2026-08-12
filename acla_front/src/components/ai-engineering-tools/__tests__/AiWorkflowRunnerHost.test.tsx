import React, { useLayoutEffect } from 'react';
import { act, render, screen } from '@testing-library/react';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefProvider,
    type AiToolComponentRefDirectory,
    useAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import AiWorkflowRunnerHost, {
    type AiWorkflowRunnerSnapshot,
} from '../AiWorkflowRunnerHost';
import { GoalRunner, type GoalExecutableRequest } from '../Goal';
import { LiveRangeTodoListRunner } from '../LiveRangeTodoList';
import { ProcedurePlanRunner, type ProcedurePlanState } from '../ProcedurePlan';
import { GoalClearedError } from 'contexts/AiToolComponentError';

const DirectoryCapture = ({
    onDirectory,
}: {
    onDirectory: (directory: AiToolComponentRefDirectory) => void;
}) => {
    const directory = useAiToolComponentRefDirectory();
    useLayoutEffect(() => onDirectory(directory), [directory, onDirectory]);
    return null;
};

const renderHost = (
    telemetry: Record<string, any> | null = null,
    onSnapshotChange = jest.fn<void, [AiWorkflowRunnerSnapshot]>(),
) => {
    let directory: AiToolComponentRefDirectory | null = null;
    const onDirectory = (next: AiToolComponentRefDirectory) => { directory = next; };
    const tree = (nextTelemetry: Record<string, any> | null) => (
        <AiToolComponentRefProvider>
            <DirectoryCapture onDirectory={onDirectory} />
            <AiWorkflowRunnerHost
                telemetry={nextTelemetry}
                onSnapshotChange={onSnapshotChange}
            />
        </AiToolComponentRefProvider>
    );
    const view = render(tree(telemetry));
    return {
        getDirectory: () => directory!,
        onSnapshotChange,
        rerenderTelemetry: (next: Record<string, any> | null) => view.rerender(tree(next)),
        unmount: view.unmount,
    };
};

describe('AiWorkflowRunnerHost', () => {
    it('renders and observes only the registered workflow runner', async () => {
        const host = renderHost();
        const runner = new GoalRunner(AI_TOOL_COMPONENT_NAMES.GOAL);
        const request: GoalExecutableRequest = {
            name: 'Stay focused',
            steps: [{
                id: 'prepare',
                title: 'Prepare',
                name: 'prepare_tool',
                taskStart: () => ({
                    tool_name: 'prepare_tool',
                    run_id: 'pending-run',
                    status: 'running',
                    output: null,
                    final: false,
                }),
            }],
            determination: {
                tool: { name: 'determine_tool' },
                result_path: 'value',
                operator: 'eq',
                target: 1,
                taskStart: () => ({ value: 1 }),
            },
        };

        let pending!: ReturnType<GoalRunner['create']>;
        await act(async () => {
            runner.addComponentRef(host.getDirectory());
            pending = runner.create(request);
            await Promise.resolve();
        });

        expect(screen.getByText('Stay focused')).toBeInTheDocument();
        expect(screen.queryByLabelText('Live range to-do list')).not.toBeInTheDocument();
        expect(host.onSnapshotChange).toHaveBeenLastCalledWith(expect.objectContaining({
            kind: 'goal',
            runner,
            snapshot: expect.objectContaining({ status: 'running' }),
        }));

        const rejection = expect(pending).rejects.toBeInstanceOf(GoalClearedError);
        act(() => runner.clear());
        await rejection;
        expect(host.getDirectory().findBaseComponentRef()).toBeNull();
        expect(screen.queryByText('Stay focused')).not.toBeInTheDocument();
    });

    it('feeds telemetry to a live-range runner and removes it after its last event', async () => {
        const host = renderHost();
        const taskStart = jest.fn(() => undefined);
        const runner = new LiveRangeTodoListRunner(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST);

        act(() => {
            runner.addComponentRef(host.getDirectory());
            runner.replaceEvents([{
                id: 'brake',
                normalized_position: 0.2,
                lead_time_seconds: 0,
                content: { title: 'Brake now' },
                data: {},
                taskStart,
            }]);
        });
        expect(screen.getByText('Brake now')).toBeInTheDocument();

        act(() => host.rerenderTelemetry({ normalized_car_position: 0.1 }));
        await act(async () => {
            host.rerenderTelemetry({ normalized_car_position: 0.3 });
            await Promise.resolve();
            await Promise.resolve();
        });

        expect(taskStart).toHaveBeenCalledTimes(1);
        expect(host.getDirectory().findBaseComponentRef()).toBeNull();
        expect(screen.queryByText('Brake now')).not.toBeInTheDocument();
    });

    it('aborts and unregisters the remaining runner on host teardown', () => {
        const host = renderHost();
        let signal: AbortSignal | null = null;
        const runner = new ProcedurePlanRunner(AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN);
        const plan: ProcedurePlanState = {
            goal: 'Finish teardown',
            currentStep: 0,
            requests: [{
                type: 'tool_call',
                title: 'Wait',
                status: 'pending',
                taskStart: (nextSignal) => {
                    signal = nextSignal;
                    return new Promise(() => undefined);
                },
            }],
        };

        act(() => {
            runner.addComponentRef(host.getDirectory());
            runner.replace(plan);
        });
        expect(screen.getByText('Finish teardown')).toBeInTheDocument();

        host.unmount();
        expect((signal as unknown as AbortSignal).aborted).toBe(true);
        expect(host.getDirectory().findBaseComponentRef()).toBeNull();
    });
});
