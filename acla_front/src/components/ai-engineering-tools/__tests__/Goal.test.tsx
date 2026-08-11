import React, { useLayoutEffect } from 'react';
import { act, render, screen } from '@testing-library/react';
import {
    AiToolComponentRefProvider,
    type AiToolComponentRefDirectory,
    useAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import Goal, {
    GoalDisplay,
    buildGoalRequest,
    compareGoalValues,
    extractGoalResultPath,
    isSafeGoalResultPath,
    validateGoalRequest,
    type GoalHandle,
    type GoalRequest,
    type GoalSnapshot,
    type GoalStepDescriptor,
    type GoalToolOutputEnvelope,
} from '../Goal';
import type { TaskStartFunction } from '../task-start-function';

const goalRequest = (goal = 'No mistakes in the last analyzed lap') => ({
    goal,
    steps: [
        { id: 'collect', title: 'Collect baseline', name: 'collect_live_baseline' },
        { id: 'analyze', title: 'Analyze baseline', name: 'analyze_live_recorded_analysis' },
        { id: 'mistake_count', title: 'Count mistakes', name: 'get_live_analysis_mistake_count' },
    ],
    comparison: {
        step_id: 'mistake_count',
        result_path: 'mistake_count',
        operator: 'eq' as const,
        target: 0,
        metric_label: 'Mistake count',
    },
});

const envelope = (
    toolName: string,
    runId: string,
    output: unknown,
    final = true,
): GoalToolOutputEnvelope => ({
    tool_name: toolName,
    run_id: runId,
    status: final ? 'complete' : 'running',
    output,
    final,
});

const DirectoryCapture = ({
    onDirectory,
}: {
    onDirectory: (directory: AiToolComponentRefDirectory) => void;
}) => {
    const directory = useAiToolComponentRefDirectory();
    useLayoutEffect(() => onDirectory(directory), [directory, onDirectory]);
    return null;
};

const renderGoal = () => {
    let directory: AiToolComponentRefDirectory | null = null;
    render(
        <AiToolComponentRefProvider>
            <Goal name="goal" />
            <DirectoryCapture onDirectory={(next) => { directory = next; }} />
        </AiToolComponentRefProvider>,
    );
    return () => directory!.findComponentRef<GoalHandle>('goal')!.current!;
};

const withTaskStarts = (
    request: ReturnType<typeof goalRequest>,
    selectTaskStart: (step: GoalStepDescriptor) => TaskStartFunction,
): GoalRequest => ({
    ...request,
    steps: request.steps.map((step) => ({ ...step, taskStart: selectTaskStart(step) })),
    comparison: { ...request.comparison },
});

describe('Goal request and comparison helpers', () => {
    it.each([
        ['eq', 2, 2, true],
        ['neq', 2, 3, true],
        ['lt', 2, 3, true],
        ['lte', 2, 2, true],
        ['gt', 3, 2, true],
        ['gte', 2, 2, true],
    ] as const)('supports %s numeric comparisons', (operator, actual, target, expected) => {
        expect(compareGoalValues(actual, operator, target)).toBe(expected);
    });

    it('rejects duplicate ids, recursion, unavailable tools, unsafe paths, and non-final comparison steps', () => {
        const available = (name: string) => name !== 'unknown_tool';
        const selectTaskStart = (step: GoalStepDescriptor) => (
            available(step.name) ? (() => undefined) : null
        );
        expect(buildGoalRequest({
            ...goalRequest(),
            steps: [
                { id: 'same', title: 'One', name: 'collect_live_baseline' },
                { id: 'same', title: 'Two', name: 'analyze_live_recorded_analysis' },
            ],
        }, selectTaskStart)).toMatchObject({ error: 'duplicate_goal_step_id' });
        expect(buildGoalRequest({
            ...goalRequest(),
            steps: [{ id: 'recursive', title: 'Recursive', name: 'create_goal' }],
        }, selectTaskStart)).toMatchObject({ error: 'recursive_goal_step' });
        expect(buildGoalRequest({
            ...goalRequest(),
            steps: [{ id: 'unknown', title: 'Unknown', name: 'unknown_tool' }],
        }, selectTaskStart)).toMatchObject({ error: 'goal_step_task_unavailable' });
        expect(buildGoalRequest({
            ...goalRequest(),
            comparison: { ...goalRequest().comparison, result_path: '__proto__.mistake_count' },
        }, selectTaskStart)).toMatchObject({ error: 'invalid_goal_comparison' });
        expect(buildGoalRequest({
            ...goalRequest(),
            comparison: { ...goalRequest().comparison, step_id: 'collect' },
        }, selectTaskStart)).toMatchObject({ error: 'invalid_goal_comparison_step' });
    });

    it('requires TaskStartFunction steps and does not accept legacy descriptors', () => {
        expect(validateGoalRequest(goalRequest())).toMatchObject({ error: 'invalid_goal_steps' });
        const taskStart = jest.fn() as TaskStartFunction;
        const built = buildGoalRequest(goalRequest(), () => taskStart);
        expect(built).toHaveProperty('request');
        if ('request' in built) {
            expect(built.request.steps.every((step) => step.taskStart === taskStart)).toBe(true);
        }
    });

    it('extracts only own values through safe object and array paths', () => {
        expect(extractGoalResultPath({ metrics: [{ count: 4 }] }, 'metrics.0.count')).toBe(4);
        expect(extractGoalResultPath({}, 'constructor.name')).toBeUndefined();
        expect(isSafeGoalResultPath('metrics.0.count')).toBe(true);
        expect(isSafeGoalResultPath('metrics[0].count')).toBe(false);
    });
});

describe('Goal workflow runner', () => {
    it.each([
        [0, 'achieved'],
        [3, 'missed'],
    ] as const)('executes ordered steps and maps mistake count %s to %s', async (actual, status) => {
        const calls: string[] = [];
        const getHandle = renderGoal();
        let run = 0;
        const request = withTaskStarts(goalRequest(), (step) => async (signal) => {
            if (signal.aborted) return;
            calls.push(step.name);
            run += 1;
            getHandle().acceptToolOutput(envelope(
                step.name,
                `run-${run}`,
                step.id === 'mistake_count' ? { mistake_count: actual, page_id: 'newest' } : {},
            ));
        });
        let result: Awaited<ReturnType<GoalHandle['createGoal']>> | undefined;
        await act(async () => { result = await getHandle().createGoal(request); });

        expect(calls).toEqual([
            'collect_live_baseline',
            'analyze_live_recorded_analysis',
            'get_live_analysis_mistake_count',
        ]);
        expect(result).toMatchObject({
            status,
            target: 0,
            actual,
            completed_steps: ['collect', 'analyze', 'mistake_count'],
            source_result: {
                step_id: 'mistake_count',
                tool_name: 'get_live_analysis_mistake_count',
                status: 'complete',
                final: true,
            },
        });
        expect(screen.getByText(new RegExp(`GOAL · ${status}`, 'i'))).toBeInTheDocument();
    });

    it('waits for a matching final envelope before advancing', async () => {
        const calls: string[] = [];
        const getHandle = renderGoal();
        const request = withTaskStarts(goalRequest(), (step) => async () => {
            calls.push(step.id);
            if (step.id === 'collect') {
                getHandle().acceptToolOutput(envelope(
                    step.name,
                    'collect-run',
                    { progress_percent: 20 },
                    false,
                ));
                return;
            }
            getHandle().acceptToolOutput(envelope(
                step.name,
                `${step.id}-run`,
                step.id === 'mistake_count' ? { mistake_count: 0 } : {},
            ));
        });
        let pending!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            pending = getHandle().createGoal(request);
            await Promise.resolve();
        });
        expect(calls).toEqual(['collect']);
        expect(screen.getByText(/GOAL · running/i)).toBeInTheDocument();

        let result: Awaited<typeof pending> | undefined;
        await act(async () => {
            getHandle().acceptToolOutput(envelope(
                'collect_live_baseline',
                'collect-run',
                { status: 'complete' },
            ));
            result = await pending;
        });
        expect(calls).toEqual(['collect', 'analyze', 'mistake_count']);
        expect(result?.status).toBe('achieved');
    });

    it('retries once with identical arguments and retains the failed card after the second failure', async () => {
        jest.useFakeTimers();
        const taskStart = jest.fn<ReturnType<TaskStartFunction>, Parameters<TaskStartFunction>>(
            async () => { throw new Error('offline'); },
        );
        const getHandle = renderGoal();
        const request: GoalRequest = {
            goal: 'Retry goal',
            steps: [{
                id: 'count',
                title: 'Count',
                name: 'get_live_analysis_mistake_count',
                arguments: { limit: 1 },
                taskStart,
            }],
            comparison: {
                step_id: 'count', result_path: 'mistake_count', operator: 'eq', target: 0, metric_label: 'Mistakes',
            },
        };
        let pending!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            pending = getHandle().createGoal(request);
            await Promise.resolve();
        });
        expect(screen.getByText(/retrying/i)).toBeInTheDocument();
        await act(async () => { jest.advanceTimersByTime(1000); });
        let result: Awaited<typeof pending> | undefined;
        await act(async () => { result = await pending; });

        expect(taskStart).toHaveBeenCalledTimes(2);
        expect(getHandle().getSnapshot()?.steps[0].arguments).toEqual({ limit: 1 });
        expect(result).toMatchObject({ status: 'error', failed_step: 'count', error: 'offline' });
        expect(screen.getByText(/GOAL · error/i)).toBeInTheDocument();
        jest.useRealTimers();
    });

    it.each([
        ['empty', null],
        ['missing', {}],
        ['string', { mistake_count: '0' }],
        ['malformed', { mistake_count: [] }],
    ])('retries and fails %s comparison results', async (_label, output) => {
        jest.useFakeTimers();
        const getHandle = renderGoal();
        const taskStart: TaskStartFunction = async () => {
            getHandle().acceptToolOutput(envelope(
                'get_live_analysis_mistake_count',
                'comparison-run',
                output,
            ));
        };
        const request: GoalRequest = {
            goal: 'Comparison validation',
            steps: [{
                id: 'count',
                title: 'Count',
                name: 'get_live_analysis_mistake_count',
                taskStart,
            }],
            comparison: {
                step_id: 'count', result_path: 'mistake_count', operator: 'eq', target: 0, metric_label: 'Mistakes',
            },
        };
        let pending!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            pending = getHandle().createGoal(request);
            await Promise.resolve();
        });
        await act(async () => { jest.advanceTimersByTime(1000); });
        await expect(pending).resolves.toMatchObject({
            status: 'error',
            failed_step: 'count',
            error: 'goal_comparison_value_not_numeric',
        });
        jest.useRealTimers();
    });

    it('invalidates the previous runner when a second goal replaces it', async () => {
        const getHandle = renderGoal();
        const singleStep = (goal: string, final: boolean): GoalRequest => ({
            goal,
            steps: [{
                id: 'count',
                title: 'Count',
                name: 'get_live_analysis_mistake_count',
                taskStart: async () => {
                    getHandle().acceptToolOutput(envelope(
                        'get_live_analysis_mistake_count',
                        `${goal}-run`,
                        final ? { mistake_count: 0 } : {},
                        final,
                    ));
                },
            }],
            comparison: {
                step_id: 'count', result_path: 'mistake_count', operator: 'eq', target: 0, metric_label: 'Mistakes',
            },
        });
        let first!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            first = getHandle().createGoal(singleStep('Old goal', false));
            await Promise.resolve();
        });
        let secondResult: Awaited<ReturnType<GoalHandle['createGoal']>> | undefined;
        await act(async () => {
            secondResult = await getHandle().createGoal(singleStep('New goal', true));
        });

        await expect(first).resolves.toMatchObject({ status: 'error', error: 'goal_replaced' });
        expect(secondResult?.status).toBe('achieved');
        expect(getHandle().getSnapshot()?.goal).toBe('New goal');
    });
});

describe('GoalDisplay', () => {
    const snapshot = (status: GoalSnapshot['status'], stepStatus: GoalSnapshot['steps'][number]['status']): GoalSnapshot => ({
        goal: 'Visible goal',
        status,
        steps: [{ id: 'one', title: 'Visible step', name: 'tool', status: stepStatus, attempts: stepStatus === 'retrying' ? 2 : 1 }],
        comparison: { step_id: 'one', result_path: 'value', operator: 'eq', target: 0, metric_label: 'Value' },
        target: 0,
        actual: status === 'running' ? null : 1,
        completed_steps: [],
        source_result: null,
        ...(status === 'error' ? { failed_step: 'one', error: 'failed' } : {}),
    });

    it.each([
        ['running', 'running'],
        ['running', 'retrying'],
        ['achieved', 'completed'],
        ['missed', 'completed'],
        ['error', 'error'],
    ] as const)('renders %s goals with a %s step', (status, stepStatus) => {
        const { unmount } = render(<GoalDisplay snapshot={snapshot(status, stepStatus)} />);
        expect(screen.getByText(new RegExp(`GOAL · ${status}`, 'i'))).toBeInTheDocument();
        expect(screen.getAllByText(new RegExp(stepStatus, 'i')).length).toBeGreaterThan(0);
        unmount();

        const overlay = render(
            <GoalDisplay snapshot={snapshot(status, stepStatus)} surface="pill" />,
        );
        expect(screen.getByLabelText('Goal')).toHaveClass('ai-chat__goal--pill');
        overlay.unmount();
    });
});
