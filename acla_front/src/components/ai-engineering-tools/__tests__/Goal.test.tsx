import React, { useLayoutEffect } from 'react';
import { act, render, screen, within } from '@testing-library/react';
import {
    AiToolComponentRefProvider,
    createAiToolComponentRefDirectory,
    type AiToolComponentRefDirectory,
    useAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import {
    GoalClearedError,
    GoalComponentError,
    GoalDeterminationFailedError,
    GoalDeterminationTaskUnavailableError,
    GoalDeterminationValueNotNumericError,
    GoalReplacedError,
    GoalStepFailedError,
    GoalStepOutputToolMismatchError,
    GoalStepTaskUnavailableError,
    GoalTaskRetryUnavailableError,
    InvalidGoalStepsError,
    InvalidGoalDeterminationError,
    DuplicateGoalStepIdError,
    RecursiveGoalDeterminationError,
    RecursiveGoalStepError,
} from 'contexts/AiToolComponentError';
import Goal, {
    GoalDisplay,
    GoalRunner,
    buildGoalRequest,
    compareGoalValues,
    extractGoalResultPath,
    isSafeGoalResultPath,
    validateGoalRequest,
    type GoalExecutableRequest,
    type GoalHandle,
    type GoalRequest,
    type GoalSnapshot,
    type GoalTaskDescriptor,
    type GoalToolOutputEnvelope,
} from '../Goal';
import type { TaskStartFunction } from '../task-start-function';

const goalRequest = (name = 'No mistakes in the last analyzed lap'): GoalRequest => ({
    name,
    steps: [
        { id: 'collect', title: 'Collect baseline', name: 'collect_live_baseline' },
        { id: 'analyze', title: 'Analyze baseline', name: 'analyze_live_recorded_analysis' },
    ],
    determination: {
        tool: { name: 'get_live_analysis_mistake_count' },
        result_path: 'mistake_count',
        operator: 'eq',
        target: 0,
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

const executableGoal = (
    request: GoalRequest,
    selectTaskStart: (task: GoalTaskDescriptor) => TaskStartFunction,
): GoalExecutableRequest => {
    const built = buildGoalRequest(request, selectTaskStart);
    if ('error' in built) throw built.error;
    return built.request;
};

const captureGoalFailure = async (run: () => Promise<unknown>): Promise<GoalComponentError> => {
    let failure: unknown;
    await act(async () => {
        try {
            await run();
        } catch (error) {
            failure = error;
        }
    });
    expect(failure).toBeInstanceOf(GoalComponentError);
    return failure as GoalComponentError;
};

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

    it('validates and clones the nested public request', () => {
        const input = goalRequest();
        const validated = validateGoalRequest(input);
        expect(validated).toEqual({ request: input });
        if ('request' in validated) {
            expect(validated.request).not.toBe(input);
            expect(validated.request.steps).not.toBe(input.steps);
            expect(validated.request.determination).not.toBe(input.determination);
        }
    });

    it('requires preparation work and rejects legacy and unknown fields', () => {
        expect((validateGoalRequest({
            ...goalRequest(),
            steps: [],
        }) as any).error).toBeInstanceOf(InvalidGoalStepsError);
        expect(validateGoalRequest({
            goal: 'Legacy goal',
            steps: [{ id: 'count', title: 'Count', name: 'get_live_analysis_mistake_count' }],
            comparison: {
                step_id: 'count',
                result_path: 'mistake_count',
                operator: 'eq',
                target: 0,
                metric_label: 'Mistakes',
                unit: 'mistakes',
            },
        })).toHaveProperty('error');
        expect(validateGoalRequest({
            ...goalRequest(),
            metric_label: 'Mistakes',
        })).toHaveProperty('error');
        expect((validateGoalRequest({
            ...goalRequest(),
            determination: { ...goalRequest().determination, unit: 'mistakes' },
        }) as any).error).toBeInstanceOf(InvalidGoalDeterminationError);
    });

    it('rejects duplicate ids, unsafe paths, invalid targets, and recursion in either phase', () => {
        expect((validateGoalRequest({
            ...goalRequest(),
            steps: [
                { id: 'same', title: 'One', name: 'collect_live_baseline' },
                { id: 'same', title: 'Two', name: 'analyze_live_recorded_analysis' },
            ],
        }) as any).error).toBeInstanceOf(DuplicateGoalStepIdError);
        expect((validateGoalRequest({
            ...goalRequest(),
            steps: [{ id: 'recursive', title: 'Recursive', name: 'create_goal' }],
        }) as any).error).toBeInstanceOf(RecursiveGoalStepError);
        expect((validateGoalRequest({
            ...goalRequest(),
            determination: {
                ...goalRequest().determination,
                tool: { name: 'retry_goal_task' },
            },
        }) as any).error).toBeInstanceOf(RecursiveGoalDeterminationError);
        expect((validateGoalRequest({
            ...goalRequest(),
            determination: {
                ...goalRequest().determination,
                result_path: '__proto__.mistake_count',
            },
        }) as any).error).toBeInstanceOf(InvalidGoalDeterminationError);
        expect((validateGoalRequest({
            ...goalRequest(),
            determination: { ...goalRequest().determination, target: Number.NaN },
        }) as any).error).toBeInstanceOf(InvalidGoalDeterminationError);
    });

    it('resolves preparation and determination functions with the goal name as its title', () => {
        const taskStart = jest.fn() as TaskStartFunction;
        const selector = jest.fn(() => taskStart);
        const built = buildGoalRequest(goalRequest('Clean-lap target'), selector);
        expect(built).toHaveProperty('request');
        if ('request' in built) {
            expect(built.request.steps.every((step) => step.taskStart === taskStart)).toBe(true);
            expect(built.request.determination.taskStart).toBe(taskStart);
        }
        expect(selector).toHaveBeenLastCalledWith({
            title: 'Clean-lap target',
            name: 'get_live_analysis_mistake_count',
        });
    });

    it('rejects unavailable preparation and determination tools independently', () => {
        const unavailableStep = buildGoalRequest(goalRequest(), (task) => (
            task.name === 'analyze_live_recorded_analysis' ? null : () => undefined
        ));
        const unavailableDetermination = buildGoalRequest(goalRequest(), (task) => (
            task.name === 'get_live_analysis_mistake_count' ? null : () => undefined
        ));
        expect((unavailableStep as any).error).toBeInstanceOf(GoalStepTaskUnavailableError);
        expect((unavailableDetermination as any).error)
            .toBeInstanceOf(GoalDeterminationTaskUnavailableError);
    });

    it('extracts only own values through safe object and array paths', () => {
        expect(extractGoalResultPath({ metrics: [{ count: 4 }] }, 'metrics.0.count')).toBe(4);
        expect(extractGoalResultPath({}, 'constructor.name')).toBeUndefined();
        expect(isSafeGoalResultPath('metrics.0.count')).toBe(true);
        expect(isSafeGoalResultPath('metrics[0].count')).toBe(false);
    });
});

describe('Goal workflow runner', () => {
    it('executes preparation in order, then reports a separate determination', async () => {
        const calls: string[] = [];
        const getHandle = renderGoal();
        let run = 0;
        const request = executableGoal(goalRequest(), (task) => async (signal) => {
            if (signal.aborted) return;
            calls.push(task.name);
            run += 1;
            getHandle().acceptToolOutput(envelope(
                task.name,
                `run-${run}`,
                task.name === 'get_live_analysis_mistake_count'
                    ? { mistake_count: 0, page_id: 'newest' }
                    : { status: 'ready' },
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
            name: goalRequest().name,
            status: 'achieved',
            target: 0,
            actual: 0,
            completed_steps: ['collect', 'analyze'],
            task_results: [
                expect.objectContaining({ step_id: 'collect', attempt: 1, status: 'completed' }),
                expect.objectContaining({ step_id: 'analyze', attempt: 1, status: 'completed' }),
            ],
            determination_result: {
                tool_name: 'get_live_analysis_mistake_count',
                attempt: 1,
                status: 'completed',
                value: 0,
                source_result: expect.objectContaining({ run_id: 'run-3' }),
            },
        });
        expect(result?.task_results).toHaveLength(2);
        expect(result?.determination_result?.source_result).not.toHaveProperty('step_id');
        expect(screen.getByText(/GOAL · achieved/i)).toBeInTheDocument();
    });

    it('normalizes immediate preparation values and extracts the determination value', async () => {
        const getHandle = renderGoal();
        const request = executableGoal(goalRequest(), (task) => (
            task.name === 'get_live_analysis_mistake_count'
                ? async () => ({ mistake_count: 0 })
                : () => undefined
        ));
        let result: Awaited<ReturnType<GoalHandle['createGoal']>> | undefined;
        await act(async () => { result = await getHandle().createGoal(request); });
        expect(result).toMatchObject({
            status: 'achieved',
            task_results: [
                expect.objectContaining({ step_id: 'collect', value: null }),
                expect.objectContaining({ step_id: 'analyze', value: null }),
            ],
            determination_result: expect.objectContaining({ value: 0 }),
        });
    });

    it.each([
        ['eq', 2, 2, 'achieved'],
        ['neq', 2, 3, 'achieved'],
        ['lt', 2, 3, 'achieved'],
        ['lte', 2, 2, 'achieved'],
        ['gt', 3, 2, 'achieved'],
        ['gte', 2, 2, 'achieved'],
    ] as const)('runs the %s determination comparison', async (operator, actual, target, status) => {
        const getHandle = renderGoal();
        const request = executableGoal({
            ...goalRequest(`${operator} goal`),
            determination: { ...goalRequest().determination, operator, target },
        }, (task) => () => (
            task.name === 'get_live_analysis_mistake_count' ? { mistake_count: actual } : null
        ));
        let result: Awaited<ReturnType<GoalHandle['createGoal']>> | undefined;
        await act(async () => { result = await getHandle().createGoal(request); });
        expect(result).toMatchObject({ status, actual });
    });

    it('reruns all preparation and determination after the first miss', async () => {
        jest.useFakeTimers();
        const calls: string[] = [];
        const getHandle = renderGoal();
        let determinationAttempt = 0;
        const request = executableGoal(goalRequest(), (task) => () => {
            calls.push(task.name);
            if (task.name === 'get_live_analysis_mistake_count') {
                determinationAttempt += 1;
                return { mistake_count: determinationAttempt === 1 ? 3 : 0 };
            }
            return { status: 'ready' };
        });
        let pending!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            pending = getHandle().createGoal(request);
            await Promise.resolve();
        });
        expect(getHandle().getSnapshot()).toMatchObject({
            status: 'missed',
            actual: 3,
            completed_steps: ['collect', 'analyze'],
        });

        let result: Awaited<typeof pending> | undefined;
        await act(async () => {
            jest.advanceTimersByTime(1000);
            result = await pending;
        });
        expect(calls).toEqual([
            'collect_live_baseline', 'analyze_live_recorded_analysis', 'get_live_analysis_mistake_count',
            'collect_live_baseline', 'analyze_live_recorded_analysis', 'get_live_analysis_mistake_count',
        ]);
        expect(result).toMatchObject({
            status: 'achieved',
            actual: 0,
            determination_result: expect.objectContaining({ attempt: 2 }),
        });
        jest.useRealTimers();
    });

    it('returns missed after the full retry and never adds determination to step results', async () => {
        jest.useFakeTimers();
        const getHandle = renderGoal();
        const request = executableGoal(goalRequest(), (task) => () => (
            task.name === 'get_live_analysis_mistake_count'
                ? { mistake_count: 2 }
                : { status: 'ready' }
        ));
        let pending!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            pending = getHandle().createGoal(request);
            await Promise.resolve();
        });
        await act(async () => { jest.advanceTimersByTime(1000); });
        await expect(pending).resolves.toMatchObject({
            status: 'missed',
            actual: 2,
            completed_steps: ['collect', 'analyze'],
            task_results: [
                expect.objectContaining({ step_id: 'collect', attempt: 1 }),
                expect.objectContaining({ step_id: 'analyze', attempt: 1 }),
                expect.objectContaining({ step_id: 'collect', attempt: 2 }),
                expect.objectContaining({ step_id: 'analyze', attempt: 2 }),
            ],
            determination_result: expect.objectContaining({ attempt: 2, value: 2 }),
        });
        jest.useRealTimers();
    });

    it('deletes a directly registered runner after a missed goal completes', async () => {
        jest.useFakeTimers();
        const directory = createAiToolComponentRefDirectory();
        const runner = new GoalRunner('goal');
        runner.addComponentRef(directory);
        const request = executableGoal(goalRequest('Registered miss'), (task) => () => (
            task.name === 'get_live_analysis_mistake_count'
                ? { mistake_count: 2 }
                : { status: 'ready' }
        ));

        let pending!: ReturnType<GoalRunner['create']>;
        await act(async () => {
            pending = runner.create(request);
            await Promise.resolve();
        });
        expect(directory.findBaseComponentRef()?.current).toBe(runner);
        await act(async () => { jest.advanceTimersByTime(1000); });
        await expect(pending).resolves.toMatchObject({ status: 'missed' });
        expect(directory.findBaseComponentRef()).toBeNull();
        jest.useRealTimers();
    });

    it('waits for matching final envelopes in both phases', async () => {
        const calls: string[] = [];
        const getHandle = renderGoal();
        const request = executableGoal(goalRequest(), (task) => async () => {
            calls.push(task.name);
            getHandle().acceptToolOutput(envelope(
                task.name,
                `${task.name}-run`,
                { progress_percent: 20 },
                false,
            ));
        });
        let pending!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            pending = getHandle().createGoal(request);
            await Promise.resolve();
        });
        expect(calls).toEqual(['collect_live_baseline']);
        await act(async () => {
            getHandle().acceptToolOutput(envelope(
                'collect_live_baseline',
                'collect_live_baseline-run',
                { status: 'ready' },
            ));
            await Promise.resolve();
        });
        expect(calls).toEqual(['collect_live_baseline', 'analyze_live_recorded_analysis']);
        await act(async () => {
            getHandle().acceptToolOutput(envelope(
                'analyze_live_recorded_analysis',
                'analyze_live_recorded_analysis-run',
                { status: 'ready' },
            ));
            await Promise.resolve();
        });
        expect(calls).toEqual([
            'collect_live_baseline',
            'analyze_live_recorded_analysis',
            'get_live_analysis_mistake_count',
        ]);
        await act(async () => {
            getHandle().acceptToolOutput(envelope(
                'get_live_analysis_mistake_count',
                'get_live_analysis_mistake_count-run',
                { mistake_count: 0 },
            ));
            await pending;
        });
        await expect(pending).resolves.toMatchObject({ status: 'achieved' });
    });

    it('publishes preparation errors and resumes at only the failed step', async () => {
        const getHandle = renderGoal();
        const calls: string[] = [];
        let analyzeAttempts = 0;
        const request = executableGoal(goalRequest(), (task) => () => {
            calls.push(task.name);
            if (task.name === 'analyze_live_recorded_analysis') {
                analyzeAttempts += 1;
                if (analyzeAttempts === 1) throw new Error('analysis_offline');
            }
            return task.name === 'get_live_analysis_mistake_count'
                ? { mistake_count: 0 }
                : { status: 'ready' };
        });
        const firstFailure = await captureGoalFailure(() => getHandle().createGoal(request));
        expect(firstFailure).toBeInstanceOf(GoalStepFailedError);
        expect(firstFailure).toMatchObject({ message: 'analysis_offline' });
        expect(getHandle().getSnapshot()).toMatchObject({
            status: 'error',
            failed_step: 'analyze',
            error: 'analysis_offline',
            determination_result: expect.objectContaining({ status: 'pending' }),
        });

        let result: Awaited<ReturnType<GoalHandle['retryFailedTask']>> | undefined;
        await act(async () => { result = await getHandle().retryFailedTask(); });
        expect(calls).toEqual([
            'collect_live_baseline',
            'analyze_live_recorded_analysis',
            'analyze_live_recorded_analysis',
            'get_live_analysis_mistake_count',
        ]);
        expect(result).toMatchObject({
            status: 'achieved',
            completed_steps: ['collect', 'analyze'],
            task_results: [
                expect.objectContaining({ step_id: 'collect', attempt: 1, status: 'completed' }),
                expect.objectContaining({ step_id: 'analyze', attempt: 1, status: 'error' }),
                expect.objectContaining({ step_id: 'analyze', attempt: 2, status: 'completed' }),
            ],
        });
    });

    it('represents determination failures separately and retries only determination', async () => {
        const getHandle = renderGoal();
        const calls: string[] = [];
        let determinationAttempts = 0;
        const request = executableGoal(goalRequest(), (task) => () => {
            calls.push(task.name);
            if (task.name === 'get_live_analysis_mistake_count') {
                determinationAttempts += 1;
                if (determinationAttempts === 1) throw new Error('count_offline');
                return { mistake_count: 0 };
            }
            return { status: 'ready' };
        });
        const failure = await captureGoalFailure(() => getHandle().createGoal(request));
        expect(failure).toBeInstanceOf(GoalDeterminationFailedError);
        expect(failure).toMatchObject({ message: 'count_offline' });
        const failedSnapshot = getHandle().getSnapshot();
        expect(failedSnapshot).toMatchObject({
            status: 'error',
            completed_steps: ['collect', 'analyze'],
            determination_result: {
                tool_name: 'get_live_analysis_mistake_count',
                attempt: 1,
                status: 'error',
                value: null,
                error: 'count_offline',
            },
        });
        expect(failedSnapshot).not.toHaveProperty('failed_step');
        expect(failedSnapshot).not.toHaveProperty('error');

        let result: Awaited<ReturnType<GoalHandle['retryFailedTask']>> | undefined;
        await act(async () => { result = await getHandle().retryFailedTask(); });
        expect(calls).toEqual([
            'collect_live_baseline',
            'analyze_live_recorded_analysis',
            'get_live_analysis_mistake_count',
            'get_live_analysis_mistake_count',
        ]);
        expect(result).toMatchObject({
            status: 'achieved',
            task_results: expect.arrayContaining([
                expect.objectContaining({ step_id: 'collect' }),
                expect.objectContaining({ step_id: 'analyze' }),
            ]),
            determination_result: expect.objectContaining({ attempt: 2, value: 0 }),
        });
        expect(result?.task_results).toHaveLength(2);
    });

    it.each([
        ['empty', null],
        ['missing', {}],
        ['string', { mistake_count: '0' }],
        ['infinite', { mistake_count: Number.POSITIVE_INFINITY }],
    ])('fails %s determination values without marking a preparation step', async (_label, output) => {
        const getHandle = renderGoal();
        const determinationTask = jest.fn(() => envelope(
            'get_live_analysis_mistake_count',
            'determination-run',
            output,
        ));
        const request = executableGoal(goalRequest(), (task) => (
            task.name === 'get_live_analysis_mistake_count'
                ? determinationTask
                : () => ({ status: 'ready' })
        ));
        const failure = await captureGoalFailure(() => getHandle().createGoal(request));
        expect(failure).toBeInstanceOf(GoalDeterminationValueNotNumericError);
        expect(failure).toMatchObject({ componentName: 'goal' });
        const snapshot = getHandle().getSnapshot();
        expect(snapshot).toMatchObject({
            status: 'error',
            determination_result: {
                status: 'error',
                error: "Goal determination path 'mistake_count' did not resolve to a finite number.",
                source_result: expect.objectContaining({ run_id: 'determination-run' }),
            },
        });
        expect(snapshot).not.toHaveProperty('failed_step');
        expect(determinationTask).toHaveBeenCalledTimes(1);
    });

    it('rejects mismatched final envelopes with phase-specific failures', async () => {
        const getHandle = renderGoal();
        const request = executableGoal(goalRequest(), (task) => () => envelope(
            task.name === 'collect_live_baseline' ? 'wrong_tool' : task.name,
            'mismatch-run',
            { mistake_count: 0 },
        ));
        const failure = await captureGoalFailure(() => getHandle().createGoal(request));
        expect(failure).toBeInstanceOf(GoalStepOutputToolMismatchError);
        expect(getHandle().getSnapshot()).toMatchObject({ failed_step: 'collect' });
    });

    it('rejects explicit retry while running, after success, and after a miss', async () => {
        const runningHandle = renderGoal();
        const runningRequest = executableGoal(goalRequest('Running goal'), (task) => () => envelope(
            task.name,
            'pending-run',
            { progress: 20 },
            false,
        ));
        let running!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            running = runningHandle().createGoal(runningRequest);
            await Promise.resolve();
        });
        await expect(runningHandle().retryFailedTask()).rejects.toBeInstanceOf(
            GoalTaskRetryUnavailableError,
        );
        act(() => runningHandle().clear());
        await expect(running).rejects.toBeInstanceOf(GoalClearedError);

        const achievedHandle = renderGoal();
        const achieved = executableGoal(goalRequest('Achieved'), (task) => () => (
            task.name === 'get_live_analysis_mistake_count' ? { mistake_count: 0 } : null
        ));
        await act(async () => { await achievedHandle().createGoal(achieved); });
        await expect(achievedHandle().retryFailedTask()).rejects.toBeInstanceOf(
            GoalTaskRetryUnavailableError,
        );

        jest.useFakeTimers();
        const missedHandle = renderGoal();
        const missed = executableGoal(goalRequest('Missed'), (task) => () => (
            task.name === 'get_live_analysis_mistake_count' ? { mistake_count: 2 } : null
        ));
        let pendingMiss!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            pendingMiss = missedHandle().createGoal(missed);
            await Promise.resolve();
        });
        await act(async () => {
            jest.advanceTimersByTime(1000);
            await pendingMiss;
        });
        await expect(missedHandle().retryFailedTask()).rejects.toBeInstanceOf(
            GoalTaskRetryUnavailableError,
        );
        jest.useRealTimers();
    });

    it('cancels a cleared goal and rejects a replaced goal without corrupting the new run', async () => {
        const getHandle = renderGoal();
        const pendingRequest = executableGoal(goalRequest('Old goal'), (task) => () => envelope(
            task.name,
            'pending-run',
            { progress: 20 },
            false,
        ));
        let cleared!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            cleared = getHandle().createGoal(pendingRequest);
            await Promise.resolve();
        });
        const clearedRejection = expect(cleared).rejects.toBeInstanceOf(GoalClearedError);
        act(() => getHandle().clear());
        await clearedRejection;

        let replaced!: ReturnType<GoalHandle['createGoal']>;
        await act(async () => {
            replaced = getHandle().createGoal(pendingRequest);
            await Promise.resolve();
        });
        const replacedRejection = expect(replaced).rejects.toBeInstanceOf(GoalReplacedError);
        const newRequest = executableGoal(goalRequest('New goal'), (task) => () => (
            task.name === 'get_live_analysis_mistake_count' ? { mistake_count: 0 } : null
        ));
        let result: Awaited<ReturnType<GoalHandle['createGoal']>> | undefined;
        await act(async () => { result = await getHandle().createGoal(newRequest); });
        await replacedRejection;
        expect(result).toMatchObject({ status: 'achieved', name: 'New goal' });
        expect(getHandle().getSnapshot()?.name).toBe('New goal');
    });

    it('publishes validation errors while preserving component-error rejection behavior', async () => {
        const getHandle = renderGoal();
        const invalid = {
            ...executableGoal(goalRequest(), () => () => undefined),
            steps: [],
        } as unknown as GoalExecutableRequest;
        const failure = await captureGoalFailure(() => getHandle().createGoal(invalid));
        expect(failure).toBeInstanceOf(InvalidGoalStepsError);
        expect(getHandle().getSnapshot()).toMatchObject({
            status: 'error',
            error: 'Provide at least one valid goal step.',
        });
    });
});

describe('GoalDisplay', () => {
    const snapshot = (
        status: GoalSnapshot['status'],
        stepStatus: GoalSnapshot['steps'][number]['status'],
        determinationStatus: NonNullable<GoalSnapshot['determination_result']>['status'],
    ): GoalSnapshot => ({
        name: 'Visible goal',
        status,
        steps: [{
            id: 'one',
            title: 'Preparation step',
            name: 'prepare_tool',
            status: stepStatus,
            attempts: 1,
        }],
        determination: {
            tool: { name: 'determine_tool' },
            result_path: 'value',
            operator: 'lte',
            target: 3,
        },
        determination_result: {
            tool_name: 'determine_tool',
            attempt: 2,
            status: determinationStatus,
            value: status === 'running' ? null : 2,
            ...(determinationStatus === 'error' ? { error: 'determination failed' } : {}),
        },
        target: 3,
        actual: status === 'running' || status === 'error' ? null : 2,
        completed_steps: stepStatus === 'completed' ? ['one'] : [],
        ...(status === 'error' && stepStatus === 'error'
            ? { failed_step: 'one', error: 'preparation failed' }
            : {}),
    });

    it.each([
        ['running', 'running', 'pending'],
        ['achieved', 'completed', 'completed'],
        ['missed', 'completed', 'completed'],
        ['error', 'error', 'pending'],
    ] as const)('renders %s chat and pill goals with separate determination', (
        status,
        stepStatus,
        determinationStatus,
    ) => {
        const { unmount } = render(
            <GoalDisplay snapshot={snapshot(status, stepStatus, determinationStatus)} />,
        );
        expect(screen.getByText(new RegExp(`GOAL · ${status}`, 'i'))).toBeInTheDocument();
        expect(screen.getByText('Preparation step').closest('li')).toBeInTheDocument();
        const determination = screen.getByLabelText('Determination');
        expect(determination).toHaveTextContent('determine_tool');
        expect(determination).toHaveTextContent(determinationStatus);
        expect(determination).toHaveTextContent('attempt 2');
        expect(determination).toHaveTextContent(status === 'running' || status === 'error'
            ? '— lte 3'
            : '2 lte 3');
        expect(within(screen.getByRole('list')).queryByText(/Determination/)).not.toBeInTheDocument();
        expect(screen.queryByText(/metric|unit/i)).not.toBeInTheDocument();
        unmount();

        const overlay = render(
            <GoalDisplay
                snapshot={snapshot(status, stepStatus, determinationStatus)}
                surface="pill"
            />,
        );
        expect(screen.getByLabelText('Goal')).toHaveClass('ai-chat__goal--pill');
        expect(screen.getByLabelText('Determination')).toBeInTheDocument();
        overlay.unmount();
    });

    it('renders determination errors without a failed preparation marker', () => {
        const value = snapshot('error', 'completed', 'error');
        const { container } = render(<GoalDisplay snapshot={value} />);
        expect(screen.getByLabelText('Determination')).toHaveTextContent('determination failed');
        expect(container.querySelector('.ai-chat__goal-step--error')).toBeNull();
    });
});
