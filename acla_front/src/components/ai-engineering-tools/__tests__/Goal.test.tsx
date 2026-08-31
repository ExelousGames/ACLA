import React from 'react';
import { render, screen } from '@testing-library/react';
import {
    GoalDisplay,
    GoalRunner,
    goalOverlayRenderer,
    buildGoalRequest,
    compareGoalValues,
    validateGoalRequest,
    type AiToolDispatcher,
    type GoalRequest,
    type NestedAiToolResult,
} from '../Goal';
import {
    createAiToolDeferred,
    createAiToolOperation,
    createAiToolOperationFrom,
    resolvedAiToolOperation,
} from '../ai-tool-operation';
import { InvalidGoalDeterminationError } from '../../../contexts/AiToolComponentError';
import { isJsonSafe } from '../../../views/floating-chat/ai-overlay-types';

const request = (): GoalRequest => ({
    name: 'Drive a clean lap',
    steps: [
        { id: 'collect', title: 'Collect baseline', name: 'collect' },
        { id: 'analyze', title: 'Analyze baseline', name: 'analyze', arguments: { limit: 4 } },
    ],
    determination: {
        tool: { name: 'determine' },
        operator: 'eq',
        target: 0,
    },
});

const operationWithValue = (value: unknown) => (
    resolvedAiToolOperation(value as NestedAiToolResult)
);

describe('GoalDisplay', () => {
    it('removes an achieved goal from chat while keeping it available on pill surfaces', () => {
        const achievedSnapshot = {
            name: 'Drive a clean lap',
            status: 'achieved' as const,
            steps: [],
            determination: request().determination,
            determination_result: {
                tool_name: 'determine',
                attempt: 1,
                status: 'completed' as const,
                value: 0,
            },
            target: 0,
            actual: 0,
            completed_steps: [],
        };
        const { rerender } = render(
            <GoalDisplay snapshot={{ ...achievedSnapshot, status: 'running' }} surface="chat" />,
        );

        expect(screen.getByLabelText('Goal')).toBeInTheDocument();

        rerender(<GoalDisplay snapshot={achievedSnapshot} surface="chat" />);
        expect(screen.queryByLabelText('Goal')).not.toBeInTheDocument();

        rerender(<GoalDisplay snapshot={achievedSnapshot} surface="pill" />);
        expect(screen.getByLabelText('Goal')).toBeInTheDocument();
    });

    it('uses a dedicated overlay card that only renders the active step', () => {
        const snapshot = {
            name: 'Drive a clean lap',
            status: 'running' as const,
            steps: [
                {
                    id: 'collect',
                    title: 'Collect baseline',
                    name: 'collect',
                    status: 'completed' as const,
                    attempts: 1,
                    run_id: 'run-1',
                    error: null,
                },
                {
                    id: 'analyze',
                    title: 'Analyze baseline',
                    name: 'analyze',
                    status: 'running' as const,
                    attempts: 2,
                    run_id: 'run-2',
                    error: null,
                },
                {
                    id: 'report',
                    title: 'Build report',
                    name: 'report',
                    status: 'pending' as const,
                    attempts: 0,
                    run_id: null,
                    error: null,
                },
            ],
            determination: request().determination,
            determination_result: {
                tool_name: 'determine',
                attempt: 0,
                status: 'pending' as const,
                value: null,
            },
            target: 0,
            actual: null,
            completed_steps: ['collect'],
        };

        const overlay = goalOverlayRenderer.renderOverlay(snapshot, 'expanded', {
            componentName: 'goal',
            revision: 1,
            emitRendererEvent: jest.fn(),
        });
        render(<>{overlay}</>);

        expect(screen.getByTestId('goal-overlay')).toBeInTheDocument();
        expect(screen.getByText('Analyze baseline')).toBeInTheDocument();
        expect(screen.getByText('Step 2 of 3 · Attempt 2')).toBeInTheDocument();
        expect(screen.queryByText('Collect baseline')).not.toBeInTheDocument();
        expect(screen.queryByText('Build report')).not.toBeInTheDocument();
        expect(screen.getByTestId('goal-overlay')).not.toHaveClass('ai-chat__goal');
    });
});

describe('Goal descriptors', () => {
    it('validates the clean-break determination shape and configurable tool names', () => {
        expect(buildGoalRequest(request())).toEqual({ request: request() });
        expect(buildGoalRequest({
            ...request(),
            determination: {
                ...request().determination,
                tool: { name: 'session_configured_numeric_tool' },
            },
        })).toHaveProperty('request');
        expect(validateGoalRequest({
            ...request(),
            steps: [{ id: 'nested', title: 'Nested', name: 'create_goal' }],
        })).toHaveProperty('error');
        expect(compareGoalValues(2, 'lte', 2)).toBe(true);
    });

    it('rejects unexpected determination properties', () => {
        const validation = validateGoalRequest({
            ...request(),
            determination: {
                ...request().determination,
                unexpected: true,
            },
        });

        expect(validation).toHaveProperty('error');
        if ('error' in validation) {
            expect(validation.error).toBeInstanceOf(InvalidGoalDeterminationError);
        }
    });
});

describe('GoalRunner central dispatch callback', () => {
    it('publishes overlay-safe running steps with a stable run id and defined error', async () => {
        const collect = createAiToolDeferred<NestedAiToolResult>();
        const dispatch: AiToolDispatcher = jest.fn((name: string) => (
            name === 'collect'
                ? createAiToolOperation(collect.promise)
                : operationWithValue(name === 'determine'
                    ? { status: 'ready', data: 0 }
                    : { status: 'complete' })
        ));
        const runner = new GoalRunner('goal', dispatch);
        const operation = runner.createGoal(request());

        expect(runner.getComponentName()).toBe('goal');
        expect(runner.getComponentType()).toBe('goal');
        expect(runner.getOverlayBehavior(null)).toEqual({
            placement: 'flow',
            requestedStatus: 'expanded',
            remove: true,
        });
        expect(runner.getOverlayMetadata()).toEqual({});
        expect(runner.handleOverlayRendererEvent({} as any)).toBeUndefined();

        const runningSnapshot = runner.getSnapshot();
        expect(isJsonSafe(runningSnapshot)).toBe(true);
        expect(runningSnapshot?.steps[0]).toMatchObject({
            status: 'running',
            run_id: expect.stringMatching(/^goal-/),
            error: null,
        });
        const runningRunId = runningSnapshot?.steps[0].run_id;

        collect.resolve({ status: 'complete' } as NestedAiToolResult);
        const result = await operation.result;
        if (result instanceof Error) throw result;

        expect(result).toMatchObject({ goal: 'Drive a clean lap', status: 'achieved' });
        expect(result).not.toHaveProperty('name');

        const completedSnapshot = runner.getSnapshot();
        expect(isJsonSafe(completedSnapshot)).toBe(true);
        expect(runner.getOverlayBehavior(completedSnapshot)).toMatchObject({ remove: true });
        expect(completedSnapshot?.steps[0]).toMatchObject({
            status: 'completed',
            run_id: runningRunId,
            error: null,
        });
    });

    it('executes ordered steps and achieves a goal from a numeric query envelope', async () => {
        const input: GoalRequest = {
            ...request(),
            determination: {
                ...request().determination,
                tool: {
                    name: 'query_analysis_result',
                    arguments: { query: '$count(analyses)' },
                },
            },
        };
        const order: string[] = [];
        const dispatch = jest.fn((name: string, args?: Record<string, unknown>) => {
            order.push(args?.limit ? `${name}:${args.limit}` : name);
            return operationWithValue(name === 'query_analysis_result'
                ? { status: 'ready', data: 0 }
                : { status: 'complete' });
        });
        const runner = new GoalRunner('goal', dispatch);
        const operation = runner.create(input);

        const result = await operation.result;
        if (result instanceof Error) throw result;

        expect(result).toMatchObject({
            name: 'Drive a clean lap',
            status: 'achieved',
            actual: 0,
            completed_steps: ['collect', 'analyze'],
            determination: {
                tool: {
                    name: 'query_analysis_result',
                    arguments: { query: '$count(analyses)' },
                },
                operator: 'eq',
                target: 0,
            },
        });
        expect(result.determination).toEqual(input.determination);
        expect(runner.getSnapshot()?.determination).toEqual(input.determination);
        expect(result.task_results).toEqual([
            {
                step_id: 'collect',
                tool_name: 'collect',
                attempt: 1,
                status: 'completed',
                source_result: {
                    step_id: 'collect',
                    tool_name: 'collect',
                    run_id: expect.any(String),
                    status: 'complete',
                },
            },
            {
                step_id: 'analyze',
                tool_name: 'analyze',
                attempt: 1,
                status: 'completed',
                source_result: {
                    step_id: 'analyze',
                    tool_name: 'analyze',
                    run_id: expect.any(String),
                    status: 'complete',
                },
            },
        ]);
        expect(order).toEqual(['collect', 'analyze:4', 'query_analysis_result']);
        expect(dispatch).toHaveBeenLastCalledWith('query_analysis_result', {
            query: '$count(analyses)',
        });
    });

    it('keeps rerunning a missed numeric-envelope goal until it is achieved', async () => {
        let determinations = 0;
        const dispatch = jest.fn((name: string) => operationWithValue(
            name === 'determine'
                ? { status: 'ready', data: ++determinations < 3 ? 1 : 0 }
                : { status: 'complete' },
        ));
        const snapshots: Array<{ status: string; actual: number | null }> = [];
        const runner = new GoalRunner('goal', dispatch, (snapshot) => {
            if (snapshot) snapshots.push({ status: snapshot.status, actual: snapshot.actual });
        });
        const operation = runner.create(request());

        const result = await operation.result;
        if (result instanceof Error) throw result;

        expect(result.status).toBe('achieved');
        expect(determinations).toBe(3);
        expect(result.task_results).toHaveLength(6);
        expect(snapshots).toEqual(expect.arrayContaining([
            { status: 'missed', actual: 1 },
            { status: 'achieved', actual: 0 },
        ]));
    });

    it.each([
        ['missing data', { status: 'ready' }],
        ['legacy top-level count', { status: 'ready', mistake_count: 0 }],
        ['structured telemetry data', { status: 'ready', data: { speed: { avg: 120 } } }],
        ['invalid status', { status: 'complete', data: 0 }],
        ['missing status', { data: 0 }],
        ['numeric string', { status: 'ready', data: '0' }],
        ['NaN', { status: 'ready', data: Number.NaN }],
        ['positive infinity', { status: 'ready', data: Number.POSITIVE_INFINITY }],
        ['negative infinity', { status: 'ready', data: Number.NEGATIVE_INFINITY }],
        ['null', null],
        ['ordinary non-query output', { status: 'complete' }],
    ])('fails determination for incompatible %s output', async (_description, output) => {
        const dispatch: AiToolDispatcher = jest.fn((name: string) => operationWithValue(
            name === 'determine' ? output : { status: 'complete' },
        ));
        const runner = new GoalRunner('goal', dispatch);

        const result = await runner.create(request()).result;
        if (result instanceof Error) throw result;

        expect(result).toMatchObject({
            status: 'failed',
            actual: null,
            completed_steps: ['collect', 'analyze'],
            error: 'Goal determination requires a ready query result with finite numeric data.',
            determination_result: {
                tool_name: 'determine',
                attempt: 1,
                status: 'error',
                value: null,
                error: 'Goal determination requires a ready query result with finite numeric data.',
            },
        });
    });

    it('retries only determination after incompatible input', async () => {
        let determinationAttempts = 0;
        const dispatch = jest.fn((name: string) => operationWithValue(
            name === 'determine'
                ? (++determinationAttempts === 1
                    ? { status: 'ready', data: '0' }
                    : { status: 'ready', data: 0 })
                : { status: 'complete' },
        ));
        const runner = new GoalRunner('goal', dispatch);

        const failedResult = await runner.create(request()).result;
        if (failedResult instanceof Error) throw failedResult;
        expect(failedResult.status).toBe('failed');

        const retryResult = await runner.retryFailedTask().result;
        if (retryResult instanceof Error) throw retryResult;

        expect(retryResult).toMatchObject({
            status: 'achieved',
            actual: 0,
            determination_result: { attempt: 2, value: 0 },
        });
        expect(dispatch.mock.calls.map(([name]) => name)).toEqual([
            'collect',
            'analyze',
            'determine',
            'determine',
        ]);
        expect(retryResult.task_results).toHaveLength(2);
    });

    it('reports a rejected determination operation as an execution failure', async () => {
        const dispatch = jest.fn((name: string) => createAiToolOperationFrom(() => {
            if (name === 'determine') throw new Error('determination exploded');
            return { status: 'complete' };
        }));
        const runner = new GoalRunner('goal', dispatch);

        const result = await runner.create(request()).result;
        if (result instanceof Error) throw result;

        expect(result).toMatchObject({
            status: 'failed',
            actual: null,
            error: 'determination exploded',
            determination_result: {
                status: 'error',
                error: 'determination exploded',
                source_result: { status: 'failed' },
            },
        });
    });

    it('retains a failed step and retries it through the same dispatcher', async () => {
        let attempts = 0;
        const dispatch = jest.fn((name: string) => createAiToolOperationFrom(() => {
            if (name === 'collect' && ++attempts === 1) throw new Error('not ready');
            return name === 'determine'
                ? { status: 'ready', data: 0 }
                : { status: 'complete' };
        }));
        const runner = new GoalRunner('goal', dispatch);

        const failedOperation = runner.create(request());
        const failedResult = await failedOperation.result;
        if (failedResult instanceof Error) throw failedResult;

        expect(failedOperation.statuses).toEqual([]);
        expect(failedResult).toMatchObject({
            status: 'failed',
            failed_step: 'collect',
            error: 'not ready',
        });
        expect(failedResult.task_results).toEqual([
            {
                step_id: 'collect',
                tool_name: 'collect',
                attempt: 1,
                status: 'error',
                source_result: {
                    step_id: 'collect',
                    tool_name: 'collect',
                    run_id: expect.any(String),
                    status: 'failed',
                },
                error: {
                    name: 'GoalStepFailedError',
                    message: 'not ready',
                    cause: {
                        name: 'Error',
                        message: 'not ready',
                    },
                },
            },
        ]);
        const failedSnapshot = runner.getSnapshot();
        expect(failedSnapshot).toMatchObject({
            failed_step: 'collect',
            error: 'not ready',
        });
        expect(failedSnapshot?.steps[0]).toMatchObject({ id: 'collect', error: 'not ready' });

        const retryResult = await runner.retryFailedTask().result;
        if (retryResult instanceof Error) throw retryResult;

        expect(retryResult.status).toBe('achieved');
        expect(retryResult.task_results.map(({ source_result: _sourceResult, ...result }) => result))
            .toEqual([
                {
                    step_id: 'collect',
                    tool_name: 'collect',
                    attempt: 1,
                    status: 'error',
                    error: {
                        name: 'GoalStepFailedError',
                        message: 'not ready',
                        cause: {
                            name: 'Error',
                            message: 'not ready',
                        },
                    },
                },
                { step_id: 'collect', tool_name: 'collect', attempt: 2, status: 'completed' },
                { step_id: 'analyze', tool_name: 'analyze', attempt: 1, status: 'completed' },
            ]);
        expect(attempts).toBe(2);
    });
});
