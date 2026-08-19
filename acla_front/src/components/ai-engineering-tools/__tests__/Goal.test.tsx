import {
    GoalRunner,
    buildGoalRequest,
    compareGoalValues,
    validateGoalRequest,
    type AiToolDispatcher,
    type GoalRequest,
    type NestedAiToolResult,
} from '../Goal';
import { createAiToolOperationFrom, resolvedAiToolOperation } from '../ai-tool-operation';
import { InvalidGoalDeterminationError } from '../../../contexts/AiToolComponentError';

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
