import {
    GoalRunner,
    buildGoalRequest,
    compareGoalValues,
    extractGoalResultPath,
    validateGoalRequest,
    type GoalRequest,
} from '../Goal';
import { createAiToolOperationFrom, resolvedAiToolOperation } from '../ai-tool-operation';

const request = (): GoalRequest => ({
    name: 'Drive a clean lap',
    steps: [
        { id: 'collect', title: 'Collect baseline', name: 'collect' },
        { id: 'analyze', title: 'Analyze baseline', name: 'analyze', arguments: { limit: 4 } },
    ],
    determination: {
        tool: { name: 'determine' },
        result_path: 'mistake_count',
        operator: 'eq',
        target: 0,
    },
});

describe('Goal descriptors', () => {
    it('validates descriptor-only requests and safe result paths', () => {
        expect(buildGoalRequest(request())).toEqual({ request: request() });
        expect(validateGoalRequest({
            ...request(),
            steps: [{ id: 'nested', title: 'Nested', name: 'create_goal' }],
        })).toHaveProperty('error');
        expect(extractGoalResultPath({ result: [{ value: 2 }] }, 'result.0.value')).toBe(2);
        expect(compareGoalValues(2, 'lte', 2)).toBe(true);
    });
});

describe('GoalRunner central dispatch callback', () => {
    it('executes ordered steps and determination through the injected dispatcher', async () => {
        const order: string[] = [];
        const dispatch = jest.fn((name: string, args?: Record<string, unknown>) => {
            order.push(args?.limit ? `${name}:${args.limit}` : name);
            return resolvedAiToolOperation(name === 'determine'
                ? { status: 'ready', mistake_count: 0 }
                : { status: 'complete' });
        });
        const runner = new GoalRunner('goal', dispatch);
        const operation = runner.create(request());

        const result = await operation.result;
        if (result instanceof Error) throw result;

        expect(result).toMatchObject({
            name: 'Drive a clean lap',
            status: 'achieved',
            actual: 0,
            completed_steps: ['collect', 'analyze'],
        });
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
        expect(order).toEqual(['collect', 'analyze:4', 'determine']);
    });

    it('retains a failed step and retries it through the same dispatcher', async () => {
        let attempts = 0;
        const dispatch = jest.fn((name: string) => createAiToolOperationFrom(() => {
            if (name === 'collect' && ++attempts === 1) throw new Error('not ready');
            return name === 'determine'
                ? { status: 'ready', mistake_count: 0 }
                : { status: 'complete' };
        }));
        const runner = new GoalRunner('goal', dispatch);

        const failedOperation = runner.create(request());
        const failedResult = await failedOperation.result;
        if (failedResult instanceof Error) throw failedResult;
        const failedProgress = await Promise.all(failedOperation.statuses);

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
            },
        ]);
        expect(failedProgress[0]).toMatchObject({
            id: 'collect',
            status: 'failed',
            error: 'not ready',
        });
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
                { step_id: 'collect', tool_name: 'collect', attempt: 1, status: 'error' },
                { step_id: 'collect', tool_name: 'collect', attempt: 2, status: 'completed' },
                { step_id: 'analyze', tool_name: 'analyze', attempt: 1, status: 'completed' },
            ]);
        expect(attempts).toBe(2);
    });
});
