import {
    ProcedurePlanRunner,
    advanceProcedurePlan,
    buildProcedurePlan,
    getProcedurePlanToolArguments,
    isProcedurePlanOptOutRequest,
    type ProcedurePlanState,
} from '../ProcedurePlan';
import { ProcedurePlanStepFailedError } from '../../../contexts/AiToolComponentError';
import { createAiToolOperationFrom, resolvedAiToolOperation } from '../ai-tool-operation';

const plan = (): ProcedurePlanState => ({
    goal: 'Review the lap',
    currentStep: 0,
    requests: [
        {
            type: 'tool_call',
            title: 'Read telemetry',
            name: 'read',
            status: 'pending',
            payload: { arguments: { lap: 2 } },
        },
        {
            type: 'tool_call',
            title: 'Compare',
            name: 'compare',
            status: 'pending',
        },
    ],
});

describe('ProcedurePlan descriptors', () => {
    it('builds stable requests and advances the active request', () => {
        expect(buildProcedurePlan({
            goal: 'Review',
            requests: [{ title: 'Read', name: 'read' }],
        })).toMatchObject({
            goal: 'Review',
            currentStep: 0,
            requests: [{ type: 'tool_call', title: 'Read', name: 'read', status: 'pending' }],
        });
        expect(getProcedurePlanToolArguments(plan().requests[0])).toEqual({ lap: 2 });
        expect(advanceProcedurePlan(plan(), 'done')).toMatchObject({
            status: 'advanced',
            reason: 'done',
        });
        expect(isProcedurePlanOptOutRequest('Please stop the plan.')).toBe(true);
    });
});

describe('ProcedurePlanRunner central dispatch callback', () => {
    it('executes requests in order and returns dispatcher outputs unchanged', async () => {
        const dispatch = jest.fn((name: string, args?: Record<string, unknown>) => resolvedAiToolOperation({
            status: 'complete',
            name,
            lap: args?.lap,
        }));
        const runner = new ProcedurePlanRunner('procedure-plan', dispatch);

        const operation = runner.createProcedurePlan(plan());
        expect(runner.getComponentName()).toBe('procedure-plan');
        expect(runner.getComponentType()).toBe('procedure_plan');
        expect(runner.getOverlayBehavior(null)).toEqual({
            placement: 'flow',
            requestedStatus: 'expanded',
            remove: true,
        });
        expect(runner.getOverlayMetadata()).toEqual({});
        expect(runner.handleOverlayRendererEvent({} as any)).toBeUndefined();
        const result = await operation.result;
        expect(result).not.toBeInstanceOf(Error);
        if (result instanceof Error) throw result;

        expect(operation.statuses).toEqual([]);
        expect(dispatch).toHaveBeenNthCalledWith(1, 'read', { lap: 2 });
        expect(dispatch).toHaveBeenNthCalledWith(2, 'compare', {});
        expect(result).toMatchObject({ status: 'complete', request_count: 2 });
        expect(result.task_results[0].output).toEqual({
            status: 'complete',
            name: 'read',
            lap: 2,
        });
        expect(runner.getProcedurePlan()).toMatchObject({
            goal: 'Review the lap',
            currentStep: 2,
        });

        const cleared = await runner.clearProcedurePlan('finished').result;
        expect(cleared).toMatchObject({ status: 'cleared', reason: 'finished' });
        expect(runner.getSnapshot()).toBeNull();
    });

    it('keeps a failed step available for retry', async () => {
        let attempts = 0;
        const onError = jest.fn();
        const rootError = new Error('offline');
        const dispatch = jest.fn((name: string) => createAiToolOperationFrom(() => {
            if (name === 'read' && ++attempts === 1) throw rootError;
            return { status: 'complete' };
        }));
        const runner = new ProcedurePlanRunner('procedure-plan', dispatch, undefined, onError);

        const failedResult = await runner.createProcedurePlan(plan()).result;
        expect(failedResult).not.toBeInstanceOf(Error);
        if (failedResult instanceof Error) throw failedResult;
        expect(failedResult).toMatchObject({
            status: 'failed',
            task_results: [{
                title: 'Read telemetry',
                tool_name: 'read',
                status: 'failed',
                error: {
                    name: 'ProcedurePlanStepFailedError',
                    message: 'offline',
                    cause: {
                        name: 'Error',
                        message: 'offline',
                    },
                },
            }],
        });
        expect(onError).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Read telemetry' }),
            expect.any(ProcedurePlanStepFailedError),
        );
        expect(onError.mock.calls[0][1]).toMatchObject({
            componentName: 'procedure-plan',
            cause: rootError,
        });
        await expect(runner.retryFailedStep()).resolves.toMatchObject({ status: 'complete' });
        expect(attempts).toBe(2);
    });
});
