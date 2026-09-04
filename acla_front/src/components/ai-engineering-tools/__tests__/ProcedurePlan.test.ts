import {
    ProcedurePlanRunner,
    advanceProcedurePlan,
    buildProcedurePlan,
    getProcedurePlanToolArguments,
    isProcedurePlanOptOutRequest,
    type ProcedurePlanState,
} from '../ProcedurePlan';
import { ProcedurePlanStepFailedError } from '../../../contexts/AiToolComponentError';
import {
    createAiToolDeferred,
    createAiToolOperation,
    createAiToolOperationFrom,
    resolvedAiToolOperation,
} from '../ai-tool-operation';

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
    it('aborts the active nested tool when the plan operation is aborted', async () => {
        const nested = createAiToolOperation(
            new Promise<Record<string, unknown>>(() => undefined),
            'complete',
        );
        const nestedAbort = jest.spyOn(nested, 'abort');
        const dispatch = jest.fn(() => nested);
        const runner = new ProcedurePlanRunner('procedure-plan', dispatch);
        const operation = runner.createProcedurePlan(plan());
        const termination = new Promise((resolve) => operation.notifyTerminated(resolve));

        operation.abort();

        expect(nestedAbort).toHaveBeenCalledTimes(1);
        await expect(operation.result).rejects.toMatchObject({ name: 'AbortError' });
        await expect(termination).resolves.toMatchObject({
            status: 'aborted',
            result: { name: 'AbortError' },
        });
        await Promise.resolve();
        expect(dispatch).toHaveBeenCalledTimes(1);
    });

    it('executes requests in order and returns dispatcher outputs unchanged', async () => {
        const dispatch = jest.fn((name: string, args?: Record<string, unknown>) => resolvedAiToolOperation({
            status: 'complete',
            name,
            lap: args?.lap,
        }, 'complete'));
        const onChange = jest.fn();
        const runner = new ProcedurePlanRunner('procedure-plan', dispatch, onChange);

        const operation = runner.createProcedurePlan(plan());
        const termination = new Promise((resolve) => operation.notifyTerminated(resolve));
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
        expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
            goal: 'Review the lap',
            currentStep: 2,
        }));
        expect(onChange).toHaveBeenLastCalledWith(null);
        expect(runner.getProcedurePlan()).toBeNull();
        expect(runner.getSnapshot()).toBeNull();
        await expect(termination).resolves.toMatchObject({
            status: 'complete',
            result: { status: 'complete' },
        });

        const clearOperation = runner.clearProcedurePlan('finished');
        const clearTermination = new Promise((resolve) => (
            clearOperation.notifyTerminated(resolve)
        ));
        const cleared = await clearOperation.result;
        expect(cleared).toMatchObject({ status: 'cleared', reason: 'finished' });
        await expect(clearTermination).resolves.toMatchObject({ status: 'cleared' });
        expect(runner.getSnapshot()).toBeNull();
    });

    it('notifies a superseded operation as replaced', async () => {
        const nestedResult = createAiToolDeferred<{ status: string }>();
        const dispatch = jest.fn(() => createAiToolOperation(nestedResult.promise, 'complete'));
        const runner = new ProcedurePlanRunner('procedure-plan', dispatch);
        const original = runner.createProcedurePlan(plan());
        const termination = new Promise((resolve) => original.notifyTerminated(resolve));

        const replacement = runner.replace(null);

        await expect(original.result).rejects.toMatchObject({
            name: 'ProcedurePlanReplacedError',
        });
        await expect(termination).resolves.toMatchObject({
            status: 'replaced',
            result: { name: 'ProcedurePlanReplacedError' },
        });
        await expect(replacement.result).resolves.toMatchObject({ status: 'cleared' });
    });

    it('returns failure details and removes a failed plan', async () => {
        const onError = jest.fn();
        const onChange = jest.fn();
        const rootError = new Error('offline');
        const dispatch = jest.fn((name: string) => createAiToolOperationFrom(() => {
            if (name === 'read') throw rootError;
            return { status: 'complete' };
        }, 'complete'));
        const runner = new ProcedurePlanRunner('procedure-plan', dispatch, onChange, onError);

        const failedResult = await runner.createProcedurePlan(plan()).result;
        expect(failedResult).not.toBeInstanceOf(Error);
        if (failedResult instanceof Error) throw failedResult;
        expect(failedResult).toMatchObject({
            status: 'failed',
            request: {
                title: 'Read telemetry',
                status: 'failed',
            },
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
            expect.objectContaining({ title: 'Read telemetry', status: 'failed' }),
            expect.any(ProcedurePlanStepFailedError),
        );
        expect(onError.mock.calls[0][1]).toMatchObject({
            componentName: 'procedure-plan',
            cause: rootError,
        });
        expect(onChange).toHaveBeenLastCalledWith(null);
        expect(runner.getProcedurePlan()).toBeNull();
        expect(runner.getSnapshot()).toBeNull();
    });
});
