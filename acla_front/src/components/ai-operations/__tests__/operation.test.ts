import {
    OperationAbortedError,
    createControlledOperation,
    createOperationDeferred,
    createOperation,
    createOperationFrom,
    mapOperation,
} from '../operation';
import { asTool, type Tool } from '../tool';
import { asWorkflow, type Workflow } from '../workflow';
import type { Operation } from '../operation';

const assertOperationTypes = (tool: Tool<number>, workflow: Workflow<number>) => {
    const operations: Operation<number>[] = [tool, workflow];
    // @ts-expect-error Workflows are a distinct operation category from tools.
    const invalidTool: Tool<number> = workflow;
    // @ts-expect-error Tools cannot be used where a workflow is required.
    const invalidWorkflow: Workflow<number> = tool;
    return { operations, invalidTool, invalidWorkflow };
};
void assertOperationTypes;

describe.each([
    { kind: 'tool', classify: asTool },
    { kind: 'workflow', classify: asWorkflow },
])('$kind operation lifecycle', ({ kind, classify }) => {
    it('preserves identity, progress, and the terminal result when classified', async () => {
        const status = createOperationDeferred<{ progress: number }>();
        const controller = createControlledOperation<number, { progress: number }, 'complete'>([status.promise]);
        const operation = classify(controller.operation);
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);

        expect(operation).toBe(controller.operation);
        expect(operation.kind).toBe(kind);
        controller.resolve('complete', 42);
        await expect(operation.result).resolves.toBe(42);
        expect(terminated).toHaveBeenCalledWith({ status: 'complete', result: 42 });
        expect(status.settled).toBe(false);
        status.resolve({ progress: 100 });

        await expect(operation.statuses[0]).resolves.toEqual({ progress: 100 });
        expect(terminated).toHaveBeenCalledTimes(1);
    });

    it('waits for nested work and async cleanup regardless of status values', async () => {
        const work = createOperationDeferred<number>();
        const cleanup = createOperationDeferred<void>();
        const cleanupStarted = createOperationDeferred<void>();
        const progress = createOperationDeferred<{ status: string }>();
        const child = asTool(createOperation(work.promise, 'still-running'));
        const operation = classify(createOperationFrom(async () => {
            try {
                return await child.result;
            } finally {
                cleanupStarted.resolve();
                await cleanup.promise;
            }
        }, [progress.promise], 'still-running'));
        const lifecycle: string[] = [];
        operation.notifyTerminated(() => lifecycle.push('terminated'));
        void operation.result.then(() => lifecycle.push('result'));

        progress.resolve({ status: 'complete' });
        await operation.statuses[0];
        expect(lifecycle).toEqual([]);
        work.resolve(42);
        await cleanupStarted.promise;
        expect(lifecycle).toEqual([]);

        cleanup.resolve();
        await expect(operation.result).resolves.toBe(42);
        expect(lifecycle).toEqual(['terminated', 'result']);
    });

    it('retains abort cleanup and termination delivery', async () => {
        const cleanup = jest.fn();
        const controller = createControlledOperation<number>([], cleanup);
        const operation = classify(controller.operation);
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);

        operation.abort();
        operation.abort();

        expect(cleanup).toHaveBeenCalledTimes(1);
        expect(controller.signal.aborted).toBe(true);
        await expect(operation.result).rejects.toBeInstanceOf(OperationAbortedError);
        expect(terminated).toHaveBeenCalledTimes(1);
        expect(terminated).toHaveBeenCalledWith({ status: 'aborted', result: expect.any(OperationAbortedError) });
    });
});

describe('createOperation', () => {
    it('completes without waiting for pending status promises', async () => {
        const status = createOperationDeferred<{ progress: number }>();
        const operation = createOperation(
            Promise.resolve({ status: 'complete' }),
            [status.promise],
            'complete',
        );
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);

        await expect(operation.result).resolves.toEqual({ status: 'complete' });
        expect(terminated).toHaveBeenCalledTimes(1);
        expect(status.settled).toBe(false);
    });

    it('ignores rejected statuses when determining final success', async () => {
        const operation = createOperation(
            Promise.resolve({ status: 'complete' }),
            [Promise.reject(new Error('status failed'))],
            'complete',
        );

        await expect(operation.result).resolves.toEqual({ status: 'complete' });
    });

    it('emits once, supports unsubscribe, and replays the original termination', async () => {
        const result = { status: 'payload-status', value: 7 };
        const controller = createControlledOperation<
            typeof result,
            never,
            'notified-status'
        >();
        const subscribed = jest.fn();
        const unsubscribed = jest.fn();
        controller.operation.notifyTerminated(subscribed);
        const unsubscribe = controller.operation.notifyTerminated(unsubscribed);
        unsubscribe();

        controller.resolve('notified-status', result);
        controller.resolve('notified-status', { ...result, value: 8 });
        await expect(controller.operation.result).resolves.toBe(result);
        await Promise.resolve();

        expect(subscribed).toHaveBeenCalledTimes(1);
        expect(subscribed).toHaveBeenCalledWith({ status: 'notified-status', result });
        expect(unsubscribed).not.toHaveBeenCalled();

        const late = jest.fn();
        controller.operation.notifyTerminated(late);
        expect(late).toHaveBeenCalledTimes(1);
        expect(late).toHaveBeenCalledWith({ status: 'notified-status', result });
    });

    it('notifies failed with the corresponding Error while preserving rejection', async () => {
        const error = new Error('broken');
        const operation = createOperationFrom(() => { throw error; }, 'complete');
        const termination = new Promise((resolve) => operation.notifyTerminated(resolve));

        await expect(operation.result).rejects.toBe(error);
        await expect(termination).resolves.toEqual({ status: 'failed', result: error });
    });

    it('preserves result and event delivery when a termination listener throws', async () => {
        const error = new Error('listener failed');
        const logged = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        try {
            const operation = createOperation(42, 'complete');
            const throwingListener = () => { throw error; };
            const notified = jest.fn();
            operation.notifyTerminated(throwingListener);
            operation.notifyTerminated(notified);

            await expect(operation.result).resolves.toBe(42);
            expect(notified).toHaveBeenCalledWith({ status: 'complete', result: 42 });
            expect(() => operation.notifyTerminated(throwingListener)).not.toThrow();
            expect(logged).toHaveBeenCalledTimes(2);
        } finally {
            logged.mockRestore();
        }
    });

    it('waits for failure cleanup but not pending progress before terminating', async () => {
        const cleanup = createOperationDeferred<void>();
        const cleanupStarted = createOperationDeferred<void>();
        const progress = createOperationDeferred<{ progress: number }>();
        const error = new Error('broken');
        const operation = createOperationFrom(async () => {
            try {
                throw error;
            } finally {
                cleanupStarted.resolve();
                await cleanup.promise;
            }
        }, [progress.promise], 'complete');
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);

        await cleanupStarted.promise;
        expect(terminated).not.toHaveBeenCalled();
        cleanup.resolve();

        await expect(operation.result).rejects.toBe(error);
        expect(terminated).toHaveBeenCalledWith({ status: 'failed', result: error });
        expect(progress.settled).toBe(false);
    });

    it('maps results while preserving the source termination status', async () => {
        const source = createOperation({ status: 'conflicting', value: 3 }, 'source-status');
        const mapped = mapOperation(source, ({ value }) => ({ doubled: value * 2 }));
        const termination = new Promise((resolve) => mapped.notifyTerminated(resolve));

        await expect(mapped.result).resolves.toEqual({ doubled: 6 });
        await expect(termination).resolves.toEqual({
            status: 'source-status',
            result: { doubled: 6 },
        });
    });

    it('waits for async result mapping independently of mapped progress', async () => {
        const mappedValue = createOperationDeferred<number>();
        const mappedProgress = createOperationDeferred<{ progress: number }>();
        const mappingStarted = createOperationDeferred<void>();
        const source = createOperation({ value: 3 }, [Promise.resolve({ progress: 100 })], 'working');
        const mapped = mapOperation(source, () => {
            mappingStarted.resolve();
            return mappedValue.promise;
        }, () => mappedProgress.promise);
        const terminated = jest.fn();
        mapped.notifyTerminated(terminated);

        await mappingStarted.promise;
        expect(terminated).not.toHaveBeenCalled();
        mappedValue.resolve(6);

        await expect(mapped.result).resolves.toBe(6);
        expect(terminated).toHaveBeenCalledWith({ status: 'working', result: 6 });
        expect(mappedProgress.settled).toBe(false);
    });

    it('runs safe cleanup before aborting result, statuses, and termination', async () => {
        const result = createOperationDeferred<{ status: string }>();
        const status = createOperationDeferred<{ progress: number }>();
        const lifecycle: string[] = [];
        const operation = createOperation(
            result.promise,
            [status.promise],
            'complete',
            () => lifecycle.push('cleanup'),
        );
        let terminationResult: Error | null = null;
        operation.notifyTerminated((termination) => {
            lifecycle.push('terminated');
            expect(termination.status).toBe('aborted');
            terminationResult = termination.result as Error;
        });

        operation.abort();
        operation.abort();

        expect(lifecycle).toEqual(['cleanup', 'terminated']);
        expect(terminationResult).toBeInstanceOf(OperationAbortedError);
        await expect(operation.result).rejects.toBe(terminationResult);
        await expect(operation.statuses[0]).rejects.toBe(terminationResult);

        result.resolve({ status: 'complete' });
        status.resolve({ progress: 100 });
        await Promise.resolve();
        expect(lifecycle).toEqual(['cleanup', 'terminated']);

        const late = jest.fn();
        operation.notifyTerminated(late);
        expect(late).toHaveBeenCalledWith({ status: 'aborted', result: terminationResult });
    });

    it('aborts the signal supplied to factory work', async () => {
        let signal: AbortSignal | null = null;
        const operation = createOperationFrom((operationSignal) => {
            signal = operationSignal;
            return new Promise<Record<string, never>>(() => undefined);
        }, 'complete');
        await Promise.resolve();

        operation.abort();

        expect((signal as unknown as AbortSignal).aborted).toBe(true);
        await expect(operation.result).rejects.toBeInstanceOf(OperationAbortedError);
    });

    it('propagates mapped operation aborts to their source', async () => {
        const source = createOperation(
            new Promise<{ value: number }>(() => undefined),
            'complete',
            [],
            jest.fn(),
        );
        const sourceAbort = jest.spyOn(source, 'abort');
        const mapped = mapOperation(source, ({ value }) => value * 2);

        mapped.abort();

        expect(sourceAbort).toHaveBeenCalledTimes(1);
        await expect(mapped.result).rejects.toBeInstanceOf(OperationAbortedError);
    });

    it('does not replace a completed operation with an abort', async () => {
        const operation = createOperation({ value: 1 }, 'complete');
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);
        await expect(operation.result).resolves.toEqual({ value: 1 });
        await Promise.resolve();

        operation.abort();

        expect(terminated).toHaveBeenCalledTimes(1);
        expect(terminated).toHaveBeenCalledWith({
            status: 'complete',
            result: { value: 1 },
        });
    });
});
