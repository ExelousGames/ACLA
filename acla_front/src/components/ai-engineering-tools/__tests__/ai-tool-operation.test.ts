import {
    AiToolOperationAbortedError,
    createControlledAiToolOperation,
    createAiToolDeferred,
    createAiToolOperation,
    createAiToolOperationFrom,
    mapAiToolOperation,
} from '../ai-tool-operation';

describe('createAiToolOperation', () => {
    it('keeps result behind every status promise', async () => {
        const status = createAiToolDeferred<{ progress: number }>();
        const operation = createAiToolOperation(
            Promise.resolve({ status: 'complete' }),
            [status.promise],
            'complete',
        );
        let settled = false;
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);
        void operation.result.then(() => { settled = true; });

        await Promise.resolve();
        expect(settled).toBe(false);
        expect(terminated).not.toHaveBeenCalled();
        status.resolve({ progress: 100 });

        await expect(operation.result).resolves.toEqual({ status: 'complete' });
        await Promise.resolve();
        expect(terminated).toHaveBeenCalledTimes(1);
    });

    it('ignores rejected statuses when determining final success', async () => {
        const operation = createAiToolOperation(
            Promise.resolve({ status: 'complete' }),
            [Promise.reject(new Error('status failed'))],
            'complete',
        );

        await expect(operation.result).resolves.toEqual({ status: 'complete' });
    });

    it('emits once, supports unsubscribe, and replays the original termination', async () => {
        const result = { status: 'payload-status', value: 7 };
        const controller = createControlledAiToolOperation<
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
        const operation = createAiToolOperationFrom(() => { throw error; }, 'complete');
        const termination = new Promise((resolve) => operation.notifyTerminated(resolve));

        await expect(operation.result).rejects.toBe(error);
        await expect(termination).resolves.toEqual({ status: 'failed', result: error });
    });

    it('maps results while preserving the source termination status', async () => {
        const source = createAiToolOperation({ status: 'conflicting', value: 3 }, 'source-status');
        const mapped = mapAiToolOperation(source, ({ value }) => ({ doubled: value * 2 }));
        const termination = new Promise((resolve) => mapped.notifyTerminated(resolve));

        await expect(mapped.result).resolves.toEqual({ doubled: 6 });
        await expect(termination).resolves.toEqual({
            status: 'source-status',
            result: { doubled: 6 },
        });
    });

    it('runs safe cleanup before aborting result, statuses, and termination', async () => {
        const result = createAiToolDeferred<{ status: string }>();
        const status = createAiToolDeferred<{ progress: number }>();
        const lifecycle: string[] = [];
        const operation = createAiToolOperation(
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
        expect(terminationResult).toBeInstanceOf(AiToolOperationAbortedError);
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
        const operation = createAiToolOperationFrom((operationSignal) => {
            signal = operationSignal;
            return new Promise<Record<string, never>>(() => undefined);
        }, 'complete');
        await Promise.resolve();

        operation.abort();

        expect((signal as unknown as AbortSignal).aborted).toBe(true);
        await expect(operation.result).rejects.toBeInstanceOf(AiToolOperationAbortedError);
    });

    it('propagates mapped operation aborts to their source', async () => {
        const source = createAiToolOperation(
            new Promise<{ value: number }>(() => undefined),
            'complete',
            [],
            jest.fn(),
        );
        const sourceAbort = jest.spyOn(source, 'abort');
        const mapped = mapAiToolOperation(source, ({ value }) => value * 2);

        mapped.abort();

        expect(sourceAbort).toHaveBeenCalledTimes(1);
        await expect(mapped.result).rejects.toBeInstanceOf(AiToolOperationAbortedError);
    });

    it('does not replace a completed operation with an abort', async () => {
        const operation = createAiToolOperation({ value: 1 }, 'complete');
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
