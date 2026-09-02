import {
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
});
