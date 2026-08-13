import {
    createAiToolDeferred,
    createAiToolOperation,
} from '../ai-tool-operation';

describe('createAiToolOperation', () => {
    it('keeps result behind every status promise', async () => {
        const status = createAiToolDeferred<{ progress: number }>();
        const operation = createAiToolOperation(
            Promise.resolve({ status: 'complete' }),
            [status.promise],
        );
        let settled = false;
        void operation.result.then(() => { settled = true; });

        await Promise.resolve();
        expect(settled).toBe(false);
        status.resolve({ progress: 100 });

        await expect(operation.result).resolves.toEqual({ status: 'complete' });
    });

    it('ignores rejected statuses when determining final success', async () => {
        const operation = createAiToolOperation(
            Promise.resolve({ status: 'complete' }),
            [Promise.reject(new Error('status failed'))],
        );

        await expect(operation.result).resolves.toEqual({ status: 'complete' });
    });
});
