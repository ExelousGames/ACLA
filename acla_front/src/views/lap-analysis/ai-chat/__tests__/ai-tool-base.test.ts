import {
    ToolExecutionError,
    type AiToolDefinition,
    createToolOutputController,
    executeAiToolDefinition,
} from '../ai-tool-base';

type TestHandlerContext = {
    toolRunId?: string;
    toolName?: string;
};

const definition = (
    execute: AiToolDefinition<object, TestHandlerContext>['execute'],
): AiToolDefinition<object, TestHandlerContext> => ({
    name: 'test_tool',
    schema: { properties: {}, required: [] },
    required: [],
    execute,
});

describe('AI tool output execution', () => {
    it('keeps successful and expected-state statuses as ordinary envelopes', async () => {
        for (const status of ['ready', 'loading', 'empty', 'unavailable', 'complete']) {
            const result = await executeAiToolDefinition(
                definition(() => ({ status })),
                {},
                {},
                { toolRunId: `run-${status}` },
            );

            expect(result).toMatchObject({
                tool_name: 'test_tool',
                run_id: `run-${status}`,
                status,
                final: true,
            });
            expect(result).not.toHaveProperty('error');
        }
    });

    it('propagates typed failures without generating an error envelope', async () => {
        const failure = new ToolExecutionError(
            'The input is invalid.',
        );

        await expect(executeAiToolDefinition(
            definition(() => { throw failure; }),
            {},
            {},
            { toolRunId: 'failed-run' },
        )).rejects.toBe(failure);
    });

    it('exposes progress and final controls only', () => {
        const output = createToolOutputController('test_tool', 'run-1');

        expect(Object.keys(output).sort()).toEqual([
            'final',
            'getFinalOutput',
            'progress',
        ]);
        expect(output.progress({ status: 'loading' })).toMatchObject({
            status: 'loading',
            final: false,
        });
    });
});
