import {
    executeSubscribedFrontendTool,
    extractInlineFunctionCalls,
    type FrontendToolHandler,
} from '../use-voice-conversation';
import {
    createAiToolOperation,
    createAiToolOperationFrom,
} from '../ai-tool-base';

const execute = async (handler: FrontendToolHandler) => {
    const frames: any[] = [];
    const events: any[] = [];
    const result = await executeSubscribedFrontendTool({
        call: { id: 'call-1', name: 'test_tool', title: 'Test tool' },
        handlers: { test_tool: handler },
        sendText: (frame) => frames.push(frame),
        emitEvent: (event) => events.push(event),
    });
    return { events, frames, result };
};

describe('executeSubscribedFrontendTool', () => {
    it('emits started, non-final statuses, then exactly one successful final frame', async () => {
        const { frames, events, result } = await execute(() => createAiToolOperation(
            Promise.resolve({ status: 'complete', value: 7 }),
            [Promise.resolve({ status: 'working', progress: 50 })],
        ));

        expect(frames).toEqual([
            expect.objectContaining({ type: 'tool_result', id: 'call-1', name: 'test_tool', final: false, result: { status: 'started' } }),
            expect.objectContaining({ final: false, result: { status: 'working', progress: 50 } }),
            expect.objectContaining({ final: true, result: { status: 'complete', value: 7 } }),
        ]);
        expect(frames.filter((frame) => frame.final)).toHaveLength(1);
        expect(events.at(-1)).toMatchObject({ status: 'completed', final: true, ok: true });
        expect(result).toMatchObject({ ok: true });
    });

    it('turns rejected statuses into non-terminal errors without changing final success', async () => {
        const { frames, result } = await execute(() => createAiToolOperation(
            Promise.resolve({ status: 'complete' }),
            [Promise.reject(new Error('progress unavailable'))],
        ));

        expect(frames[1]).toMatchObject({
            final: false,
            result: { ok: false, status: 'status_failed', message: 'progress unavailable' },
        });
        expect(frames.at(-1)).toMatchObject({ final: true, result: { status: 'complete' } });
        expect(result).toMatchObject({ ok: true });
    });

    it.each([
        ['resolved Error', () => createAiToolOperation(new Error('broken'))],
        ['rejected promise', () => createAiToolOperationFrom(() => { throw new Error('broken'); })],
    ])('normalizes a %s into the same failed final frame', async (_label, handler) => {
        const { frames, result } = await execute(handler as any);

        expect(frames.at(-1)).toMatchObject({
            final: true,
            result: { ok: false, name: 'ToolExecutionError', message: 'broken' },
        });
        expect(result).toMatchObject({ ok: false, message: 'broken' });
    });
});

describe('extractInlineFunctionCalls', () => {
    it('extracts a structured inline call without leaking the marker into chat', () => {
        expect(extractInlineFunctionCalls('Before <function=show_map>{"id":"spa"}</function> after')).toEqual({
            cleanText: 'Before  after',
            calls: [{ name: 'show_map', arguments: { id: 'spa' } }],
        });
    });
});
