import {
    executeSubscribedFrontendTool,
    extractInlineFunctionCalls,
    type FrontendToolHandler,
} from '../use-voice-conversation';
import {
    createAiToolOperation,
    createAiToolOperationFrom,
    createControlledAiToolOperation,
} from '../ai-tool-base';
import { AnalysisResultsQueryError } from '../../visualization/charts/analysisResultsQuery';

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
    it('emits started, progress, and completion statuses without a final flag', async () => {
        const { frames, events, result } = await execute(() => createAiToolOperation(
            Promise.resolve({ status: 'complete', value: 7 }),
            [Promise.resolve({ status: 'working', progress: 50 })],
            'complete',
        ));

        expect(frames).toEqual([
            { type: 'tool_result', id: 'call-1', name: 'test_tool', result: { status: 'started' } },
            { type: 'tool_result', id: 'call-1', name: 'test_tool', result: { status: 'working', progress: 50 } },
            { type: 'tool_result', id: 'call-1', name: 'test_tool', result: { status: 'complete', value: 7 } },
        ]);
        events.forEach((event) => expect(event).not.toHaveProperty('final'));
        expect(events.at(-1)).toMatchObject({ status: 'completed', ok: true });
        expect(result).toMatchObject({ ok: true });
    });

    it('reports rejected progress delivery without changing the operation result', async () => {
        const { frames, result } = await execute(() => createAiToolOperation(
            Promise.resolve({ status: 'complete' }),
            [Promise.reject(new Error('progress unavailable'))],
            'complete',
        ));

        expect(frames[1]).toMatchObject({
            result: { ok: false, status: 'status_failed', message: 'progress unavailable' },
        });
        expect(frames.at(-1)).toMatchObject({ result: { status: 'complete' } });
        expect(result).toMatchObject({ ok: true });
    });

    it('uses the notified terminal status when the result is missing or conflicts', async () => {
        const conflicting = await execute(() => createAiToolOperation(
            { status: 'payload-status', value: 7 },
            'notified-status',
        ));
        const missing = await execute(() => createAiToolOperation(
            { value: 8 },
            'explicit-status',
        ));

        expect(conflicting.frames.at(-1)).toMatchObject({
            result: { status: 'notified-status', value: 7 },
        });
        expect(missing.frames.at(-1)).toMatchObject({
            result: { status: 'explicit-status', value: 8 },
        });
        expect(conflicting.frames).toHaveLength(2);
        expect(missing.frames).toHaveLength(2);
    });

    it.each([
        ['resolved Error', () => createAiToolOperation(new Error('broken'), 'failed')],
        ['rejected promise', () => createAiToolOperationFrom(() => { throw new Error('broken'); }, 'failed')],
    ])('normalizes a %s into the same failed status frame', async (_label, handler) => {
        const { frames, result } = await execute(handler as any);

        expect(frames.at(-1)).toMatchObject({
            result: { status: 'failed', ok: false, name: 'ToolExecutionError', message: 'broken' },
        });
        expect(result).toMatchObject({ ok: false, message: 'broken' });
    });

    it.each(['cancelled', 'replaced'])('preserves the producer error status %s', async (status) => {
        const control = createControlledAiToolOperation<Record<string, unknown>>();
        const execution = execute(() => control.operation);
        control.reject(status, new Error('operation stopped'));

        const { frames, result } = await execution;
        expect(frames.at(-1)).toMatchObject({
            result: { status, ok: false, message: 'operation stopped' },
        });
        expect(result).toMatchObject({ ok: false });
    });

    it('reports an aborted operation with its status', async () => {
        const control = createControlledAiToolOperation<Record<string, unknown>>();
        const execution = execute(() => control.operation);
        control.operation.abort();

        const { frames } = await execution;
        expect(frames.at(-1)).toMatchObject({ result: { status: 'aborted', ok: false } });
    });

    it.each([
        [undefined, 'InvalidToolCallError'],
        ['missing_tool', 'ToolNotRegisteredError'],
    ])('reports an invalid call %s with a failed status', async (name, errorName) => {
        const frames: object[] = [];
        await executeSubscribedFrontendTool({
            call: { id: 'invalid-call', name },
            handlers: {},
            sendText: (frame) => frames.push(frame),
        });
        expect(frames.at(-1)).toMatchObject({
            result: { status: 'failed', ok: false, name: errorName },
        });
    });

    it('preserves normalized JSONata detail in the failed status frame', async () => {
        const detail = {
            code: 'S0202',
            position: 12,
            token: ']',
            message: 'Expected a closing bracket.',
        };
        const { frames, result } = await execute(() => createAiToolOperationFrom(() => {
            throw new AnalysisResultsQueryError(detail);
        }, 'failed'));

        expect(frames.at(-1)).toMatchObject({
            result: {
                status: 'failed',
                ok: false,
                name: 'ToolExecutionError',
                message: detail.message,
                cause: {
                    name: 'AnalysisResultsQueryError',
                    message: detail.message,
                    detail,
                },
            },
        });
        expect(result).toMatchObject({
            ok: false,
            message: detail.message,
            cause: { detail },
        });
        expect(frames.at(-1).result.cause).not.toHaveProperty('stack');
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
