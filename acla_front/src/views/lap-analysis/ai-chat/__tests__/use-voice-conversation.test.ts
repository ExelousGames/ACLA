import {
    buildVoiceSessionMetadata,
    executeSubscribedFrontendTool,
    extractInlineFunctionCalls,
} from '../use-voice-conversation';
import { createToolOutputController } from '../ai-tool-base';

describe('buildVoiceSessionMetadata', () => {
    it('defaults to a main conversation session', () => {
        expect(buildVoiceSessionMetadata({
            clientSessionId: 'main-1',
        })).toEqual({
            conversation_role: 'main',
            client_session_id: 'main-1',
            parent_client_session_id: null,
            agent_mode: null,
        });
    });

    it('includes child agent session identity', () => {
        expect(buildVoiceSessionMetadata({
            conversationRole: 'agent',
            clientSessionId: 'agent-1',
            parentClientSessionId: 'main-1',
            agentMode: 'live_performance_analyst',
        })).toEqual({
            conversation_role: 'agent',
            client_session_id: 'agent-1',
            parent_client_session_id: 'main-1',
            agent_mode: 'live_performance_analyst',
        });
    });
});

describe('extractInlineFunctionCalls', () => {
    it('strips inline function tags and parses JSON arguments', () => {
        const result = extractInlineFunctionCalls(
            'Let me start a live performance analyst session for you.\n<function=start_agent_session>{"agent_mode":"live_performance_analyst"}</function>',
        );

        expect(result.cleanText).toBe('Let me start a live performance analyst session for you.');
        expect(result.calls).toEqual([
            {
                name: 'start_agent_session',
                arguments: { agent_mode: 'live_performance_analyst' },
            },
        ]);
    });

    it('handles inline function tags that are missing the closing name bracket', () => {
        const result = extractInlineFunctionCalls(
            '<function=start_agent_session{"agent_mode":"live_performance_analyst"}</function>',
        );

        expect(result.cleanText).toBe('');
        expect(result.calls).toEqual([
            {
                name: 'start_agent_session',
                arguments: { agent_mode: 'live_performance_analyst' },
            },
        ]);
    });

    it('keeps malformed arguments available for diagnostics', () => {
        const result = extractInlineFunctionCalls(
            '<function=start_agent_session>agent_mode=track_guide</function>',
        );

        expect(result.cleanText).toBe('');
        expect(result.calls).toEqual([
            {
                name: 'start_agent_session',
                arguments: { raw: 'agent_mode=track_guide' },
            },
        ]);
    });
});

describe('executeSubscribedFrontendTool', () => {
    it('emits lifecycle events and a single final tool_result for direct tool calls', async () => {
        const frames: object[] = [];
        const events: object[] = [];

        const result = await executeSubscribedFrontendTool({
            call: {
                id: 'tool-1',
                name: 'read_context',
                title: 'Read context',
                arguments: { session_id: 's1' },
            },
            handlers: {
                read_context: async (args) => ({ status: 'ready', args }),
            },
            baseContext: {
                sendToolStatus: (data) => frames.push({ type: 'tool_status', data }),
            },
            sendText: (payload) => frames.push(payload),
            emitEvent: (event) => events.push(event),
        });

        expect(result).toMatchObject({ id: 'tool-1', name: 'read_context', ok: true });
        expect(events).toMatchObject([
            { kind: 'tool_call', runId: 'tool-1', name: 'read_context', title: 'Read context', status: 'started' },
            {
                kind: 'tool_call',
                runId: 'tool-1',
                name: 'read_context',
                title: 'Read context',
                status: 'completed',
                ok: true,
                result: {
                    status: 'ready',
                    args: { session_id: 's1' },
                },
            },
        ]);
        expect(frames).toEqual([
            expect.objectContaining({
                type: 'tool_result',
                id: 'tool-1',
                name: 'read_context',
                result: {
                    status: 'ready',
                    args: { session_id: 's1' },
                },
            }),
        ]);
        expect((frames[0] as any).messages).toBeUndefined();
    });

    it('emits lifecycle events and a failed tool_result when the handler fails', async () => {
        const frames: object[] = [];
        const events: object[] = [];

        const result = await executeSubscribedFrontendTool({
            call: { id: 'tool-2', name: 'explode' },
            handlers: {
                explode: async () => {
                    throw new Error('boom');
                },
            },
            baseContext: {
                sendToolStatus: (data) => frames.push({ type: 'tool_status', data }),
            },
            sendText: (payload) => frames.push(payload),
            emitEvent: (event) => events.push(event),
        });

        expect(result).toEqual({ id: 'tool-2', name: 'explode', ok: false, error: 'boom' });
        expect(events).toMatchObject([
            { kind: 'tool_call', runId: 'tool-2', name: 'explode', status: 'started' },
            { kind: 'tool_call', runId: 'tool-2', name: 'explode', status: 'completed', ok: false, error: 'boom' },
        ]);
        expect(frames).toContainEqual(expect.objectContaining({
            type: 'tool_result',
            id: 'tool-2',
            name: 'explode',
            result: { ok: false, error: 'boom' },
        }));
        expect((frames[0] as any).messages).toBeUndefined();
    });

    it('does not expose a secondary tool output callback', async () => {
        const frames: object[] = [];
        const contextKeys: string[][] = [];

        await executeSubscribedFrontendTool({
            call: { id: 'tool-3', name: 'single_output' },
            handlers: {
                single_output: async (_args, ctx) => {
                    contextKeys.push(Object.keys(ctx).sort());
                    return { status: 'done' };
                },
            },
            baseContext: {
                sendToolStatus: (data) => frames.push({ type: 'tool_status', data }),
            },
            sendText: (payload) => frames.push(payload),
        });

        expect(contextKeys).toEqual([['sendToolStatus', 'toolName', 'toolRunId']]);
        expect(frames).toEqual([
            expect.objectContaining({
                type: 'tool_result',
                id: 'tool-3',
                name: 'single_output',
                result: { status: 'done' },
            }),
        ]);
        expect((frames[0] as any).messages).toBeUndefined();
    });

    it('sends only envelope ai_output to the AI tool_result frame', async () => {
        const frames: object[] = [];
        const events: object[] = [];
        const aiOutput = {
            name: 'collect_live_baseline',
            status: 'complete',
            message: 'Baseline complete.',
        };
        const uiOutput = {
            status: 'complete',
            message: 'Baseline complete.',
            snapshot: { baseline_ready: true },
        };
        const envelope = createToolOutputController(
            'collect_live_baseline',
            'tool-4',
        ).final(uiOutput, { aiOutput });

        await executeSubscribedFrontendTool({
            call: { id: 'tool-4', name: 'collect_live_baseline' },
            handlers: {
                collect_live_baseline: async () => envelope,
            },
            baseContext: {
                sendToolStatus: (data) => frames.push({ type: 'tool_status', data }),
            },
            sendText: (payload) => frames.push(payload),
            emitEvent: (event) => events.push(event),
        });

        expect(frames).toEqual([
            expect.objectContaining({
                type: 'tool_result',
                id: 'tool-4',
                name: 'collect_live_baseline',
                result: aiOutput,
            }),
        ]);
        expect((frames[0] as any).messages).toBeUndefined();
        expect((frames[0] as any).result).not.toHaveProperty('snapshot');
        expect(events).toContainEqual(expect.objectContaining({
            kind: 'tool_call',
            runId: 'tool-4',
            result: uiOutput,
        }));
        expect(events).toContainEqual(expect.objectContaining({
            kind: 'tool_call',
            runId: 'tool-4',
            name: 'collect_live_baseline',
            status: 'started',
        }));
    });

    it('does not mark a non-final envelope completed', async () => {
        const frames: object[] = [];
        const events: object[] = [];
        const uiOutput = {
            status: 'started',
            message: 'Baseline collection started.',
        };
        const envelope = createToolOutputController(
            'collect_live_baseline',
            'tool-5',
        ).progress(uiOutput, {
            aiOutput: {
                name: 'collect_live_baseline',
                status: 'started',
                message: 'Baseline collection started.',
            },
        });

        await executeSubscribedFrontendTool({
            call: { id: 'tool-5', name: 'collect_live_baseline' },
            handlers: {
                collect_live_baseline: async () => envelope,
            },
            baseContext: {
                sendToolStatus: (data) => frames.push({ type: 'tool_status', data }),
            },
            sendText: (payload) => frames.push(payload),
            emitEvent: (event) => events.push(event),
        });

        expect(frames).toEqual([
            expect.objectContaining({
                type: 'tool_result',
                id: 'tool-5',
                name: 'collect_live_baseline',
                result: envelope.output,
            }),
        ]);
        expect((frames[0] as any).messages).toBeUndefined();
        expect(events).toContainEqual(expect.objectContaining({
            kind: 'tool_call',
            runId: 'tool-5',
            name: 'collect_live_baseline',
            status: 'started',
            result: uiOutput,
        }));
        expect(events).not.toContainEqual(expect.objectContaining({
            kind: 'tool_call',
            runId: 'tool-5',
            status: 'completed',
        }));
    });

    it('supports plan-triggered calls with generated run ids', async () => {
        const frames: object[] = [];

        await executeSubscribedFrontendTool({
            call: {
                name: 'show_map',
                title: 'Show the current map',
                arguments: { map_id: 'spa' },
            },
            handlers: {
                show_map: async () => ({ status: 'displayed' }),
            },
            baseContext: {
                sendToolStatus: (data) => frames.push({ type: 'tool_status', data }),
            },
            sendText: (payload) => frames.push(payload),
            makeRunId: () => 'plan-1',
        });

        expect(frames).toContainEqual(expect.objectContaining({
            type: 'tool_result',
            id: 'plan-1',
            name: 'show_map',
            result: { status: 'displayed' },
        }));
        expect((frames[0] as any).messages).toBeUndefined();
    });
});
