import {
    buildVoiceSessionMetadata,
    executeSubscribedFrontendTool,
    extractInlineFunctionCalls,
    mapBackendToolEventForUi,
} from '../use-voice-conversation';

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
                arguments: { session_id: 's1' },
            },
            handlers: {
                read_context: async (args) => ({ status: 'ready', args }),
            },
            baseContext: {
                sendObservation: (data) => frames.push({ type: 'observation', data }),
            },
            sendText: (payload) => frames.push(payload),
            emitEvent: (event) => events.push(event),
        });

        expect(result).toMatchObject({ id: 'tool-1', name: 'read_context', ok: true });
        expect(events).toMatchObject([
            { kind: 'tool_event', runId: 'tool-1', name: 'read_context', status: 'started' },
            {
                kind: 'tool_event',
                runId: 'tool-1',
                name: 'read_context',
                status: 'completed',
                ok: true,
                result: {
                    status: 'ready',
                    args: { session_id: 's1' },
                },
            },
        ]);
        expect(frames).toEqual([
            {
                type: 'tool_result',
                id: 'tool-1',
                result: {
                    status: 'ready',
                    args: { session_id: 's1' },
                },
            },
        ]);
    });

    it('emits lifecycle events and a final tool_error when the handler fails', async () => {
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
                sendObservation: (data) => frames.push({ type: 'observation', data }),
            },
            sendText: (payload) => frames.push(payload),
            emitEvent: (event) => events.push(event),
        });

        expect(result).toEqual({ id: 'tool-2', name: 'explode', ok: false, error: 'boom' });
        expect(events).toMatchObject([
            { kind: 'tool_event', runId: 'tool-2', name: 'explode', status: 'started' },
            { kind: 'tool_event', runId: 'tool-2', name: 'explode', status: 'completed', ok: false, error: 'boom' },
        ]);
        expect(frames).toContainEqual({ type: 'tool_error', id: 'tool-2', error: 'boom' });
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
                sendObservation: (data) => frames.push({ type: 'observation', data }),
            },
            sendText: (payload) => frames.push(payload),
        });

        expect(contextKeys).toEqual([['sendObservation', 'toolName', 'toolRunId']]);
        expect(frames).toEqual([
            { type: 'tool_result', id: 'tool-3', result: { status: 'done' } },
        ]);
    });

    it('sends only envelope ai_output to the AI tool_result frame', async () => {
        const frames: object[] = [];
        const events: object[] = [];
        const aiOutput = {
            name: 'collect_live_baseline',
            status: 'complete',
            message: 'Baseline complete.',
        };
        const envelope = {
            tool_name: 'collect_live_baseline',
            run_id: 'tool-4',
            status: 'complete',
            ui_output: {
                status: 'complete',
                message: 'Baseline complete.',
                snapshot: { baseline_ready: true },
            },
            ai_output: aiOutput,
            final: true,
        };

        await executeSubscribedFrontendTool({
            call: { id: 'tool-4', name: 'collect_live_baseline' },
            handlers: {
                collect_live_baseline: async () => envelope,
            },
            baseContext: {
                sendObservation: (data) => frames.push({ type: 'observation', data }),
            },
            sendText: (payload) => frames.push(payload),
            emitEvent: (event) => events.push(event),
        });

        expect(frames).toEqual([
            { type: 'tool_result', id: 'tool-4', result: aiOutput },
        ]);
        expect((frames[0] as any).result).not.toHaveProperty('snapshot');
        expect(events).toContainEqual(expect.objectContaining({
            kind: 'tool_event',
            runId: 'tool-4',
            result: envelope,
        }));
        expect(events).toContainEqual(expect.objectContaining({
            kind: 'tool_event',
            runId: 'tool-4',
            name: 'collect_live_baseline',
            status: 'started',
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
                sendObservation: (data) => frames.push({ type: 'observation', data }),
            },
            sendText: (payload) => frames.push(payload),
            makeRunId: () => 'plan-1',
        });

        expect(frames).toContainEqual({
            type: 'tool_result',
            id: 'plan-1',
            result: { status: 'displayed' },
        });
    });
});

describe('mapBackendToolEventForUi', () => {
    it('keeps backend tool status frames out of the visible transcript', () => {
        expect(mapBackendToolEventForUi({
            type: 'tool_event',
            name: 'query_telemetry_metric',
            title: 'Querying telemetry',
            status: 'started',
        })).toBeNull();
    });
});
