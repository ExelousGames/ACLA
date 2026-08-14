import { VoiceGateway } from './voice.gateway';

describe('VoiceGateway', () => {
    const gateway = new VoiceGateway({ verify: jest.fn() } as any) as any;

    it('builds the exact create URL while preserving telemetry and model parameters', () => {
        const upstreamUrl = gateway.buildUpstreamUrl(
            'user-1',
            'session-1',
            'openai:gpt-4.1',
            'create',
        );

        expect(upstreamUrl).toBe(
            'ws://localhost:8000/voice/stream?user_id=user-1&session_id=session-1&chat_llm_model=openai%3Agpt-4.1&chat_session_action=create',
        );
    });

    it('builds the exact resume URL and URL-encodes the server-issued ID', () => {
        const upstreamUrl = gateway.buildUpstreamUrl(
            'user-1',
            'session/1',
            'hosted:qwen/qwen3-32b',
            'resume',
            'server/id + 1',
        );

        expect(upstreamUrl).toBe(
            'ws://localhost:8000/voice/stream?user_id=user-1&session_id=session%2F1&chat_llm_model=hosted%3Aqwen%2Fqwen3-32b&chat_session_action=resume&chat_session_id=server%2Fid+%2B+1',
        );
    });

    it('forwards browser chat-session query parameters into the upstream bridge', () => {
        const forwardingGateway = new VoiceGateway({
            verify: jest.fn().mockReturnValue({ id: 'user-1' }),
        } as any) as any;
        forwardingGateway.bridge = jest.fn();

        forwardingGateway.handleConnection(
            { close: jest.fn() } as any,
            {
                url: '/voice/stream?token=jwt&session_id=session-1&chat_llm_model=openai%3Agpt-4.1&chat_session_action=resume&chat_session_id=server%2Fid',
            } as any,
        );

        expect(forwardingGateway.bridge).toHaveBeenCalledWith(
            expect.anything(),
            'user-1',
            'session-1',
            'openai:gpt-4.1',
            'resume',
            'server/id',
        );
    });

    it('normalizes chat LLM model selectors without restricting provider-specific names', () => {
        expect(gateway.normalizeChatLlmModel(' hosted:qwen/qwen3-32b ')).toBe('hosted:qwen/qwen3-32b');
        expect(gateway.normalizeChatLlmModel('')).toBeNull();
        expect(gateway.normalizeChatLlmModel(null)).toBeNull();
    });

    it('sanitizes session_info without injecting or accepting tool fields', () => {
        const frame = gateway.sanitizeContextFrame(Buffer.from(JSON.stringify({
            type: 'session_info',
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
            tools: [{ name: 'client_tool' }],
            tool_metadata: { client_tool: { title: 'Untrusted' } },
            query_scope_schema: { type: 'client_schema' },
            tool_result_handling: 'client handling',
            session_context: {
                session_mode: 'recorded',
                context_kind: 'live',
                active_agent_session: { agent_mode: 'live_performance_analyst' },
                agent_session: { agent_mode: 'live_performance_analyst' },
                agent_modes: { active: ['live_performance_analyst'] },
                active_screen: { assistant_mode: 'live' },
            },
        })), false);

        expect(frame.isBinary).toBe(false);
        expect(JSON.parse(frame.data.toString())).toEqual({
            type: 'session_info',
            session_context: { session_mode: 'recorded' },
        });
    });

    it('sanitizes context updates while preserving direct canonical modes', () => {
        const raw = Buffer.from(JSON.stringify({
            type: 'session_context',
            agent_mode: 'overtake',
            session_context: {
                session_mode: 'live',
                agent_mode: 'track_guide',
                context_kind: 'recorded',
                active_screen: { assistant_mode: 'recorded', label: 'Live' },
                agent_session: { agent_mode: 'live_performance_analyst' },
            },
        }));
        const frame = gateway.sanitizeContextFrame(raw, false);

        expect(JSON.parse(frame.data.toString())).toEqual({
            type: 'session_context',
            session_context: {
                session_mode: 'live',
                agent_mode: 'track_guide',
            },
        });
    });

    it('does not strip operational agent_mode fields from tool results', () => {
        const raw = Buffer.from(JSON.stringify({
            type: 'tool_result',
            result: { agent_mode: 'live_performance_analyst' },
        }));
        const frame = gateway.sanitizeContextFrame(raw, false);

        expect(frame).toEqual({ data: raw, isBinary: false });
    });
});
