import {
    buildVoiceSessionMetadata,
    extractInlineFunctionCalls,
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
