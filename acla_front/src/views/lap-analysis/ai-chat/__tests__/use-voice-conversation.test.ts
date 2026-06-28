import { buildVoiceSessionMetadata } from '../use-voice-conversation';

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
