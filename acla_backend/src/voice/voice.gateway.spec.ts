import { VoiceGateway } from './voice.gateway';
import {
    FRONTEND_APPLICATION_QUERY_SCOPE_SCHEMA,
    getAiToolMetadataForSessionContext,
    getFrontendApplicationToolsForSessionContext,
} from '../shared/ai/frontend-application-tool-registry';

describe('VoiceGateway', () => {
    const gateway = new VoiceGateway({ verify: jest.fn() } as any) as any;

    it('replaces frontend_info tool metadata with the backend frontend application registry', () => {
        const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
            type: 'frontend_info',
            tools: [{ name: 'client_filtered_tool' }],
            query_scope_schema: { type: 'client_schema' },
            session_context: { session_mode: 'recorded' },
        })), false);

        const payload = JSON.parse(frame.data.toString());

        expect(frame.isBinary).toBe(false);
        expect(payload.session_context).toEqual({ session_mode: 'recorded' });
        expect(payload.tools).toEqual(getFrontendApplicationToolsForSessionContext({
            session_mode: 'recorded',
        }));
        expect(payload.tool_metadata).toEqual(getAiToolMetadataForSessionContext({
            session_mode: 'recorded',
        }));
        expect(payload.tool_metadata.run_recorded_ai_analysis.title)
            .toBe('Running recorded session AI analysis');
        expect(payload.query_scope_schema).toEqual(FRONTEND_APPLICATION_QUERY_SCOPE_SCHEMA);
    });

    it('keeps live-only tools out of recorded frontend_info frames', () => {
        const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
            type: 'frontend_info',
            session_context: { session_mode: 'recorded' },
        })), false);

        const payload = JSON.parse(frame.data.toString());
        const toolNames = payload.tools.map((tool: { name: string }) => tool.name);

        expect(toolNames).toEqual(expect.arrayContaining([
            'run_recorded_ai_analysis',
            'get_recorded_session_analysis',
            'stop_agent_session',
        ]));
        expect(toolNames).not.toEqual(expect.arrayContaining([
            'start_agent_session',
            'get_live_session_snapshot',
            'restart_live_baseline',
        ]));
    });

    it('passes non-handshake text frames through unchanged', () => {
        const raw = Buffer.from(JSON.stringify({ type: 'session_context', session_context: {} }));
        const frame = gateway.withBackendToolRegistry(raw, false);

        expect(frame).toEqual({ data: raw, isBinary: false });
    });
});
