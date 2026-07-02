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
            tool_result_handling: 'client handling',
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
        expect(payload.tool_result_handling).toBeUndefined();
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
            'classify_live_section',
            'restart_live_baseline',
        ]));
    });

    it('keeps disabled live helpers out of live frontend_info frames', () => {
        const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
            type: 'frontend_info',
            session_context: { session_mode: 'live' },
        })), false);

        const payload = JSON.parse(frame.data.toString());
        const toolNames = payload.tools.map((tool: { name: string }) => tool.name);

        expect(toolNames).toEqual(expect.arrayContaining([
            'start_agent_session',
            'analyze_telemetry',
            'get_next_corner',
            'query_telemetry_metric',
            'get_event_log',
        ]));
        expect(toolNames).not.toEqual(expect.arrayContaining([
            'set_live_range_tracker',
            'update_live_range_tracker',
            'get_live_range_tracker',
            'collect_live_baseline',
            'restart_live_baseline',
            'analyze_live_recorded_analysis',
            'classify_live_section',
        ]));
    });

    it('advertises query_telemetry_metric field guidance in live frontend_info frames', () => {
        const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
            type: 'frontend_info',
            session_context: { session_mode: 'live' },
        })), false);

        const payload = JSON.parse(frame.data.toString());
        const metadata = payload.tool_metadata.query_telemetry_metric;

        expect(metadata.description).toContain('selected fields over a live-session scope');
        expect(metadata.parameters.fields.description).toContain('Physics_speed_kmh');
        expect(metadata.parameters.fields.description).toContain('Physics_brake_pressure_front_left');
        expect(metadata.parameters.fields.description).toContain('Graphics_current_time');
        expect(metadata.parameters.fields.description).toContain('Graphics_fuel_per_lap');
        expect(metadata.parameters.fields.description).toContain('Static_track');
        expect(metadata.parameters.fields.description).toContain('Static_car_model');
        expect(metadata.parameters.fields.description).toContain('do not invent unlisted names');
        expect(metadata.parameters.scope.description).toContain('type="now"');
        expect(metadata.parameters.scope.description).toContain('last_seconds, event, lap, or range');
        expect(metadata.parameters.reduce.description).toContain('avg, min, max, or stats');
        expect(metadata.parameters.reduce.description).toContain('Prefer stats');
    });

    it('advertises only frontend-held AI request helpers as frontend tools', () => {
        const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
            type: 'frontend_info',
            session_context: { session_mode: 'recorded' },
        })), false);

        const payload = JSON.parse(frame.data.toString());
        const toolNames = payload.tools.map((tool: { name: string }) => tool.name);

        expect(toolNames).toEqual(expect.arrayContaining([
            'analyze_telemetry',
        ]));
        expect(toolNames).not.toEqual(expect.arrayContaining([
            'explain_label',
            'get_track_knowledge',
            'search_racing_knowledge',
        ]));
        expect(payload.tool_metadata.analyze_telemetry.title).toBe('Analyzing telemetry');
    });

    it('passes non-handshake text frames through unchanged', () => {
        const raw = Buffer.from(JSON.stringify({ type: 'session_context', session_context: {} }));
        const frame = gateway.withBackendToolRegistry(raw, false);

        expect(frame).toEqual({ data: raw, isBinary: false });
    });
});
