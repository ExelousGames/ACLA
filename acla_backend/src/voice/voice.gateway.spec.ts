import { VoiceGateway } from './voice.gateway';
import {
    FRONTEND_APPLICATION_QUERY_SCOPE_SCHEMA,
    getAiToolMetadataForSessionContext,
    getFrontendApplicationToolsForSessionContext,
} from '../shared/ai/frontend-application-tool-registry';

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

    it('advertises front desk tools without falling back to live mode', () => {
        const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
            type: 'frontend_info',
            session_context: { session_mode: 'front_desk' },
        })), false);

        const payload = JSON.parse(frame.data.toString());
        const toolNames = payload.tools.map((tool: { name: string }) => tool.name);

        expect(payload.session_context).toEqual({ session_mode: 'front_desk' });
        expect(toolNames).toEqual(expect.arrayContaining([
            'show_map',
            'set_procedure_plan',
            'advance_plan_step',
            'clear_procedure_plan',
            'get_available_user_summary_maps',
            'search_user_summary_map_level',
        ]));
        expect(toolNames).not.toEqual(expect.arrayContaining([
            'start_agent_session',
            'analyze_telemetry',
            'get_next_corner',
            'query_telemetry_metric',
            'run_recorded_ai_analysis',
            'get_recorded_session_context',
        ]));
        expect(payload.tool_metadata.get_available_user_summary_maps.title)
            .toBe('Listing user summary maps');
        expect(payload.tool_metadata.start_agent_session).toBeUndefined();
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
            'set_live_range_todo_list',
            'update_live_range_todo_list',
            'get_live_range_todo_list',
            'collect_live_baseline',
            'restart_live_baseline',
            'analyze_live_recorded_analysis',
            'classify_live_section',
        ]));
    });

    it('advertises live analyst helper tools to child agent frontend_info frames', () => {
        const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
            type: 'frontend_info',
            session_context: {
                session_mode: 'live',
                conversation_role: 'agent',
                agent_mode: 'live_performance_analyst',
            },
        })), false);

        const payload = JSON.parse(frame.data.toString());
        const toolNames = payload.tools.map((tool: { name: string }) => tool.name);

        expect(toolNames).toEqual(expect.arrayContaining([
            'collect_live_baseline',
            'restart_live_baseline',
            'analyze_live_recorded_analysis',
            'get_live_analysis_mistake_count',
            'create_goal',
            'retry_goal_task',
            'set_live_range_todo_list',
            'update_live_range_todo_list',
            'get_live_range_todo_list',
            'classify_live_section',
            'analyze_telemetry',
            'stop_agent_session',
        ]));
        expect(toolNames).not.toEqual(expect.arrayContaining([
            'set_live_range_tracker',
            'update_live_range_tracker',
            'get_live_range_tracker',
        ]));
        expect(toolNames).not.toContain('start_agent_session');
        expect(payload.tool_metadata.set_live_range_todo_list.description)
            .toContain('AI Chat mounts the list');
        expect(payload.tool_metadata.set_live_range_todo_list.description)
            .toContain('attaches its notification callback');
        expect(payload.tool_metadata.get_live_analysis_mistake_count.title)
            .toBe('Counting live analysis mistakes');
        expect(payload.tool_metadata.create_goal.title).toBe('Creating goal');
        expect(payload.tool_metadata.retry_goal_task.title).toBe('Retrying failed goal task');
        const createGoal = payload.tools.find((tool: { name: string }) => tool.name === 'create_goal');
        expect(createGoal.properties.steps.items.properties.name.enum).toEqual(
            toolNames.filter((name: string) => (
                name !== 'create_goal' && name !== 'retry_goal_task'
            )),
        );
        expect(createGoal.properties.steps.items.properties.name.enum)
            .not.toContain('retry_goal_task');
    });

    it('keeps live analysis mistake counting out of non-analyst child frames', () => {
        for (const agentMode of ['track_guide', 'overtake']) {
            const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
                type: 'frontend_info',
                session_context: {
                    session_mode: 'live',
                    conversation_role: 'agent',
                    agent_mode: agentMode,
                },
            })), false);
            const payload = JSON.parse(frame.data.toString());
            expect(payload.tools.map((tool: { name: string }) => tool.name))
                .not.toContain('get_live_analysis_mistake_count');
            expect(payload.tools.map((tool: { name: string }) => tool.name))
                .not.toContain('create_goal');
            expect(payload.tools.map((tool: { name: string }) => tool.name))
                .not.toContain('retry_goal_task');
        }
    });

    it('advertises query_telemetry_metric field guidance in live frontend_info frames', () => {
        const frame = gateway.withBackendToolRegistry(Buffer.from(JSON.stringify({
            type: 'frontend_info',
            session_context: { session_mode: 'live' },
        })), false);

        const payload = JSON.parse(frame.data.toString());
        const metadata = payload.tool_metadata.query_telemetry_metric;

        expect(metadata.description).toContain('selected fields over a live-session scope');
        expect(metadata.description).toContain('Do not use `query_telemetry_metric`');
        expect(metadata.description).toContain('performance checking, pace diagnosis');
        expect(metadata.description).toContain('use `live_performance_analyst`');
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
