import {
    BadRequestException,
    RequestMethod,
} from '@nestjs/common';
import {
    GUARDS_METADATA,
    METHOD_METADATA,
    PATH_METADATA,
} from '@nestjs/common/constants';
import { SessionToolsController } from './session-tools.controller';

describe('SessionToolsController', () => {
    const controller = new SessionToolsController();

    it('requires the JWT auth guard', () => {
        const guards = Reflect.getMetadata(
            GUARDS_METADATA,
            SessionToolsController.prototype.getSessionTools,
        );
        expect(guards).toHaveLength(1);
    });

    it('exposes POST /session-tools', () => {
        expect(Reflect.getMetadata(
            PATH_METADATA,
            SessionToolsController.prototype.getSessionTools,
        )).toBe('session-tools');
        expect(Reflect.getMetadata(
            METHOD_METADATA,
            SessionToolsController.prototype.getSessionTools,
        )).toBe(RequestMethod.POST);
    });

    it('relies on the JWT guard without a controller-level identity check', () => {
        expect(controller.getSessionTools({
            session_context: { session_mode: 'live' },
        }).length).toBeGreaterThan(0);
    });

    it.each([
        undefined,
        {},
        { session_context: {} },
        { session_context: { session_mode: 'unknown' } },
        { session_context: { session_mode: 'live', agent_mode: 'unknown' } },
    ])('rejects invalid session context: %p', (body) => {
        expect(() => controller.getSessionTools(body))
            .toThrow(BadRequestException);
    });

    it('preserves the recorded-session allowlist', () => {
        const tools = controller.getSessionTools({
            session_context: { session_mode: 'recorded' },
        });
        const names = tools.map(({ name }) => name);

        expect(names).toEqual(expect.arrayContaining([
            'run_recorded_ai_analysis',
            'get_recorded_session_analysis',
            'apply_query_to_analysis_result',
            'stop_agent_session',
        ]));
        expect(names).not.toEqual(expect.arrayContaining([
            'start_agent_session',
            'restart_live_baseline',
        ]));
    });

    it('preserves the live analyst allowlist and exact response shape', () => {
        const tools = controller.getSessionTools({
            session_context: {
                session_mode: 'live',
                agent_mode: 'live_performance_analyst',
            },
        });
        const names = tools.map(({ name }) => name);

        expect(names).toEqual(expect.arrayContaining([
            'collect_live_baseline',
            'apply_query_to_analysis_result',
            'query_analysis_result',
            'create_goal',
            'retry_goal_task',
            'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
        ]));
        expect(names).not.toContain('start_agent_session');
        expect(tools.every((tool) => (
            Object.keys(tool).sort().join(',')
            === 'description,name,properties,required'
        ))).toBe(true);
        expect(tools.every(({ description }) => typeof description === 'string'))
            .toBe(true);
        expect(tools.some((tool) => 'title' in tool)).toBe(false);
    });

    it('returns the JSONata analysis-result query contract', () => {
        const tools = controller.getSessionTools({
            session_context: { session_mode: 'recorded' },
        });
        const tool = tools.find(({ name }) => name === 'query_analysis_result') as any;

        expect(tool.required).toEqual(['query']);
        expect(Object.keys(tool.properties)).toEqual(['query']);
        expect(tool.properties.query).toMatchObject({
            type: 'string',
            minLength: 1,
            pattern: '\\S',
        });
        expect(tool.properties.query).not.toHaveProperty('enum');
        expect(tool.description).toContain('actual JSON-safe JSONata value');
        expect(tool.description).toContain('$count(analyses)');
        expect(tool.description).toContain('exactly one root structure');
        expect(tool.description).not.toContain('result_count');
        expect(tool.description).not.toContain('mistake_count');
    });

    it('returns the JSONata analysis-result apply contract', () => {
        const tools = controller.getSessionTools({
            session_context: { session_mode: 'recorded' },
        });
        const tool = tools.find(({ name }) => name === 'apply_query_to_analysis_result') as any;

        expect(tool.required).toEqual(['query']);
        expect(Object.keys(tool.properties)).toEqual(['query', 'page_number']);
        expect(tool.properties.query).toMatchObject({
            type: 'string',
            minLength: 1,
            pattern: '\\S',
        });
        expect(tool.properties.page_number).toMatchObject({ type: 'integer' });
        expect(tool.description).toContain('returns only its status');
        expect(tool.description).not.toContain('matched element count');
        expect(tool.description).toContain('retained-page array order');
        expect(tool.description).toContain('manual Apply');
    });

    it.each(['front_desk', 'live', 'recorded', 'user_summary'])(
        'accepts session mode %s',
        (sessionMode) => {
            expect(controller.getSessionTools({
                session_context: { session_mode: sessionMode },
            }).length).toBeGreaterThan(0);
        },
    );
});
