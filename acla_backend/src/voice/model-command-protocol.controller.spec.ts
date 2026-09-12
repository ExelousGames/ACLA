import {
    BadRequestException,
    RequestMethod,
} from '@nestjs/common';
import {
    GUARDS_METADATA,
    METHOD_METADATA,
    PATH_METADATA,
} from '@nestjs/common/constants';
import { ModelCommandProtocolController } from './model-command-protocol.controller';

describe('ModelCommandProtocolController', () => {
    const controller = new ModelCommandProtocolController();

    it('requires the JWT auth guard', () => {
        const guards = Reflect.getMetadata(
            GUARDS_METADATA,
            ModelCommandProtocolController.prototype.getModelCommands,
        );
        expect(guards).toHaveLength(1);
    });

    it('exposes POST /model-command-protocol', () => {
        expect(Reflect.getMetadata(
            PATH_METADATA,
            ModelCommandProtocolController.prototype.getModelCommands,
        )).toBe('model-command-protocol');
        expect(Reflect.getMetadata(
            METHOD_METADATA,
            ModelCommandProtocolController.prototype.getModelCommands,
        )).toBe(RequestMethod.POST);
    });

    it('relies on the JWT guard without a controller-level identity check', () => {
        expect(controller.getModelCommands({
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
        expect(() => controller.getModelCommands(body))
            .toThrow(BadRequestException);
    });

    it('preserves the recorded-session allowlist', () => {
        const tools = controller.getModelCommands({
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
        const tools = controller.getModelCommands({
            session_context: {
                session_mode: 'live',
                agent_mode: 'live_performance_analyst',
            },
        });
        const names = tools.map(({ name }) => name);

        expect(names).toEqual(expect.arrayContaining([
            'collect_live_baseline',
            'apply_query_to_analysis_result',
            'query_lap_analysis_result',
            'create_repeatable_plan',
            'add_analysis_result_to_do_list',
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

    it('returns workflow envelopes for creation, reads, and controls', () => {
        const commands = controller.getModelCommands({
            session_context: { session_mode: 'live', agent_mode: 'live_performance_analyst' },
        });
        const creationNames = [
            'set_procedure_plan',
            'create_repeatable_plan',
            'add_event_to_live_range_todo_list',
        ];
        creationNames.forEach((name) => {
            const command = commands.find((entry) => entry.name === name) as any;
            expect(Object.keys(command.properties)).toEqual(['workflow']);
            expect(command.required).toEqual(['workflow']);
            expect(command.properties.workflow.additionalProperties).toBe(false);
            expect(command.properties.workflow.properties.operations.items.properties.operation.oneOf.length).toBeGreaterThan(0);
        });
        [
            'get_live_range_todo_list',
        ].forEach((name) => {
            expect(commands.find((entry) => entry.name === name))
                .toMatchObject({ properties: { workflow: { required: ['name', 'operations'], properties: { operations: { maxItems: 0 } } } }, required: ['workflow'] });
        });
        ['advance_plan_step', 'clear_procedure_plan'].forEach((name) => {
            const command = commands.find((entry) => entry.name === name) as any;
            expect(Object.keys(command.properties.workflow.properties)).toEqual(['name', 'operations', 'reason']);
            expect(command.required).toEqual(['workflow']);
        });
    });

    it('returns a no-argument tool envelope for transferring analysis comparison graphs', () => {
        const commands = controller.getModelCommands({
            session_context: { session_mode: 'live', agent_mode: 'live_performance_analyst' },
        });
        const command = commands.find(({ name }) => name === 'add_analysis_result_to_do_list') as any;
        expect(Object.keys(command.properties)).toEqual(['tool']);
        expect(command.required).toEqual(['tool']);
        expect(command.properties.tool).toMatchObject({
            required: ['name'],
            properties: {
                name: { enum: ['add_analysis_result_to_do_list'] },
                arguments: { properties: {}, required: [], additionalProperties: false },
            },
        });
    });

    it('returns the JSONata analysis-result query contract', () => {
        const tools = controller.getModelCommands({
            session_context: { session_mode: 'recorded' },
        });
        const tool = tools.find(({ name }) => name === 'query_lap_analysis_result') as any;

        expect(tool.properties.tool.properties.arguments.required).toEqual(['query']);
        expect(Object.keys(tool.properties.tool.properties.arguments.properties)).toEqual(['query']);
        expect(tool.properties.tool.properties.arguments.properties.query).toMatchObject({
            type: 'string',
            minLength: 1,
            pattern: '\\S',
        });
        expect(tool.properties.tool.properties.arguments.properties.query).not.toHaveProperty('enum');
        expect(tool.description).toContain('actual JSON-safe JSONata value');
        expect(tool.description).toContain('$count(analyses)');
        expect(tool.description).toContain('exactly one root structure');
        expect(tool.description).not.toContain('result_count');
        expect(tool.description).not.toContain('mistake_count');
    });

    it('returns the JSONata analysis-result apply contract', () => {
        const tools = controller.getModelCommands({
            session_context: { session_mode: 'recorded' },
        });
        const tool = tools.find(({ name }) => name === 'apply_query_to_analysis_result') as any;

        expect(tool.properties.tool.properties.arguments.required).toEqual(['query']);
        expect(Object.keys(tool.properties.tool.properties.arguments.properties)).toEqual(['query', 'page_number']);
        expect(tool.properties.tool.properties.arguments.properties.query).toMatchObject({
            type: 'string',
            minLength: 1,
            pattern: '\\S',
        });
        expect(tool.properties.tool.properties.arguments.properties.page_number).toMatchObject({ type: 'integer' });
        expect(tool.description).toContain('returns only its status');
        expect(tool.description).not.toContain('matched element count');
        expect(tool.description).toContain('retained-page array order');
        expect(tool.description).toContain('manual Apply');
    });

    it.each(['front_desk', 'live', 'recorded', 'user_summary'])(
        'accepts session mode %s',
        (sessionMode) => {
            expect(controller.getModelCommands({
                session_context: { session_mode: sessionMode },
            }).length).toBeGreaterThan(0);
        },
    );
});
