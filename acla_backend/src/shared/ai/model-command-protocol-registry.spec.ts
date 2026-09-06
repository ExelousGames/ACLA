import Ajv = require('ajv');
import {
    MODEL_COMMAND_PROTOCOL,
    getModelCommandsForSessionContext,
} from './model-command-protocol-registry';
import {
    TELEMETRY_METRIC_FIELD_DEFINITIONS,
    TELEMETRY_METRIC_FIELD_SCHEMA,
} from './telemetry-metric-fields';

const WORKFLOW_NAMES = [
    'create_repeatable_plan',
    'retry_repeatable_plan_task',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    'add_event_to_live_range_todo_list',
    'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
    'get_live_range_todo_list',
];
const getWorkflowSchema = (command: any) => command.properties[command.name];
const getCallNames = (schema: any): string[] => schema.oneOf.map((branch: any) => branch.required[0]);
const getCallMetadata = (schema: any, name: string) => (
    schema.oneOf.find((branch: any) => branch.required[0] === name).properties[name]
);

describe('live baseline tools', () => {
    it('directs new recordings through collect and limits restart to active recordings', () => {
        const collect = MODEL_COMMAND_PROTOCOL.find(({ name }) => name === 'collect_live_baseline') as any;
        const restart = MODEL_COMMAND_PROTOCOL.find(({ name }) => name === 'restart_live_baseline') as any;

        expect(collect?.description).toContain('holds only one recording at a time');
        expect(collect?.description).toContain('starts a new recording');
        expect(restart?.description).toContain('only while it is waiting for its start condition or actively collecting');
        expect(restart?.description).toContain('use collect_live_baseline');
    });

    it('requires either the full-lap preset or mutually exclusive custom conditions', () => {
        const collect = MODEL_COMMAND_PROTOCOL.find(({ name }) => name === 'collect_live_baseline') as any;
        const [presetQuery, customQuery] = collect.properties.query.oneOf;

        expect(collect.required).toEqual(['query']);
        expect(presetQuery).toMatchObject({
            required: ['preset'],
            additionalProperties: false,
            properties: {
                preset: { enum: ['full_lap'] },
            },
        });
        expect(Object.keys(presetQuery.properties)).toEqual(['preset']);
        expect(customQuery).toMatchObject({
            required: ['start_query', 'end_query'],
            additionalProperties: false,
        });
        expect(Object.keys(customQuery.properties).sort()).toEqual(['end_query', 'start_query']);
        [customQuery.properties.start_query, customQuery.properties.end_query]
            .forEach((condition) => {
                expect(condition).toMatchObject({
                    required: ['field', 'operator', 'value'],
                    additionalProperties: false,
                    properties: {
                        field: TELEMETRY_METRIC_FIELD_SCHEMA,
                        operator: { enum: ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'] },
                        value: { type: 'number' },
                    },
                });
            });
    });
});

describe('session live range to-do tools', () => {
    it('exposes one strict executable-event batch schema plus the read tool', () => {
        const names = MODEL_COMMAND_PROTOCOL.map((tool) => tool.name);
        expect(names.filter((name) => name.endsWith('_live_range_todo_list'))).toEqual([
            'add_event_to_live_range_todo_list',
            'get_live_range_todo_list',
            'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
        ]);

        const addTool = MODEL_COMMAND_PROTOCOL.find((tool) => (
            tool.name === 'add_event_to_live_range_todo_list'
        )) as any;
        expect(addTool.required).toEqual(['add_event_to_live_range_todo_list']);
        const workflowSchema = getWorkflowSchema(addTool);
        expect(workflowSchema).toMatchObject({
            required: ['tools'],
            additionalProperties: false,
            properties: { tools: { minItems: 1 } },
        });
        const metadata = getCallMetadata(workflowSchema.properties.tools.items, 'show_map');
        const eventSchema = metadata.properties.event;
        const contentSchema = eventSchema.properties.content;
        expect(Object.keys(metadata.properties)).toEqual(['event', 'arguments']);
        expect(metadata.required).toEqual(['event', 'arguments']);
        expect(metadata.additionalProperties).toBe(false);
        expect(Object.keys(eventSchema.properties)).toEqual([
            'id',
            'normalized_position',
            'lead_time_seconds',
            'content',
        ]);
        expect(eventSchema.required).toEqual(['id', 'normalized_position', 'content']);
        expect(eventSchema.additionalProperties).toBe(false);
        expect(Object.keys(contentSchema.properties)).toEqual(['title', 'description']);
        expect(contentSchema.required).toEqual(['title']);
        expect(contentSchema.additionalProperties).toBe(false);
        expect(eventSchema.properties.lead_time_seconds.description).toContain('Defaults to 2 seconds');
    });

    it('advertises only add/read to child live agents and derives tool-only alternatives', () => {
        const liveMainNames = getModelCommandsForSessionContext({
            session_mode: 'live',
        }).map((tool) => tool.name);
        const liveAgentTools = getModelCommandsForSessionContext({
            session_mode: 'live',
            agent_mode: 'track_guide',
        });
        const liveAgentNames = liveAgentTools.map((tool) => tool.name);

        expect(liveMainNames.filter((name) => name.endsWith('_live_range_todo_list')))
            .toEqual([]);
        expect(liveAgentNames.filter((name) => name.endsWith('_live_range_todo_list')))
            .toEqual([
                'add_event_to_live_range_todo_list',
                'get_live_range_todo_list',
            ]);

        const addTool = liveAgentTools.find(({ name }) => (
            name === 'add_event_to_live_range_todo_list'
        )) as any;
        const nestedNames = getCallNames(getWorkflowSchema(addTool).properties.tools.items);
        expect(nestedNames).toEqual(expect.arrayContaining([
            'analyze_telemetry',
            'query_telemetry_metric',
        ]));
        WORKFLOW_NAMES.forEach((name) => expect(nestedNames).not.toContain(name));
        const liveAgentNameSet = new Set<string>(liveAgentNames);
        expect(nestedNames.every((name: string) => liveAgentNameSet.has(name))).toBe(true);
        expect(addTool.description).toContain('AI Chat mounts the list');
        expect(addTool.description).toContain('returns the updated list summary immediately');

        const analystAddTool = getModelCommandsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        }).find(({ name }) => name === 'add_event_to_live_range_todo_list') as any;
        expect(getCallNames(getWorkflowSchema(analystAddTool).properties.tools.items))
            .toEqual(nestedNames);
    });

    it('selects tools only from direct canonical mode fields', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getModelCommandsForSessionContext(context).map(({ name }) => name)
        );

        expect(namesFor({
            session_mode: 'live',
            conversation_role: 'agent',
            active_agent_session: { agent_mode: 'live_performance_analyst' },
            agent_session: { agent_mode: 'live_performance_analyst' },
            agent_modes: { active: ['live_performance_analyst'] },
        })).not.toContain('add_event_to_live_range_todo_list');
        expect(namesFor({
            session_mode: 'recorded',
            context_kind: 'live',
            active_screen: { assistant_mode: 'live' },
        })).not.toContain('start_agent_session');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'track_guide',
            agent_session: { agent_mode: 'live_performance_analyst' },
        })).not.toContain('create_repeatable_plan');
    });
});

describe('filtered Driver/Expert comparison queue tool', () => {
    const namesFor = (context: Record<string, unknown>) => (
        getModelCommandsForSessionContext(context).map(({ name }) => name)
    );

    it('defines a strict no-argument schema only for the live performance analyst', () => {
        const tool = MODEL_COMMAND_PROTOCOL.find(({ name }) => (
            name === 'add_filtered_driver_expert_comparisons_to_live_range_todo_list'
        ));
        expect(tool).toMatchObject({ properties: {}, required: [] });

        expect(namesFor({ session_mode: 'live' }))
            .not.toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
        expect(namesFor({ session_mode: 'live', agent_mode: 'track_guide' }))
            .not.toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
        expect(namesFor({ session_mode: 'live', agent_mode: 'overtake' }))
            .not.toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
        expect(namesFor({
            session_mode: 'recorded',
            agent_mode: 'live_performance_analyst',
        })).not.toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        })).toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
    });

    it('keeps automatic comparison queueing standalone and excludes it from child calls', () => {
        const analystTools = getModelCommandsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const repeatablePlan = analystTools.find(({ name }) => name === 'create_repeatable_plan') as any;
        const addEvents = analystTools.find(({ name }) => (
            name === 'add_event_to_live_range_todo_list'
        )) as any;
        const repeatablePlanNames = getCallNames(getWorkflowSchema(repeatablePlan).properties.tools.items);
        const nestedLiveRangeNames = getCallNames(getWorkflowSchema(addEvents).properties.tools.items);

        expect(repeatablePlanNames)
            .not.toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
        expect(nestedLiveRangeNames)
            .not.toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
        expect(analystTools.map(({ name }) => name)).toContain('set_procedure_plan');
    });
});

describe('analysis result query tool', () => {
    const namesFor = (context: Record<string, unknown>) => (
        getModelCommandsForSessionContext(context).map(({ name }) => name)
    );
    const eligibleContexts = [
        { session_mode: 'live' },
        { session_mode: 'recorded' },
        { session_mode: 'live', agent_mode: 'track_guide' },
        { session_mode: 'live', agent_mode: 'overtake' },
        { session_mode: 'live', agent_mode: 'live_performance_analyst' },
    ];

    it('requires one non-blank JSONata expression without a legacy enum', () => {
        const tool = MODEL_COMMAND_PROTOCOL.find(({ name }) => (
            name === 'query_analysis_result'
        )) as any;

        expect(MODEL_COMMAND_PROTOCOL.filter(({ name }) => name === 'query_analysis_result'))
            .toHaveLength(1);
        expect(tool).toMatchObject({
            description: expect.any(String),
            properties: {
                query: {
                    type: 'string',
                    minLength: 1,
                    pattern: '\\S',
                },
            },
            required: ['query'],
        });
        expect(Object.keys(tool.properties)).toEqual(['query']);
        expect(tool.properties.query).not.toHaveProperty('enum');
        expect(new RegExp(tool.properties.query.pattern).test('   ')).toBe(false);
        expect(new RegExp(tool.properties.query.pattern).test('$count(analyses)'))
            .toBe(true);
    });

    it('describes one all-analysis root in every eligible context', () => {
        eligibleContexts.forEach((context) => {
            const tool = getModelCommandsForSessionContext(context).find(({ name }) => (
                name === 'query_analysis_result'
            ));
            const description = tool?.description ?? '';

            expect(description).toContain('exactly one root structure');
            expect(description).toContain('"analyses"');
            expect(description).toContain('"elements"');
            expect(description).toContain('"normalizedPositionRange"');
            expect(description).toContain('actual JSON-safe JSONata value');
            expect(description).toContain('not a count unless the expression returns one');
            expect(description).toContain('$count(analyses) counts analyses');
            expect(description).toContain('$count(analyses.elements)');
            expect(description).toContain('analyses.elements[labels[$ = "Lockup"]].{ "id": id, "section": section }');
            expect(description).not.toContain('active lap analysis');
        });
    });

    it('does not advertise the old identifiers as aliases', () => {
        const tool = MODEL_COMMAND_PROTOCOL.find(({ name }) => (
            name === 'query_analysis_result'
        ));
        const serializedTool = JSON.stringify(tool);

        expect(serializedTool).not.toContain('result_count');
        expect(serializedTool).not.toContain('mistake_count');
    });

    it('is advertised in live, live-agent, and recorded contexts only', () => {
        expect(namesFor({ session_mode: 'live' }))
            .toContain('query_analysis_result');
        expect(namesFor({ session_mode: 'recorded' }))
            .toContain('query_analysis_result');
        expect(namesFor({ session_mode: 'live', agent_mode: 'track_guide' }))
            .toContain('query_analysis_result');
        expect(namesFor({ session_mode: 'live', agent_mode: 'overtake' }))
            .toContain('query_analysis_result');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        })).toContain('query_analysis_result');
        expect(namesFor({ session_mode: 'front_desk' }))
            .not.toContain('query_analysis_result');
        expect(namesFor({ session_mode: 'user_summary' }))
            .not.toContain('query_analysis_result');
        expect(namesFor({ session_mode: 'front_desk', agent_mode: 'track_guide' }))
            .not.toContain('query_analysis_result');
        expect(namesFor({ session_mode: 'user_summary', agent_mode: 'track_guide' }))
            .not.toContain('query_analysis_result');
    });

    it('is available to compatible live analyst repeatable plan steps and stop conditions', () => {
        const tools = getModelCommandsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const repeatablePlan = tools.find(({ name }) => name === 'create_repeatable_plan') as any;

        expect(getCallNames(getWorkflowSchema(repeatablePlan).properties.tools.items))
            .toContain('query_analysis_result');
        expect(getCallNames(getWorkflowSchema(repeatablePlan).properties.stop_when.properties.tool))
            .toContain('query_analysis_result');
    });
});

describe('analysis result query apply tool', () => {
    const namesFor = (context: Record<string, unknown>) => (
        getModelCommandsForSessionContext(context).map(({ name }) => name)
    );
    const eligibleContexts = [
        { session_mode: 'live' },
        { session_mode: 'recorded' },
        { session_mode: 'live', agent_mode: 'track_guide' },
        { session_mode: 'live', agent_mode: 'overtake' },
        { session_mode: 'live', agent_mode: 'live_performance_analyst' },
    ];

    it('requires final non-blank JSONata and accepts only an optional integer page number', () => {
        const tool = MODEL_COMMAND_PROTOCOL.find(({ name }) => (
            name === 'apply_query_to_analysis_result'
        )) as any;

        expect(MODEL_COMMAND_PROTOCOL.filter(({ name }) => name === 'apply_query_to_analysis_result'))
            .toHaveLength(1);
        expect(Object.keys(tool.properties)).toEqual(['query', 'page_number']);
        expect(tool.required).toEqual(['query']);
        expect(tool.properties.query).toMatchObject({
            type: 'string',
            minLength: 1,
            pattern: '\\S',
        });
        expect(tool.properties.page_number).toMatchObject({ type: 'integer' });
        expect(tool.description).toContain('returns only its status');
        expect(tool.description).not.toContain('matched element count');
        expect(tool.description).toContain('receives only { "elements"');
        expect(tool.description).toContain('one element ID string');
        expect(tool.description).toContain('Unknown IDs and nested arrays are rejected');
        expect(tool.description).toContain('1-based retained-page array order');
        expect(tool.description).toContain('highest page number is the most recent analysis');
        expect(tool.description).toContain('same commit path as manual Apply');
    });

    it('is advertised exactly wherever the read query is available', () => {
        eligibleContexts.forEach((context) => {
            expect(namesFor(context)).toEqual(expect.arrayContaining([
                'apply_query_to_analysis_result',
                'query_analysis_result',
            ]));
        });
        [
            { session_mode: 'front_desk' },
            { session_mode: 'user_summary' },
            { session_mode: 'front_desk', agent_mode: 'track_guide' },
            { session_mode: 'user_summary', agent_mode: 'track_guide' },
        ].forEach((context) => {
            expect(namesFor(context)).not.toContain('apply_query_to_analysis_result');
        });
    });

    it('is available in analyst repeatable plans, stop conditions, and nested live-range workflows', () => {
        const tools = getModelCommandsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const repeatablePlan = tools.find(({ name }) => name === 'create_repeatable_plan') as any;
        const addEvents = tools.find(({ name }) => (
            name === 'add_event_to_live_range_todo_list'
        )) as any;

        expect(getCallNames(getWorkflowSchema(repeatablePlan).properties.tools.items))
            .toContain('apply_query_to_analysis_result');
        expect(getCallNames(getWorkflowSchema(repeatablePlan).properties.stop_when.properties.tool))
            .toContain('apply_query_to_analysis_result');
        expect(getCallNames(getWorkflowSchema(addEvents).properties.tools.items))
            .toContain('apply_query_to_analysis_result');
    });
});

describe('set_procedure_plan tool', () => {
    it('requires a same-name envelope with a goal and ordered titled tool calls', () => {
        const command = MODEL_COMMAND_PROTOCOL.find(({ name }) => name === 'set_procedure_plan') as any;
        expect(command.required).toEqual(['set_procedure_plan']);
        expect(Object.keys(command.properties)).toEqual(['set_procedure_plan']);
        const workflow = getWorkflowSchema(command);
        expect(Object.keys(workflow.properties)).toEqual(['goal', 'tools']);
        expect(workflow.required).toEqual(['goal', 'tools']);
        expect(workflow.additionalProperties).toBe(false);
        expect(workflow.properties.tools.minItems).toBe(1);
        expect(workflow.properties.goal.type).toBe('string');
        expect(workflow.properties.goal).not.toHaveProperty('minLength');
        expect(workflow.properties.goal).not.toHaveProperty('pattern');
        const metadata = getCallMetadata(workflow.properties.tools.items, 'show_map');
        expect(Object.keys(metadata.properties)).toEqual(['title', 'arguments']);
        expect(metadata.required).toEqual(['title', 'arguments']);
        expect(metadata.additionalProperties).toBe(false);
        expect(metadata.properties.title).toEqual({ type: 'string', minLength: 1, pattern: '\\S' });
    });
});

describe('create_repeatable_plan tool', () => {
    it('defines ordered keyed calls and a keyed numeric stop_when call', () => {
        const commands = getModelCommandsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const command = commands.find(({ name }) => name === 'create_repeatable_plan') as any;
        expect(command.required).toEqual(['create_repeatable_plan']);
        expect(Object.keys(command.properties)).toEqual(['create_repeatable_plan']);
        const workflow = getWorkflowSchema(command);
        expect(workflow).toMatchObject({
            required: ['name', 'tools', 'stop_when'],
            additionalProperties: false,
            properties: {
                name: { type: 'string' },
                tools: { minItems: 1 },
                stop_when: {
                    required: ['tool', 'operator', 'target'],
                    additionalProperties: false,
                    properties: {
                        operator: { enum: ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'] },
                        target: { type: 'number', description: expect.any(String) },
                    },
                },
            },
        });
        expect(Object.keys(workflow.properties)).toEqual(['name', 'tools', 'stop_when']);
        const step = getCallMetadata(workflow.properties.tools.items, 'query_analysis_result');
        expect(Object.keys(step.properties)).toEqual(['id', 'title', 'arguments']);
        expect(step.required).toEqual(['id', 'title']);
        expect(step.additionalProperties).toBe(false);
        expect(step.properties.arguments).toMatchObject({ type: 'object', default: {} });
        const stopWhen = workflow.properties.stop_when;
        const stopCall = getCallMetadata(stopWhen.properties.tool, 'query_analysis_result');
        expect(Object.keys(stopCall.properties)).toEqual(['arguments']);
        expect(stopCall.required).toEqual([]);
        expect(stopCall.additionalProperties).toBe(false);
        expect(stopCall.properties.arguments).toMatchObject({ type: 'object', default: {} });
        expect(Object.keys(stopWhen.properties)).toEqual(['tool', 'operator', 'target']);
        expect(stopWhen.description).toContain('{ "status": "ready", "data": finiteNumber }');
        expect(command.description).toContain('both values must be finite numbers');
        expect(stopWhen.description).toContain('comparable with the target');
        expect(stopWhen.properties.target.description)
            .toContain('comparable to the value returned by the stop-when tool');
    });

    it('exposes create_repeatable_plan only to the live performance analyst', () => {
        [
            { session_mode: 'live' },
            { session_mode: 'live', agent_mode: 'track_guide' },
            { session_mode: 'live', agent_mode: 'overtake' },
            { session_mode: 'recorded', agent_mode: 'live_performance_analyst' },
        ].forEach((context) => {
            expect(getModelCommandsForSessionContext(context).map(({ name }) => name))
                .not.toContain('create_repeatable_plan');
        });
        expect(getModelCommandsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        }).map(({ name }) => name)).toContain('create_repeatable_plan');
    });
});

describe('telemetry metric query tool', () => {
    it('requires the described supported fields, scope, and a summarized reduction', () => {
        const tool = MODEL_COMMAND_PROTOCOL.find(({ name }) => (
            name === 'query_telemetry_metric'
        ));

        expect(tool).toMatchObject({
            properties: {
                fields: {
                    type: 'array',
                    items: TELEMETRY_METRIC_FIELD_SCHEMA,
                },
                scope: {
                    type: 'object',
                    required: ['type'],
                },
                reduce: {
                    type: 'string',
                    enum: ['avg', 'min', 'max', 'stats'],
                },
            },
            required: ['fields', 'scope', 'reduce'],
        });
        expect(TELEMETRY_METRIC_FIELD_SCHEMA.enum).toEqual(
            TELEMETRY_METRIC_FIELD_DEFINITIONS.map(({ name }) => name),
        );
        TELEMETRY_METRIC_FIELD_DEFINITIONS.forEach(({ name, description }) => {
            expect(TELEMETRY_METRIC_FIELD_SCHEMA.description)
                .toContain(`${name}: ${description}`);
        });
        expect(TELEMETRY_METRIC_FIELD_SCHEMA.enum).toEqual([
            'Physics_speed_kmh',
            'Physics_gear',
            'Physics_rpm',
            'Physics_brake',
            'Physics_gas',
            'Graphics_normalized_car_position',
        ]);
    });
});

describe('retry_repeatable_plan_task tool', () => {
    it('defines a no-argument schema and is exposed only to the Live Performance Analyst', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getModelCommandsForSessionContext(context).map(({ name }) => name)
        );
        const tool = MODEL_COMMAND_PROTOCOL.find(({ name }) => (
            name === 'retry_repeatable_plan_task'
        ));
        expect(tool).toMatchObject({ properties: {}, required: [] });

        expect(namesFor({ session_mode: 'live' })).not.toContain('retry_repeatable_plan_task');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'track_guide',
        })).not.toContain('retry_repeatable_plan_task');
        expect(namesFor({
            session_mode: 'recorded',
            agent_mode: 'live_performance_analyst',
        })).not.toContain('retry_repeatable_plan_task');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        })).toContain('retry_repeatable_plan_task');
    });
});


describe('tool-only workflow schema validation', () => {
    const context = { session_mode: 'live', agent_mode: 'live_performance_analyst' };
    const commands = getModelCommandsForSessionContext(context);
    const ajv = new Ajv({ allErrors: true, strictNumbers: true });
    const event = {
        id: 'corner-1',
        normalized_position: 0.25,
        lead_time_seconds: 3,
        content: { title: 'Review corner', description: 'Compare the driving line.' },
    };
    const cases = [
        {
            name: 'set_procedure_plan',
            body: {
                goal: 'Review the session',
                tools: [{ query_analysis_result: { title: 'Count analyses', arguments: { query: '$count(analyses)' } } }],
            },
            metadata: { title: 'Count analyses', arguments: { query: '$count(analyses)' } },
            legacyKey: 'requests',
        },
        {
            name: 'create_repeatable_plan',
            body: {
                name: 'Practice a clean lap',
                tools: [{ query_analysis_result: { id: 'count', title: 'Count analyses', arguments: { query: '$count(analyses)' } } }],
                stop_when: { tool: { query_analysis_result: { arguments: { query: '$count(analyses)' } } }, operator: 'gte', target: 3 },
            },
            metadata: { id: 'count', title: 'Count analyses', arguments: { query: '$count(analyses)' } },
            legacyKey: 'steps',
        },
        {
            name: 'add_event_to_live_range_todo_list',
            body: {
                tools: [{ query_analysis_result: { event, arguments: { query: '$count(analyses)' } } }],
            },
            metadata: { event, arguments: { query: '$count(analyses)' } },
            legacyKey: 'events',
        },
    ];
    const compileCommand = (command: any) => ajv.compile({
        type: 'object',
        properties: command.properties,
        required: command.required,
        // Native tool adapters supply the strict parameter-object boundary.
        additionalProperties: false,
    });

    describe.each(cases)('$name', ({ name, body, metadata, legacyKey }) => {
        const command = commands.find((entry) => entry.name === name) as any;
        const validate = compileCommand(command);
        const envelope = (value: unknown) => ({ [name]: value });
        const withTools = (tools: unknown) => envelope({ ...body, tools });

        it('accepts ordered repeated tool calls and preserves metadata and arguments', () => {
            const second = JSON.parse(JSON.stringify(metadata));
            if ('id' in second) second.id = 'count-again';
            if ('event' in second) second.event.id = 'corner-2';
            second.arguments = { query: '$count(analyses.elements)', nested: { payload: ['kept', 2] } };
            const input = withTools([
                { query_analysis_result: metadata },
                { query_analysis_result: second },
            ]);
            const before = JSON.stringify(input);
            expect(validate(input)).toBe(true);
            expect(JSON.stringify(input)).toBe(before);
        });

        it('rejects malformed, missing, mismatched, and mixed wrappers', () => {
            [
                undefined, null, [], {}, body,
                { wrong_workflow: body },
                envelope(null), envelope([]), envelope({}),
                { ...envelope(body), extra: {} },
                { ...envelope(body), [legacyKey]: [] },
                envelope({ ...body, [legacyKey]: [] }),
                envelope({ ...body, current_request: 0 }),
            ].forEach((input) => expect(validate(input)).toBe(false));
            Object.keys(body).forEach((key) => {
                const missing: any = { ...body };
                delete missing[key];
                expect(validate(envelope(missing))).toBe(false);
            });
        });

        it('rejects empty or multiple tool keys, name descriptors, and metadata aliases', () => {
            const invalidEntries: unknown[] = [
                null, [], 'query_analysis_result', {},
                { query_analysis_result: metadata, show_map: metadata },
                { name: 'query_analysis_result', ...metadata },
                { tool: { name: 'query_analysis_result', arguments: {} } },
                { query_analysis_result: { ...metadata, name: 'query_analysis_result' } },
                { query_analysis_result: null },
                { query_analysis_result: [] },
            ];
            ['payload', 'args', 'parameters'].forEach((alias) => {
                const missingArguments: any = { ...metadata, [alias]: {} };
                delete missingArguments.arguments;
                invalidEntries.push(
                    { query_analysis_result: missingArguments },
                    { query_analysis_result: { ...metadata, [alias]: {} } },
                );
            });
            [null, [], 'query'].forEach((argumentsValue) => {
                invalidEntries.push({ query_analysis_result: { ...metadata, arguments: argumentsValue } });
            });
            invalidEntries.forEach((entry) => {
                // A malformed later entry invalidates the entire batch.
                expect(validate(withTools([{ query_analysis_result: metadata }, entry]))).toBe(false);
            });
            [null, {}, 'tools'].forEach((tools) => expect(validate(withTools(tools))).toBe(false));
        });

        it.each(WORKFLOW_NAMES)('rejects workflow child %s, including reads and controls', (childName) => {
            expect(validate(withTools([{ [childName]: metadata }]))).toBe(false);
        });

        it.each(['unknown_tool', 'start_agent_session', 'run_recorded_ai_analysis'])(
            'rejects unavailable child %s',
            (childName) => expect(validate(withTools([{ [childName]: metadata }]))).toBe(false),
        );

        it('retains metadata requirements and permits omitted arguments only for repeatable calls', () => {
            const missingArguments: any = { ...metadata };
            delete missingArguments.arguments;
            expect(validate(withTools([{ query_analysis_result: missingArguments }])))
                .toBe(name === 'create_repeatable_plan');
            Object.keys(metadata).filter((key) => key !== 'arguments').forEach((key) => {
                const missing: any = { ...metadata };
                delete missing[key];
                expect(validate(withTools([{ query_analysis_result: missing }]))).toBe(false);
            });
            expect(validate(withTools([]))).toBe(false);
            if (name === 'set_procedure_plan') {
                ['', ' \t\n '].forEach((blank) => {
                    expect(validate(envelope({ ...body, goal: blank }))).toBe(true);
                    expect(validate(withTools([
                        { query_analysis_result: metadata },
                        { query_analysis_result: { ...metadata, title: blank } },
                    ]))).toBe(false);
                });
            }
        });
    });

    it('keeps keyed stop arguments optional and rejects invalid stop conditions', () => {
        const repeatable = cases.find(({ name }) => name === 'create_repeatable_plan')!;
        const validate = compileCommand(commands.find(({ name }) => name === repeatable.name));
        const withStop = (stop_when: unknown) => ({ [repeatable.name]: { ...repeatable.body, stop_when } });
        const stop = { tool: { query_analysis_result: {} }, operator: 'gte', target: 3 };
        ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'].forEach((operator) => {
            expect(validate(withStop({ ...stop, operator }))).toBe(true);
        });
        [
            null, [], {},
            { ...stop, tool: {} },
            { ...stop, tool: { query_analysis_result: {}, show_map: {} } },
            { ...stop, tool: { name: 'query_analysis_result', arguments: {} } },
            { ...stop, tool: { query_analysis_result: { title: 'Not stop metadata' } } },
            { ...stop, tool: { query_analysis_result: { arguments: null } } },
            { ...stop, operator: 'equals' },
            { ...stop, target: '3' },
            { ...stop, target: Infinity },
            { ...stop, target: NaN },
            { ...stop, metric_label: 'analyses' },
        ].forEach((invalid) => expect(validate(withStop(invalid))).toBe(false));
        Object.keys(stop).forEach((key) => {
            const missing: any = { ...stop };
            delete missing[key];
            expect(validate(withStop(missing))).toBe(false);
        });
        [...WORKFLOW_NAMES, 'unknown_tool', 'start_agent_session', 'run_recorded_ai_analysis']
            .forEach((name) => expect(validate(withStop({ ...stop, tool: { [name]: {} } }))).toBe(false));
    });

    it('retains live-range event bounds, required content, and optional lead time', () => {
        const liveRange = cases.find(({ name }) => name === 'add_event_to_live_range_todo_list')!;
        const validate = compileCommand(commands.find(({ name }) => name === liveRange.name));
        const withEvent = (value: unknown) => ({
            [liveRange.name]: { tools: [{ query_analysis_result: { event: value, arguments: {} } }] },
        });
        expect(validate(withEvent({ id: 'corner', normalized_position: 0, content: { title: 'Start' } }))).toBe(true);
        expect(validate(withEvent({ ...event, normalized_position: 1 }))).toBe(true);
        [
            { ...event, id: ' ' },
            { ...event, normalized_position: -0.1 },
            { ...event, normalized_position: 1.1 },
            { ...event, lead_time_seconds: -1 },
            { ...event, content: {} },
            { ...event, content: { title: ' ' } },
            { ...event, content: { title: 'Corner', extra: true } },
            { ...event, extra: true },
        ].forEach((invalid) => expect(validate(withEvent(invalid))).toBe(false));
    });

    it.each(['front_desk', 'live', 'recorded', 'user_summary'].flatMap((session_mode) => (
        [undefined, 'track_guide', 'overtake', 'live_performance_analyst'].map((agent_mode) => ({
            session_mode,
            ...(agent_mode ? { agent_mode } : {}),
        }))
    )))('derives exactly session-available tools for every workflow and stop check: %p', (sessionContext) => {
        const available = getModelCommandsForSessionContext(sessionContext);
        const expectedNames = available.map(({ name }) => name).filter((name) => !WORKFLOW_NAMES.includes(name));
        available.forEach((command) => {
            expect(Object.keys(command).sort()).toEqual(['description', 'name', 'properties', 'required']);
            if (!cases.some(({ name }) => name === command.name)) {
                const originalCommand = MODEL_COMMAND_PROTOCOL.find(({ name }) => name === command.name)!;
                expect(command.properties).toEqual(originalCommand.properties);
                expect(command.required).toEqual(originalCommand.required);
                expect(command.description).toBe('description' in originalCommand ? originalCommand.description : '');
                return;
            }
            const workflow = getWorkflowSchema(command);
            expect(getCallNames(workflow.properties.tools.items)).toEqual(expectedNames);
            if (command.name === 'create_repeatable_plan') {
                expect(getCallNames(workflow.properties.stop_when.properties.tool)).toEqual(expectedNames);
            }
        });
    });

    it('does not mutate the global catalog or earlier session schemas when deriving another session', () => {
        const originalCatalog = JSON.stringify(MODEL_COMMAND_PROTOCOL);
        const recorded = getModelCommandsForSessionContext({ session_mode: 'recorded' });
        const originalRecorded = JSON.stringify(recorded);
        getModelCommandsForSessionContext(context);
        getModelCommandsForSessionContext({ session_mode: 'front_desk' });
        expect(JSON.stringify(recorded)).toBe(originalRecorded);
        expect(JSON.stringify(MODEL_COMMAND_PROTOCOL)).toBe(originalCatalog);
    });
});

describe('catalog workflow guidance', () => {
    const contexts = [
        { session_mode: 'front_desk' },
        { session_mode: 'live' },
        { session_mode: 'recorded' },
        { session_mode: 'user_summary' },
        { session_mode: 'live', agent_mode: 'track_guide' },
        { session_mode: 'live', agent_mode: 'overtake' },
        { session_mode: 'live', agent_mode: 'live_performance_analyst' },
    ];
    const ajv = new Ajv({ allErrors: true, strictNumbers: true });
    const validateArguments = (command: any, argumentsValue: unknown) => {
        const validate = ajv.compile({
            type: 'object',
            properties: command.properties,
            required: command.required,
            additionalProperties: false,
        });
        expect(validate(argumentsValue)).toBe(true);
    };

    it.each(contexts)('supplies executable workflow examples for %p', (context) => {
        const commands = getModelCommandsForSessionContext(context);
        const workflows = commands.filter(({ name }) => [
            'set_procedure_plan', 'create_repeatable_plan', 'add_event_to_live_range_todo_list',
        ].includes(name));
        expect(workflows.length).toBeGreaterThan(0);
        workflows.forEach((command) => {
            expect(command.description).toContain('single outer key');
            expect(command.description).toContain('All workflow categories are forbidden as children');
            expect(command.description).toContain('No legacy compatibility');
            const example = command.description.match(/```json\s*([\s\S]*?)```/);
            expect(example).not.toBeNull();
            const argumentsValue = JSON.parse(example![1]);
            validateArguments(command, argumentsValue);
            const body = argumentsValue[command.name];
            const children = [...body.tools, ...(body.stop_when ? [body.stop_when.tool] : [])];
            children.forEach((child) => {
                const [name] = Object.keys(child);
                expect(WORKFLOW_NAMES).not.toContain(name);
                const tool = commands.find((candidate) => candidate.name === name);
                expect(tool).toBeDefined();
                validateArguments(tool, child[name].arguments ?? {});
            });
        });
    });

    it('keeps procedure lifecycle and cancellation guidance with its commands', () => {
        const commands = getModelCommandsForSessionContext({ session_mode: 'live' });
        const description = (name: string) => commands.find((command) => command.name === name)!.description;
        expect(description('set_procedure_plan')).toContain('The application owns visible plan state');
        expect(description('set_procedure_plan')).toContain('Tool calls are fire-and-forget');
        expect(description('set_procedure_plan')).toContain('Do not skip, clear, replace, or abandon an active plan');
        expect(description('advance_plan_step')).toContain('confirm completion before advancing');
        expect(description('clear_procedure_plan')).toContain('only when the driver explicitly asks');
        expect(description('clear_procedure_plan')).not.toContain('when the plan is no longer useful');
    });
});
