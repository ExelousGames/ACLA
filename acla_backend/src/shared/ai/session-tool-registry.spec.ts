import {
    SESSION_TOOLS,
    getSessionToolsForSessionContext,
} from './session-tool-registry';
import {
    TELEMETRY_METRIC_FIELD_DEFINITIONS,
    TELEMETRY_METRIC_FIELD_SCHEMA,
} from './telemetry-metric-fields';

describe('live baseline tools', () => {
    it('directs new recordings through collect and limits restart to active recordings', () => {
        const collect = SESSION_TOOLS.find(({ name }) => name === 'collect_live_baseline') as any;
        const restart = SESSION_TOOLS.find(({ name }) => name === 'restart_live_baseline') as any;

        expect(collect?.description).toContain('If a previous baseline exists');
        expect(collect?.description).toContain('starts a new recording');
        expect(restart?.description).toContain('only while it is waiting for its start condition or actively collecting');
        expect(restart?.description).toContain('use collect_live_baseline');
    });

    it('requires either the full-lap preset or mutually exclusive custom conditions', () => {
        const collect = SESSION_TOOLS.find(({ name }) => name === 'collect_live_baseline') as any;
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
        const names = SESSION_TOOLS.map((tool) => tool.name);
        expect(names.filter((name) => name.endsWith('_live_range_todo_list'))).toEqual([
            'add_event_to_live_range_todo_list',
            'get_live_range_todo_list',
            'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
        ]);

        const addTool = SESSION_TOOLS.find((tool) => (
            tool.name === 'add_event_to_live_range_todo_list'
        )) as any;
        expect(addTool).toMatchObject({
            required: ['events'],
            properties: {
                events: {
                    minItems: 1,
                    items: {
                        required: ['event', 'tool'],
                        additionalProperties: false,
                    },
                },
            },
        });
        const itemSchema = addTool.properties.events.items;
        const eventSchema = itemSchema.properties.event;
        const contentSchema = eventSchema.properties.content;
        const nestedToolSchema = itemSchema.properties.tool;
        expect(Object.keys(itemSchema.properties)).toEqual(['event', 'tool']);
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
        expect(Object.keys(nestedToolSchema.properties)).toEqual(['name', 'arguments']);
        expect(nestedToolSchema.required).toEqual(['name', 'arguments']);
        expect(nestedToolSchema.additionalProperties).toBe(false);
    });

    it('advertises only add/read to child live agents and derives the nested-tool enum', () => {
        const liveMainNames = getSessionToolsForSessionContext({
            session_mode: 'live',
        }).map((tool) => tool.name);
        const liveAgentTools = getSessionToolsForSessionContext({
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
        const nestedNames = addTool.properties.events.items
            .properties.tool.properties.name.enum;
        expect(nestedNames).toEqual(expect.arrayContaining([
            'analyze_telemetry',
            'query_telemetry_metric',
            'get_live_range_todo_list',
        ]));
        expect(nestedNames).not.toEqual(expect.arrayContaining([
            'create_goal',
            'retry_goal_task',
            'set_procedure_plan',
            'advance_plan_step',
            'clear_procedure_plan',
            'add_event_to_live_range_todo_list',
            'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
        ]));
        const liveAgentNameSet = new Set<string>(liveAgentNames);
        expect(nestedNames.every((name: string) => liveAgentNameSet.has(name))).toBe(true);
        expect(addTool.description).toContain('AI Chat mounts the list');
        expect(addTool.description).toContain('returns the updated list summary immediately');

        const analystAddTool = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        }).find(({ name }) => name === 'add_event_to_live_range_todo_list') as any;
        expect(analystAddTool.properties.events.items.properties.tool.properties.name.enum)
            .toEqual(nestedNames);
    });

    it('selects tools only from direct canonical mode fields', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getSessionToolsForSessionContext(context).map(({ name }) => name)
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
        })).not.toContain('create_goal');
    });
});

describe('filtered Driver/Expert comparison queue tool', () => {
    const namesFor = (context: Record<string, unknown>) => (
        getSessionToolsForSessionContext(context).map(({ name }) => name)
    );

    it('defines a strict no-argument schema only for the live performance analyst', () => {
        const tool = SESSION_TOOLS.find(({ name }) => (
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

    it('includes the tool in analyst goals and excludes recursive Live Range scheduling', () => {
        const analystTools = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const createGoal = analystTools.find(({ name }) => name === 'create_goal') as any;
        const addEvents = analystTools.find(({ name }) => (
            name === 'add_event_to_live_range_todo_list'
        )) as any;
        const goalNames = createGoal.properties.steps.items.properties.name.enum;
        const nestedLiveRangeNames = addEvents.properties.events.items
            .properties.tool.properties.name.enum;

        expect(goalNames)
            .toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
        expect(nestedLiveRangeNames)
            .not.toContain('add_filtered_driver_expert_comparisons_to_live_range_todo_list');
        expect(analystTools.map(({ name }) => name)).toContain('set_procedure_plan');
    });
});

describe('analysis result query tool', () => {
    const namesFor = (context: Record<string, unknown>) => (
        getSessionToolsForSessionContext(context).map(({ name }) => name)
    );
    const eligibleContexts = [
        { session_mode: 'live' },
        { session_mode: 'recorded' },
        { session_mode: 'live', agent_mode: 'track_guide' },
        { session_mode: 'live', agent_mode: 'overtake' },
        { session_mode: 'live', agent_mode: 'live_performance_analyst' },
    ];

    it('requires one non-blank JSONata expression without a legacy enum', () => {
        const tool = SESSION_TOOLS.find(({ name }) => (
            name === 'query_analysis_result'
        )) as any;

        expect(SESSION_TOOLS.filter(({ name }) => name === 'query_analysis_result'))
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
            const tool = getSessionToolsForSessionContext(context).find(({ name }) => (
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
        const tool = SESSION_TOOLS.find(({ name }) => (
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

    it('is available to compatible live analyst goal steps and stop conditions', () => {
        const tools = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const createGoal = tools.find(({ name }) => name === 'create_goal') as any;

        expect(createGoal.properties.steps.items.properties.name.enum)
            .toContain('query_analysis_result');
        expect(createGoal.properties.stop_when.properties.tool.properties.name.enum)
            .toContain('query_analysis_result');
    });
});

describe('analysis result query apply tool', () => {
    const namesFor = (context: Record<string, unknown>) => (
        getSessionToolsForSessionContext(context).map(({ name }) => name)
    );
    const eligibleContexts = [
        { session_mode: 'live' },
        { session_mode: 'recorded' },
        { session_mode: 'live', agent_mode: 'track_guide' },
        { session_mode: 'live', agent_mode: 'overtake' },
        { session_mode: 'live', agent_mode: 'live_performance_analyst' },
    ];

    it('requires final non-blank JSONata and accepts only an optional integer page number', () => {
        const tool = SESSION_TOOLS.find(({ name }) => (
            name === 'apply_query_to_analysis_result'
        )) as any;

        expect(SESSION_TOOLS.filter(({ name }) => name === 'apply_query_to_analysis_result'))
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

    it('is available in analyst goals, stop conditions, and nested live-range workflows', () => {
        const tools = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const createGoal = tools.find(({ name }) => name === 'create_goal') as any;
        const addEvents = tools.find(({ name }) => (
            name === 'add_event_to_live_range_todo_list'
        )) as any;

        expect(createGoal.properties.steps.items.properties.name.enum)
            .toContain('apply_query_to_analysis_result');
        expect(createGoal.properties.stop_when.properties.tool.properties.name.enum)
            .toContain('apply_query_to_analysis_result');
        expect(addEvents.properties.events.items.properties.tool.properties.name.enum)
            .toContain('apply_query_to_analysis_result');
    });
});

describe('set_procedure_plan tool', () => {
    it('does not expose a current_request argument', () => {
        const tool = SESSION_TOOLS.find(({ name }) => (
            name === 'set_procedure_plan'
        )) as any;

        expect(tool.properties).not.toHaveProperty('current_request');
        expect(Object.keys(tool.properties).sort()).toEqual(['goal', 'requests']);
    });

    it('exposes only assistant-authored request inputs', () => {
        const tool = SESSION_TOOLS.find(({ name }) => (
            name === 'set_procedure_plan'
        )) as any;
        const requestItems = tool.properties.requests.items;

        expect(Object.keys(requestItems.properties).sort())
            .toEqual(['name', 'payload', 'title']);
        expect(requestItems.required).toEqual(['title']);
    });
});

describe('create_goal tool', () => {
    it('defines the canonical preparation workflow and numeric stop_when schema', () => {
        const tools = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const tool = tools.find(({ name }) => name === 'create_goal') as any;
        expect(tool).toMatchObject({
            required: ['name', 'steps', 'stop_when'],
            properties: {
                name: { type: 'string' },
                steps: {
                    minItems: 1,
                    items: { required: ['id', 'title', 'name'] },
                },
                stop_when: {
                    required: ['tool', 'operator', 'target'],
                    properties: {
                        tool: {
                            required: ['name'],
                            properties: {
                                name: { type: 'string' },
                                arguments: { type: 'object' },
                            },
                        },
                        operator: { enum: ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'] },
                        target: {
                            type: 'number',
                            description: expect.any(String),
                        },
                    },
                },
            },
        });
        expect(tool.properties).not.toHaveProperty('goal');
        expect(tool.properties).not.toHaveProperty('comparison');
        expect(tool.properties).not.toHaveProperty('determination');
        expect(tool.properties.stop_when.properties).not.toHaveProperty('step_id');
        expect(tool.properties.stop_when.properties).not.toHaveProperty('metric_label');
        expect(tool.properties.stop_when.properties).not.toHaveProperty('unit');
        expect(Object.keys(tool.properties.stop_when.properties).sort())
            .toEqual(['operator', 'target', 'tool']);
        expect(tool.properties.stop_when.description)
            .toContain('{ "status": "ready", "data": finiteNumber }');
        expect(tool.description).toContain('both values must be finite numbers');
        expect(tool.description).not.toContain('query_analysis_result');
        expect(tool.properties.stop_when.description).toContain('comparable with the target');
        expect(tool.properties.stop_when.description).not.toContain('query_analysis_result');
        expect(tool.properties.stop_when.properties.tool.description)
            .not.toContain('query_analysis_result');
        expect(tool.properties.stop_when.properties.target.description)
            .toContain('comparable to the value returned by the stop-when tool');
    });

    it('exposes create_goal only to the Live Performance Analyst and constrains nested tools by session', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getSessionToolsForSessionContext(context).map(({ name }) => name)
        );
        expect(namesFor({ session_mode: 'live' })).not.toContain('create_goal');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'track_guide',
        })).not.toContain('create_goal');

        const analystTools = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const createGoal = analystTools.find(({ name }) => name === 'create_goal') as any;
        const nestedNames = createGoal.properties.steps.items.properties.name.enum;
        const stopWhenNames = createGoal
            .properties.stop_when.properties.tool.properties.name.enum;
        expect(nestedNames).toEqual(analystTools
            .map(({ name }) => name)
            .filter((name) => name !== 'create_goal' && name !== 'retry_goal_task'));
        expect(nestedNames).toEqual(expect.arrayContaining([
            'collect_live_baseline',
            'analyze_live_recorded_analysis',
            'query_analysis_result',
        ]));
        expect(nestedNames).not.toContain('create_goal');
        expect(nestedNames).not.toContain('retry_goal_task');
        expect(stopWhenNames).toEqual(nestedNames);
        expect(stopWhenNames).not.toContain('create_goal');
        expect(stopWhenNames).not.toContain('retry_goal_task');

    });
});

describe('telemetry metric query tool', () => {
    it('requires the described supported fields, scope, and a summarized reduction', () => {
        const tool = SESSION_TOOLS.find(({ name }) => (
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

describe('retry_goal_task tool', () => {
    it('defines a no-argument schema and is exposed only to the Live Performance Analyst', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getSessionToolsForSessionContext(context).map(({ name }) => name)
        );
        const tool = SESSION_TOOLS.find(({ name }) => (
            name === 'retry_goal_task'
        ));
        expect(tool).toMatchObject({ properties: {}, required: [] });

        expect(namesFor({ session_mode: 'live' })).not.toContain('retry_goal_task');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'track_guide',
        })).not.toContain('retry_goal_task');
        expect(namesFor({
            session_mode: 'recorded',
            agent_mode: 'live_performance_analyst',
        })).not.toContain('retry_goal_task');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        })).toContain('retry_goal_task');
    });
});
