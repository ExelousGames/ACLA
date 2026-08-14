import {
    SESSION_TOOLS,
    getSessionToolsForSessionContext,
} from './session-tool-registry';

describe('session live range to-do tools', () => {
    it('atomically exposes only the new tool definitions and schemas', () => {
        const names = SESSION_TOOLS.map((tool) => tool.name);
        expect(names).toEqual(expect.arrayContaining([
            'set_live_range_todo_list',
            'update_live_range_todo_list',
            'get_live_range_todo_list',
        ]));
        expect(names).not.toEqual(expect.arrayContaining([
            'set_live_range_tracker',
            'update_live_range_tracker',
            'get_live_range_tracker',
        ]));

        const setTool = SESSION_TOOLS.find((tool) => (
            tool.name === 'set_live_range_todo_list'
        ));
        const updateTool = SESSION_TOOLS.find((tool) => (
            tool.name === 'update_live_range_todo_list'
        ));
        expect(setTool).toMatchObject({
            required: ['events'],
            properties: {
                events: {
                    items: {
                        required: ['id', 'normalized_position', 'content'],
                    },
                },
            },
        });
        expect(updateTool).toMatchObject({
            required: ['action'],
            properties: {
                action: {
                    enum: ['add_events', 'update_events', 'remove_events', 'reset_events', 'clear'],
                },
            },
        });
    });

    it('advertises the tools only to child live-agent sessions with AI Chat ownership guidance', () => {
        const liveMainNames = getSessionToolsForSessionContext({
            session_mode: 'live',
        }).map((tool) => tool.name);
        const liveAgentTools = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'track_guide',
        });
        const liveAgentNames = liveAgentTools.map((tool) => tool.name);

        expect(liveMainNames).not.toContain('set_live_range_todo_list');
        expect(liveAgentNames).toEqual(expect.arrayContaining([
            'set_live_range_todo_list',
            'update_live_range_todo_list',
            'get_live_range_todo_list',
        ]));

        const setTool = liveAgentTools.find(({ name }) => name === 'set_live_range_todo_list');
        expect(setTool?.description).toContain('AI Chat mounts the list');
        expect(setTool?.description).toContain('attaches its notification callback');
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
        })).not.toContain('set_live_range_todo_list');
        expect(namesFor({
            session_mode: 'recorded',
            context_kind: 'live',
            active_screen: { assistant_mode: 'live' },
        })).not.toContain('start_agent_session');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'track_guide',
            agent_session: { agent_mode: 'live_performance_analyst' },
        })).not.toContain('get_live_analysis_mistake_count');
    });
});

describe('live analysis mistake count tool', () => {
    it('defines the no-argument schema and description', () => {
        const tool = SESSION_TOOLS.find(({ name }) => (
            name === 'get_live_analysis_mistake_count'
        ));
        expect(tool).toMatchObject({
            description: expect.any(String),
            properties: {},
            required: [],
        });
    });

    it('exposes the tool only to a Live Performance Analyst child session', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getSessionToolsForSessionContext(context).map(({ name }) => name)
        );

        expect(namesFor({ session_mode: 'live' }))
            .not.toContain('get_live_analysis_mistake_count');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'track_guide',
        })).not.toContain('get_live_analysis_mistake_count');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'overtake',
        })).not.toContain('get_live_analysis_mistake_count');
        expect(namesFor({
            session_mode: 'recorded',
            agent_mode: 'live_performance_analyst',
        })).not.toContain('get_live_analysis_mistake_count');
        expect(namesFor({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        })).toContain('get_live_analysis_mistake_count');
    });
});

describe('analysis result query tool', () => {
    const namesFor = (context: Record<string, unknown>) => (
        getSessionToolsForSessionContext(context).map(({ name }) => name)
    );

    it('replaces the count tool with a required result_count query schema', () => {
        const tool = SESSION_TOOLS.find(({ name }) => (
            name === 'query_analysis_result'
        ));

        expect(SESSION_TOOLS.map(({ name }) => name))
            .not.toContain('get_analysis_result_count');
        expect(tool).toMatchObject({
            description: expect.any(String),
            properties: {
                query: {
                    type: 'string',
                    enum: ['result_count'],
                },
            },
            required: ['query'],
        });
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

    it('is available to compatible live analyst goal steps and determinations', () => {
        const tools = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const createGoal = tools.find(({ name }) => name === 'create_goal') as any;

        expect(createGoal.properties.steps.items.properties.name.enum)
            .toContain('query_analysis_result');
        expect(createGoal.properties.determination.properties.tool.properties.name.enum)
            .toContain('query_analysis_result');
    });
});

describe('create_goal tool', () => {
    it('defines the canonical preparation workflow and numeric determination schema', () => {
        const tools = getSessionToolsForSessionContext({
            session_mode: 'live',
            agent_mode: 'live_performance_analyst',
        });
        const tool = tools.find(({ name }) => name === 'create_goal') as any;
        expect(tool).toMatchObject({
            required: ['name', 'steps', 'determination'],
            properties: {
                name: { type: 'string' },
                steps: {
                    minItems: 1,
                    items: { required: ['id', 'title', 'name'] },
                },
                determination: {
                    required: ['tool', 'result_path', 'operator', 'target'],
                    properties: {
                        tool: {
                            required: ['name'],
                            properties: {
                                name: { type: 'string' },
                                arguments: { type: 'object' },
                            },
                        },
                        result_path: { type: 'string' },
                        operator: { enum: ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'] },
                        target: { type: 'number' },
                    },
                },
            },
        });
        expect(tool.properties).not.toHaveProperty('goal');
        expect(tool.properties).not.toHaveProperty('comparison');
        expect(tool.properties.determination.properties).not.toHaveProperty('step_id');
        expect(tool.properties.determination.properties).not.toHaveProperty('metric_label');
        expect(tool.properties.determination.properties).not.toHaveProperty('unit');
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
        const determinationNames = createGoal
            .properties.determination.properties.tool.properties.name.enum;
        expect(nestedNames).toEqual(analystTools
            .map(({ name }) => name)
            .filter((name) => name !== 'create_goal' && name !== 'retry_goal_task'));
        expect(nestedNames).toEqual(expect.arrayContaining([
            'collect_live_baseline',
            'analyze_live_recorded_analysis',
            'get_live_analysis_mistake_count',
        ]));
        expect(nestedNames).not.toContain('create_goal');
        expect(nestedNames).not.toContain('retry_goal_task');
        expect(determinationNames).toEqual(nestedNames);
        expect(determinationNames).not.toContain('create_goal');
        expect(determinationNames).not.toContain('retry_goal_task');

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
