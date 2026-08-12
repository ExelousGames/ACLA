import {
    FRONTEND_APPLICATION_TOOLS,
    getAiToolMetadataForSessionContext,
    getFrontendApplicationToolsForSessionContext,
} from './frontend-application-tool-registry';

describe('frontend application live range to-do tools', () => {
    it('atomically exposes only the new tool definitions and schemas', () => {
        const names = FRONTEND_APPLICATION_TOOLS.map((tool) => tool.name);
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

        const setTool = FRONTEND_APPLICATION_TOOLS.find((tool) => (
            tool.name === 'set_live_range_todo_list'
        ));
        const updateTool = FRONTEND_APPLICATION_TOOLS.find((tool) => (
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
        const liveMainNames = getFrontendApplicationToolsForSessionContext({
            session_mode: 'live',
        }).map((tool) => tool.name);
        const liveAgentNames = getFrontendApplicationToolsForSessionContext({
            session_mode: 'live',
            conversation_role: 'agent',
        }).map((tool) => tool.name);

        expect(liveMainNames).not.toContain('set_live_range_todo_list');
        expect(liveAgentNames).toEqual(expect.arrayContaining([
            'set_live_range_todo_list',
            'update_live_range_todo_list',
            'get_live_range_todo_list',
        ]));

        const metadata = getAiToolMetadataForSessionContext({
            session_mode: 'live',
            conversation_role: 'agent',
        });
        expect(metadata.set_live_range_todo_list.title).toBe('Setting live range to-do list');
        expect(metadata.set_live_range_todo_list.description).toContain('AI Chat mounts the list');
        expect(metadata.set_live_range_todo_list.description).toContain('attaches its notification callback');
    });
});

describe('live analysis mistake count tool', () => {
    it('defines the no-argument schema and requested metadata', () => {
        const tool = FRONTEND_APPLICATION_TOOLS.find(({ name }) => (
            name === 'get_live_analysis_mistake_count'
        ));
        expect(tool).toMatchObject({
            properties: {},
            required: [],
        });

        const metadata = getAiToolMetadataForSessionContext({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        });
        expect(metadata.get_live_analysis_mistake_count.title)
            .toBe('Counting live analysis mistakes');
    });

    it('exposes the tool only to a Live Performance Analyst child session', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getFrontendApplicationToolsForSessionContext(context).map(({ name }) => name)
        );

        expect(namesFor({ session_mode: 'live' }))
            .not.toContain('get_live_analysis_mistake_count');
        expect(namesFor({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'track_guide',
        })).not.toContain('get_live_analysis_mistake_count');
        expect(namesFor({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'overtake',
        })).not.toContain('get_live_analysis_mistake_count');
        expect(namesFor({
            session_mode: 'recorded',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        })).not.toContain('get_live_analysis_mistake_count');
        expect(namesFor({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        })).toContain('get_live_analysis_mistake_count');
    });
});

describe('create_goal tool', () => {
    it('defines the ordered workflow and numeric comparison schema', () => {
        const tools = getFrontendApplicationToolsForSessionContext({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        });
        const tool = tools.find(({ name }) => name === 'create_goal') as any;
        expect(tool).toMatchObject({
            required: ['goal', 'steps', 'comparison'],
            properties: {
                steps: {
                    minItems: 1,
                    items: { required: ['id', 'title', 'name'] },
                },
                comparison: {
                    required: ['step_id', 'result_path', 'operator', 'target', 'metric_label'],
                    properties: {
                        operator: { enum: ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'] },
                        target: { type: 'number' },
                    },
                },
            },
        });
    });

    it('exposes create_goal only to the Live Performance Analyst and constrains nested tools by session', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getFrontendApplicationToolsForSessionContext(context).map(({ name }) => name)
        );
        expect(namesFor({ session_mode: 'live' })).not.toContain('create_goal');
        expect(namesFor({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'track_guide',
        })).not.toContain('create_goal');

        const analystTools = getFrontendApplicationToolsForSessionContext({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        });
        const createGoal = analystTools.find(({ name }) => name === 'create_goal') as any;
        const nestedNames = createGoal.properties.steps.items.properties.name.enum;
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

        const metadata = getAiToolMetadataForSessionContext({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        });
        expect(metadata.create_goal.title).toBe('Creating goal');
    });
});

describe('retry_goal_task tool', () => {
    it('defines a no-argument schema and is exposed only to the Live Performance Analyst', () => {
        const namesFor = (context: Record<string, unknown>) => (
            getFrontendApplicationToolsForSessionContext(context).map(({ name }) => name)
        );
        const tool = FRONTEND_APPLICATION_TOOLS.find(({ name }) => (
            name === 'retry_goal_task'
        ));
        expect(tool).toMatchObject({ properties: {}, required: [] });

        expect(namesFor({ session_mode: 'live' })).not.toContain('retry_goal_task');
        expect(namesFor({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'track_guide',
        })).not.toContain('retry_goal_task');
        expect(namesFor({
            session_mode: 'recorded',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        })).not.toContain('retry_goal_task');
        expect(namesFor({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        })).toContain('retry_goal_task');

        const metadata = getAiToolMetadataForSessionContext({
            session_mode: 'live',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
        });
        expect(metadata.retry_goal_task.title).toBe('Retrying failed goal task');
    });
});
