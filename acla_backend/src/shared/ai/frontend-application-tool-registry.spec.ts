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

    it('advertises the tools only to child live-agent sessions with panel guidance', () => {
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
        expect(metadata.set_live_range_todo_list.description).toContain('panel must already be open');
        expect(metadata.set_live_range_todo_list.description).toContain('AI adapter attaches its notification callback');
    });
});
