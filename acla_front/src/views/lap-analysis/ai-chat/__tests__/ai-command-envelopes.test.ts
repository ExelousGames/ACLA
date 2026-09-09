import { asTool, createOperation } from 'components/ai-operations';
import { createAiCommandRegistry, createWorkflowToolDispatcher, frontendOperationRegistry } from '../ai-command-registry';

const definitions = Object.values(frontendOperationRegistry);

describe('explicit AI command envelopes', () => {
    it.each(definitions.filter(({ kind }) => kind === 'tool'))('unwraps $name without changing its arguments', async (definition) => {
        const args = { tool: { name: 'literal argument' }, arguments: { nested: true }, payload: [1, 2] };
        const execute = jest.spyOn(definition, 'execute').mockImplementation(() => asTool(createOperation({ status: 'ready' }, 'ready')) as any);
        try {
            const registry = createAiCommandRegistry({ sessionMode: 'live' });
            await (registry[definition.name] as any)({ tool: { name: definition.name, arguments: args } }).result;
            expect(execute).toHaveBeenCalledWith(expect.anything(), args, expect.any(Function), undefined);
            expect((execute.mock.calls[0] as unknown[])[1]).toBe(args);
        } finally {
            execute.mockRestore();
        }
    });

    it.each([
        { tool: { name: 'show_map' } },
        { tool: { name: 'get_next_corner', arguments: [] } },
        { tool: { name: 'get_next_corner', args: {} } },
        { tool: { name: 'get_next_corner' }, extra: true },
    ])('rejects mismatched or malformed tool envelopes before execution: %p', async (input) => {
        const execute = jest.spyOn(frontendOperationRegistry.get_next_corner, 'execute');
        try {
            await expect(createAiCommandRegistry({ sessionMode: 'live' }).get_next_corner(input as any).result).rejects.toThrow();
            expect(execute).not.toHaveBeenCalled();
        } finally {
            execute.mockRestore();
        }
    });

    it('passes a nested tool argument literally through the child dispatcher', async () => {
        const args = { tool: { name: 'literal data' } };
        const execute = jest.spyOn(frontendOperationRegistry.get_next_corner, 'execute').mockImplementation(() => asTool(createOperation({ status: 'ready' }, 'ready')) as any);
        try {
            await createWorkflowToolDispatcher({ sessionMode: 'live' })('get_next_corner', args).result;
            expect((execute.mock.calls[0] as unknown[])[1]).toBe(args);
        } finally {
            execute.mockRestore();
        }
    });

    it.each(definitions.filter(({ name, kind }) => kind === 'workflow' && ![
        'set_procedure_plan', 'create_repeatable_plan', 'add_event_to_live_range_todo_list',
        'append_procedure_plan', 'append_repeatable_plan', 'create_live_range_todo_list',
    ].includes(name)))('requires the named empty tools list for $name', async (definition) => {
        const execute = jest.spyOn(definition, 'execute').mockImplementation(() => createOperation({ status: 'ready' }, 'ready') as any);
        try {
            const handler = createAiCommandRegistry({ sessionMode: 'live' })[definition.name] as any;
            await handler({ workflow: { name: definition.name, tools: [] } }).result;
            expect(execute).toHaveBeenCalledTimes(1);
            execute.mockClear();
            for (const input of [{}, { workflow: { name: definition.name } }, { workflow: { name: definition.name, tools: [{ tool: { name: 'show_map' } }] } }]) {
                await expect(handler(input).result).rejects.toThrow();
            }
            expect(execute).not.toHaveBeenCalled();
        } finally {
            execute.mockRestore();
        }
    });
});
