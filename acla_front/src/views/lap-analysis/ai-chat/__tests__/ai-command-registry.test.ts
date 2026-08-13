import {
    FRONTEND_AI_TOOL_NAMES,
    createAiCommandRegistry,
    frontendAiToolRegistry,
} from '../ai-command-registry';
import {
    AI_TOOL_COMPONENT_NAMES,
    createAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import { resolvedAiToolOperation } from 'components/ai-engineering-tools';
import type { AiChatHandle } from '../ai-chat';

const register = (name: string, handle: object) => {
    const directory = createAiToolComponentRefDirectory();
    directory.reserveComponentRef(name, Symbol(name), {
        getComponentName: () => name,
        ...handle,
    } as any);
    return directory;
};

describe('frontend AI tool registry', () => {
    it('is a name-keyed definition object covering every advertised tool', () => {
        expect(Object.keys(frontendAiToolRegistry).sort()).toEqual(
            [...FRONTEND_AI_TOOL_NAMES].sort(),
        );
        FRONTEND_AI_TOOL_NAMES.forEach((name) => {
            expect(frontendAiToolRegistry[name].componentName).toEqual(expect.any(String));
        });
    });

    it('preserves the component operation instead of awaiting or wrapping its result', async () => {
        const componentOperation = resolvedAiToolOperation({
            status: 'started' as const,
            conversation_role: 'agent' as const,
            agent_mode: 'overtake' as const,
        });
        const handle: Partial<AiChatHandle> = {
            startAgentSession: jest.fn(() => componentOperation),
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, handle),
        });

        const returned = registry.start_agent_session({ agent_mode: 'overtake' });

        expect(returned).toBe(componentOperation);
        await expect(returned.result).resolves.toMatchObject({ status: 'started' });
        expect(returned.statuses).toEqual([]);
    });

    it('returns a rejected operation for an unavailable component', async () => {
        const registry = createAiCommandRegistry({
            componentRefs: createAiToolComponentRefDirectory(),
        });

        const operation = registry.show_map({});

        await expect(operation.result).rejects.toMatchObject({
            name: 'ComponentRefUnavailableError',
        });
    });
});
