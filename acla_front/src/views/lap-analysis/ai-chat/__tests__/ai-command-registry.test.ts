import {
    FRONTEND_AI_TOOL_NAMES,
    createAiCommandRegistry,
    frontendAiToolRegistry,
    isGoalStepAvailableForContext,
} from '../ai-command-registry';
import {
    AI_TOOL_COMPONENT_NAMES,
    createAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import { resolvedAiToolOperation } from 'components/ai-engineering-tools';
import type { AiChatHandle } from '../ai-chat';
import type { AnalysisResultsChartHandle } from '../../visualization/charts/AnalysisResultsChart';

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
        expect(FRONTEND_AI_TOOL_NAMES).toContain('query_analysis_result');
        expect(Object.keys(frontendAiToolRegistry)).not.toContain('get_analysis_result_count');
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

    it('dispatches the analysis result count and preserves its component operation', async () => {
        const componentOperation = resolvedAiToolOperation({
            status: 'ready' as const,
            analysis_result_count: 4,
        });
        const componentName = 'visualization:analysis-results';
        const handle: Partial<AnalysisResultsChartHandle> = {
            getAnalysisResultCount: jest.fn(() => componentOperation),
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(componentName, handle),
        });

        const returned = registry.query_analysis_result({ query: 'result_count' });

        expect(returned).toBe(componentOperation);
        await expect(returned.result).resolves.toEqual({
            status: 'ready',
            analysis_result_count: 4,
        });
    });

    it.each([
        {},
        { query: 'unsupported' },
    ])('rejects an invalid analysis result query: %p', async (args) => {
        const componentName = 'visualization:analysis-results';
        const handle: Partial<AnalysisResultsChartHandle> = {
            getAnalysisResultCount: jest.fn(),
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(componentName, handle),
        });

        const operation = registry.query_analysis_result(args);

        await expect(operation.result).rejects.toMatchObject({
            name: 'InvalidToolCallError',
        });
        expect(handle.getAnalysisResultCount).not.toHaveBeenCalled();
    });

    it('rejects the analysis result count when its tab is not mounted', async () => {
        const registry = createAiCommandRegistry({
            componentRefs: createAiToolComponentRefDirectory(),
        });

        const operation = registry.query_analysis_result({ query: 'result_count' });

        await expect(operation.result).rejects.toMatchObject({
            name: 'ComponentRefUnavailableError',
            componentName: 'visualization:analysis-results',
        });
    });

    it('allows the analysis result count in compatible live analyst goal steps', () => {
        expect(isGoalStepAvailableForContext({
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        }, 'query_analysis_result')).toBe(true);
        expect(isGoalStepAvailableForContext({
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'track_guide',
        }, 'query_analysis_result')).toBe(false);
    });
});
