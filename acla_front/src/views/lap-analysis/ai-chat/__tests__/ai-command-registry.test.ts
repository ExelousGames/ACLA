import {
    FRONTEND_AI_TOOL_NAMES,
    createAiCommandRegistry,
    frontendAiToolRegistry,
    isGoalStepAvailableForContext,
} from '../ai-command-registry';
import type {
    AiCommandRegistry,
    FrontendAiQueryContractCoverage,
    QueryAnalysisResultArguments,
    QueryAnalysisResultResult,
    QueryTelemetryMetricArguments,
    QueryTelemetryMetricResult,
} from '../ai-command-registry';
import {
    AI_TOOL_COMPONENT_NAMES,
    createAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import { resolvedAiToolOperation } from 'components/ai-engineering-tools';
import type { AiToolOperation, AiToolQueryResult } from 'components/ai-engineering-tools';
import type { AiChatHandle } from '../ai-chat';
import type { AnalysisResultsChartHandle } from '../../visualization/charts/AnalysisResultsChart';
import type { LiveSessionHandle } from 'views/live-session/LiveSessionView';

// @ts-expect-error AiToolQueryResult requires an explicit data type.
type MissingAiToolQueryResultGeneric = AiToolQueryResult;

// @ts-expect-error QueryAnalysisResultArguments requires an explicit query.
type MissingAnalysisArgumentsGeneric = QueryAnalysisResultArguments;

// @ts-expect-error QueryAnalysisResultResult requires an explicit query.
type MissingAnalysisResultGeneric = QueryAnalysisResultResult;

// @ts-expect-error QueryTelemetryMetricArguments requires an explicit reduction.
type MissingTelemetryArgumentsGeneric = QueryTelemetryMetricArguments;

// @ts-expect-error QueryTelemetryMetricResult requires an explicit reduction.
type MissingTelemetryResultGeneric = QueryTelemetryMetricResult;

const queryContractCoverage: FrontendAiQueryContractCoverage = true;

const assertQueryContractTypes = (registry: AiCommandRegistry) => {
    const resultCount: AiToolOperation<QueryAnalysisResultResult<'result_count'>> = (
        registry.query_analysis_result({ query: 'result_count' })
    );
    const mistakeCount: AiToolOperation<QueryAnalysisResultResult<'mistake_count'>> = (
        registry.query_analysis_result({ query: 'mistake_count' })
    );
    const avg: AiToolOperation<QueryTelemetryMetricResult<'avg'>> = registry.query_telemetry_metric({
        fields: ['speed'],
        scope: { type: 'now' },
        reduce: 'avg',
    });
    const min: AiToolOperation<QueryTelemetryMetricResult<'min'>> = registry.query_telemetry_metric({
        fields: ['speed'],
        scope: { type: 'now' },
        reduce: 'min',
    });
    const max: AiToolOperation<QueryTelemetryMetricResult<'max'>> = registry.query_telemetry_metric({
        fields: ['speed'],
        scope: { type: 'now' },
        reduce: 'max',
    });
    const stats: AiToolOperation<QueryTelemetryMetricResult<'stats'>> = registry.query_telemetry_metric({
        fields: ['speed'],
        scope: { type: 'now' },
        reduce: 'stats',
    });

    // @ts-expect-error Analysis result queries accept only declared query names.
    registry.query_analysis_result({ query: 'unsupported' });
    // @ts-expect-error The model-facing telemetry query does not expose raw values.
    registry.query_telemetry_metric({ fields: ['speed'], scope: { type: 'now' }, reduce: 'raw' });
    // @ts-expect-error Stats results cannot be assigned to scalar telemetry results.
    const mismatchedReduction: AiToolOperation<QueryTelemetryMetricResult<'avg'>> = stats;

    return {
        resultCount,
        mistakeCount,
        avg,
        min,
        max,
        stats,
        mismatchedReduction,
        queryContractCoverage,
    };
};

void assertQueryContractTypes;

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

    it('preserves a reduction-specific telemetry component operation', async () => {
        const componentOperation = resolvedAiToolOperation({
            status: 'ready' as const,
            data: { Physics_speed_kmh: 123 },
        });
        const handle: Partial<LiveSessionHandle> = {
            queryTelemetryMetricForAi: jest.fn(() => componentOperation) as any,
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION, handle),
        });

        const returned = registry.query_telemetry_metric({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'avg',
        });

        expect(returned).toBe(componentOperation);
        await expect(returned.result).resolves.toEqual({
            status: 'ready',
            data: { Physics_speed_kmh: 123 },
        });
    });

    it('dispatches both analysis result queries and preserves their component operations', async () => {
        const resultCountOperation = resolvedAiToolOperation({
            status: 'ready' as const,
            data: 4,
        });
        const mistakeCountOperation = resolvedAiToolOperation({
            status: 'ready' as const,
            data: 3,
        });
        const componentName = 'visualization:analysis-results';
        const handle: Partial<AnalysisResultsChartHandle> = {
            queryAnalysisResult: jest.fn(({ query }) => (
                query === 'result_count' ? resultCountOperation : mistakeCountOperation
            )) as any,
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(componentName, handle),
        });

        const returnedResultCount = registry.query_analysis_result({ query: 'result_count' });
        const returnedMistakeCount = registry.query_analysis_result({ query: 'mistake_count' });

        expect(returnedResultCount).toBe(resultCountOperation);
        expect(returnedMistakeCount).toBe(mistakeCountOperation);
        expect(handle.queryAnalysisResult).toHaveBeenNthCalledWith(1, { query: 'result_count' });
        expect(handle.queryAnalysisResult).toHaveBeenNthCalledWith(2, { query: 'mistake_count' });
        await expect(returnedResultCount.result).resolves.toEqual({
            status: 'ready',
            data: 4,
        });
        await expect(returnedMistakeCount.result).resolves.toEqual({
            status: 'ready',
            data: 3,
        });
    });

    it.each([
        {},
        { query: 'unsupported' },
        { query: 'result_count', extra: true },
    ])('rejects an invalid analysis result query: %p', async (args) => {
        const componentName = 'visualization:analysis-results';
        const handle: Partial<AnalysisResultsChartHandle> = {
            queryAnalysisResult: jest.fn(),
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(componentName, handle),
        });

        const operation = registry.query_analysis_result(args as any);

        await expect(operation.result).rejects.toMatchObject({
            name: 'InvalidToolCallError',
        });
        expect(handle.queryAnalysisResult).not.toHaveBeenCalled();
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
