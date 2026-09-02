import {
    FRONTEND_AI_TOOL_NAMES,
    createAiCommandRegistry,
    frontendAiToolRegistry,
    isGoalStepAvailableForContext,
    startAgentRuntime,
} from '../ai-command-registry';
import type {
    AiCommandRegistry,
    FrontendAiQueryContractCoverage,
    QueryAnalysisResultOutput,
    QueryTelemetryMetricArguments,
    QueryTelemetryMetricResult,
} from '../ai-command-registry';
import {
    AI_TOOL_COMPONENT_NAMES,
    createAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import {
    createControlledAiToolOperation,
    LiveRangeTodoListRunner,
    resolvedAiToolOperation,
} from 'components/ai-engineering-tools';
import type {
    AiToolOperation,
    AiToolQueryResult,
    LiveRangeTodoEventInput,
    LiveRangeTodoListHandle,
} from 'components/ai-engineering-tools';
import type { AiChatHandle } from '../ai-chat';
import type { AnalysisResultsChartHandle } from '../../visualization/charts/AnalysisResultsChart';
import type { FilteredAnalysisSegmentsSnapshot } from '../../visualization/charts/AnalysisResultsChart';
import type { LiveSessionHandle } from 'views/live-session/LiveSessionView';
import { RecordingState } from 'views/lap-analysis/recording-state';

// @ts-expect-error AiToolQueryResult requires an explicit data type.
type MissingAiToolQueryResultGeneric = AiToolQueryResult;

// @ts-expect-error QueryTelemetryMetricArguments requires an explicit reduction.
type MissingTelemetryArgumentsGeneric = QueryTelemetryMetricArguments;

// @ts-expect-error QueryTelemetryMetricResult requires an explicit reduction.
type MissingTelemetryResultGeneric = QueryTelemetryMetricResult;

const queryContractCoverage: FrontendAiQueryContractCoverage = true;

const assertQueryContractTypes = (registry: AiCommandRegistry) => {
    const analysisResult: AiToolOperation<QueryAnalysisResultOutput> = (
        registry.query_analysis_result({ query: '$count(analyses)' })
    );
    // @ts-expect-error Analysis result queries require an expression.
    registry.query_analysis_result({});
    // @ts-expect-error Analysis result queries accept no extra arguments.
    registry.query_analysis_result({ query: 'analyses', extra: true });
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
    const display: AiToolOperation<'graph shown'> = registry.display_specific_result_in_overlay({
        page_id: 'page-id',
        result_id: 'result-id',
    });
    // @ts-expect-error Specific result displays require both exact ids.
    registry.display_specific_result_in_overlay({ page_id: 'page-id' });
    // @ts-expect-error Specific result displays accept no extra arguments.
    registry.display_specific_result_in_overlay({ page_id: 'page-id', result_id: 'result-id', extra: true });

    // @ts-expect-error The model-facing telemetry query does not expose raw values.
    registry.query_telemetry_metric({ fields: ['speed'], scope: { type: 'now' }, reduce: 'raw' });
    // @ts-expect-error Stats results cannot be assigned to scalar telemetry results.
    const mismatchedReduction: AiToolOperation<QueryTelemetryMetricResult<'avg'>> = stats;

    return {
        analysisResult,
        avg,
        min,
        max,
        stats,
        display,
        mismatchedReduction,
        queryContractCoverage,
    };
};

void assertQueryContractTypes;

const register = (name: string, handle: object) => {
    const directory = createAiToolComponentRefDirectory();
    directory.registerComponentRef({ current: {
        getComponentName: () => name,
        ...handle,
    } as any });
    return directory;
};

describe('frontend AI tool registry', () => {
    it('publishes live analyst runtime statuses without routing them through session intelligence', async () => {
        const publishStatus = jest.fn();
        const livePerformanceAnalystState = {
            enabled: false,
        };

        await expect(startAgentRuntime('live_performance_analyst', {
            sessionMode: 'live',
            recordingState: RecordingState.RECORDING,
            getLiveSessionSnapshot: () => ({
                status: 'ready',
                track: 'brands_hatch',
                car: '',
                current_lap: 0,
                completed_laps: 0,
                normalized_position: 0,
                sample_count: 1,
                live_session_type: 'unknown',
                completed_lap_count: 0,
            }),
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            livePerformanceAnalystState,
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
        }, {}, publishStatus)).resolves.toMatchObject({
            status: 'started',
            agent_mode: 'live_performance_analyst',
        });

        expect(publishStatus).toHaveBeenCalledWith(expect.objectContaining({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'live_analysis_started',
            snapshot: expect.objectContaining({ track: 'brands_hatch' }),
        }));
    });

    it('is a name-keyed definition object covering every advertised tool', () => {
        expect(Object.keys(frontendAiToolRegistry).sort()).toEqual(
            [...FRONTEND_AI_TOOL_NAMES].sort(),
        );
        expect(FRONTEND_AI_TOOL_NAMES).toContain('query_analysis_result');
        expect(FRONTEND_AI_TOOL_NAMES).toContain('apply_query_to_analysis_result');
        expect(FRONTEND_AI_TOOL_NAMES).toContain('display_specific_result_in_overlay');
        FRONTEND_AI_TOOL_NAMES.forEach((name) => {
            expect(frontendAiToolRegistry[name].componentName).toEqual(expect.any(String));
        });
    });

    it('preserves the component operation instead of awaiting or wrapping its result', async () => {
        const componentOperation = resolvedAiToolOperation({
            status: 'started' as const,
            conversation_role: 'agent' as const,
            agent_mode: 'overtake' as const,
        }, 'started');
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
        }, 'ready');
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

    it('dispatches JSONata expressions and preserves actual JSON result types', async () => {
        const operations = new Map<string, AiToolOperation<QueryAnalysisResultOutput>>([
            ['$count(analyses)', resolvedAiToolOperation({ status: 'ready' as const, data: 4 }, 'ready')],
            ['{"count": $count(analyses.elements)}', resolvedAiToolOperation({
                status: 'ready' as const,
                data: { count: 4 },
            }, 'ready')],
            ['[analyses.elements.id]', resolvedAiToolOperation({
                status: 'ready' as const,
                data: ['first', 'second'],
            }, 'ready')],
            ['analyses.elements[id = "missing"]', resolvedAiToolOperation({
                status: 'ready' as const,
                data: null,
            }, 'ready')],
        ]);
        const componentName = 'visualization:analysis-results';
        const handle: Partial<AnalysisResultsChartHandle> = {
            queryAnalysisResult: jest.fn(({ query }) => operations.get(query)!) as any,
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(componentName, handle),
        });

        const results = Array.from(operations, ([query, componentOperation]) => {
            const returned = registry.query_analysis_result({ query });
            expect(returned).toBe(componentOperation);
            return Promise.all([returned.result, componentOperation.result]).then(([
                returnedResult,
                componentResult,
            ]) => expect(returnedResult).toEqual(componentResult));
        });
        await Promise.all(results);
        expect(handle.queryAnalysisResult).toHaveBeenCalledTimes(operations.size);
        expect(handle.queryAnalysisResult).toHaveBeenNthCalledWith(1, {
            query: '$count(analyses)',
        });
    });

    it.each([
        {},
        { query: '' },
        { query: '   ' },
        { query: 4 },
        { query: '$count(analyses)', extra: true },
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

    it('validates and dispatches an Analysis Results query apply operation unchanged', async () => {
        const componentOperation = resolvedAiToolOperation({
            status: 'ready' as const,
        }, 'ready');
        const handle: Partial<AnalysisResultsChartHandle> = {
            applyAnalysisResultQuery: jest.fn(() => componentOperation) as any,
        };
        const registry = createAiCommandRegistry({
            componentRefs: register('visualization:analysis-results', handle),
        });

        const returned = registry.apply_query_to_analysis_result({
            query: 'elements',
            page_number: -1,
        });

        expect(returned).toBe(componentOperation);
        await expect(returned.result).resolves.toEqual(await componentOperation.result);
        expect(handle.applyAnalysisResultQuery).toHaveBeenCalledWith({
            query: 'elements',
            page_number: -1,
        });
    });

    it.each([
        {},
        { query: '' },
        { query: '   ' },
        { query: 4 },
        { query: 'elements', page_number: 1.5 },
        { query: 'elements', page_number: '1' },
        { query: 'elements', page_number: undefined },
        { query: 'elements', extra: true },
    ])('rejects invalid Analysis Results apply arguments: %p', async (args) => {
        const handle: Partial<AnalysisResultsChartHandle> = {
            applyAnalysisResultQuery: jest.fn(),
        };
        const registry = createAiCommandRegistry({
            componentRefs: register('visualization:analysis-results', handle),
        });

        await expect(registry.apply_query_to_analysis_result(args).result).rejects.toMatchObject({
            name: 'InvalidToolCallError',
        });
        expect(handle.applyAnalysisResultQuery).not.toHaveBeenCalled();
    });

    it('rejects an analysis result expression when its tab is not mounted', async () => {
        const registry = createAiCommandRegistry({
            componentRefs: createAiToolComponentRefDirectory(),
        });

        const operation = registry.query_analysis_result({ query: '$count(analyses)' });

        await expect(operation.result).rejects.toMatchObject({
            name: 'ComponentRefUnavailableError',
            componentName: 'visualization:analysis-results',
        });
    });

    it('allows analysis result expressions in compatible live analyst goal steps', () => {
        expect(isGoalStepAvailableForContext({
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        }, 'query_analysis_result')).toBe(true);
        expect(isGoalStepAvailableForContext({
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        }, 'apply_query_to_analysis_result')).toBe(true);
        expect(isGoalStepAvailableForContext({
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'track_guide',
        }, 'query_analysis_result')).toBe(false);
    });
});

const reserve = (
    directory: ReturnType<typeof createAiToolComponentRefDirectory>,
    name: string,
    handle: object,
) => {
    directory.registerComponentRef({ current: {
        getComponentName: () => name,
        ...handle,
    } as any });
};

const comparisonData = (durationMs: number) => ({
    samples: durationMs > 0 ? [{
        driverTimeMs: 0,
        expertTimeMs: 0,
        driverTrackPosition: 0.1,
        expertTrackPosition: 0.1,
        driverGas: 0.2,
        expertGas: 0.3,
    }, {
        driverTimeMs: durationMs,
        expertTimeMs: durationMs,
        driverTrackPosition: 0.2,
        expertTrackPosition: 0.2,
        driverGas: 0.4,
        expertGas: 0.5,
    }] : [{
        driverTimeMs: 0,
        expertTimeMs: 0,
        driverTrackPosition: 0.1,
        expertTrackPosition: 0.1,
        driverGas: 0.2,
        expertGas: 0.3,
    }],
});

const createMockComparisonOperation = () => {
    const controller = createControlledAiToolOperation<
        'graph shown',
        never,
        'complete' | 'cancelled' | 'replaced' | 'failed'
    >();
    return {
        operation: controller.operation,
        complete: () => controller.resolve('complete', 'graph shown'),
        terminate: (status: 'cancelled' | 'replaced' | 'failed') => {
            controller.reject(status, new Error(status));
        },
    };
};

describe('specific Analysis Results overlay tool', () => {
    const args = { page_id: 'retained-page', result_id: 'braking-result' };

    const setup = () => {
        const directory = createAiToolComponentRefDirectory();
        const displayController = createMockComparisonOperation();
        const displaySpecificResultInOverlay = jest.fn(() => displayController.operation);
        reserve(directory, 'visualization:analysis-results', {
            displaySpecificResultInOverlay,
        } satisfies Partial<AnalysisResultsChartHandle>);
        return {
            displayController,
            displaySpecificResultInOverlay,
            registry: createAiCommandRegistry({ componentRefs: directory, sessionGame: 'acc' }),
        };
    };

    it('delegates exact ids and the operation lifecycle to Analysis Results', async () => {
        const test = setup();
        const operation = test.registry.display_specific_result_in_overlay(args);
        const terminated = new Promise((resolve) => operation.notifyTerminated(resolve));
        let resultSettled = false;
        void operation.result.then(() => { resultSettled = true; });

        await Promise.resolve();
        expect(resultSettled).toBe(false);
        expect(test.displaySpecificResultInOverlay).toHaveBeenCalledWith(
            'retained-page',
            'braking-result',
            undefined,
        );

        test.displayController.complete();
        test.displayController.complete();

        await expect(operation.result).resolves.toBe('graph shown');
        await expect(terminated).resolves.toEqual({
            status: 'complete',
            result: 'graph shown',
        });
    });

    it.each([
        {},
        { page_id: 'retained-page' },
        { result_id: 'braking-result' },
        { page_id: '', result_id: 'braking-result' },
        { page_id: 'retained-page', result_id: ' ' },
        { page_id: 'retained-page', result_id: 'braking-result', extra: true },
    ])('rejects invalid exact arguments without resolving or publishing: %p', async (invalidArgs) => {
        const test = setup();

        await expect(test.registry.display_specific_result_in_overlay(invalidArgs as any).result)
            .rejects.toMatchObject({ name: 'InvalidToolCallError' });
        expect(test.displaySpecificResultInOverlay).not.toHaveBeenCalled();
    });

    it.each(['cancelled', 'replaced', 'failed'] as const)(
        'preserves the Analysis Results %s termination path',
        async (status) => {
            const test = setup();
            const operation = test.registry.display_specific_result_in_overlay(args);
            const terminated = new Promise((resolve) => operation.notifyTerminated(resolve));

            test.displayController.terminate(status);

            await expect(operation.result).rejects.toThrow(status);
            await expect(terminated).resolves.toMatchObject({ status });
        },
    );

    it('forwards the task abort signal to Analysis Results', () => {
        const test = setup();
        const abortController = new AbortController();
        (test.registry.display_specific_result_in_overlay as any)(
            args,
            abortController.signal,
        );
        expect(test.displaySpecificResultInOverlay).toHaveBeenCalledWith(
            'retained-page',
            'braking-result',
            abortController.signal,
        );
    });
});

const todoResult = (events: readonly { id: string }[] = []) => ({
    status: events.length > 0 ? 'ready' as const : 'empty' as const,
    todo_list: {
        events: events.map(({ id }) => ({ id })),
        current_position: null,
        rolling_rate: null,
        created_at: 1,
        updated_at: 1,
    },
});

const scheduledItem = (
    id: string,
    toolName = 'analyze_telemetry',
    args: Record<string, unknown> = { scope: { type: 'now' } },
) => ({
    event: {
        id,
        normalized_position: 0.5,
        lead_time_seconds: 0,
        content: { title: id, description: `Run ${id}` },
    },
    tool: { name: toolName, arguments: args },
});

const childLiveRegistry = (
    directory: ReturnType<typeof createAiToolComponentRefDirectory>,
) => createAiCommandRegistry({
    componentRefs: directory,
    sessionMode: 'live',
    conversationRole: 'agent',
    agentMode: 'track_guide',
});

const analystLiveRegistry = (
    directory: ReturnType<typeof createAiToolComponentRefDirectory>,
) => createAiCommandRegistry({
    componentRefs: directory,
    sessionMode: 'live',
    conversationRole: 'agent',
    agentMode: 'live_performance_analyst',
    sessionGame: 'acc',
});

describe('live range to-do add tool', () => {
    it('reuses the mounted list, appends every event, and completes immediately', async () => {
        const directory = createAiToolComponentRefDirectory();
        const inserted: LiveRangeTodoEventInput[] = [];
        const addEvent = jest.fn((event: LiveRangeTodoEventInput) => {
            inserted.push(event);
            return todoResult(inserted) as any;
        });
        const handle: Partial<LiveRangeTodoListHandle> = {
            addEvent,
            get: () => todoResult([{ id: 'existing' }, ...inserted]) as any,
            getForAi: () => resolvedAiToolOperation({
                status: 'ready' as const,
                event_count: inserted.length + 1,
                pending_count: inserted.length + 1,
                running_count: 0,
            }, 'complete'),
        };
        reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, handle);

        const operation = childLiveRegistry(directory).add_event_to_live_range_todo_list({
            events: [scheduledItem('first'), scheduledItem('second')],
        });

        expect(addEvent).toHaveBeenCalledTimes(2);
        expect(addEvent).toHaveBeenNthCalledWith(1, expect.objectContaining({
            id: 'first',
            taskStart: expect.any(Function),
        }));
        expect(addEvent).toHaveBeenNthCalledWith(2, expect.objectContaining({
            id: 'second',
            taskStart: expect.any(Function),
        }));
        expect(operation.statuses).toEqual([]);
        await expect(operation.result).resolves.toEqual({
            status: 'ready',
            event_count: 3,
            pending_count: 3,
            running_count: 0,
        });
    });

    it('asks AI Chat to initialize the list when it is missing', async () => {
        const directory = createAiToolComponentRefDirectory();
        const addEvent = jest.fn(() => todoResult([{ id: 'mounted' }]) as any);
        const todoHandle: Partial<LiveRangeTodoListHandle> = {
            addEvent,
            get: () => todoResult() as any,
            getForAi: () => resolvedAiToolOperation({
                status: 'ready' as const,
                event_count: 1,
                pending_count: 1,
                running_count: 0,
            }, 'complete'),
        };
        const initializeLiveRangeTodoList = jest.fn(() => {
            reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, todoHandle);
            return todoHandle as LiveRangeTodoListHandle;
        });
        reserve(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, {
            initializeLiveRangeTodoList,
        } satisfies Partial<AiChatHandle>);

        const operation = childLiveRegistry(directory).add_event_to_live_range_todo_list({
            events: [scheduledItem('mounted')],
        });

        expect(initializeLiveRangeTodoList).toHaveBeenCalledTimes(1);
        expect(addEvent).toHaveBeenCalledTimes(1);
        await expect(operation.result).resolves.toMatchObject({ event_count: 1 });
    });

    it('dispatches stored nested-tool arguments only when telemetry makes the event due', async () => {
        const directory = createAiToolComponentRefDirectory();
        const runner = new LiveRangeTodoListRunner(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST);
        const summarize = () => {
            const events = runner.get().todo_list?.events ?? [];
            return resolvedAiToolOperation({
                status: events.length > 0 ? 'ready' as const : 'empty' as const,
                event_count: events.length,
                pending_count: events.filter(({ status }) => status === 'pending').length,
                running_count: events.filter(({ status }) => status === 'running').length,
            }, 'complete');
        };
        reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, {
            addEvent: (event: LiveRangeTodoEventInput) => runner.addEvent(event),
            get: () => runner.get(),
            getForAi: summarize,
        } satisfies Partial<LiveRangeTodoListHandle>);
        const analyzeTelemetryForAi = jest.fn(() => resolvedAiToolOperation({ status: 'ready' }, 'ready'));
        reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION, { analyzeTelemetryForAi });
        const storedArguments = { scope: { type: 'now' } };

        const operation = childLiveRegistry(directory).add_event_to_live_range_todo_list({
            events: [scheduledItem('deferred', 'analyze_telemetry', storedArguments)],
        });
        storedArguments.scope.type = 'last_seconds';

        await expect(operation.result).resolves.toMatchObject({ event_count: 1 });
        expect(analyzeTelemetryForAi).not.toHaveBeenCalled();
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0, Graphics_completed_laps: 1 });
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.6, Graphics_completed_laps: 1 });
        expect(analyzeTelemetryForAi).toHaveBeenCalledWith({ scope: { type: 'now' } });
        for (let index = 0; index < 6; index += 1) await Promise.resolve();
        expect(runner.get().todo_list?.events).toHaveLength(0);
    });

    it.each([
        ['empty batch', { events: [] }],
        ['malformed event', { events: [{ ...scheduledItem('bad'), event: { id: 'bad' } }] }],
        ['missing tool', { events: [{ event: scheduledItem('bad').event }] }],
        ['unavailable tool', { events: [scheduledItem('bad', 'run_recorded_ai_analysis')] }],
        ['recursive tool', { events: [scheduledItem('bad', 'add_event_to_live_range_todo_list')] }],
        ['filtered comparison recursion', {
            events: [scheduledItem(
                'bad',
                'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
            )],
        }],
        ['invalid arguments', { events: [scheduledItem('bad', 'analyze_telemetry', { value: undefined })] }],
        ['AI-provided ETA', {
            events: [{
                ...scheduledItem('bad'),
                event: { ...scheduledItem('bad').event, eta_seconds: 10 },
            }],
        }],
        ['duplicate ids', { events: [scheduledItem('same'), scheduledItem('same')] }],
    ])('rejects an invalid atomic batch: %s', async (_label, payload) => {
        const directory = createAiToolComponentRefDirectory();
        const addEvent = jest.fn();
        reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, {
            addEvent,
            get: () => todoResult() as any,
            getForAi: jest.fn(),
        } satisfies Partial<LiveRangeTodoListHandle>);

        const operation = childLiveRegistry(directory)
            .add_event_to_live_range_todo_list(payload as any);

        await expect(operation.result).rejects.toMatchObject({
            name: 'InvalidLiveRangeTodoListError',
        });
        expect(addEvent).not.toHaveBeenCalled();
    });

    it.each([
        'create_goal',
        'retry_goal_task',
        'set_procedure_plan',
        'advance_plan_step',
        'clear_procedure_plan',
    ])('rejects unsafe nested tool %s without mutating the list', async (toolName) => {
        const directory = createAiToolComponentRefDirectory();
        const addEvent = jest.fn();
        reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, {
            addEvent,
            get: () => todoResult() as any,
            getForAi: jest.fn(),
        } satisfies Partial<LiveRangeTodoListHandle>);

        const operation = childLiveRegistry(directory).add_event_to_live_range_todo_list({
            events: [scheduledItem('unsafe', toolName)],
        });

        await expect(operation.result).rejects.toMatchObject({
            name: 'InvalidLiveRangeTodoListError',
        });
        expect(addEvent).not.toHaveBeenCalled();
    });

    it('rejects collisions with existing events before adding any batch item', async () => {
        const directory = createAiToolComponentRefDirectory();
        const addEvent = jest.fn();
        reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, {
            addEvent,
            get: () => todoResult([{ id: 'existing' }]) as any,
            getForAi: jest.fn(),
        } satisfies Partial<LiveRangeTodoListHandle>);

        const operation = childLiveRegistry(directory).add_event_to_live_range_todo_list({
            events: [scheduledItem('new'), scheduledItem('existing')],
        });

        await expect(operation.result).rejects.toThrow(/Duplicate/);
        expect(addEvent).not.toHaveBeenCalled();
    });
});

describe('filtered Driver/Expert comparison queue tool', () => {
    afterEach(() => {
        jest.useRealTimers();
    });

    it('appends eligible segments in filtered order and publishes overlays only when due', async () => {
        jest.useFakeTimers();
        const directory = createAiToolComponentRefDirectory();
        const runner = new LiveRangeTodoListRunner(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST);
        const existingTask = jest.fn(() => resolvedAiToolOperation({}, 'complete'));
        const duplicateTask = jest.fn(() => resolvedAiToolOperation({}, 'complete'));
        runner.addEvent({
            id: 'existing-user-event',
            normalized_position: 0.95,
            lead_time_seconds: 0,
            content: { title: 'Existing user event' },
            taskStart: existingTask,
        });
        runner.addEvent({
            id: 'analysis-comparison:duplicate',
            normalized_position: 0.9,
            lead_time_seconds: 0,
            content: { title: 'Already queued comparison' },
            taskStart: duplicateTask,
        });
        reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, {
            addEvent: (event: LiveRangeTodoEventInput) => runner.addEvent(event),
            get: () => runner.get(),
        } satisfies Partial<LiveRangeTodoListHandle>);

        const fiveSecondComparison = comparisonData(5_000);
        const secondComparison = comparisonData(2_000);
        const filteredSnapshot: FilteredAnalysisSegmentsSnapshot = {
            status: 'ready',
            activePageId: 'analysis-page-7',
            appliedView: 'custom',
            committedQuery: 'elements^(>normalizedPositionRange.start)',
            segments: [{
                id: 'late-first',
                labels: ['MSP'],
                title: 'Late braking',
                section: 'Turn 1',
                normalizedPositionRange: { start: 0.4, end: 0.45 },
                comparison: fiveSecondComparison,
            }, {
                id: 'early-second',
                labels: ['MSR'],
                normalizedPositionRange: { start: 0.7, end: 0.75 },
                comparison: secondComparison,
            }, {
                id: 'duplicate',
                labels: ['MSP'],
                normalizedPositionRange: { start: 0.8, end: 0.85 },
                comparison: secondComparison,
            }, {
                id: 'bad-position',
                labels: ['MSP'],
                normalizedPositionRange: { start: 1.2, end: 1.3 },
                comparison: secondComparison,
            }, {
                id: 'missing-comparison',
                labels: ['MSP'],
                normalizedPositionRange: { start: 0.2, end: 0.25 },
            }, {
                id: 'zero-duration',
                labels: ['MSP'],
                normalizedPositionRange: { start: 0.3, end: 0.35 },
                comparison: comparisonData(0),
            }],
        };
        const displayControllers: ReturnType<typeof createMockComparisonOperation>[] = [];
        const displaySpecificResultInOverlay = jest.fn(() => {
            const display = createMockComparisonOperation();
            displayControllers.push(display);
            return display.operation;
        });
        reserve(directory, 'visualization:analysis-results', {
            getFilteredSegments: () => filteredSnapshot,
            displaySpecificResultInOverlay,
        } satisfies Partial<AnalysisResultsChartHandle>);
        reserve(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, {
            getOpportunityTelemetryRows: () => [{
                Graphics_normalized_car_position: 0.1,
                Graphics_estimated_lap_time: 100_000,
            }],
        } satisfies Partial<AiChatHandle>);

        const operation = analystLiveRegistry(directory)
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({});
        const result = await operation.result;

        expect(result).toMatchObject({
            status: 'ready',
            active_page_id: 'analysis-page-7',
            applied_view: 'custom',
            committed_query: 'elements^(>normalizedPositionRange.start)',
            matched_count: 6,
            queued_count: 2,
            skipped_count: 4,
            queued_timing: [{
                segment_id: 'late-first',
                event_id: 'analysis-comparison:late-first',
                normalized_position: 0.4,
                replay_duration_ms: 5_000,
                lead_time_seconds: 7,
            }, {
                segment_id: 'early-second',
                event_id: 'analysis-comparison:early-second',
                normalized_position: 0.7,
                replay_duration_ms: 2_000,
                lead_time_seconds: 4,
            }],
        });
        expect((result as any).skipped_segments).toEqual([
            expect.objectContaining({ segment_id: 'bad-position', reason_code: 'invalid_start_position' }),
            expect.objectContaining({ segment_id: 'missing-comparison', reason_code: 'comparison_unavailable' }),
            expect.objectContaining({ segment_id: 'zero-duration', reason_code: 'invalid_replay_duration' }),
            expect.objectContaining({ segment_id: 'duplicate', reason_code: 'already_queued' }),
        ]);
        expect(runner.get().todo_list?.events.map(({ id }) => id)).toEqual([
            'existing-user-event',
            'analysis-comparison:duplicate',
            'analysis-comparison:late-first',
            'analysis-comparison:early-second',
        ]);
        const queuedEvents = runner.get().todo_list?.events.slice(2) ?? [];
        expect(queuedEvents.map(({ id }) => id)).toEqual([
            'analysis-comparison:late-first',
            'analysis-comparison:early-second',
        ]);
        expect(queuedEvents[0].eta_seconds).toBeCloseTo(30);
        expect(queuedEvents[1].eta_seconds).toBeCloseTo(60);
        expect(displaySpecificResultInOverlay).not.toHaveBeenCalled();

        runner.acceptTelemetry({ Graphics_normalized_car_position: 0, Graphics_completed_laps: 1 });
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.5, Graphics_completed_laps: 1 });
        expect(displaySpecificResultInOverlay).toHaveBeenNthCalledWith(
            1,
            'analysis-page-7',
            'late-first',
            expect.any(AbortSignal),
        );

        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.8, Graphics_completed_laps: 1 });
        expect(displaySpecificResultInOverlay).toHaveBeenCalledTimes(1);
        runner.removeEvents(['existing-user-event', 'analysis-comparison:duplicate']);

        jest.advanceTimersByTime(60_000);
        for (let index = 0; index < 4; index += 1) await Promise.resolve();
        expect(displaySpecificResultInOverlay).toHaveBeenCalledTimes(1);

        displayControllers[0].complete();
        for (let index = 0; index < 6; index += 1) await Promise.resolve();
        expect(displaySpecificResultInOverlay).toHaveBeenCalledTimes(1);

        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.65, Graphics_completed_laps: 2 });
        expect(displaySpecificResultInOverlay).toHaveBeenNthCalledWith(
            2,
            'analysis-page-7',
            'early-second',
            expect.any(AbortSignal),
        );
        displayControllers[0].complete();
        expect(displaySpecificResultInOverlay).toHaveBeenCalledTimes(2);
        expect(existingTask).not.toHaveBeenCalled();
        expect(duplicateTask).not.toHaveBeenCalled();
        runner.dispose();
    });

    it('asks AI Chat to initialize the list for an eligible comparison', async () => {
        const directory = createAiToolComponentRefDirectory();
        const addEvent = jest.fn();
        const todoHandle: Partial<LiveRangeTodoListHandle> = {
            addEvent,
            get: () => todoResult() as any,
        };
        const initializeLiveRangeTodoList = jest.fn(() => {
            reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, todoHandle);
            return todoHandle as LiveRangeTodoListHandle;
        });
        reserve(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, {
            initializeLiveRangeTodoList,
        } satisfies Partial<AiChatHandle>);
        reserve(directory, 'visualization:analysis-results', {
            getFilteredSegments: () => ({
                status: 'ready',
                activePageId: 'mounted-page',
                appliedView: 'mistakes',
                committedQuery: 'elements',
                segments: [{
                    id: 'mounted-comparison',
                    labels: ['MSP'],
                    normalizedPositionRange: { start: 0.25, end: 0.3 },
                    comparison: comparisonData(1_000),
                }],
            }),
        } satisfies Partial<AnalysisResultsChartHandle>);

        await expect(analystLiveRegistry(directory)
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({}).result)
            .resolves.toMatchObject({ queued_count: 1 });

        expect(initializeLiveRangeTodoList).toHaveBeenCalledTimes(1);
        expect(addEvent).toHaveBeenCalledWith(expect.objectContaining({
            id: 'analysis-comparison:mounted-comparison',
            normalized_position: 0.25,
            lead_time_seconds: 3,
            taskStart: expect.any(Function),
        }));
    });

    it.each([
        ['main live session', { sessionMode: 'live', conversationRole: 'main' }],
        ['other child agent', {
            sessionMode: 'live', conversationRole: 'agent', agentMode: 'track_guide',
        }],
        ['recorded analyst', {
            sessionMode: 'recorded', conversationRole: 'agent', agentMode: 'live_performance_analyst',
        }],
    ])('rejects %s availability', async (_label, context) => {
        const registry = createAiCommandRegistry({
            ...context,
            componentRefs: createAiToolComponentRefDirectory(),
        } as any);

        await expect(registry
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({}).result)
            .rejects.toMatchObject({ name: 'ToolNotRegisteredError' });
    });

    it('returns busy without mounting a list and rejects arguments', async () => {
        const directory = createAiToolComponentRefDirectory();
        reserve(directory, 'visualization:analysis-results', {
            getFilteredSegments: () => ({
                status: 'busy',
                activePageId: 'busy-page',
                appliedView: 'mistakes',
                committedQuery: 'elements',
                segments: [],
            }),
        } satisfies Partial<AnalysisResultsChartHandle>);
        const registry = analystLiveRegistry(directory);

        await expect(registry
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({}).result)
            .resolves.toMatchObject({
                status: 'busy',
                matched_count: 0,
                queued_count: 0,
                skipped_count: 0,
            });
        await expect(registry
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({ extra: true }).result)
            .rejects.toMatchObject({ name: 'InvalidToolCallError' });
    });

    it('fails a matched batch when none of its results has a showable graph', async () => {
        const directory = createAiToolComponentRefDirectory();
        reserve(directory, 'visualization:analysis-results', {
            getFilteredSegments: () => ({
                status: 'ready',
                activePageId: 'unsupported-page',
                appliedView: 'mistakes',
                committedQuery: 'elements',
                segments: [{
                    id: 'unsupported-result',
                    labels: ['MSP'],
                    normalizedPositionRange: { start: 0.2, end: 0.3 },
                }],
            }),
        } satisfies Partial<AnalysisResultsChartHandle>);

        await expect(analystLiveRegistry(directory)
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({}).result)
            .rejects.toMatchObject({ name: 'ToolExecutionError' });
        expect(directory.findComponentRef(
            AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        )).toBeNull();
    });

    it('is available in analyst goals but never as a nested Live Range task', () => {
        expect(isGoalStepAvailableForContext({
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        }, 'add_filtered_driver_expert_comparisons_to_live_range_todo_list')).toBe(true);
        expect(isGoalStepAvailableForContext({
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'track_guide',
        }, 'add_filtered_driver_expert_comparisons_to_live_range_todo_list')).toBe(false);
        expect(isGoalStepAvailableForContext({
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        }, 'display_specific_result_in_overlay')).toBe(true);
    });
});
