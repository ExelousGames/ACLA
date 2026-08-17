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
    it('publishes live analyst runtime statuses without routing them through session intelligence', async () => {
        const publishStatus = jest.fn();
        const livePerformanceAnalystState = {
            enabled: false,
        };

        await expect(startAgentRuntime('live_performance_analyst', {
            sessionMode: 'live',
            recordingState: RecordingState.RECORDING,
            sessionIntelligence: {
                getLiveSessionSnapshot: () => ({
                    status: 'ready',
                    track: 'brands_hatch',
                }),
            } as any,
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
        expect(FRONTEND_AI_TOOL_NAMES).toContain('apply_analysis_result_query');
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

    it('dispatches JSONata expressions and preserves actual JSON result types', async () => {
        const operations = new Map<string, AiToolOperation<QueryAnalysisResultOutput>>([
            ['$count(analyses)', resolvedAiToolOperation({ status: 'ready' as const, data: 4 })],
            ['{"count": $count(analyses.elements)}', resolvedAiToolOperation({
                status: 'ready' as const,
                data: { count: 4 },
            })],
            ['[analyses.elements.id]', resolvedAiToolOperation({
                status: 'ready' as const,
                data: ['first', 'second'],
            })],
            ['analyses.elements[id = "missing"]', resolvedAiToolOperation({
                status: 'ready' as const,
                data: null,
            })],
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
            data: 2,
            applied_query: 'elements',
            applied_page_id: 'page-3',
            applied_page_number: 3,
            requested_page_number: -1,
            used_most_recent_fallback: true,
        });
        const handle: Partial<AnalysisResultsChartHandle> = {
            applyAnalysisResultQuery: jest.fn(() => componentOperation) as any,
        };
        const registry = createAiCommandRegistry({
            componentRefs: register('visualization:analysis-results', handle),
        });

        const returned = registry.apply_analysis_result_query({
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

        await expect(registry.apply_analysis_result_query(args).result).rejects.toMatchObject({
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
        }, 'apply_analysis_result_query')).toBe(true);
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
    directory.reserveComponentRef(name, Symbol(name), {
        getComponentName: () => name,
        ...handle,
    } as any);
};

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
            }),
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

    it('asks AI Chat to mount only when the list is missing', async () => {
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
            }),
        };
        const mountLiveRangeTodoList = jest.fn(() => {
            reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, todoHandle);
        });
        reserve(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, {
            mountLiveRangeTodoList,
        } satisfies Partial<AiChatHandle>);

        const operation = childLiveRegistry(directory).add_event_to_live_range_todo_list({
            events: [scheduledItem('mounted')],
        });

        expect(mountLiveRangeTodoList).toHaveBeenCalledTimes(1);
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
            });
        };
        reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, {
            addEvent: (event: LiveRangeTodoEventInput) => runner.addEvent(event),
            get: () => runner.get(),
            getForAi: summarize,
        } satisfies Partial<LiveRangeTodoListHandle>);
        const analyzeTelemetryForAi = jest.fn(() => resolvedAiToolOperation({ status: 'ready' }));
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
    const comparison = (durationMs: number) => ({
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

    it('appends eligible segments in filtered order and publishes overlays only when due', async () => {
        const directory = createAiToolComponentRefDirectory();
        const runner = new LiveRangeTodoListRunner(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST);
        const existingTask = jest.fn();
        const duplicateTask = jest.fn();
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

        const fiveSecondComparison = comparison(5_000);
        const secondComparison = comparison(2_000);
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
                comparison: comparison(0),
            }],
        };
        reserve(directory, 'visualization:analysis-results', {
            getFilteredSegments: () => filteredSnapshot,
        } satisfies Partial<AnalysisResultsChartHandle>);
        const displayDriverExpertComparison = jest.fn();
        reserve(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, {
            displayDriverExpertComparison,
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
        expect(displayDriverExpertComparison).not.toHaveBeenCalled();

        runner.acceptTelemetry({ Graphics_normalized_car_position: 0, Graphics_completed_laps: 1 });
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.5, Graphics_completed_laps: 1 });
        expect(displayDriverExpertComparison).toHaveBeenNthCalledWith(1, {
            title: 'Late braking: Driver vs Expert',
            comparison: fiveSecondComparison,
            game: 'acc',
        });
        for (let index = 0; index < 4; index += 1) await Promise.resolve();

        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.8, Graphics_completed_laps: 1 });
        expect(displayDriverExpertComparison).toHaveBeenNthCalledWith(2, {
            title: 'Driver vs Expert',
            comparison: secondComparison,
            game: 'acc',
        });
        expect(existingTask).not.toHaveBeenCalled();
        expect(duplicateTask).not.toHaveBeenCalled();
    });

    it('asks AI Chat to mount the list when an eligible comparison has no active list', async () => {
        const directory = createAiToolComponentRefDirectory();
        const addEvent = jest.fn();
        const todoHandle: Partial<LiveRangeTodoListHandle> = {
            addEvent,
            get: () => todoResult() as any,
        };
        const mountLiveRangeTodoList = jest.fn(() => {
            reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, todoHandle);
        });
        reserve(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, {
            mountLiveRangeTodoList,
            displayDriverExpertComparison: jest.fn(),
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
                    comparison: comparison(1_000),
                }],
            }),
        } satisfies Partial<AnalysisResultsChartHandle>);

        await expect(analystLiveRegistry(directory)
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({}).result)
            .resolves.toMatchObject({ queued_count: 1 });

        expect(mountLiveRangeTodoList).toHaveBeenCalledTimes(1);
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
    });
});
