import { parseProcedurePlanInput } from 'components/ai-operations/ProcedurePlan';
import { parseRepeatablePlanInput } from 'components/ai-operations/RepeatablePlan';
import type { WorkflowDispatcher } from 'components/ai-operations/tool';
import type { WorkflowPanelHandle } from 'components/ai-operations/WorkflowPanel';
import {
    createAiCommandRegistry,
    createWorkflowToolDispatcher,
    frontendOperationRegistry,
    startAgentRuntime,
} from '../ai-command-registry';
import type {
    AiCommandRegistry,
    FrontendAiCommandContext,
    FrontendAiQueryContractCoverage,
    QueryAnalysisResultOutput,
    QueryTelemetryMetricArguments,
    QueryTelemetryMetricResult,
    FrontendWorkflowName,
    FrontendToolName,
} from '../ai-command-registry';
import {
    OPERATION_COMPONENT_NAMES,
    createOperationComponentRefDirectory,
} from 'contexts/OperationComponentRefContext';
import {
    asTool,
    asWorkflow,
    createControlledOperation,
    LiveRangeTodoListRunner,
    resolvedOperation,
} from 'components/ai-operations';
import type {
    Operation,
    OperationQueryResult,
    LiveRangeTodoEventInput,
    LiveRangeTodoListHandle,
    Workflow,
    Tool,
    ProcedurePlanRunResult,
    ProcedurePlanInput,
    RepeatablePlanInput,
    LiveRangeTodoListInput,
} from 'components/ai-operations';
import type { AiChatHandle } from '../ai-chat';
import type { AnalysisResultsChartHandle } from '../../visualization/charts/AnalysisResultsChart';
import type { FilteredAnalysisSegmentsSnapshot } from '../../visualization/charts/AnalysisResultsChart';
import type { LiveSessionHandle } from 'views/live-session/LiveSessionView';
import { RecordingState } from 'views/lap-analysis/recording-state';

// @ts-expect-error OperationQueryResult requires an explicit data type.
type MissingOperationQueryResultGeneric = OperationQueryResult;

// @ts-expect-error QueryTelemetryMetricArguments requires an explicit reduction.
type MissingTelemetryArgumentsGeneric = QueryTelemetryMetricArguments;

// @ts-expect-error QueryTelemetryMetricResult requires an explicit reduction.
type MissingTelemetryResultGeneric = QueryTelemetryMetricResult;

const queryContractCoverage: FrontendAiQueryContractCoverage = true;

const assertQueryContractTypes = (registry: AiCommandRegistry) => {
    const input: ProcedurePlanInput = { workflow: { name: 'set_procedure_plan',
        goal: 'Count', operations: [{ operation: { name: 'query_analysis_result', title: 'Count', arguments: { query: '1' } } }],
    } };
    const workflow: Workflow<ProcedurePlanRunResult> = registry.set_procedure_plan(input);
    const tool: Tool<QueryAnalysisResultOutput> = registry.query_analysis_result({ query: 'analyses' });
    // @ts-expect-error Workflow commands cannot return the distinct Tool type.
    const invalidTool: Tool<ProcedurePlanRunResult> = registry.set_procedure_plan(input);
    // @ts-expect-error Workflow creation requires its named envelope.
    registry.set_procedure_plan({ goal: 'Legacy', requests: [] });
    // @ts-expect-error Workflow names are excluded from tool names.
    const invalidToolName: FrontendToolName = 'create_repeatable_plan';
    // @ts-expect-error Tool names are excluded from workflow names.
    const invalidWorkflowName: FrontendWorkflowName = 'query_analysis_result';
    const analysisResult: Operation<QueryAnalysisResultOutput> = (
        registry.query_analysis_result({ query: '$count(analyses)' })
    );
    // @ts-expect-error Analysis result queries require an expression.
    registry.query_analysis_result({});
    // @ts-expect-error Analysis result queries accept no extra arguments.
    registry.query_analysis_result({ query: 'analyses', extra: true });
    const avg: Operation<QueryTelemetryMetricResult<'avg'>> = registry.query_telemetry_metric({
        fields: ['speed'],
        scope: { type: 'now' },
        reduce: 'avg',
    });
    const min: Operation<QueryTelemetryMetricResult<'min'>> = registry.query_telemetry_metric({
        fields: ['speed'],
        scope: { type: 'now' },
        reduce: 'min',
    });
    const max: Operation<QueryTelemetryMetricResult<'max'>> = registry.query_telemetry_metric({
        fields: ['speed'],
        scope: { type: 'now' },
        reduce: 'max',
    });
    const stats: Operation<QueryTelemetryMetricResult<'stats'>> = registry.query_telemetry_metric({
        fields: ['speed'],
        scope: { type: 'now' },
        reduce: 'stats',
    });
    const display: Operation<'graph shown'> = registry.display_specific_result_in_overlay({
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
    const mismatchedReduction: Operation<QueryTelemetryMetricResult<'avg'>> = stats;

    return {
        workflow, tool, invalidTool, invalidToolName, invalidWorkflowName,
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
    const directory = createOperationComponentRefDirectory();
    directory.registerComponentRef({ current: {
        getComponentName: () => name,
        ...handle,
    } as any });
    return directory;
};

describe('frontend operation registry', () => {
    const workflowNames: FrontendWorkflowName[] = [
        'create_repeatable_plan',
        'append_repeatable_plan',
        'append_procedure_plan',
        'create_live_range_todo_list',
        'set_procedure_plan',
        'advance_plan_step',
        'clear_procedure_plan',
        'add_event_to_live_range_todo_list',
        'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
        'get_live_range_todo_list',
    ];

    it('classifies plan and live range operations as workflows and other operations as tools', () => {
        const workflows = Object.values(frontendOperationRegistry)
            .filter(({ kind }) => kind === 'workflow')
            .map(({ name }) => name);
        expect(workflows.sort()).toEqual([...workflowNames].sort());
        expect(frontendOperationRegistry.query_analysis_result.kind).toBe('tool');
    });

    it.each(workflowNames)('retains workflow classification when %s fails before execution', async (name) => {
        const workflow = createAiCommandRegistry({})[name]({} as any);
        expect(workflow.kind).toBe('workflow');
        await expect(workflow.result).rejects.toBeInstanceOf(Error);
    });

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

    it('creates a handler for every name-keyed operation definition', () => {
        const registry = createAiCommandRegistry({});
        expect(Object.keys(registry).sort()).toEqual(
            Object.keys(frontendOperationRegistry).sort(),
        );
        expect(registry).toHaveProperty('query_analysis_result');
        expect(registry).toHaveProperty('apply_query_to_analysis_result');
        expect(registry).toHaveProperty('display_specific_result_in_overlay');
        Object.entries(frontendOperationRegistry).forEach(([name, definition]) => {
            expect(definition.name).toBe(name);
            expect(definition.componentName).toEqual(expect.any(String));
        });
    });

    it('preserves the component operation instead of awaiting or wrapping its result', async () => {
        const componentOperation = resolvedOperation({
            status: 'started' as const,
            conversation_role: 'agent' as const,
            agent_mode: 'overtake' as const,
        }, 'started');
        const handle: Partial<AiChatHandle> = {
            startAgentSession: jest.fn(() => componentOperation),
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT, handle),
        });

        const returned = registry.start_agent_session({ agent_mode: 'overtake' });

        expect(returned).toBe(componentOperation);
        expect(returned.kind).toBe('tool');
        await expect(returned.result).resolves.toMatchObject({ status: 'started' });
        expect(returned.statuses).toEqual([]);
    });

    it('returns a rejected operation for an unavailable component', async () => {
        const registry = createAiCommandRegistry({
            componentRefs: createOperationComponentRefDirectory(),
        });

        const operation = registry.show_map({});

        await expect(operation.result).rejects.toMatchObject({
            name: 'ComponentRefUnavailableError',
        });
    });

    it('preserves a reduction-specific telemetry component operation', async () => {
        const componentOperation = resolvedOperation({
            status: 'ready' as const,
            data: { Physics_speed_kmh: 123 },
        }, 'ready');
        const handle: Partial<LiveSessionHandle> = {
            queryTelemetryMetricForAi: jest.fn(() => componentOperation) as any,
        };
        const registry = createAiCommandRegistry({
            componentRefs: register(OPERATION_COMPONENT_NAMES.LIVE_SESSION, handle),
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
        const operations = new Map<string, Operation<QueryAnalysisResultOutput>>([
            ['$count(analyses)', resolvedOperation({ status: 'ready' as const, data: 4 }, 'ready')],
            ['{"count": $count(analyses.elements)}', resolvedOperation({
                status: 'ready' as const,
                data: { count: 4 },
            }, 'ready')],
            ['[analyses.elements.id]', resolvedOperation({
                status: 'ready' as const,
                data: ['first', 'second'],
            }, 'ready')],
            ['analyses.elements[id = "missing"]', resolvedOperation({
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
            name: 'InvalidOperationCallError',
        });
        expect(handle.queryAnalysisResult).not.toHaveBeenCalled();
    });

    it('validates and dispatches an Analysis Results query apply operation unchanged', async () => {
        const componentOperation = resolvedOperation({
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
            name: 'InvalidOperationCallError',
        });
        expect(handle.applyAnalysisResultQuery).not.toHaveBeenCalled();
    });

    it('rejects an analysis result expression when its tab is not mounted', async () => {
        const registry = createAiCommandRegistry({
            componentRefs: createOperationComponentRefDirectory(),
        });

        const operation = registry.query_analysis_result({ query: '$count(analyses)' });

        await expect(operation.result).rejects.toMatchObject({
            name: 'ComponentRefUnavailableError',
            componentName: 'visualization:analysis-results',
        });
    });
});

const reserve = (
    directory: ReturnType<typeof createOperationComponentRefDirectory>,
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
    const controller = createControlledOperation<
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
        const directory = createOperationComponentRefDirectory();
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
            .rejects.toMatchObject({ name: 'InvalidOperationCallError' });
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

    it('forwards the task abort signal and aborts the returned operation', async () => {
        const test = setup();
        const abortController = new AbortController();
        const operation = (test.registry.display_specific_result_in_overlay as any)(
            args,
            abortController.signal,
        );
        const termination = new Promise((resolve) => operation.notifyTerminated(resolve));
        expect(test.displaySpecificResultInOverlay).toHaveBeenCalledWith(
            'retained-page',
            'braking-result',
            abortController.signal,
        );

        abortController.abort();

        await expect(operation.result).rejects.toMatchObject({ name: 'AbortError' });
        await expect(termination).resolves.toMatchObject({ status: 'aborted' });
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
    operation: { name: toolName,
        event: {
            id,
            normalized_position: 0.5,
            lead_time_seconds: 0,
            content: { title: id, description: `Run ${id}` },
        },
        arguments: args,
    },
});

const scheduledPayload = (operations: ReturnType<typeof scheduledItem>[]) => ({
    workflow: { name: 'add_event_to_live_range_todo_list', operations },
} as unknown as LiveRangeTodoListInput);

const childLiveRegistry = (
    directory: ReturnType<typeof createOperationComponentRefDirectory>,
) => createAiCommandRegistry({
    componentRefs: directory,
    sessionMode: 'live',
    conversationRole: 'agent',
    agentMode: 'track_guide',
});

const analystLiveRegistry = (
    directory: ReturnType<typeof createOperationComponentRefDirectory>,
) => createAiCommandRegistry({
    componentRefs: directory,
    sessionMode: 'live',
    conversationRole: 'agent',
    agentMode: 'live_performance_analyst',
    sessionGame: 'acc',
});

describe('strict workflow creation and tool dispatch', () => {
    const procedure = (): ProcedurePlanInput => ({ workflow: { name: 'set_procedure_plan',
        goal: 'Review the session',
        operations: [
            { operation: { name: 'query_analysis_result', title: 'Count analyses', arguments: { query: '$count(analyses)' } } },
            { operation: { name: 'query_analysis_result', title: 'Read analyses', arguments: { query: 'analyses' } } },
        ],
    } });
    const repeatable = (): RepeatablePlanInput => ({ workflow: { name: 'create_repeatable_plan',
        goal: 'Review until ready',
        operations: [
            { operation: { name: 'query_analysis_result', id: 'one', title: 'First count', arguments: { query: '1' } } },
            { operation: { name: 'query_analysis_result', id: 'two', title: 'Second count', arguments: { query: '2' } } },
        ],
        stop_when: { tool: { name: 'query_analysis_result', arguments: { query: '3' }  }, operator: 'gte', target: 3 },
    } });
    const setup = (context: FrontendAiCommandContext = {
        sessionMode: 'live', conversationRole: 'agent', agentMode: 'live_performance_analyst',
    }) => {
        const createProcedurePlan = jest.fn((_input: unknown, _dispatch: unknown) => asWorkflow(resolvedOperation({ status: 'complete' }, 'complete')));
        const createRepeatablePlan = jest.fn((_input: unknown, _dispatch: unknown) => asWorkflow(resolvedOperation({ status: 'achieved' }, 'complete')));
        const directory = register(OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL, {
            createProcedurePlan: (input: unknown, dispatch: WorkflowDispatcher) => {
                parseProcedurePlanInput(input).requests.forEach((step) => dispatch.validate(step.name!));
                return createProcedurePlan(input, dispatch);
            },
            createRepeatablePlan: (input: unknown, dispatch: WorkflowDispatcher) => {
                const request = parseRepeatablePlanInput(input);
                request.steps.forEach((step) => dispatch.validate(step.name));
                dispatch.validate(request.stop_when.tool.name);
                return createRepeatablePlan(input, dispatch);
            },
        });
        return { registry: createAiCommandRegistry({ ...context, componentRefs: directory }), createProcedurePlan, createRepeatablePlan };
    };

    it.each(['live', 'recorded', 'front_desk', 'user_summary'] as const)(
        'forwards full plan envelopes, repeated operations, and stop calls in a main %s session', async (sessionMode) => {
            const test = setup({ sessionMode, conversationRole: 'main' });
            const plan = procedure();
            const goal = repeatable();
            await expect(test.registry.set_procedure_plan(plan).result).resolves.toMatchObject({ status: 'complete' });
            await expect(test.registry.create_repeatable_plan(goal).result).resolves.toMatchObject({ status: 'achieved' });
            expect(test.createProcedurePlan).toHaveBeenCalledWith(plan, expect.any(Function));
            expect(test.createRepeatablePlan).toHaveBeenCalledWith(goal, expect.any(Function));
            expect(test.createProcedurePlan.mock.calls[0][0]).toBe(plan);
            expect(test.createRepeatablePlan.mock.calls[0][0]).toBe(goal);
        },
    );

    it.each(Object.values(frontendOperationRegistry)
        .filter(({ kind }) => kind === 'workflow')
        .map(({ name }) => name))(
        'forwards workflow children in %s to component-owned validation and execution', async (name) => {
            const test = setup();
            const plan: any = procedure();
            plan.workflow.operations.push({ operation: { name: name, title: 'Invalid later call', arguments: {} } });
            const goal: any = repeatable();
            goal.workflow.operations.push({ operation: { name: name, id: 'invalid', title: 'Invalid later call' } });
            const stop: any = repeatable();
            stop.workflow.stop_when.tool = { name };
            await expect(test.registry.set_procedure_plan(plan).result).resolves.toBeDefined();
            await expect(test.registry.create_repeatable_plan(goal).result).resolves.toBeDefined();
            await expect(test.registry.create_repeatable_plan(stop).result).rejects.toThrow(/stop condition/);
            expect(test.createProcedurePlan).toHaveBeenCalled();
            expect(test.createRepeatablePlan).toHaveBeenCalled();
        },
    );

    it.each([
        'missing', 'toString',
    ])('preflights unregistered later tool %s before either plan is created', async (name) => {
        const test = setup();
        const plan: any = procedure();
        plan.workflow.operations.push({ operation: { name: name, title: 'Later', arguments: {} } });
        const goal: any = repeatable();
        goal.workflow.stop_when.tool = { name, arguments: {} };
        await expect(test.registry.set_procedure_plan(plan).result).rejects.toBeInstanceOf(Error);
        await expect(test.registry.create_repeatable_plan(goal).result).rejects.toBeInstanceOf(Error);
        expect(test.createProcedurePlan).not.toHaveBeenCalled();
        expect(test.createRepeatablePlan).not.toHaveBeenCalled();
    });

    it.each(['run_recorded_ai_analysis', 'start_agent_session'])(
        'accepts registered tool %s in both plans and stop checks', async (name) => {
            const test = setup();
            const plan: any = procedure();
            plan.workflow.operations.push({ operation: { name: name, title: 'Later', arguments: {} } });
            const goal: any = repeatable();
            goal.workflow.operations.push({ operation: { name: name, id: 'later', title: 'Later', arguments: {} } });
            goal.workflow.stop_when.tool = { name, arguments: {} };

            await expect(test.registry.set_procedure_plan(plan).result).resolves.toMatchObject({ status: 'complete' });
            await expect(test.registry.create_repeatable_plan(goal).result).resolves.toMatchObject({ status: 'achieved' });
            expect(test.createProcedurePlan).toHaveBeenCalledWith(plan, expect.any(Function));
            expect(test.createRepeatablePlan).toHaveBeenCalledWith(goal, expect.any(Function));
        },
    );

    it.each<[string, string, unknown]>([
        ['unwrapped procedure', 'set_procedure_plan', { goal: 'Legacy', operations: [] }],
        ['legacy procedure', 'set_procedure_plan', { goal: 'Legacy', requests: [] }],
        ['wrong procedure wrapper', 'set_procedure_plan', repeatable()],
        ['mixed procedure', 'set_procedure_plan', { ...procedure(), requests: [] }],
        ...['payload', 'args', 'parameters'].map((alias): [string, string, unknown] => [
            `procedure ${alias}`, 'set_procedure_plan', { workflow: { name: 'set_procedure_plan',
                goal: 'Invalid', operations: [{ operation: { name: 'query_analysis_result', title: 'Count', arguments: {}, [alias]: {} } }],
            } },
        ]),
        ['unwrapped repeatable', 'create_repeatable_plan', repeatable().workflow],
        ['legacy repeatable', 'create_repeatable_plan', { name: 'Legacy', steps: [], stop_when: {} }],
        ['wrong repeatable wrapper', 'create_repeatable_plan', procedure()],
        ['mixed repeatable', 'create_repeatable_plan', { workflow: { ...repeatable().workflow, steps: [] } }],
        ['name stop descriptor', 'create_repeatable_plan', { workflow: { ...repeatable().workflow,
            stop_when: { tool: { query_analysis_result: { arguments: { query: '3' } } }, operator: 'gte', target: 3 },
        } }],
    ])('rejects %s without invoking creation handlers', async (_label, name, input) => {
        const test = setup();
        await expect((test.registry as any)[name as string](input).result).rejects.toBeInstanceOf(Error);
        expect(test.createProcedurePlan).not.toHaveBeenCalled();
        expect(test.createRepeatablePlan).not.toHaveBeenCalled();
    });

    it('rejects every workflow before its handler through the tool dispatcher', () => {
        const dispatcher = createWorkflowToolDispatcher({
            sessionMode: 'live', conversationRole: 'agent', agentMode: 'live_performance_analyst',
        });
        Object.values(frontendOperationRegistry).filter(({ kind }) => kind === 'workflow')
            .forEach((definition) => {
                const handler = jest.spyOn(definition, 'execute');
                expect(() => dispatcher.validate(definition.name)).not.toThrow();
                handler.mockRestore();
            });
    });

    it('lets the tool handler report its own session error', async () => {
        const error = new Error('Live telemetry is unavailable.');
        const getNextCornerForAi = jest.fn(() => { throw error; });
        const dispatcher = createWorkflowToolDispatcher({
            sessionMode: 'recorded',
            componentRefs: register(OPERATION_COMPONENT_NAMES.LIVE_SESSION, { getNextCornerForAi }),
        });

        await expect(dispatcher('get_next_corner').result).rejects.toBe(error);
        expect(getNextCornerForAi).toHaveBeenCalledTimes(1);
    });

    it.each(['live', 'recorded', 'front_desk', 'user_summary'] as const)(
        'dispatches registered recorded operations in a %s agent context', async (sessionMode) => {
            const operation = asTool(resolvedOperation({ status: 'ready' }, 'ready'));
            const runRecordedAnalysisForAi = jest.fn(() => operation);
            const dispatcher = createWorkflowToolDispatcher({
                sessionMode, conversationRole: 'agent', agentMode: 'track_guide',
                componentRefs: register(OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS, { runRecordedAnalysisForAi }),
            });
            const args = { session_id: 'recorded-session' };

            expect(dispatcher('run_recorded_ai_analysis', args)).toBe(operation);
            await expect(operation.result).resolves.toEqual({ status: 'ready' });
            expect(runRecordedAnalysisForAi).toHaveBeenCalledWith(args);
        },
    );

    it('does not reclassify a Workflow returned by a tool handler', async () => {
        const workflow = asWorkflow(resolvedOperation({ status: 'ready' }, 'ready'));
        const dispatcher = createWorkflowToolDispatcher({
            sessionMode: 'live',
            componentRefs: register(OPERATION_COMPONENT_NAMES.LIVE_SESSION, { getNextCornerForAi: () => workflow }),
        });
        await expect(dispatcher('get_next_corner').result).rejects.toThrow(/returned a workflow/);
        expect(workflow.kind).toBe('workflow');
    });
});

const mountLiveQueue = (directory: ReturnType<typeof createOperationComponentRefDirectory>, runner = new LiveRangeTodoListRunner(OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)) => {
    runner.addComponentRef(directory);
    reserve(directory, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL, {
        appendLiveRangeTodoList: (input: LiveRangeTodoListInput, dispatch: WorkflowDispatcher) => runner.appendLiveRangeTodoList(input, dispatch),
    });
    return runner;
};

describe('live range to-do workflow forwarding', () => {
    it('forwards the complete native envelope and immediate append operation to the component', async () => {
        const result = asWorkflow(resolvedOperation({ status: 'ready', event_count: 2 }, 'complete'));
        const appendLiveRangeTodoList = jest.fn(() => result);
        const registry = childLiveRegistry(register(OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL, { appendLiveRangeTodoList }));
        const payload = scheduledPayload([scheduledItem('first'), scheduledItem('second')]);
        const operation = registry.add_event_to_live_range_todo_list(payload);
        expect(appendLiveRangeTodoList).toHaveBeenCalledWith(payload, expect.any(Function));
        expect(operation).toBe(result);
        await expect(operation.result).resolves.toMatchObject({ event_count: 2 });
    });

    it.each(['analyze_telemetry', 'run_recorded_ai_analysis'])('preserves literal %s arguments until telemetry makes the event due', async (toolName) => {
        const directory = createOperationComponentRefDirectory();
        const runner = mountLiveQueue(directory);
        const toolHandler = jest.fn(() => asTool(resolvedOperation({ status: 'ready' }, 'ready')));
        reserve(directory, OPERATION_COMPONENT_NAMES.LIVE_SESSION, { analyzeTelemetryForAi: toolHandler });
        reserve(directory, OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS, { runRecordedAnalysisForAi: toolHandler });
        const args = { scope: { type: 'now' }, tool: { name: 'literal data' } };
        await expect(childLiveRegistry(directory).add_event_to_live_range_todo_list(scheduledPayload([scheduledItem('deferred', toolName, args)])).result)
            .resolves.toMatchObject({ event_count: 1 });
        args.scope.type = 'changed';
        expect(toolHandler).not.toHaveBeenCalled();
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0 });
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.6 });
        expect(toolHandler).toHaveBeenCalledWith({ scope: { type: 'now' }, tool: { name: 'literal data' } });
        runner.dispose();
    });

    it.each([
        ['unwrapped', { operations: [scheduledItem('bad')] }],
        ['legacy events', { events: [scheduledItem('bad')] }],
        ['wrong wrapper', { workflow: { name: 'set_procedure_plan', operations: [scheduledItem('bad')] } }],
        ['mixed envelope', { ...scheduledPayload([scheduledItem('bad')]), events: [] }],
        ['mixed body', { workflow: { name: 'add_event_to_live_range_todo_list', operations: [scheduledItem('bad')], events: [] } }],
        ['empty batch', scheduledPayload([])],
        ['zero names', scheduledPayload([{} as any])],
        ['multiple names', scheduledPayload([{
            ...scheduledItem('first'), get_event_log: {},
        } as any])],
        ['malformed later event', scheduledPayload([
            scheduledItem('first'), { operation: { name: 'analyze_telemetry', event: { id: 'bad' }, arguments: {} } } as any,
        ])],
        ['missing arguments', scheduledPayload([{
            operation: { name: 'analyze_telemetry', event: scheduledItem('bad').operation.event },
        } as any])],
        ['name descriptor', scheduledPayload([{
            event: scheduledItem('bad').operation.event,
            operation: { name: 'analyze_telemetry', arguments: {} },
        } as any])],
        ['unknown later tool', scheduledPayload([scheduledItem('first'), scheduledItem('bad', 'missing')])],
        ['inherited name', scheduledPayload([scheduledItem('bad', 'toString')])],
        ['invalid arguments', scheduledPayload([scheduledItem('bad', 'analyze_telemetry', { value: undefined })])],
        ['AI-provided ETA', scheduledPayload([{
            operation: {
                ...scheduledItem('bad').operation,
                event: { ...scheduledItem('bad').operation.event, eta_seconds: 10 },
            },
        } as any])],
        ['duplicate ids', scheduledPayload([scheduledItem('same'), scheduledItem('same')])],
    ])('rejects an invalid atomic batch: %s', async (_label, payload) => {
        const directory = createOperationComponentRefDirectory();
        const runner = mountLiveQueue(directory);
        const added = jest.spyOn(runner, 'addEvent');
        await expect(childLiveRegistry(directory).add_event_to_live_range_todo_list(payload as any).result)
            .rejects.toMatchObject({ name: 'InvalidLiveRangeTodoListError' });
        expect(added).not.toHaveBeenCalled();
        runner.dispose();
    });

    it('rejects existing event IDs before adding any part of a batch', async () => {
        const directory = createOperationComponentRefDirectory();
        const runner = mountLiveQueue(directory);
        const registry = childLiveRegistry(directory);
        await registry.add_event_to_live_range_todo_list(scheduledPayload([scheduledItem('existing')])).result;
        const before = runner.getSnapshot();
        await expect(registry.add_event_to_live_range_todo_list(scheduledPayload([scheduledItem('new'), scheduledItem('existing')])).result).rejects.toThrow(/Duplicate/);
        expect(runner.getSnapshot()).toEqual(before);
        runner.dispose();
    });
});

describe('filtered Driver/Expert comparison queue workflow', () => {
    afterEach(() => {
        jest.useRealTimers();
    });

    it.each(['failure', 'abort'] as const)('does not queue comparisons after voice generation %s', async (outcome) => {
        const directory = createOperationComponentRefDirectory();
        const addEvent = jest.fn();
        reserve(directory, OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, {
            addEvent, get: () => todoResult() as any,
        } satisfies Partial<LiveRangeTodoListHandle>);
        let finish!: (value: Record<string, number>) => void;
        let fail!: (error: Error) => void;
        const preparation = new Promise<Record<string, number>>((resolve, reject) => { finish = resolve; fail = reject; });
        const prepareComparisonVoices = jest.fn(() => preparation);
        reserve(directory, 'visualization:analysis-results', {
            prepareComparisonVoices,
            getFilteredSegments: () => ({
                status: 'ready', activePageId: 'page', appliedView: 'mistakes', committedQuery: 'elements',
                segments: [{ id: 'corner', labels: [], normalizedPositionRange: { start: 0.5, end: 0.6 }, comparison: comparisonData(1000) }],
            }),
        } satisfies Partial<AnalysisResultsChartHandle>);
        const operation = analystLiveRegistry(directory).add_filtered_driver_expert_comparisons_to_live_range_todo_list({ workflow: { name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list', operations: [],  } });
        await Promise.resolve();
        expect(prepareComparisonVoices).toHaveBeenCalled();
        expect(addEvent).not.toHaveBeenCalled();
        if (outcome === 'abort') {
            operation.abort();
            finish({ corner: 8000 });
        } else {
            fail(new Error('Speech unavailable'));
        }
        await expect(operation.result).rejects.toBeInstanceOf(Error);
        expect(addEvent).not.toHaveBeenCalled();
    });

    it('appends eligible segments in filtered order and publishes overlays only when due', async () => {
        jest.useFakeTimers();
        const directory = createOperationComponentRefDirectory();
        const runner = new LiveRangeTodoListRunner(OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST);
        const existingTask = jest.fn(() => asTool(resolvedOperation({}, 'complete')));
        const duplicateTask = jest.fn(() => asTool(resolvedOperation({}, 'complete')));
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
        mountLiveQueue(directory, runner);

        const fiveSecondComparison = comparisonData(5_000);
        const secondComparison = comparisonData(2_000);
        const filteredSnapshot: FilteredAnalysisSegmentsSnapshot = {
            status: 'ready',
            activePageId: 'analysis-page-7',
            appliedView: 'custom',
            committedQuery: 'elements^(>normalizedPositionRange.start)',
            segments: [{
                id: 'late-first',
                labels: labelRanges('MSP'),
                title: 'Late braking',
                section: 'Turn 1',
                normalizedPositionRange: { start: 0.4, end: 0.45 },
                comparison: fiveSecondComparison,
            }, {
                id: 'early-second',
                labels: labelRanges('MSR'),
                normalizedPositionRange: { start: 0.7, end: 0.75 },
                comparison: secondComparison,
            }, {
                id: 'duplicate',
                labels: labelRanges('MSP'),
                normalizedPositionRange: { start: 0.8, end: 0.85 },
                comparison: secondComparison,
            }, {
                id: 'bad-position',
                labels: labelRanges('MSP'),
                normalizedPositionRange: { start: 1.2, end: 1.3 },
                comparison: secondComparison,
            }, {
                id: 'missing-comparison',
                labels: labelRanges('MSP'),
                normalizedPositionRange: { start: 0.2, end: 0.25 },
            }, {
                id: 'zero-duration',
                labels: labelRanges('MSP'),
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
            prepareComparisonVoices: jest.fn(async (_pageId, ids) => {
                expect(ids).toEqual(['late-first', 'early-second']);
                expect(runner.get().todo_list?.events).toHaveLength(2);
                return { 'late-first': 1_000, 'early-second': 1_000 };
            }),
            displaySpecificResultInOverlay,
        } satisfies Partial<AnalysisResultsChartHandle>);
        reserve(directory, OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT, {
            getOpportunityTelemetryRows: () => [{
                Graphics_normalized_car_position: 0.1,
                Graphics_estimated_lap_time: 100_000,
            }],
        } satisfies Partial<AiChatHandle>);

        const operation = analystLiveRegistry(directory)
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({ workflow: { name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list', operations: [],  } });
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);
        const result = await operation.result;

        expect(terminated).toHaveBeenCalledWith({ status: 'ready', result });
        expect(result).toMatchObject({
            status: 'ready',
            active_page_id: 'analysis-page-7',
            applied_view: 'custom',
            committed_query: 'elements^(>normalizedPositionRange.start)',
            matched_count: 6,
            queued_count: 2,
            skipped_count: 4,
        });
        expect(result).not.toHaveProperty('queued_timing');
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
        expect(queuedEvents).toMatchObject([
            { normalized_position: 0.4, lead_time_seconds: 7 },
            { normalized_position: 0.7, lead_time_seconds: 4 },
        ]);
        expect(queuedEvents[0].eta_seconds).toBeNull();
        expect(queuedEvents[1].eta_seconds).toBeNull();
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
        expect(displaySpecificResultInOverlay).toHaveBeenCalledTimes(1);
        jest.advanceTimersByTime(1_000);
        runner.acceptTelemetry({ Graphics_normalized_car_position: 0.68, Graphics_completed_laps: 2 });
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

    it('routes eligible comparisons through the panel append command', async () => {
        const directory = createOperationComponentRefDirectory();
        const appendLiveRangeTodoList = jest.fn(() => asWorkflow(resolvedOperation({ status: 'ready', event_count: 1 }, 'complete')));
        reserve(directory, OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL, { appendLiveRangeTodoList });
        reserve(directory, 'visualization:analysis-results', {
            prepareComparisonVoices: async () => ({ 'mounted-comparison': 8_000 }),
            getFilteredSegments: () => ({
                status: 'ready',
                activePageId: 'mounted-page',
                appliedView: 'mistakes',
                committedQuery: 'elements',
                segments: [{
                    id: 'mounted-comparison',
                    labels: labelRanges('MSP'),
                    normalizedPositionRange: { start: 0.25, end: 0.3 },
                    comparison: comparisonData(1_000),
                }],
            }),
        } satisfies Partial<AnalysisResultsChartHandle>);

        await expect(analystLiveRegistry(directory)
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({ workflow: { name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list', operations: [],  } }).result)
            .resolves.toMatchObject({ queued_count: 1 });

        expect(appendLiveRangeTodoList).toHaveBeenCalledWith({ workflow: {
            name: 'add_event_to_live_range_todo_list',
            operations: [{ operation: { name: 'display_specific_result_in_overlay',
                event: { id: 'analysis-comparison:mounted-comparison', normalized_position: 0.25, lead_time_seconds: 10,
                    content: { title: 'Driver vs Expert' } },
                arguments: { page_id: 'mounted-page', result_id: 'mounted-comparison' } } }],
        } }, expect.any(Function));
    });

    it.each([
        ['main live session', { sessionMode: 'live', conversationRole: 'main' }],
        ['other child agent', {
            sessionMode: 'live', conversationRole: 'agent', agentMode: 'track_guide',
        }],
        ['recorded analyst', {
            sessionMode: 'recorded', conversationRole: 'agent', agentMode: 'live_performance_analyst',
        }],
    ])('delegates comparison scheduling in a %s', async (_label, context) => {
        const getFilteredSegments = jest.fn(() => ({
            status: 'busy' as const,
            activePageId: null,
            appliedView: null,
            committedQuery: null,
            segments: [],
        }));
        const registry = createAiCommandRegistry({
            ...context,
            componentRefs: register('visualization:analysis-results', { getFilteredSegments }),
        } as any);

        await expect(registry
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({ workflow: { name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list', operations: [],  } }).result)
            .resolves.toMatchObject({ status: 'busy' });
        expect(getFilteredSegments).toHaveBeenCalledTimes(1);
    });

    it.each(['busy', 'empty'] as const)('reports %s on termination without mounting a list and rejects arguments', async (status) => {
        const directory = createOperationComponentRefDirectory();
        reserve(directory, 'visualization:analysis-results', {
            getFilteredSegments: () => ({
                status,
                activePageId: 'busy-page',
                appliedView: 'mistakes',
                committedQuery: 'elements',
                segments: [],
            }),
        } satisfies Partial<AnalysisResultsChartHandle>);
        const registry = analystLiveRegistry(directory);

        const operation = registry.add_filtered_driver_expert_comparisons_to_live_range_todo_list({ workflow: { name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list', operations: [],  } });
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);
        await expect(operation.result)
            .resolves.toMatchObject({
                status,
                matched_count: 0,
                queued_count: 0,
                skipped_count: 0,
            });
        expect(terminated).toHaveBeenCalledWith({
            status,
            result: expect.objectContaining({ status, queued_count: 0 }),
        });
        expect(directory.findComponentRef(OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)).toBeNull();
        await expect(registry
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({ workflow: { name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list', operations: [],  extra: true  } } as any).result)
            .rejects.toMatchObject({ name: 'InvalidOperationCallError' });
    });

    it('fails a matched batch when none of its results has a showable graph', async () => {
        const directory = createOperationComponentRefDirectory();
        reserve(directory, 'visualization:analysis-results', {
            getFilteredSegments: () => ({
                status: 'ready',
                activePageId: 'unsupported-page',
                appliedView: 'mistakes',
                committedQuery: 'elements',
                segments: [{
                    id: 'unsupported-result',
                    labels: labelRanges('MSP'),
                    normalizedPositionRange: { start: 0.2, end: 0.3 },
                }],
            }),
        } satisfies Partial<AnalysisResultsChartHandle>);

        await expect(analystLiveRegistry(directory)
            .add_filtered_driver_expert_comparisons_to_live_range_todo_list({ workflow: { name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list', operations: [],  } }).result)
            .rejects.toMatchObject({ name: 'OperationExecutionError' });
        expect(directory.findComponentRef(
            OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        )).toBeNull();
    });
});

function labelRanges(...names: string[]) {
    return names.map((label_name) => ({ label_name, start_index: 0, end_index: 1 }));
}
