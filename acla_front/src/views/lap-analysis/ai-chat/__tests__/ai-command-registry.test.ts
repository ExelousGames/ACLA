import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    createAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import {
    BaselineCollectionVisualizationRequiredError,
    BaselineLapRecordRequiredError,
    ComponentMountTimeoutError,
    ComponentNameMismatchError,
    ComponentRefUnavailableError,
    DuplicateAiToolComponentError,
    GoalStepFailedError,
    InvalidGoalNameError,
    InvalidLiveRangeTodoListError,
    LiveAnalysisResultUnavailableError,
    NonLiveContextLiveToolsUnavailableError,
    RecordedAnalysisFailedError,
    SessionAnalysisFailedError,
} from 'contexts/AiToolComponentError';
import { RecordingState } from 'views/lap-analysis/recording-state';
import {
    GoalRunner,
    LiveRangeTodoListRunner,
    ProcedurePlanRunner,
} from 'components/ai-engineering-tools';
import {
    AmbiguousComponentTargetError,
    CreateGoalToolUnavailableError,
    getToolEnvelopeUiOutput,
    LivePerformanceAnalystToolUnavailableError,
    LiveSectionClassificationFailedError,
    RetryGoalTaskToolUnavailableError,
    TelemetryAnalysisFailedError,
    ToolExecutionError,
    ToolOutputEnvelope,
} from '../ai-tool-base';
import {
    AiCommandRegistryContext,
    createAiCommandRegistry,
    frontendToolDefinitions,
    isGoalStepAvailableForContext,
} from '../ai-command-registry';

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: jest.fn() },
}));

const mockPost = apiService.post as jest.Mock;
const handlerContext = { toolRunId: 'run-1', sendToolStatus: jest.fn() };
const uiOutput = (envelope: ToolOutputEnvelope) => getToolEnvelopeUiOutput(envelope) as any;

const analysisResult = {
    status: 'success',
    session_id: 'session-1',
    samples_analyzed: 2,
    parent_segment_count: 1,
    segments: [{
        id: 'segment-1',
        labels: ['MSP'],
        track_section: 'Turn 1',
        start_index: 0,
        end_index: 1,
        expert_reference_data: [],
    }],
};

const reserve = (directory: AiToolComponentRefDirectory, name: string, value: Record<string, any>) => {
    directory.reserveComponentRef(name, Symbol(name), { getComponentName: () => name, ...value } as any);
};

const createHarness = (mode: 'live' | 'recorded' | 'user_summary' = 'live') => {
    const directory = createAiToolComponentRefDirectory();
    const rows = [{ Physics_speed_kmh: 201, Physics_brake: 0.4 }, { Physics_speed_kmh: 205, Physics_brake: 0.1 }];
    const comparison = { delta: -0.2 };
    const classification = { sectionId: 'turn-1', sectionName: 'Turn 1', severity: 1 };
    const baseline = { id: 'baseline-1', lap: 2, track: 'Monza', car: 'BMW', sample_count: 2, captured_at: 'now', records: rows };
    const chat = {
        getSessionMode: jest.fn(() => mode),
        getRecordingState: jest.fn(() => RecordingState.RECORDING),
        startAgentSession: jest.fn(() => ({ status: 'started', conversation_role: 'agent', agent_mode: 'track_guide' })),
        stopAgentSession: jest.fn(() => ({ status: 'stopped', conversation_role: 'agent' })),
        startTrackGuide: jest.fn(), setTrackGuideEnabled: jest.fn(), setLivePerformanceAnalystEnabled: jest.fn(),
        advanceProcedurePlanStep: jest.fn(() => ({ status: 'ready' })), getProcedurePlan: jest.fn(() => null),
        clearProcedurePlan: jest.fn(), setProcedurePlan: jest.fn(),
        selectTaskStartFunction: jest.fn(() => jest.fn()),
        selectGoalTaskStartFunction: jest.fn(() => jest.fn()),
        setAgentTagActive: jest.fn(),
        getOpportunityTelemetryRows: jest.fn(() => rows), getOpportunityAgentState: jest.fn(() => ({})),
        getLivePerformanceAnalystState: jest.fn(() => ({})), getLabelName: jest.fn((id: string) => id),
        getCategoryLabels: jest.fn(() => []), getCircuitMapById: jest.fn(async () => null),
        getCircuitMapByTrack: jest.fn(async () => ({ id: 'map-1', circuit_name: 'Monza', source_track_key: 'monza' })),
        displayMap: jest.fn(),
    };
    const intelligence = {
        compareFocusedSection: jest.fn(() => comparison),
        getFocusSection: jest.fn(() => ({ section: { id: 'turn-1', name: 'Turn 1', from: 0.1, to: 0.2 }, baseline: {}, selectedAt: 'now', reason: 'largest gap' })),
        getSectionTiming: jest.fn(() => ({ delta: 0.2 })),
    };
    const live = {
        getRecordingState: jest.fn(() => RecordingState.RECORDING), getSessionIntelligence: jest.fn(() => intelligence),
        getCurrentTelemetry: jest.fn(() => rows.at(-1)), queryTelemetryMetric: jest.fn(() => ({ avg: 203 })),
        getTelemetryForScope: jest.fn(() => rows), getEventLog: jest.fn(() => [{ type: 'brake' }]),
        getNextCorner: jest.fn(() => ({ name: 'Turn 1' })),
        getLiveSessionSnapshot: jest.fn(() => ({ status: 'ready', track: 'Monza', car: 'BMW', current_lap: 3 })),
        getLiveSectionHistory: jest.fn(() => [classification]),
        getLiveSectionTelemetry: jest.fn(() => ({ status: 'ready', section: { id: 'turn-1', name: 'Turn 1' }, lap: 3, startSampleIdx: 0, endSampleIdx: 1, rows })),
        recordLiveSectionClassification: jest.fn(() => classification),
        getLatestAnalysisResultPage: jest.fn<any, []>(() => null),
    };
    const recordedState = { sessionId: 'session-1', status: 'ready', result: analysisResult };
    const recorded = {
        getSelectedSession: jest.fn(() => ({ SessionId: 'session-1', session_name: 'Race', map: 'Monza', car: 'BMW' })),
        getMapSelected: jest.fn(() => 'Monza'), getRecordedAiAnalysis: jest.fn(() => recordedState),
        getRecordedPlaybackSummary: jest.fn(() => ({ sessionId: 'session-1', sampleCount: 2, durationSeconds: 1, playbackIndex: 1, playbackTimeSeconds: 0.5, activeSegment: null })),
        runRecordedAiAnalysis: jest.fn(async () => recordedState), requestSessionAnalysis: jest.fn(async () => ({ session: true })),
        requestPerformanceInsights: jest.fn(async () => ({ insights: true })), requestLapComparison: jest.fn(async () => ({ comparison: true })),
        requestExpertLineGuidance: jest.fn(async () => ({ guidance: true })), requestTelemetryData: jest.fn(async () => ({ telemetry: true })),
    };
    const summary = {
        getUserSummaryMapLevel: jest.fn(() => ({ status: 'ready', maps: [] })),
        getAvailableUserSummaryMaps: jest.fn(() => ({ status: 'ready', map_count: 1 })),
        searchUserSummaryMapLevel: jest.fn(() => ({ status: 'ready', match_count: 1 })),
    };
    const todo = {
        addEvent: jest.fn(), replaceEvents: jest.fn(() => ({ status: 'ready', todo_list: { events: [] } })),
        updateEvents: jest.fn(() => ({ status: 'ready', todo_list: { events: [] } })), removeEvents: jest.fn(),
        resetEvents: jest.fn(), clear: jest.fn(), get: jest.fn(() => ({ status: 'ready', todo_list: { events: [] } })),
    };
    const goal = {
        createGoal: jest.fn(async () => ({
            name: 'No mistakes',
            status: 'achieved',
            determination: {
                tool: { name: 'get_live_analysis_mistake_count' },
                result_path: 'mistake_count',
                operator: 'eq',
                target: 0,
            },
            target: 0,
            actual: 0,
            completed_steps: ['collect'],
            task_results: [{
                step_id: 'collect', tool_name: 'collect_live_baseline', attempt: 1,
                status: 'completed', value: { status: 'ready' },
            }],
            determination_result: {
                tool_name: 'get_live_analysis_mistake_count', attempt: 1,
                status: 'completed', value: 0,
                source_result: { tool_name: 'get_live_analysis_mistake_count', run_id: 'nested-2', status: 'complete', final: true },
            },
        })),
        retryFailedTask: jest.fn(async () => {
            throw new GoalStepFailedError(
                AI_TOOL_COMPONENT_NAMES.GOAL,
                'The failed goal task could not be retried.',
            );
        }),
        acceptToolOutput: jest.fn(),
        getSnapshot: jest.fn(() => null),
        clear: jest.fn(),
    };
    const childOwners = new Map<string, symbol>();
    const childHandles = new Map<string, any>();
    const instances: any[] = [];
    const mountChild = (name: string, type: string) => {
        if (directory.findComponentRef(name)?.current) return;
        const child = {
            getComponentName: () => name,
            updateTelemetry: jest.fn(() => true), updateLiveTelemetry: jest.fn(() => true),
            updateEvents: jest.fn(() => true), updateLiveEvents: jest.fn(() => true), updateMap: jest.fn(() => true),
            replaceAnalysisResults: jest.fn(() => true), appendAnalysisResult: jest.fn(() => ({ success: true })),
            updateAnalysisResult: jest.fn(() => ({ success: true })), removeAnalysisResult: jest.fn(() => ({ success: true })),
            updateGuidanceData: jest.fn(() => true), refreshGuidanceOnce: jest.fn(async () => ({ success: true })),
            disableTelemetry: jest.fn(() => true), disableLiveTelemetry: jest.fn(() => true),
            disableEventLog: jest.fn(() => true), disableLiveEventLog: jest.fn(() => true), disableMap: jest.fn(() => true),
            disableAnalysisResults: jest.fn(() => true), disableGuidance: jest.fn(() => true),
            startCollection: jest.fn(() => ({
                status: 'complete', progress_percent: 100, car: baseline.car, track: baseline.track,
                message: 'Baseline complete. Cached lap record is ready.',
            })),
            restartCollection: jest.fn(() => ({
                status: 'waiting_for_start', progress_percent: 0, car: null, track: null,
                message: 'Waiting for the next lap start',
            })),
            requestAnalysis: jest.fn(async () => ({
                status: 'ready',
                message: 'Telemetry analysis is ready.',
                analysis: analysisResult,
                source: 'baseline_lap_record',
                baseline,
                chartId: 'chart-1',
                component_name: 'visualization:analysis-results',
            })),
            getTag: jest.fn(() => ({ status: 'complete', progress_percent: 100 })),
            getLapRecord: jest.fn(() => baseline),
            getToolOutput: jest.fn(() => null),
            subscribeToolOutput: jest.fn(() => jest.fn()),
        };
        const owner = Symbol(name);
        childOwners.set(name, owner); childHandles.set(name, child);
        directory.reserveComponentRef(name, owner, child);
    };
    const manager = {
        getVisualizationCapabilities: jest.fn(() => ({ availableCharts: [] })),
        getCurrentVisualizations: jest.fn(() => instances),
        requestVisualization: jest.fn((options: any) => {
            const exact = instances.find((item) => item.name === options.name);
            if (!exact) instances.push({ id: `chart-${instances.length + 1}`, ...options });
            mountChild(options.name, options.type);
            const item = exact || instances.at(-1);
            return { success: true, message: exact ? 'Reused chart.' : 'Opened chart.', componentName: item.name, chartId: item.id, chartType: item.type, reused: Boolean(exact) };
        }),
        updateVisualization: jest.fn(() => ({ success: true, message: 'Updated chart.' })),
        closeVisualization: jest.fn(() => ({ success: true, message: 'Closed chart.' })),
    };

    reserve(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, chat);
    reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION, live);
    reserve(directory, AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS, recorded);
    reserve(directory, AI_TOOL_COMPONENT_NAMES.USER_SUMMARY, summary);
    mountChild(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION, 'baseline-collection');
    reserve(directory, mode === 'live' ? AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER : AI_TOOL_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER, manager);

    const context: AiCommandRegistryContext = {
        componentRefs: directory, sessionId: 'session-1', sessionMode: mode,
        opportunityAgentState: { intervalId: null, inFlight: false, lastAlertKey: null, lastAlertAt: 0 },
        startTrackGuide: jest.fn(), setTrackGuideEnabled: jest.fn(), getOpportunityTelemetryRows: jest.fn(() => rows),
    };
    return { directory, context, chat, live, recorded, summary, todo, goal, manager, instances, childOwners, childHandles, rows, classification };
};

describe('named component-ref AI command registry', () => {
    beforeEach(() => {
        mockPost.mockReset().mockResolvedValue({ data: analysisResult });
        handlerContext.sendToolStatus.mockReset();
    });

    it('publishes the static frontend map with all 44 handlers', () => {
        const names = frontendToolDefinitions.map((definition) => definition.name);
        expect(names).toHaveLength(44);
        expect(new Set(names).size).toBe(44);
        expect(names).toEqual(expect.arrayContaining([
            'analyze_telemetry',
            'classify_live_section',
            'get_live_analysis_mistake_count',
            'create_goal',
            'retry_goal_task',
        ]));
        expect(frontendToolDefinitions.find(({ name }) => name === 'retry_goal_task'))
            .toMatchObject({ schema: { properties: {}, required: [] }, required: [] });
    });

    it('creates and removes a goal runner only for the Live Performance Analyst', async () => {
        const h = createHarness('live');
        const args = {
            name: 'No mistakes',
            steps: [{ id: 'collect', title: 'Collect', name: 'collect_live_baseline' }],
            determination: {
                tool: { name: 'get_live_analysis_mistake_count' },
                result_path: 'mistake_count',
                operator: 'eq',
                target: 0,
            },
        };
        const analyst = createAiCommandRegistry({
            ...h.context,
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        });
        (h.chat.selectGoalTaskStartFunction as jest.Mock).mockImplementation((task: any) => jest.fn(() => (
            task.name === 'get_live_analysis_mistake_count'
                ? { mistake_count: 0 }
                : { status: 'ready' }
        )));
        const result = await analyst.create_goal(args, handlerContext);

        expect(h.directory.findBaseComponentRef()).toBeNull();
        expect(h.chat.selectGoalTaskStartFunction).toHaveBeenLastCalledWith({
            title: 'No mistakes',
            name: 'get_live_analysis_mistake_count',
        });
        expect(uiOutput(result)).toMatchObject({ status: 'achieved', target: 0, actual: 0 });
        expect(result.output).toMatchObject({
            status: 'achieved',
            target: 0,
            actual: 0,
            completed_steps: ['collect'],
            determination: args.determination,
            determination_result: expect.objectContaining({
                tool_name: 'get_live_analysis_mistake_count',
                value: 0,
            }),
            task_results: [expect.objectContaining({ step_id: 'collect' })],
        });

        const unavailable = createAiCommandRegistry({
            ...h.context,
            conversationRole: 'agent',
            agentMode: 'track_guide',
        }).create_goal(args, handlerContext);
        await expect(unavailable).rejects.toBeInstanceOf(CreateGoalToolUnavailableError);
    });

    it('rejects the legacy create_goal shape before invoking the goal component', async () => {
        const h = createHarness('live');
        const result = createAiCommandRegistry({
            ...h.context,
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        }).create_goal({
            goal: 'Legacy goal',
            steps: [{ id: 'count', title: 'Count', name: 'get_live_analysis_mistake_count' }],
            comparison: {
                step_id: 'count',
                result_path: 'mistake_count',
                operator: 'eq',
                target: 0,
                metric_label: 'Mistakes',
            },
        }, handlerContext);

        await expect(result).rejects.toBeInstanceOf(InvalidGoalNameError);
        expect(h.directory.findBaseComponentRef()).toBeNull();
    });

    it('retains an errored goal for retry and removes it after retry completion', async () => {
        const h = createHarness('live');
        const analystContext = {
            ...h.context,
            conversationRole: 'agent' as const,
            agentMode: 'live_performance_analyst' as const,
        };
        let preparationAttempts = 0;
        (h.chat.selectGoalTaskStartFunction as jest.Mock).mockImplementation((task: any) => jest.fn(() => {
            if (task.name === 'get_live_analysis_mistake_count') return { mistake_count: 0 };
            preparationAttempts += 1;
            if (preparationAttempts === 1) throw new Error('analysis offline');
            return { status: 'ready' };
        }));
        const registry = createAiCommandRegistry(analystContext);
        const args = {
            name: 'Recoverable goal',
            steps: [{ id: 'analyze', title: 'Analyze', name: 'analyze_live_recorded_analysis' }],
            determination: {
                tool: { name: 'get_live_analysis_mistake_count' },
                result_path: 'mistake_count',
                operator: 'eq',
                target: 0,
            },
        };

        await expect(registry.create_goal(args, handlerContext))
            .rejects.toBeInstanceOf(GoalStepFailedError);
        expect(h.directory.findBaseComponentRef()?.current).toBeInstanceOf(GoalRunner);
        expect(uiOutput(await registry.retry_goal_task({}, handlerContext)))
            .toMatchObject({ status: 'achieved' });
        expect(h.directory.findBaseComponentRef()).toBeNull();
        expect(isGoalStepAvailableForContext(analystContext, 'retry_goal_task')).toBe(false);
        expect(isGoalStepAvailableForContext(analystContext, 'create_goal')).toBe(false);

        const unavailable = createAiCommandRegistry({
            ...h.context,
            conversationRole: 'agent',
            agentMode: 'track_guide',
        }).retry_goal_task({}, handlerContext);
        await expect(unavailable).rejects.toBeInstanceOf(RetryGoalTaskToolUnavailableError);
    });

    it('routes mistake counting through the newest stored page and preserves counts for AI output', async () => {
        const h = createHarness('live');
        h.chat.getLabelName.mockImplementation((id: string) => (
            id === 'MSP' ? 'Training Error' : id === 'MSR' ? 'Race Error' : id
        ));
        h.live.getLatestAnalysisResultPage.mockReturnValue({
            id: 'latest-page',
            createdAt: 2,
            baseline: {
                id: 'baseline-2', lap: 4, lap_time_ms: 90_000, captured_at: 1,
                track: 'Monza', car: 'BMW', sample_count: 2,
            },
            elements: [
                { id: 'practice', labels: ['Training Error', 'Training Error'] },
                { id: 'racing', labels: ['Mistake (Racing)'] },
                { id: 'combined', labels: ['MSP', 'MSR'] },
                { id: 'unrelated', labels: ['MSP1', 'Telemetry'] },
            ],
        });
        const registry = createAiCommandRegistry({
            ...h.context,
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        });

        const result = await registry.get_live_analysis_mistake_count({}, handlerContext);

        expect(uiOutput(result)).toEqual({
            status: 'ready',
            mistake_count: 3,
            practice_mistake_count: 2,
            racing_mistake_count: 2,
            page_id: 'latest-page',
            baseline_lap: 4,
            track: 'Monza',
            car: 'BMW',
        });
        expect(result.output).toMatchObject({
            mistake_count: 3,
            practice_mistake_count: 2,
            racing_mistake_count: 2,
            baseline_lap: 4,
        });
        expect(mockPost).not.toHaveBeenCalled();
    });

    it('returns zero for an empty analyzed page and an explicit error when no page is stored', async () => {
        const h = createHarness('live');
        const registry = createAiCommandRegistry({
            ...h.context,
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        });
        h.live.getLatestAnalysisResultPage.mockReturnValue({
            id: 'empty-page',
            createdAt: 2,
            baseline: {
                id: 'baseline-2', lap: 4, lap_time_ms: null, captured_at: 1,
                track: 'Monza', car: 'BMW', sample_count: 2,
            },
            elements: [],
        });

        expect(uiOutput(await registry.get_live_analysis_mistake_count({}, handlerContext)))
            .toMatchObject({ status: 'ready', mistake_count: 0 });
        h.live.getLatestAnalysisResultPage.mockReturnValue(null);
        await expect(registry.get_live_analysis_mistake_count({}, handlerContext))
            .rejects.toBeInstanceOf(LiveAnalysisResultUnavailableError);
    });

    it('rejects mistake counting outside the Live Performance Analyst agent', async () => {
        const h = createHarness('live');
        const main = createAiCommandRegistry({ ...h.context, conversationRole: 'main' });
        const trackGuide = createAiCommandRegistry({
            ...h.context,
            conversationRole: 'agent',
            agentMode: 'track_guide',
        });

        await expect(main.get_live_analysis_mistake_count({}, handlerContext))
            .rejects.toBeInstanceOf(LivePerformanceAnalystToolUnavailableError);
        await expect(trackGuide.get_live_analysis_mistake_count({}, handlerContext))
            .rejects.toBeInstanceOf(LivePerformanceAnalystToolUnavailableError);
        expect(h.live.getLatestAnalysisResultPage).not.toHaveBeenCalled();
    });

    it('opens baseline visualization and drives a registered procedure-plan runner', async () => {
        const h = createHarness('live');
        const awaitComponentRef = jest.spyOn(h.directory, 'awaitComponentRef');
        const registry = createAiCommandRegistry(h.context);
        h.chat.selectTaskStartFunction.mockImplementation(() => (
            jest.fn(() => new Promise<void>(() => undefined))
        ));
        await registry.start_agent_session({ agent_mode: 'track_guide' }, handlerContext);
        await registry.stop_agent_session({ agent_session_id: 'agent-1' }, handlerContext);
        await registry.collect_live_baseline({}, handlerContext);
        await registry.restart_live_baseline({}, handlerContext);
        await registry.set_procedure_plan({
            goal: 'Improve',
            requests: [
                { type: 'tool', title: 'Brake later' },
                { type: 'tool', title: 'Accelerate sooner' },
            ],
        }, handlerContext);
        expect(h.directory.findBaseComponentRef()?.current).toBeInstanceOf(ProcedurePlanRunner);
        expect(uiOutput(await registry.advance_plan_step({ reason: 'done' }, handlerContext)))
            .toMatchObject({ status: 'advanced', current_request: 0 });
        await registry.clear_procedure_plan({}, handlerContext);
        expect(h.directory.findBaseComponentRef()).toBeNull();
        await registry.show_map({ map_id: 'monza' }, handlerContext);

        expect(h.chat.startAgentSession).toHaveBeenCalled();
        expect(h.chat.stopAgentSession).toHaveBeenCalled();
        expect(h.manager.requestVisualization).toHaveBeenCalledWith({
            name: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            type: 'baseline-collection',
        });
        expect(h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION).startCollection).toHaveBeenCalled();
        expect(h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION).startCollection)
            .toHaveBeenCalledWith('run-1');
        expect(h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION).restartCollection).toHaveBeenCalled();
        expect(awaitComponentRef).toHaveBeenCalledWith(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION, 5000);
        expect(h.chat.setProcedurePlan).not.toHaveBeenCalled();
        expect(h.chat.advanceProcedurePlanStep).not.toHaveBeenCalled();
        expect(h.chat.clearProcedurePlan).not.toHaveBeenCalled();
        expect(h.chat.displayMap).toHaveBeenCalledWith(expect.objectContaining({ status: 'ready' }));
    });

    it('keeps already-running and not-running agent states successful', async () => {
        const h = createHarness('live');
        h.chat.startAgentSession.mockReturnValueOnce({
            status: 'already_running',
            conversation_role: 'agent',
            agent_mode: 'track_guide',
            agent_session_id: 'agent-1',
            parent_client_session_id: 'main-1',
        } as any);
        h.chat.stopAgentSession.mockReturnValueOnce({
            status: 'not_running',
            conversation_role: 'agent',
            agent_mode: 'track_guide',
            agent_session_id: 'agent-1',
        } as any);
        const registry = createAiCommandRegistry(h.context);

        expect(uiOutput(await registry.start_agent_session(
            { agent_mode: 'track_guide' },
            handlerContext,
        ))).toMatchObject({ status: 'already_running', agent_session_id: 'agent-1' });
        expect(uiOutput(await registry.stop_agent_session(
            { agent_session_id: 'agent-1' },
            handlerContext,
        ))).toMatchObject({ status: 'not_running', agent_session_id: 'agent-1' });
    });

    it('keeps an unavailable circuit map as a successful domain result', async () => {
        const h = createHarness('recorded');
        h.chat.getCircuitMapByTrack.mockResolvedValueOnce(null as any);

        const result = await createAiCommandRegistry(h.context).show_map(
            { track: 'Unknown Circuit' },
            handlerContext,
        );

        expect(uiOutput(result)).toMatchObject({
            status: 'unavailable',
            requested_map: 'Monza',
        });
        expect(h.chat.displayMap).toHaveBeenCalledWith(expect.objectContaining({
            status: 'unavailable',
            requestedMap: 'Monza',
        }));
    });

    it('preserves component-state failures at the shared boundary', async () => {
        const h = createHarness('live');
        const startFailure = new NonLiveContextLiveToolsUnavailableError(
            AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT,
            'Agent sessions are only available in live session mode.',
        );
        h.chat.startAgentSession.mockImplementationOnce(() => { throw startFailure; });
        const registry = createAiCommandRegistry(h.context);

        await expect(registry.start_agent_session(
            { agent_mode: 'track_guide' },
            handlerContext,
        )).rejects.toBe(startFailure);
        await expect(registry.advance_plan_step({}, handlerContext))
            .rejects.toBeInstanceOf(ComponentRefUnavailableError);
    });

    it('requires the mounted baseline visualization for baseline-dependent commands', async () => {
        const h = createHarness('live');
        h.directory.releaseComponentRef(
            AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            h.childOwners.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)!,
        );
        const registry = createAiCommandRegistry(h.context);

        for (const result of await Promise.allSettled([
            registry.get_live_focus_section({}, handlerContext),
            registry.analyze_live_recorded_analysis({}, handlerContext),
            registry._get_live_section_telemetry({ section_id: 'turn-1' }, handlerContext),
            registry._record_live_section_classification({ section_id: 'turn-1' }, handlerContext),
            registry.classify_live_section({ section_id: 'turn-1' }, handlerContext),
        ])) {
            expect(result).toMatchObject({ status: 'rejected' });
            expect((result as PromiseRejectedResult).reason)
                .toBeInstanceOf(BaselineCollectionVisualizationRequiredError);
            expect((result as PromiseRejectedResult).reason).toMatchObject({
                message: expect.stringContaining('Baseline collection visualization required'),
            });
        }
    });

    it('resolves Live Session before delegating baseline analysis without calling the API', async () => {
        const h = createHarness('live');
        const baseline = h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION);
        const result = await createAiCommandRegistry(h.context)
            .analyze_live_recorded_analysis({ limit: 3 }, handlerContext);

        expect(h.live.getRecordingState).toHaveBeenCalled();
        expect(baseline.requestAnalysis).toHaveBeenCalledWith({ limit: 3 });
        expect(h.live.getRecordingState.mock.invocationCallOrder[0])
            .toBeLessThan(baseline.requestAnalysis.mock.invocationCallOrder[0]);
        expect(mockPost).not.toHaveBeenCalled();
        expect(uiOutput(result)).toMatchObject({
            status: 'ready',
            chartId: 'chart-1',
            component_name: 'visualization:analysis-results',
        });
    });

    it('does not resolve Baseline Collection when Live Session is missing', async () => {
        const h = createHarness('live');
        const baseline = h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION);
        h.directory.findComponentRef(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!.current = null;

        const result = createAiCommandRegistry(h.context)
            .analyze_live_recorded_analysis({}, handlerContext);

        await expect(result).rejects.toBeInstanceOf(ComponentRefUnavailableError);
        expect(baseline.requestAnalysis).not.toHaveBeenCalled();
    });

    it('preserves the Baseline Collection incomplete-lap failure', async () => {
        const h = createHarness('live');
        const baseline = h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION);
        const componentFailure = new BaselineLapRecordRequiredError(
            AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            'Live recorded analysis requires a recorded baseline lap before it can run.',
        );
        baseline.requestAnalysis.mockRejectedValueOnce(componentFailure);

        const result = createAiCommandRegistry(h.context)
            .analyze_live_recorded_analysis({}, handlerContext);

        await expect(result).rejects.toBe(componentFailure);
    });

    it('routes live telemetry, events, focus/history, and classifications through LiveSessionView', async () => {
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);
        await registry.query_telemetry_metric({ fields: ['speed'], reduce: 'avg' }, handlerContext);
        await registry._get_telemetry_for_scope({ scope: 'lap' }, handlerContext);
        await registry.get_event_log({}, handlerContext);
        await registry.get_next_corner({}, handlerContext);
        await registry.get_live_focus_section({}, handlerContext);
        await registry.get_live_section_history({ limit: 5 }, handlerContext);
        await registry._get_live_section_telemetry({ section_id: 'turn-1' }, handlerContext);
        await registry._record_live_section_classification({ section_id: 'turn-1' }, handlerContext);

        expect(h.live.queryTelemetryMetric).toHaveBeenCalled();
        expect(h.live.getTelemetryForScope).toHaveBeenCalled();
        expect(h.live.getEventLog).toHaveBeenCalled();
        expect(h.live.getNextCorner).toHaveBeenCalled();
        expect(h.live.getLiveSectionHistory).toHaveBeenCalledWith(5);
        expect(h.live.getLiveSectionTelemetry).toHaveBeenCalled();
        expect(h.live.recordLiveSectionClassification).toHaveBeenCalled();
    });

    it('routes recorded APIs and summary lookups through their concrete screen owners', async () => {
        const h = createHarness('recorded');
        const registry = createAiCommandRegistry(h.context);
        await registry.get_session_analysis({}, handlerContext);
        await registry.run_recorded_ai_analysis({}, handlerContext);
        await registry.get_recorded_session_analysis({}, handlerContext);
        await registry.get_recorded_session_context({}, handlerContext);
        await registry.get_performance_insights({}, handlerContext);
        await registry.compare_lap_times({ session_ids: ['a', 'b'] }, handlerContext);
        await registry.follow_expert_line({}, handlerContext);
        await registry.get_telemetry_data({}, handlerContext);
        await registry.get_user_summary_map_level({}, handlerContext);
        await registry.get_available_user_summary_maps({}, handlerContext);
        await registry.search_user_summary_map_level({ query: 'Monza' }, handlerContext);

        expect(h.recorded.requestSessionAnalysis).toHaveBeenCalled();
        expect(h.recorded.runRecordedAiAnalysis).toHaveBeenCalled();
        expect(h.recorded.requestPerformanceInsights).toHaveBeenCalled();
        expect(h.recorded.requestLapComparison).toHaveBeenCalled();
        expect(h.recorded.requestExpertLineGuidance).toHaveBeenCalled();
        expect(h.recorded.requestTelemetryData).toHaveBeenCalled();
        expect(h.summary.getUserSummaryMapLevel).toHaveBeenCalled();
        expect(h.summary.getAvailableUserSummaryMaps).toHaveBeenCalled();
        expect(h.summary.searchUserSummaryMapLevel).toHaveBeenCalled();
    });

    it('converts recorded API and analysis failures to stable exceptions', async () => {
        const h = createHarness('recorded');
        const registry = createAiCommandRegistry(h.context);
        const requestFailure = new SessionAnalysisFailedError(
            AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
            'Session API unavailable.',
        );
        const analysisFailure = new RecordedAnalysisFailedError(
            AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
            'Classifier unavailable.',
        );
        h.recorded.requestSessionAnalysis.mockRejectedValueOnce(requestFailure);
        h.recorded.runRecordedAiAnalysis.mockRejectedValueOnce(analysisFailure);

        await expect(registry.get_session_analysis({}, handlerContext)).rejects.toBe(requestFailure);
        await expect(registry.run_recorded_ai_analysis({}, handlerContext)).rejects.toBe(analysisFailure);
    });

    it('passes existing tool errors and unexpected errors through the component boundary unchanged', async () => {
        const h = createHarness('recorded');
        const registry = createAiCommandRegistry(h.context);
        const toolFailure = new ToolExecutionError('Tool validation failed.');
        const unexpectedFailure = new Error('Unexpected component bug.');
        h.summary.getAvailableUserSummaryMaps.mockImplementationOnce(() => { throw toolFailure; });
        h.summary.getUserSummaryMapLevel.mockImplementationOnce(() => { throw unexpectedFailure; });

        await expect(registry.get_available_user_summary_maps({}, handlerContext)).rejects.toBe(toolFailure);
        await expect(registry.get_user_summary_map_level({}, handlerContext)).rejects.toBe(unexpectedFailure);
    });

    it('creates a todo runner, reuses it for update/read, and deletes it on clear', async () => {
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);
        await registry.set_live_range_todo_list({
            events: [{
                id: 'brake',
                normalized_position: 0.5,
                content: { title: 'Brake' },
                data: {},
            }],
        }, handlerContext);
        expect(h.directory.findBaseComponentRef()?.current).toBeInstanceOf(LiveRangeTodoListRunner);
        await expect(registry.set_live_range_todo_list({ events: [] }, handlerContext))
            .rejects.toBeInstanceOf(DuplicateAiToolComponentError);
        expect(uiOutput(await registry.update_live_range_todo_list({
            action: 'update_events',
            events: [{ id: 'brake', content: { title: 'Brake now' } }],
        }, handlerContext))).toMatchObject({ status: 'ready' });
        expect(uiOutput(await registry.get_live_range_todo_list({}, handlerContext)))
            .toMatchObject({ status: 'ready' });
        await registry.update_live_range_todo_list({ action: 'clear' }, handlerContext);
        expect(h.directory.findBaseComponentRef()).toBeNull();

        const missingDirectory = createAiToolComponentRefDirectory();
        const missing = createAiCommandRegistry({ ...h.context, componentRefs: missingDirectory });
        await expect(missing.get_live_range_todo_list({}, handlerContext))
            .rejects.toBeInstanceOf(ComponentRefUnavailableError);
    });

    it('removes a newly registered todo runner when initialization fails', async () => {
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);

        await expect(registry.set_live_range_todo_list({}, handlerContext))
            .rejects.toBeInstanceOf(InvalidLiveRangeTodoListError);
        expect(h.directory.findBaseComponentRef()).toBeNull();
    });

    it('creates, awaits, updates, controls, disables, and closes exact named visualization children', async () => {
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);
        await registry.get_visualization_capabilities({}, handlerContext);
        const opened = uiOutput(await registry.open_visualization_chart({ type: 'telemetry-overview', data: { metrics: ['speed'] } }, handlerContext));
        expect(opened).toMatchObject({ success: true, componentName: 'telemetry:speed' });
        expect(h.childHandles.get('telemetry:speed').updateLiveTelemetry).toHaveBeenCalled();
        await registry.disable_ui_component({ component_name: 'telemetry:speed' }, handlerContext);
        expect(h.childHandles.get('telemetry:speed').disableLiveTelemetry).toHaveBeenCalled();
        await registry.close_visualization_chart({ component_name: 'telemetry:speed' }, handlerContext);

        await registry.add_imitation_guidance_chart({}, handlerContext);
        await registry.update_guidance_once({}, handlerContext);
        expect(h.childHandles.get('visualization:imitation-guidance-chart').refreshGuidanceOnce).toHaveBeenCalled();
        await registry.remove_imitation_guidance_chart({}, handlerContext);
        expect(h.manager.getVisualizationCapabilities).toHaveBeenCalled();
        expect(h.manager.closeVisualization).toHaveBeenCalled();
    });

    it('reuses a registered static map before asking the visualization manager to create one', async () => {
        const h = createHarness('recorded');
        const name = 'visualization:map-visualization';
        const updateMap = jest.fn(() => true);
        reserve(h.directory, name, { updateMap, disableMap: jest.fn(() => true) });

        const result = uiOutput(await createAiCommandRegistry(h.context).open_visualization_chart({
            type: 'map-visualization',
            data: { selected: true },
            config: { camera: 'fit' },
        }, handlerContext));

        expect(result).toMatchObject({
            success: true,
            componentName: name,
            chartType: 'map-visualization',
            reused: true,
        });
        expect(updateMap).toHaveBeenCalledWith({ selected: true }, { camera: 'fit' });
        expect(h.manager.requestVisualization).not.toHaveBeenCalled();
    });

    it('throws semantic candidates when a telemetry target is ambiguous', async () => {
        const h = createHarness('live');
        h.instances.push(
            { id: 'speed', name: 'telemetry:speed', type: 'telemetry-overview' },
            { id: 'brake', name: 'telemetry:brake', type: 'telemetry-overview' },
        );
        const result = createAiCommandRegistry(h.context)
            .close_visualization_chart({}, handlerContext);
        await expect(result).rejects.toBeInstanceOf(AmbiguousComponentTargetError);
    });

    it('keeps live composite raw rows API-only and records compact classification results', async () => {
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);
        const telemetryEnvelope = await registry.analyze_telemetry({ scope: 'lap' }, handlerContext);
        const classifyEnvelope = await registry.classify_live_section({ section_id: 'turn-1' }, handlerContext);

        expect(mockPost).toHaveBeenCalledWith(
            '/racing-session/analyze-live-recorded-analysis',
            expect.objectContaining({ records: h.rows }),
            { timeout: 120000 },
        );
        expect(h.live.recordLiveSectionClassification).toHaveBeenCalledWith(expect.objectContaining({ telemetry_stats: { row_count: 2, field_count: 2 } }));
        expect(JSON.stringify(telemetryEnvelope.output)).not.toContain('Physics_speed_kmh');
        expect(JSON.stringify(classifyEnvelope.output)).not.toContain('Physics_speed_kmh');
        expect(telemetryEnvelope.output).toMatchObject({ telemetry_stats: { row_count: 2, field_count: 2 } });
        expect(classifyEnvelope.output).toMatchObject({ telemetry_stats: { row_count: 2, field_count: 2 } });
    });

    it('throws stable live composite API failures', async () => {
        mockPost.mockRejectedValue({ data: { message: 'classifier unavailable' } });
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);
        const telemetry = registry.analyze_telemetry({ scope: 'lap' }, handlerContext);
        const section = registry.classify_live_section({ section_id: 'turn-1' }, handlerContext);

        await expect(telemetry).rejects.toBeInstanceOf(TelemetryAnalysisFailedError);
        await expect(telemetry).rejects.toMatchObject({ message: 'classifier unavailable' });
        await expect(section).rejects.toBeInstanceOf(LiveSectionClassificationFailedError);
        await expect(section).rejects.toMatchObject({ message: 'classifier unavailable' });
    });

    it('routes recorded telemetry analysis through manager and delegates live baseline analysis', async () => {
        const recorded = createHarness('recorded');
        const recordedResult = await createAiCommandRegistry(recorded.context).analyze_telemetry({}, handlerContext);
        expect(recorded.recorded.runRecordedAiAnalysis).toHaveBeenCalled();
        expect(recorded.manager.requestVisualization).toHaveBeenCalledWith(expect.objectContaining({ name: 'visualization:analysis-results' }));
        expect(recordedResult.output).toMatchObject({ component_name: 'visualization:analysis-results' });

        const live = createHarness('live');
        await createAiCommandRegistry(live.context).analyze_live_recorded_analysis({}, handlerContext);
        expect(live.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION).requestAnalysis)
            .toHaveBeenCalledWith({});
        expect(mockPost).not.toHaveBeenCalled();
    });

    it('uses each segment expert reference array for recorded analysis results', async () => {
            const h = createHarness('recorded');
            const records = [
                {
                    raw_index: 400,
                    Graphics_current_time: 1_000,
                    Graphics_normalized_car_position: 0.1,
                },
                {
                    Graphics_current_time: 1_100,
                    Graphics_normalized_car_position: 0.2,
                },
            ];
            const result = {
                ...analysisResult,
                parent_segment_count: 2,
                segments: [{
                    id: 'segment-1',
                    labels: ['MSP'],
                    track_section: 'Turn 1',
                    start_index: 0,
                    end_index: 0,
                    expert_reference_data: [{
                        raw_index: -30,
                        expert_optimal_time: 900,
                        Graphics_normalized_car_position: 0.1,
                        expert_optimal_throttle: 0.6,
                    }],
                }, {
                    id: 'segment-2',
                    labels: ['EA'],
                    track_section: 'Turn 2',
                    start_index: 1,
                    end_index: 1,
                    expert_reference_data: [{
                        expert_optimal_time: 990,
                        Graphics_normalized_car_position: 0.2,
                        expert_optimal_throttle: 0.7,
                    }],
                }],
                expert_reference_data: [{
                    raw_index: 0,
                    expert_optimal_time: 1,
                    Graphics_normalized_car_position: 0.9,
                }],
            };

            h.recorded.getSelectedSession.mockReturnValue({
                SessionId: 'session-1',
                session_name: 'Race',
                map: 'Monza',
                car: 'BMW',
                data: records,
            } as any);
            h.recorded.runRecordedAiAnalysis.mockResolvedValue({
                sessionId: 'session-1',
                status: 'ready',
                result: result as any,
            });
            await createAiCommandRegistry(h.context).analyze_telemetry({}, handlerContext);

            expect(h.childHandles.get('visualization:analysis-results').replaceAnalysisResults)
                .toHaveBeenCalledWith({
                    elements: [expect.objectContaining({
                        comparison: {
                            samples: [expect.objectContaining({
                                driverTimeMs: 1_000,
                                expertTimeMs: 900,
                                driverTrackPosition: 0.1,
                                expertTrackPosition: 0.1,
                            })],
                        },
                    }), expect.objectContaining({
                        comparison: {
                            samples: [expect.objectContaining({
                                driverTimeMs: 1_100,
                                expertTimeMs: 990,
                                driverTrackPosition: 0.2,
                                expertTrackPosition: 0.2,
                            })],
                        },
                    })],
                });
    });

    it('reports missing, stale-name, and five-second child mount failures with stable codes', async () => {
        const noDirectory = createHarness('live');
        const missing = createAiCommandRegistry({ ...noDirectory.context, componentRefs: undefined })
            .get_event_log({}, handlerContext);
        await expect(missing).rejects.toBeInstanceOf(ComponentRefUnavailableError);

        const stale = createHarness('live');
        const liveRef = stale.directory.findComponentRef<any>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!;
        liveRef.current.getComponentName = () => 'wrong-live-session';
        await expect(createAiCommandRegistry(stale.context).get_event_log({}, handlerContext))
            .rejects.toBeInstanceOf(ComponentNameMismatchError);

        jest.useFakeTimers();
        const timeoutHarness = createHarness('live');
        timeoutHarness.manager.requestVisualization.mockImplementation((options: any) => ({
            success: true,
            message: 'requested',
            componentName: options.name,
            chartId: 'pending-chart',
            chartType: options.type,
            reused: false,
        }));
        const pending = createAiCommandRegistry(timeoutHarness.context).open_visualization_chart({ type: 'event-log' }, handlerContext);
        jest.advanceTimersByTime(5000);
        await expect(pending).rejects.toBeInstanceOf(ComponentMountTimeoutError);
        jest.useRealTimers();
    });
});
