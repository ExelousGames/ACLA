import apiService from 'services/api.service';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefDirectory,
    createAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import { RecordingState } from 'views/lap-analysis/recording-state';
import { getToolEnvelopeUiOutput, ToolOutputEnvelope } from '../ai-tool-base';
import {
    AiCommandRegistryContext,
    createAiCommandRegistry,
    frontendToolDefinitions,
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
    reserve(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, todo);
    mountChild(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION, 'baseline-collection');
    reserve(directory, mode === 'live' ? AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER : AI_TOOL_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER, manager);

    const context: AiCommandRegistryContext = {
        componentRefs: directory, sessionId: 'session-1', sessionMode: mode,
        opportunityAgentState: { intervalId: null, inFlight: false, lastAlertKey: null, lastAlertAt: 0 },
        startTrackGuide: jest.fn(), setTrackGuideEnabled: jest.fn(), getOpportunityTelemetryRows: jest.fn(() => rows),
    };
    return { directory, context, chat, live, recorded, summary, todo, manager, instances, childOwners, childHandles, rows, classification };
};

describe('named component-ref AI command registry', () => {
    beforeEach(() => {
        mockPost.mockReset().mockResolvedValue({ data: analysisResult });
        handlerContext.sendToolStatus.mockReset();
    });

    it('publishes the unchanged static frontend map with all 41 handlers', () => {
        const names = frontendToolDefinitions.map((definition) => definition.name);
        expect(names).toHaveLength(41);
        expect(new Set(names).size).toBe(41);
        expect(names).toEqual(expect.arrayContaining(['analyze_telemetry', 'classify_live_section']));
    });

    it('opens and drives the baseline visualization while keeping assistant-owned operations on AiChat', async () => {
        const h = createHarness('live');
        const awaitComponentRef = jest.spyOn(h.directory, 'awaitComponentRef');
        const registry = createAiCommandRegistry(h.context);
        await registry.start_agent_session({ agent_mode: 'track_guide' }, handlerContext);
        await registry.stop_agent_session({ agent_session_id: 'agent-1' }, handlerContext);
        await registry.collect_live_baseline({}, handlerContext);
        await registry.restart_live_baseline({}, handlerContext);
        await registry.set_procedure_plan({ goal: 'Improve', requests: [{ type: 'tool', title: 'Brake later' }] }, handlerContext);
        await registry.advance_plan_step({ reason: 'done' }, handlerContext);
        await registry.clear_procedure_plan({}, handlerContext);
        await registry.show_map({ map_id: 'monza' }, handlerContext);

        expect(h.chat.startAgentSession).toHaveBeenCalled();
        expect(h.chat.stopAgentSession).toHaveBeenCalled();
        expect(h.manager.requestVisualization).toHaveBeenCalledWith({
            name: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            type: 'baseline-collection',
        });
        expect(h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION).startCollection).toHaveBeenCalled();
        expect(h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION).restartCollection).toHaveBeenCalled();
        expect(awaitComponentRef).toHaveBeenCalledWith(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION, 5000);
        expect(h.chat.setProcedurePlan).toHaveBeenCalled();
        expect(h.chat.advanceProcedurePlanStep).toHaveBeenCalledWith('done');
        expect(h.chat.clearProcedurePlan).toHaveBeenCalled();
        expect(h.chat.displayMap).toHaveBeenCalledWith(expect.objectContaining({ status: 'ready' }));
    });

    it('requires the mounted baseline visualization for baseline-dependent commands', async () => {
        const h = createHarness('live');
        h.directory.releaseComponentRef(
            AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
            h.childOwners.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION)!,
        );
        const registry = createAiCommandRegistry(h.context);

        for (const result of await Promise.all([
            registry.get_live_focus_section({}, handlerContext),
            registry.analyze_live_recorded_analysis({}, handlerContext),
            registry._get_live_section_telemetry({ section_id: 'turn-1' }, handlerContext),
            registry._record_live_section_classification({ section_id: 'turn-1' }, handlerContext),
            registry.classify_live_section({ section_id: 'turn-1' }, handlerContext),
        ])) {
            expect(uiOutput(result)).toMatchObject({
                error: 'baseline_collection_visualization_required',
                component_name: AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION,
                message: expect.stringContaining('Baseline collection visualization required'),
            });
        }
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

    it('requires the mounted todo panel and routes set/update/read to its exact ref', async () => {
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);
        await registry.set_live_range_todo_list({ events: [] }, handlerContext);
        await registry.update_live_range_todo_list({ action: 'update_events', events: [] }, handlerContext);
        await registry.get_live_range_todo_list({}, handlerContext);
        expect(h.todo.replaceEvents).toHaveBeenCalled();
        expect(h.todo.updateEvents).toHaveBeenCalled();
        expect(h.todo.get).toHaveBeenCalled();

        h.directory.releaseComponentRef(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, Symbol('wrong-owner'));
        const missingDirectory = createAiToolComponentRefDirectory();
        const missing = createAiCommandRegistry({ ...h.context, componentRefs: missingDirectory });
        expect(uiOutput(await missing.get_live_range_todo_list({}, handlerContext))).toMatchObject({ error: 'component_ref_unavailable' });
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

    it('returns semantic candidates when a telemetry target is ambiguous', async () => {
        const h = createHarness('live');
        h.instances.push(
            { id: 'speed', name: 'telemetry:speed', type: 'telemetry-overview' },
            { id: 'brake', name: 'telemetry:brake', type: 'telemetry-overview' },
        );
        const result = uiOutput(await createAiCommandRegistry(h.context).close_visualization_chart({}, handlerContext));
        expect(result).toMatchObject({
            error: 'ambiguous_component_target',
            semantic_candidates: ['telemetry:brake', 'telemetry:speed'],
        });
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

    it('normalizes live composite API failures without exposing raw telemetry', async () => {
        mockPost.mockRejectedValue({ data: { message: 'classifier unavailable' } });
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);
        const telemetry = await registry.analyze_telemetry({ scope: 'lap' }, handlerContext);
        const section = await registry.classify_live_section({ section_id: 'turn-1' }, handlerContext);

        expect(uiOutput(telemetry)).toMatchObject({ error: 'telemetry_analysis_failed', message: 'classifier unavailable' });
        expect(uiOutput(section)).toMatchObject({ error: 'live_section_classification_failed', message: 'classifier unavailable' });
        expect(JSON.stringify(telemetry.output)).not.toContain('Physics_speed_kmh');
        expect(JSON.stringify(section.output)).not.toContain('Physics_speed_kmh');
    });

    it('routes recorded telemetry analysis and live baseline analysis through manager and mounted chart refs', async () => {
        const recorded = createHarness('recorded');
        const recordedResult = await createAiCommandRegistry(recorded.context).analyze_telemetry({}, handlerContext);
        expect(recorded.recorded.runRecordedAiAnalysis).toHaveBeenCalled();
        expect(recorded.manager.requestVisualization).toHaveBeenCalledWith(expect.objectContaining({ name: 'visualization:analysis-results' }));
        expect(recordedResult.output).toMatchObject({ component_name: 'visualization:analysis-results' });

        const live = createHarness('live');
        await createAiCommandRegistry(live.context).analyze_live_recorded_analysis({}, handlerContext);
        expect(mockPost).toHaveBeenCalledWith(
            '/racing-session/analyze-live-recorded-analysis',
            expect.objectContaining({ records: live.rows }),
            { timeout: 120000 },
        );
        expect(live.childHandles.get('visualization:analysis-results').replaceAnalysisResults).toHaveBeenCalled();
    });

    it('refreshes a registered Analysis Results chart without requesting another visualization', async () => {
        const h = createHarness('live');
        const name = 'visualization:analysis-results';
        const replaceAnalysisResults = jest.fn(() => true);
        h.instances.push({ id: 'existing-chart', name, type: 'analysis-results' });
        reserve(h.directory, name, { replaceAnalysisResults });

        const result = await createAiCommandRegistry(h.context)
            .analyze_live_recorded_analysis({}, handlerContext);

        expect(replaceAnalysisResults).toHaveBeenCalledWith({
            elements: [expect.objectContaining({ id: 'segment-1' })],
        });
        expect(h.manager.requestVisualization).not.toHaveBeenCalled();
        expect(uiOutput(result)).toMatchObject({
            chartId: 'existing-chart',
            component_name: name,
        });
    });

    it('reuses one Analysis Results panel and replaces its content across baseline analyses', async () => {
        const h = createHarness('live');
        const registry = createAiCommandRegistry(h.context);
        const secondResult = {
            ...analysisResult,
            segments: [{
                ...analysisResult.segments[0],
                id: 'segment-2',
                start_index: 1,
                end_index: 1,
            }],
        };

        const first = await registry.analyze_live_recorded_analysis({}, handlerContext);
        mockPost.mockResolvedValueOnce({ data: secondResult });
        const second = await registry.analyze_live_recorded_analysis({}, handlerContext);

        const chart = h.childHandles.get('visualization:analysis-results');
        expect(h.manager.requestVisualization).toHaveBeenCalledTimes(1);
        expect(h.instances).toHaveLength(1);
        expect(chart.replaceAnalysisResults).toHaveBeenCalledTimes(2);
        expect(chart.replaceAnalysisResults).toHaveBeenLastCalledWith({
            elements: [expect.objectContaining({ id: 'segment-2' })],
        });
        expect(uiOutput(first)).toMatchObject({
            chartId: 'chart-1',
            component_name: 'visualization:analysis-results',
        });
        expect(uiOutput(second)).toMatchObject({
            chartId: 'chart-1',
            component_name: 'visualization:analysis-results',
        });
    });

    it('re-resolves the latest Analysis Results handle after mount registration replay', async () => {
        const h = createHarness('live');
        const name = 'visualization:analysis-results';
        const staleRef = { current: null };
        const latestReplace = jest.fn(() => true);
        jest.spyOn(h.directory, 'awaitComponentRef').mockResolvedValue(staleRef as any);
        h.manager.requestVisualization.mockImplementation((options: any) => {
            h.instances.push({ id: 'strict-chart', ...options });
            reserve(h.directory, name, { replaceAnalysisResults: latestReplace });
            return {
                success: true,
                message: 'Opened chart.',
                componentName: name,
                chartId: 'strict-chart',
                chartType: options.type,
                reused: false,
            };
        });

        const result = await createAiCommandRegistry(h.context)
            .analyze_live_recorded_analysis({}, handlerContext);

        expect(latestReplace).toHaveBeenCalledWith({
            elements: [expect.objectContaining({ id: 'segment-1' })],
        });
        expect(uiOutput(result)).toMatchObject({
            chartId: 'strict-chart',
            component_name: name,
        });
    });

    it('keeps baseline API failures classified while exposing genuine chart mount timeouts', async () => {
        mockPost.mockRejectedValueOnce({ data: { message: 'classifier unavailable' } });
        const apiFailureHarness = createHarness('live');
        const apiFailure = await createAiCommandRegistry(apiFailureHarness.context)
            .analyze_live_recorded_analysis({}, handlerContext);
        expect(uiOutput(apiFailure)).toMatchObject({
            error: 'recorded_analysis_failed',
            message: 'classifier unavailable',
        });

        jest.useFakeTimers();
        const timeoutHarness = createHarness('live');
        timeoutHarness.manager.requestVisualization.mockImplementation((options: any) => ({
            success: true,
            message: 'requested',
            componentName: options.name,
            chartId: 'pending-analysis-chart',
            chartType: options.type,
            reused: false,
        }));
        const pending = createAiCommandRegistry(timeoutHarness.context)
            .analyze_live_recorded_analysis({}, handlerContext);
        for (let index = 0; index < 10
            && timeoutHarness.manager.requestVisualization.mock.calls.length === 0; index += 1) {
            await Promise.resolve();
        }
        expect(timeoutHarness.manager.requestVisualization).toHaveBeenCalledTimes(1);
        jest.advanceTimersByTime(5000);
        expect(uiOutput(await pending)).toMatchObject({
            error: 'component_mount_timeout',
            component_name: 'visualization:analysis-results',
        });
        jest.useRealTimers();
    });

    it.each(['recorded', 'live'] as const)(
        'uses each segment expert reference array for %s analysis results',
        async (mode) => {
            const h = createHarness(mode);
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

            if (mode === 'recorded') {
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
            } else {
                h.childHandles.get(AI_TOOL_COMPONENT_NAMES.BASELINE_COLLECTION).getLapRecord.mockReturnValue({
                    id: 'baseline-1',
                    lap: 2,
                    track: 'Monza',
                    car: 'BMW',
                    sample_count: 2,
                    captured_at: 'now',
                    records,
                } as any);
                mockPost.mockResolvedValue({ data: result });
                await createAiCommandRegistry(h.context).analyze_live_recorded_analysis({}, handlerContext);
            }

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
        },
    );

    it('reports missing, stale-name, and five-second child mount failures with stable codes', async () => {
        const noDirectory = createHarness('live');
        const missing = await createAiCommandRegistry({ ...noDirectory.context, componentRefs: undefined })
            .get_event_log({}, handlerContext);
        expect(uiOutput(missing)).toMatchObject({ error: 'component_ref_unavailable' });

        const stale = createHarness('live');
        const liveRef = stale.directory.findComponentRef<any>(AI_TOOL_COMPONENT_NAMES.LIVE_SESSION)!;
        liveRef.current.getComponentName = () => 'wrong-live-session';
        expect(uiOutput(await createAiCommandRegistry(stale.context).get_event_log({}, handlerContext)))
            .toMatchObject({ error: 'component_name_mismatch' });

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
        expect(uiOutput(await pending)).toMatchObject({ error: 'component_mount_timeout' });
        jest.useRealTimers();
    });
});
