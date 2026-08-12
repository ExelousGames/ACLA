import fs from 'fs';
import path from 'path';
import {
    AI_TOOL_COMPONENT_NAMES,
    ComponentRefUnavailableError,
    createAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import {
    FRONTEND_AI_TOOL_NAMES,
    createAiCommandRegistry,
    frontendAiToolRegistry,
} from '../ai-command-registry';

const backendCatalogNames = [
    'start_agent_session',
    'stop_agent_session',
    'set_live_range_todo_list',
    'update_live_range_todo_list',
    'get_live_range_todo_list',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'get_live_analysis_mistake_count',
    'create_goal',
    'retry_goal_task',
    'advance_plan_step',
    'clear_procedure_plan',
    'set_procedure_plan',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
    'show_map',
    'run_recorded_ai_analysis',
    'get_recorded_session_analysis',
    'get_recorded_session_context',
    'analyze_telemetry',
    'classify_live_section',
];

const handlerContext = {
    toolRunId: 'run-1',
    toolName: 'test',
    sendToolStatus: jest.fn(),
};

const register = (
    directory: ReturnType<typeof createAiToolComponentRefDirectory>,
    name: string,
    methods: Record<string, jest.Mock>,
) => directory.reserveComponentRef(name, Symbol(name), {
    getComponentName: () => name,
    ...methods,
} as any);

describe('direct component AI command registry', () => {
    it('is the single immutable list of exactly the 26 backend-cataloged tools', () => {
        const backendSource = fs.readFileSync(path.resolve(
            process.cwd(),
            '../acla_backend/src/shared/ai/frontend-application-tool-registry.ts',
        ), 'utf8');
        const catalogSource = backendSource
            .split('export const FRONTEND_APPLICATION_TOOLS = [')[1]
            .split('] as const;')[0];
        const actualBackendNames = Array.from(
            catalogSource.matchAll(/\bname:\s*'([^']+)'/g),
            (match) => match[1],
        );

        expect(actualBackendNames).toEqual(backendCatalogNames);
        expect(FRONTEND_AI_TOOL_NAMES).toEqual(backendCatalogNames);
        expect(frontendAiToolRegistry.map(({ name }) => name)).toEqual(backendCatalogNames);
        expect(new Set(FRONTEND_AI_TOOL_NAMES).size).toBe(26);
        expect(Object.isFrozen(frontendAiToolRegistry)).toBe(true);
    });

    it('routes each ownership group to one component function and returns its value unchanged', async () => {
        const directory = createAiToolComponentRefDirectory();
        const result = { status: 'sentinel', value: 7 };
        const chat = {
            startAgentSession: jest.fn(() => result),
            stopAgentSession: jest.fn(() => result),
            createLiveRangeTodoList: jest.fn(async () => result),
            createGoal: jest.fn(async () => result),
            createProcedurePlan: jest.fn(async () => result),
            showMap: jest.fn(async () => result),
        };
        const live = {
            collectLiveBaselineForAi: jest.fn(async () => result),
            restartLiveBaselineForAi: jest.fn(async () => result),
            analyzeLiveRecordedAnalysisForAi: jest.fn(async () => result),
            getLiveAnalysisMistakeCountForAi: jest.fn(() => result),
            getNextCornerForAi: jest.fn(() => result),
            queryTelemetryMetricForAi: jest.fn(() => result),
            getEventLogForAi: jest.fn(() => result),
            analyzeTelemetryForAi: jest.fn(async () => result),
            classifyLiveSectionForAi: jest.fn(async () => result),
        };
        const recorded = {
            runRecordedAnalysisForAi: jest.fn(async () => result),
            getRecordedAnalysisForAi: jest.fn(() => result),
            getRecordedSessionContextForAi: jest.fn(() => result),
            analyzeTelemetryForAi: jest.fn(async () => result),
        };
        const summary = {
            getUserSummaryMapLevel: jest.fn(() => result),
            getAvailableUserSummaryMaps: jest.fn(() => result),
            searchUserSummaryMapLevel: jest.fn(() => result),
        };
        const goal = { retryFailedTask: jest.fn(async () => result) };
        const plan = {
            advancePlanStep: jest.fn(async () => result),
            clearProcedurePlan: jest.fn(() => result),
        };
        const todo = {
            updateForAi: jest.fn(() => result),
            getForAi: jest.fn(() => result),
        };
        register(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, chat);
        register(directory, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION, live);
        register(directory, AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS, recorded);
        register(directory, AI_TOOL_COMPONENT_NAMES.USER_SUMMARY, summary);
        register(directory, AI_TOOL_COMPONENT_NAMES.GOAL, goal);
        register(directory, AI_TOOL_COMPONENT_NAMES.PROCEDURE_PLAN, plan);
        register(directory, AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST, todo);

        const liveRegistry = createAiCommandRegistry({
            componentRefs: directory,
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        });
        await expect(liveRegistry.start_agent_session({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.stop_agent_session({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.create_goal({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.set_procedure_plan({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.set_live_range_todo_list({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.retry_goal_task({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.advance_plan_step({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.clear_procedure_plan({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.update_live_range_todo_list({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.get_live_range_todo_list({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.collect_live_baseline({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.restart_live_baseline({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.analyze_live_recorded_analysis({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.get_live_analysis_mistake_count({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.get_next_corner({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.query_telemetry_metric({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.get_event_log({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.get_user_summary_map_level({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.get_available_user_summary_maps({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.search_user_summary_map_level({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.show_map({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.analyze_telemetry({}, handlerContext)).resolves.toBe(result);
        await expect(liveRegistry.classify_live_section({}, handlerContext)).resolves.toBe(result);

        expect(chat.startAgentSession).toHaveBeenCalledTimes(1);
        expect(chat.stopAgentSession).toHaveBeenCalledTimes(1);
        expect(chat.createGoal).toHaveBeenCalledTimes(1);
        expect(chat.createProcedurePlan).toHaveBeenCalledTimes(1);
        expect(chat.createLiveRangeTodoList).toHaveBeenCalledTimes(1);
        expect(goal.retryFailedTask).toHaveBeenCalledTimes(1);
        expect(plan.advancePlanStep).toHaveBeenCalledTimes(1);
        expect(plan.clearProcedurePlan).toHaveBeenCalledTimes(1);
        expect(todo.updateForAi).toHaveBeenCalledTimes(1);
        expect(todo.getForAi).toHaveBeenCalledTimes(1);
        expect(live.collectLiveBaselineForAi).toHaveBeenCalledTimes(1);
        expect(live.restartLiveBaselineForAi).toHaveBeenCalledTimes(1);
        expect(live.analyzeLiveRecordedAnalysisForAi).toHaveBeenCalledTimes(1);
        expect(live.getLiveAnalysisMistakeCountForAi).toHaveBeenCalledTimes(1);
        expect(live.getNextCornerForAi).toHaveBeenCalledTimes(1);
        expect(live.queryTelemetryMetricForAi).toHaveBeenCalledTimes(1);
        expect(live.getEventLogForAi).toHaveBeenCalledTimes(1);
        expect(live.analyzeTelemetryForAi).toHaveBeenCalledTimes(1);
        expect(live.classifyLiveSectionForAi).toHaveBeenCalledTimes(1);
        expect(summary.getUserSummaryMapLevel).toHaveBeenCalledTimes(1);
        expect(summary.getAvailableUserSummaryMaps).toHaveBeenCalledTimes(1);
        expect(summary.searchUserSummaryMapLevel).toHaveBeenCalledTimes(1);
        expect(chat.showMap).toHaveBeenCalledTimes(1);

        const recordedRegistry = createAiCommandRegistry({
            componentRefs: directory,
            sessionMode: 'recorded',
            conversationRole: 'main',
        });
        await expect(recordedRegistry.run_recorded_ai_analysis({}, handlerContext)).resolves.toBe(result);
        await expect(recordedRegistry.get_recorded_session_analysis({}, handlerContext)).resolves.toBe(result);
        await expect(recordedRegistry.get_recorded_session_context({}, handlerContext)).resolves.toBe(result);
        await expect(recordedRegistry.analyze_telemetry({}, handlerContext)).resolves.toBe(result);
        expect(recorded.runRecordedAnalysisForAi).toHaveBeenCalledTimes(1);
        expect(recorded.getRecordedAnalysisForAi).toHaveBeenCalledTimes(1);
        expect(recorded.getRecordedSessionContextForAi).toHaveBeenCalledTimes(1);
        expect(recorded.analyzeTelemetryForAi).toHaveBeenCalledTimes(1);
        expect(live.analyzeTelemetryForAi).toHaveBeenCalledTimes(1);
    });

    it('keeps typed missing-component failures', async () => {
        const registry = createAiCommandRegistry({
            componentRefs: createAiToolComponentRefDirectory(),
            sessionMode: 'live',
        });

        await expect(registry.get_next_corner({}, handlerContext))
            .rejects.toBeInstanceOf(ComponentRefUnavailableError);
    });

    it('reuses central dispatch for nested goal steps and enforces recursion restrictions', async () => {
        const directory = createAiToolComponentRefDirectory();
        const liveResult = { status: 'ready', corner: { name: 'T1' } };
        const live = { getNextCornerForAi: jest.fn(() => liveResult) };
        const chat = {
            createGoal: jest.fn(async (_args, dispatch) => {
                const nested = await dispatch('get_next_corner', {});
                await expect(dispatch('create_goal', {})).rejects.toMatchObject({
                    name: 'ToolNotRegisteredError',
                });
                return nested;
            }),
        };
        register(directory, AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT, chat);
        register(directory, AI_TOOL_COMPONENT_NAMES.LIVE_SESSION, live);
        const registry = createAiCommandRegistry({
            componentRefs: directory,
            sessionMode: 'live',
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        });

        await expect(registry.create_goal({}, handlerContext)).resolves.toBe(liveResult);
        expect(live.getNextCornerForAi).toHaveBeenCalledTimes(1);
    });
});
