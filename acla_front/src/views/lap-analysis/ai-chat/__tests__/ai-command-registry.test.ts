jest.mock('views/lap-analysis/visualization/VisualizationRegistry', () => ({
    visualizationController: {
        getVisualizationAssistantContext: jest.fn(),
        openVisualization: jest.fn(),
        closeVisualization: jest.fn(),
        invokeVisualizationControl: jest.fn(),
        getCurrentInstances: jest.fn(() => []),
    },
}));

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: {
        post: jest.fn(),
    },
}));

import {
    AiCommandRegistryContext,
    createAiCommandRegistry,
    frontendToolSchemas,
    getFrontendToolSchemasForSessionMode,
    startAgentRuntime,
} from '../ai-command-registry';
import { RecordedAiAnalysisState } from 'views/lap-analysis/recorded-session-analysis';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import { buildBaselineCollectionTag } from '../BaselineCollectionTracker';

const labelNames: Record<string, string> = {
    brands_hatch: 'Brands Hatch',
    brands_hatch1: 'Paddock Hill Bend',
    brands_hatch2: 'Druids',
    monza: 'Monza',
    monza1: 'Rettifilo',
};

const categories: Record<string, string[]> = {
    brands_hatch: ['brands_hatch1', 'brands_hatch2'],
    monza: ['monza1'],
};

const createRegistry = () => createAiCommandRegistry({
    sessionMode: 'live',
    opportunityAgentState: {
        intervalId: null,
        inFlight: false,
        lastAlertKey: null,
        lastAlertAt: 0,
    },
    startTrackGuide: jest.fn(),
    setTrackGuideEnabled: jest.fn(),
    getOpportunityTelemetryRows: () => [],
    getLabelName: (labelId) => labelNames[labelId],
    getCategoryLabels: (category) => categories[category] ?? [],
    userSummary: {
        sessionAnalysis: {
            practice: {
                tracks: {
                    brands_hatch: {
                        trackName: 'Brands Hatch GP',
                        analyzedSessionCount: 3,
                        sections: {
                            brands_hatch1: {
                                analyzedTimeCount: 10,
                                mistakeCount: 1,
                                expertAdherenceCount: 9,
                                parentSegments: [],
                            },
                            brands_hatch2: {
                                analyzedTimeCount: 10,
                                mistakeCount: 6,
                                expertAdherenceCount: 4,
                                parentSegments: [
                                    {
                                        id: 'MSP',
                                        type: 'mistake',
                                        count: 6,
                                        childSegments: [
                                            { id: 'late_brake', count: 4 },
                                            { id: 'wide_exit', count: 2 },
                                        ],
                                    },
                                    {
                                        id: 'EA',
                                        type: 'expert_adherence',
                                        count: 4,
                                        childSegments: [
                                            { id: 'good_apex', count: 4 },
                                        ],
                                    },
                                ],
                            },
                        },
                    },
                    monza: {
                        trackName: 'Monza',
                        analyzedSessionCount: 1,
                        sections: {
                            monza1: {
                                analyzedTimeCount: 8,
                                mistakeCount: 0,
                                expertAdherenceCount: 8,
                                parentSegments: [],
                            },
                        },
                    },
                },
            },
        },
    },
});

describe('ai command registry user summary tools', () => {
    it('exposes a frontend schema for searching map-level user summary', () => {
        expect(frontendToolSchemas.some((tool) => tool.name === 'search_user_summary_map_level')).toBe(true);
    });

    it('exposes a frontend schema for listing available user summary maps', () => {
        expect(frontendToolSchemas.some((tool) => tool.name === 'get_available_user_summary_maps')).toBe(true);
    });

    it('exposes a frontend schema for displaying maps in chat', () => {
        expect(frontendToolSchemas.some((tool) => tool.name === 'show_map')).toBe(true);
    });

    it('displays a requested circuit map through the frontend callback', async () => {
        const displayMap = jest.fn();
        const registry = createAiCommandRegistry({
            sessionMode: 'live',
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
            getCircuitMapById: jest.fn(async () => ({
                id: 'brands_hatch',
                game: 'acc',
                circuit_name: 'Brands Hatch GP',
                source_track_key: 'brands_hatch',
                updated_at: null,
                sample_count: 4,
                resolution: 1000,
                samples: {
                    left_boundary: [
                        { bin: 0, normalized_position: 0, x: 0, y: 0, z: 0, sample_count: 1, updated_at: 'now' },
                        { bin: 100, normalized_position: 0.1, x: 10, y: 0, z: 0, sample_count: 1, updated_at: 'now' },
                    ],
                    right_boundary: [
                        { bin: 0, normalized_position: 0, x: 0, y: 0, z: 4, sample_count: 1, updated_at: 'now' },
                        { bin: 100, normalized_position: 0.1, x: 10, y: 0, z: 4, sample_count: 1, updated_at: 'now' },
                    ],
                    pit_lane: [],
                },
            })),
            displayMap,
        });

        const result = await registry.show_map(
            { map_id: 'brands_hatch', section_start: 0, section_end: 0.1, section_label: 'Paddock' },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'displayed',
            map_id: 'brands_hatch',
            circuit_name: 'Brands Hatch GP',
        });
        expect(displayMap).toHaveBeenCalledWith(expect.objectContaining({
            status: 'ready',
            section: {
                start: 0,
                end: 0.1,
                label: 'Paddock',
            },
        }));
    });

    it('shows the map unavailable fallback when no circuit map can be resolved', async () => {
        const displayMap = jest.fn();
        const registry = createAiCommandRegistry({
            sessionMode: 'live',
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
            getCircuitMapById: jest.fn(async () => null),
            getCircuitMapByTrack: jest.fn(async () => null),
            displayMap,
        });

        const result = await registry.show_map(
            { map_id: 'unknown_track' },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'unavailable',
            message: 'Map is not available',
        });
        expect(displayMap).toHaveBeenCalledWith(expect.objectContaining({
            status: 'unavailable',
            reason: 'No circuit map is available for "unknown_track".',
        }));
    });

    it('lists compact map choices from the retrieved user summary', async () => {
        const result = await createRegistry().get_available_user_summary_maps(
            {},
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            map_count: 2,
            map_options: [
                'Brands Hatch GP (brands_hatch) - 3 analyzed sessions',
                'Monza (monza) - 1 analyzed session',
            ],
            response_text:
                'Available maps in your summary:\n' +
                '- Brands Hatch GP (brands_hatch) - 3 analyzed sessions\n' +
                '- Monza (monza) - 1 analyzed session\n' +
                'Which map should I inspect?',
            maps: [
                {
                    id: 'brands_hatch',
                    name: 'Brands Hatch GP',
                    analyzed_session_count: 3,
                    section_count: 2,
                },
                {
                    id: 'monza',
                    name: 'Monza',
                    analyzed_session_count: 1,
                    section_count: 1,
                },
            ],
        });
    });

    it('searches retrieved user summary map-level rows by track name', async () => {
        const result = await createRegistry().search_user_summary_map_level(
            { query: 'brands hatch' },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            query: 'brands hatch',
            match_count: 1,
            maps: [
                {
                    id: 'brands_hatch',
                    name: 'Brands Hatch GP',
                },
            ],
        });
    });

    it('searches map-level top sections from the retrieved user summary', async () => {
        const result = await createRegistry().search_user_summary_map_level(
            { query: 'druids' },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            match_count: 1,
            maps: [
                {
                    id: 'brands_hatch',
                    matched_fields: expect.arrayContaining(['top_mistake_section']),
                },
            ],
        });
    });

    it('includes section-level mistake breakdowns when reading a known map', async () => {
        const result = await createRegistry().get_user_summary_map_level(
            { map_id: 'brands_hatch' },
            { sendObservation: jest.fn() },
        );

        expect(result.maps[0].sections).toEqual(expect.arrayContaining([
            expect.objectContaining({
                id: 'brands_hatch2',
                mistake_segments: [
                    expect.objectContaining({
                        id: 'MSP',
                        count: 6,
                        child_segments: [
                            expect.objectContaining({ id: 'late_brake', count: 4 }),
                            expect.objectContaining({ id: 'wide_exit', count: 2 }),
                        ],
                    }),
                ],
            }),
        ]));
        expect(result.maps[0].top_mistake_sections[0]).toEqual(expect.objectContaining({
            id: 'brands_hatch2',
            mistake_segments: expect.any(Array),
        }));
    });
});

describe('ai command registry recorded session tools', () => {
    const recordedAnalysisState: RecordedAiAnalysisState = {
        sessionId: 'session-1',
        status: 'ready',
        result: {
            status: 'success',
            session_id: 'session-1',
            samples_analyzed: 120,
            segment_count: 1,
            segments: [
                {
                    id: 'segment-1',
                    start_index: 10,
                    end_index: 40,
                    main_label_id: 'brands_hatch1',
                    labels: ['brands_hatch1', 'late_brake'],
                    child_segments: [
                        {
                            start_index: 20,
                            end_index: 28,
                            labels: ['late_brake'],
                        },
                    ],
                },
            ],
        },
    };

    const createRecordedRegistry = (overrides: Record<string, any> = {}) => {
        const analysisContext = {
            sessionSelected: {
                SessionId: 'session-1',
                session_name: 'Practice 1',
                map: 'brands_hatch',
                car: 'Ferrari 296',
                user_id: 'user-1',
                points: [],
                data: [],
            },
            mapSelected: 'brands_hatch',
            recordedAiAnalysis: recordedAnalysisState,
            recordedPlaybackSummary: {
                sessionId: 'session-1',
                sampleCount: 120,
                durationSeconds: 92.5,
                playbackIndex: 24,
                playbackTimeSeconds: 18.25,
                activeSegment: {
                    segmentId: 'segment-1',
                    startIndex: 10,
                    endIndex: 40,
                    parentLabel: 'Paddock Hill Bend',
                    childLabels: ['late_brake'],
                },
            },
            runRecordedAiAnalysis: jest.fn(async () => recordedAnalysisState),
            ...overrides.analysisContext,
        };

        return {
            analysisContext,
            registry: createAiCommandRegistry({
                sessionMode: 'recorded',
                analysisContext,
                opportunityAgentState: {
                    intervalId: null,
                    inFlight: false,
                    lastAlertKey: null,
                    lastAlertAt: 0,
                },
                startTrackGuide: jest.fn(),
                setTrackGuideEnabled: jest.fn(),
                getOpportunityTelemetryRows: () => [],
                getLabelName: (labelId) => labelNames[labelId] || labelId,
                getCategoryLabels: (category) => categories[category] ?? [],
                ...overrides.registryContext,
            }),
        };
    };

    it('exposes frontend schemas for recorded session analysis tools', () => {
        expect(frontendToolSchemas.some((tool) => tool.name === 'run_recorded_ai_analysis')).toBe(true);
        expect(frontendToolSchemas.some((tool) => tool.name === 'get_recorded_session_analysis')).toBe(true);
        expect(frontendToolSchemas.some((tool) => tool.name === 'get_recorded_session_context')).toBe(true);
    });

    it('advertises recorded-session and user-summary tools in recorded mode', () => {
        const toolNames = getFrontendToolSchemasForSessionMode('recorded').map((tool) => tool.name);

        expect(toolNames).toEqual(expect.arrayContaining([
            'show_map',
            'run_recorded_ai_analysis',
            'get_recorded_session_analysis',
            'get_recorded_session_context',
            'get_user_summary_map_level',
            'get_available_user_summary_maps',
            'search_user_summary_map_level',
        ]));
        expect(toolNames).not.toEqual(expect.arrayContaining([
            'query_telemetry_metric',
            'get_event_log',
            'start_live_performance_analysis',
            'stop_per_turn_coaching',
            'get_live_session_snapshot',
        ]));
    });

    it('runs recorded AI analysis through the shared context action', async () => {
        const { analysisContext, registry } = createRecordedRegistry();

        const result = await registry.run_recorded_ai_analysis(
            { force: true, limit: 5 },
            { sendObservation: jest.fn() },
        );

        expect(analysisContext.runRecordedAiAnalysis).toHaveBeenCalledWith({ force: true });
        expect(result).toMatchObject({
            status: 'ready',
            session_id: 'session-1',
            analysis: {
                segment_count: 1,
                segments: [
                    {
                        id: 'segment-1',
                        parent_label: 'Paddock Hill Bend',
                    },
                ],
            },
        });
    });

    it('returns cached shared recorded analysis without running analysis again', async () => {
        const { analysisContext, registry } = createRecordedRegistry();

        const result = await registry.get_recorded_session_analysis(
            { limit: 1 },
            { sendObservation: jest.fn() },
        );

        expect(analysisContext.runRecordedAiAnalysis).not.toHaveBeenCalled();
        expect(result).toMatchObject({
            status: 'ready',
            analysis: {
                returned_segment_count: 1,
                samples_analyzed: 120,
            },
        });
    });

    it('returns selected recorded session playback context', async () => {
        const { registry } = createRecordedRegistry();

        const result = await registry.get_recorded_session_context(
            {},
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            selected_session: {
                id: 'session-1',
                name: 'Practice 1',
            },
            recorded_telemetry: {
                sample_count: 120,
                playback_time_seconds: 18.25,
                active_segment: {
                    parentLabel: 'Paddock Hill Bend',
                },
            },
        });
    });

    it('returns a clear error when no recorded session is selected', async () => {
        const { registry } = createRecordedRegistry({
            analysisContext: {
                sessionSelected: null,
            },
        });

        const result = await registry.get_recorded_session_analysis(
            {},
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'error',
            error: 'no_recorded_session',
        });
    });

    it('keeps live-only telemetry tools unavailable in recorded mode', async () => {
        const { registry } = createRecordedRegistry();

        const result = await registry.query_telemetry_metric(
            { fields: ['Physics_speed_kmh'], scope: { type: 'now' }, reduce: 'avg' },
            { sendObservation: jest.fn() },
        );

        expect(result).toEqual({ error: 'recorded_session_live_tools_unavailable' });
    });

    it('keeps user-summary map tools available in recorded mode for comparison context', async () => {
        const { registry } = createRecordedRegistry({
            registryContext: {
                userSummary: {
                    sessionAnalysis: {
                        practice: {
                            tracks: {
                                brands_hatch: {
                                    trackName: 'Brands Hatch GP',
                                    analyzedSessionCount: 3,
                                    sections: {},
                                },
                            },
                        },
                    },
                },
            },
        });

        const result = await registry.get_user_summary_map_level(
            { map_id: 'brands_hatch' },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            map_count: 1,
            maps: [
                {
                    id: 'brands_hatch',
                    name: 'Brands Hatch GP',
                    analyzed_session_count: 3,
                },
            ],
        });
    });
});

describe('ai command registry live performance analyst tools', () => {
    const createLiveAnalystRegistry = (overrides: Partial<AiCommandRegistryContext> = {}) => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.startBaselineCollectionAtLapStart();
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.98,
        });
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.03,
        });
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 1,
            Graphics_normalized_car_position: 0,
        });

        const recordedAiAnalysis: RecordedAiAnalysisState = {
            sessionId: 'session-1',
            status: 'ready',
            result: {
                status: 'success',
                session_id: 'session-1',
                samples_analyzed: 120,
                segment_count: 1,
                segments: [
                    {
                        id: 'brands_hatch2:10-30',
                        parent_segment_id: 'brands_hatch2',
                        parent_label_id: 'brands_hatch2',
                        main_label_id: 'brands_hatch2',
                        start_index: 10,
                        end_index: 30,
                        labels: ['brands_hatch2', 'MSP', 'late_brake'],
                        sub_labels: ['late_brake'],
                        child_segments: [
                            {
                                start_index: 12,
                                end_index: 24,
                                labels: ['MSP', 'late_brake'],
                            },
                        ],
                    },
                ],
            },
        };
        const analysisContext = {
            sessionSelected: {
                SessionId: 'session-1',
                session_name: 'Practice 1',
                map: 'brands_hatch',
                car: 'Ferrari 296',
                user_id: 'user-1',
                points: [],
                data: [],
            },
            mapSelected: 'brands_hatch',
            recordedAiAnalysis,
            runRecordedAiAnalysis: jest.fn(async () => recordedAiAnalysis),
        };

        const livePerformanceAnalystState: any = {
            intervalId: null,
            inFlight: false,
            enabled: false,
            lastObservationKey: null,
            lastObservationAt: 0,
            lastSpokenAt: 0,
        };

        const context: AiCommandRegistryContext = {
            sessionMode: 'live',
            analysisContext,
            sessionIntelligence,
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            livePerformanceAnalystState,
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            setLivePerformanceAnalystEnabled: jest.fn((enabled) => {
                livePerformanceAnalystState.enabled = enabled;
            }),
            setAgentTagActive: jest.fn(),
            getOpportunityTelemetryRows: () => [],
            getBaselineCollectionTag: () => buildBaselineCollectionTag(
                sessionIntelligence.getLiveSessionSnapshot() as unknown as Record<string, any>,
            ),
            getLabelName: (labelId) => labelNames[labelId] || labelId,
            getCategoryLabels: (category) => categories[category] ?? [],
            ...overrides,
        };

        return {
            analysisContext,
            sessionIntelligence,
            livePerformanceAnalystState,
            context,
            registry: createAiCommandRegistry(context),
        };
    };

    it('advertises generic live agent session tools only in live mode', () => {
        const liveToolNames = getFrontendToolSchemasForSessionMode('live').map((tool) => tool.name);
        const recordedToolNames = getFrontendToolSchemasForSessionMode('recorded').map((tool) => tool.name);

        expect(liveToolNames).toEqual(expect.arrayContaining([
            'start_agent_session',
            'stop_agent_session',
            'get_live_session_snapshot',
            'get_live_focus_section',
            'get_live_section_history',
            'advance_plan_step',
            'set_procedure_plan',
            'clear_procedure_plan',
        ]));
        expect(liveToolNames).not.toEqual(expect.arrayContaining([
            '_get_live_section_telemetry',
            '_record_live_section_classification',
            'start_per_turn_coaching',
            'stop_per_turn_coaching',
            'start_overtake_agent',
            'stop_overtake_agent',
            'start_live_performance_analysis',
            'stop_live_performance_analysis',
        ]));
        expect(recordedToolNames).not.toEqual(expect.arrayContaining([
            'start_live_performance_analysis',
            'get_live_session_snapshot',
        ]));
        expect(recordedToolNames).toEqual(expect.arrayContaining([
            'stop_agent_session',
            'advance_plan_step',
            'set_procedure_plan',
            'clear_procedure_plan',
        ]));
    });

    it('starts every live agent through the generic child agent session tool', async () => {
        const startAgentSession = jest.fn(() => ({
            status: 'started' as const,
            conversation_role: 'agent' as const,
            agent_mode: 'live_performance_analyst' as const,
            agent_session_id: 'agent-1',
            parent_client_session_id: 'main-1',
        }));
        const { registry, livePerformanceAnalystState } = createLiveAnalystRegistry({
            conversationRole: 'main',
            startAgentSession,
        });

        const result = await registry.start_agent_session(
            { agent_mode: 'live_performance_analyst', interval_seconds: 3 },
            { sendObservation: jest.fn() },
        );

        expect(startAgentSession).toHaveBeenCalledWith(
            'live_performance_analyst',
            { agent_mode: 'live_performance_analyst', interval_seconds: 3 },
        );
        expect(result).toMatchObject({
            status: 'started',
            conversation_role: 'agent',
            agent_mode: 'live_performance_analyst',
            agent_session_id: 'agent-1',
        });
        expect(livePerformanceAnalystState.enabled).toBe(false);
        expect(livePerformanceAnalystState.intervalId).toBeNull();
    });

    it('starts track guide through the generic child agent session tool', async () => {
        const startAgentSession = jest.fn((agentMode) => ({
            status: 'started' as const,
            conversation_role: 'agent' as const,
            agent_mode: agentMode,
            agent_session_id: `agent-${agentMode}`,
            parent_client_session_id: 'main-1',
        }));
        const { registry } = createLiveAnalystRegistry({
            conversationRole: 'main',
            startAgentSession,
        });

        const result = await registry.start_agent_session(
            { agent_mode: 'track_guide' },
            { sendObservation: jest.fn() },
        );

        expect(startAgentSession).toHaveBeenCalledWith('track_guide', { agent_mode: 'track_guide' });
        expect(result).toMatchObject({
            status: 'started',
            conversation_role: 'agent',
            agent_mode: 'track_guide',
            agent_session_id: 'agent-track_guide',
        });
    });

    it('advertises shared runtime tools inside an agent session without recursive or dedicated agent controls', () => {
        const toolNames = getFrontendToolSchemasForSessionMode('live', {
            conversationRole: 'agent',
            agentMode: 'live_performance_analyst',
        }).map((tool) => tool.name);

        expect(toolNames).toEqual(expect.arrayContaining([
            'get_live_session_snapshot',
            'stop_agent_session',
        ]));
        expect(toolNames).not.toEqual(expect.arrayContaining([
            'start_agent_session',
            'start_per_turn_coaching',
            'stop_per_turn_coaching',
            'start_overtake_agent',
            'stop_overtake_agent',
            'start_live_performance_analysis',
            'stop_live_performance_analysis',
        ]));
    });

    it('lets the assistant advance the visible procedure plan request', async () => {
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'advanced',
            current_step: 1,
            step: 'Compare the next pass.',
        }));
        const { registry } = createLiveAnalystRegistry({
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Run live analysis from a clean baseline.',
                requests: [
                    { type: 'request', subscriber: 'driver', status: 'complete', title: 'Collect a complete baseline lap.' },
                    { type: 'request', subscriber: 'driver', status: 'pending', title: 'Compare the next pass.' },
                    { type: 'request', subscriber: 'driver', status: 'pending', title: 'Select the focus section.' },
                ],
                currentStep: 0,
                sourceEvent: 'live_analysis_plan_started',
            }),
        });

        const result = await registry.advance_plan_step(
            { reason: 'first step completed' },
            { sendObservation: jest.fn() },
        );

        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('first step completed');
        expect(result).toEqual({
            status: 'advanced',
            current_step: 1,
            step: 'Compare the next pass.',
        });
    });

    it('lets the assistant set the visible procedure plan request list', async () => {
        const setProcedurePlan = jest.fn();
        const { registry } = createLiveAnalystRegistry({
            setProcedurePlan,
        });

        const result = await registry.set_procedure_plan(
            {
                goal: 'Improve Druids entry.',
                requests: [
                    {
                        type: 'tool_call',
                        name: 'show_map',
                        title: 'Show the focus map',
                        result_visibility: 'tag',
                        payload: { tool: 'show_map', section_name: 'T2 Druids' },
                    },
                    {
                        type: 'driver_action',
                        subscriber: 'driver',
                        title: 'Brake earlier on the next approach',
                    },
                ],
            },
            { sendObservation: jest.fn() },
        );

        expect(setProcedurePlan).toHaveBeenCalledWith(expect.objectContaining({
            goal: 'Improve Druids entry.',
            requests: [
                expect.objectContaining({
                    type: 'tool_call',
                    name: 'show_map',
                    title: 'Show the focus map',
                    result_visibility: 'tag',
                    payload: { tool: 'show_map', section_name: 'T2 Druids' },
                    status: 'pending',
                }),
                expect.objectContaining({
                    type: 'driver_action',
                    subscriber: 'driver',
                    title: 'Brake earlier on the next approach',
                    status: 'pending',
                }),
            ],
            currentStep: 0,
        }));
        expect(result).toMatchObject({
            status: 'ready',
            goal: 'Improve Druids entry.',
            request_count: 2,
            current_request: 0,
        });
    });

    it('lets the assistant clear the visible procedure plan', async () => {
        const clearProcedurePlan = jest.fn();
        const { registry } = createLiveAnalystRegistry({
            clearProcedurePlan,
        });

        const result = await registry.clear_procedure_plan(
            { reason: 'plan is complete' },
            { sendObservation: jest.fn() },
        );

        expect(clearProcedurePlan).toHaveBeenCalledTimes(1);
        expect(result).toEqual({
            status: 'cleared',
            reason: 'plan is complete',
        });
    });

    it('advances the current step without executing the next request', async () => {
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'advanced',
            current_step: 1,
            step: 'Run the worker.',
        }));
        const { registry } = createLiveAnalystRegistry({
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Run a delegated workflow.',
                requests: [
                    { type: 'request', subscriber: 'driver', status: 'complete', title: 'Complete the first task.' },
                    { type: 'request', status: 'pending', title: 'Run the worker.' },
                ],
                currentStep: 0,
                sourceEvent: 'procedure_plan_started',
            }),
        });

        const result = await registry.advance_plan_step(
            { reason: 'first task complete' },
            { sendObservation: jest.fn() },
        );

        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('first task complete');
        expect(result).toEqual({
            status: 'advanced',
            current_step: 1,
            step: 'Run the worker.',
        });
    });

    it('executes the active tool_call request and returns AI-visible output before advancing', async () => {
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'advanced',
            current_step: 1,
            step: 'Use the snapshot.',
        }));
        const { registry } = createLiveAnalystRegistry({
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Fetch live session state.',
                requests: [
                    {
                        type: 'tool_call',
                        name: 'get_live_session_snapshot',
                        status: 'pending',
                        title: 'Read live session state.',
                        result_visibility: 'ai',
                    },
                    { type: 'driver_action', subscriber: 'driver', status: 'pending', title: 'Use the snapshot.' },
                ],
                currentStep: 0,
                sourceEvent: 'procedure_plan_started',
            }),
        });

        const result = await registry.advance_plan_step(
            { reason: 'snapshot requested' },
            { sendObservation: jest.fn() },
        );

        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('snapshot requested');
        expect(result).toMatchObject({
            status: 'advanced',
            current_step: 1,
            executed_tool: {
                name: 'get_live_session_snapshot',
                arguments: {},
                result_visibility: 'ai',
            },
            tool_result: {
                status: 'ready',
                agent_mode: 'live_performance_analyst',
                snapshot: {
                    baseline_ready: true,
                },
            },
        });
    });

    it('executes tag-only active tool_call requests without returning full tool output', async () => {
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'complete',
            current_step: 0,
            step: 'Show current state.',
        }));
        const { registry } = createLiveAnalystRegistry({
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Update the UI.',
                requests: [
                    {
                        type: 'tool_call',
                        name: 'get_live_session_snapshot',
                        status: 'pending',
                        title: 'Show current state.',
                        result_visibility: 'tag',
                    },
                ],
                currentStep: 0,
                sourceEvent: 'procedure_plan_started',
            }),
        });

        const result = await registry.advance_plan_step(
            { reason: 'ui tag updated' },
            { sendObservation: jest.fn() },
        );

        expect(result).toEqual({
            status: 'complete',
            current_step: 0,
            step: 'Show current state.',
            executed_tool: {
                name: 'get_live_session_snapshot',
                arguments: {},
                result_visibility: 'tag',
            },
            tool_result: {
                status: 'completed',
                result_visibility: 'tag',
            },
        });
    });

    it('blocks plan advancement until the baseline prerequisite is complete', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.2,
        });
        const advanceProcedurePlanStep = jest.fn();
        const context: AiCommandRegistryContext = {
            sessionMode: 'live',
            sessionIntelligence,
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            livePerformanceAnalystState: {
                intervalId: null,
                inFlight: false,
                enabled: true,
                lastObservationKey: null,
                lastObservationAt: 0,
                lastSpokenAt: 0,
            },
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Run live analysis from a clean baseline.',
                requests: [
                    { type: 'request', subscriber: 'driver', status: 'pending', title: 'Collect a complete baseline lap.' },
                    { type: 'request', subscriber: 'live_recorded_analysis', status: 'pending', title: 'Analyze the baseline.' },
                    { type: 'request', subscriber: 'driver', status: 'pending', title: 'Select the focus section.' },
                ],
                currentStep: 1,
                sourceEvent: 'live_analysis_plan_started',
            }),
        };
        const registry = createAiCommandRegistry(context);

        const result = await registry.advance_plan_step(
            { reason: 'skip ahead' },
            { sendObservation: jest.fn() },
        );

        expect(advanceProcedurePlanStep).not.toHaveBeenCalled();
        expect(result).toMatchObject({
            status: 'error',
            error: 'baseline_collection_incomplete',
            snapshot: {
                baseline_ready: false,
            },
        });
    });

    it('does not mark the baseline collection step complete while baseline progress is still zero', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.2,
        });
        sessionIntelligence.startBaselineCollectionAtLapStart();

        const advanceProcedurePlanStep = jest.fn();
        const context: AiCommandRegistryContext = {
            sessionMode: 'live',
            sessionIntelligence,
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            livePerformanceAnalystState: {
                intervalId: null,
                inFlight: false,
                enabled: true,
                lastObservationKey: null,
                lastObservationAt: 0,
                lastSpokenAt: 0,
            },
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
            getBaselineCollectionTag: () => buildBaselineCollectionTag(
                sessionIntelligence.getLiveSessionSnapshot() as unknown as Record<string, any>,
            ),
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Collect a baseline and use recorded-session analysis to choose a focus.',
                requests: [
                    {
                        type: 'driver_action',
                        subscriber: 'baseline_collection',
                        status: 'pending',
                        title: 'Collect a clean baseline lap',
                        detail: 'Complete one full lap before requesting classifier analysis.',
                    },
                    {
                        type: 'frontend_request',
                        subscriber: 'live_recorded_analysis',
                        status: 'pending',
                        title: 'Request recorded-session classifier',
                    },
                ],
                currentStep: 0,
                sourceEvent: 'procedure_plan_started',
            }),
        };
        const registry = createAiCommandRegistry(context);

        const result = await registry.advance_plan_step(
            { reason: 'baseline step completed' },
            { sendObservation: jest.fn() },
        );

        expect(sessionIntelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: false,
            baseline_collection_started: false,
            baseline_progress_percent: 0,
        });
        expect(advanceProcedurePlanStep).not.toHaveBeenCalled();
        expect(result).toMatchObject({
            status: 'error',
            error: 'baseline_collection_incomplete',
            snapshot: {
                baseline_ready: false,
                baseline_progress_percent: 0,
            },
            tag: {
                subscriber: 'baseline_collection',
                ready: false,
                progress_percent: 0,
            },
        });
    });

    it('does not execute an unregistered next request while advancing the current step', async () => {
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'advanced',
            current_step: 1,
            step: 'Run the worker.',
        }));
        const { registry } = createLiveAnalystRegistry({
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Run a delegated workflow.',
                requests: [
                    { type: 'request', subscriber: 'driver', status: 'complete', title: 'Complete the first task.' },
                    { type: 'request', subscriber: 'unregistered_worker', status: 'pending', title: 'Run the worker.' },
                ],
                currentStep: 0,
                sourceEvent: 'procedure_plan_started',
            }),
        });

        const result = await registry.advance_plan_step(
            { reason: 'first task complete' },
            { sendObservation: jest.fn() },
        );

        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('first task complete');
        expect(result).toEqual({
            status: 'advanced',
            current_step: 1,
            step: 'Run the worker.',
        });
    });

    it('returns compact live session snapshot state', async () => {
        const { registry } = createLiveAnalystRegistry();

        const result = await registry.get_live_session_snapshot(
            {},
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            snapshot: {
                track: 'brands_hatch',
                car: 'Ferrari 296',
                completed_laps: 1,
                baseline_ready: true,
                baseline_lap: 0,
                live_session_type: 'solo_practice',
            },
        });
    });

    it('does not expose a focus section while baseline collection is still in progress', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.2,
        });
        sessionIntelligence.recordSectionClassification({
            section_name: 'T2 Druids',
            lap: 0,
            start_sample_idx: 1,
            end_sample_idx: 2,
            mistake_count: 3,
            expert_adherence_count: 0,
            severity: 2,
            confidence: 0.9,
            child_labels: ['late brake'],
        });

        const context: AiCommandRegistryContext = {
            sessionMode: 'live',
            sessionIntelligence,
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            livePerformanceAnalystState: {
                intervalId: null,
                inFlight: false,
                enabled: true,
                lastObservationKey: null,
                lastObservationAt: 0,
                lastSpokenAt: 0,
            },
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
        };
        const registry = createAiCommandRegistry(context);

        const focusResult = await registry.get_live_focus_section(
            {},
            { sendObservation: jest.fn() },
        );
        const telemetryResult = await registry._get_live_section_telemetry(
            { section_name: 'T2 Druids', lap: 0 },
            { sendObservation: jest.fn() },
        );

        expect(focusResult).toMatchObject({
            status: 'error',
            error: 'baseline_collection_incomplete',
            snapshot: {
                baseline_ready: false,
            },
        });
        expect(telemetryResult).toMatchObject({
            status: 'error',
            error: 'baseline_collection_incomplete',
            snapshot: {
                baseline_ready: false,
            },
        });
    });

    it('starts the visible procedure plan while collecting baseline', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.45,
        });

        const livePerformanceAnalystState: any = {
            intervalId: null,
            inFlight: false,
            enabled: false,
            lastObservationKey: null,
            lastObservationAt: 0,
            lastSpokenAt: 0,
        };
        const context: AiCommandRegistryContext = {
            sessionMode: 'live',
            sessionIntelligence,
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            livePerformanceAnalystState,
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            setLivePerformanceAnalystEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
        };
        const sendObservation = jest.fn();
        sessionIntelligence.onLiveAnalystObservation(sendObservation);

        const result = await startAgentRuntime(
            'live_performance_analyst',
            context,
            {},
            { sendObservation },
        );

        expect(result).toMatchObject({
            status: 'started',
            initial: {
                status: 'checked',
                snapshot: {
                    baseline_ready: false,
                    baseline_collection_started: false,
                    baseline_progress_percent: 0,
                },
            },
        });
        expect(sendObservation).toHaveBeenCalledTimes(1);
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'live_analysis_plan_started',
            message: expect.stringContaining('Collect a baseline first'),
            snapshot: expect.objectContaining({
                baseline_ready: false,
            }),
        }));
        const startupPlanObservation = sendObservation.mock.calls.find(([payload]) => (
            payload.event === 'live_analysis_plan_started'
        ))?.[0];
        expect(startupPlanObservation).not.toHaveProperty('goal');
        expect(startupPlanObservation).not.toHaveProperty('requests');
        expect(startupPlanObservation).not.toHaveProperty('current_request');
        expect(startupPlanObservation).not.toHaveProperty('internal_tool_hint');
        expect(startupPlanObservation).not.toHaveProperty('sections');

        if (livePerformanceAnalystState.intervalId) {
            clearInterval(livePerformanceAnalystState.intervalId);
        }
    });

    it('does not run classifier analysis while advancing into the classifier request', async () => {
        const {
            analysisContext,
            context,
            registry,
            livePerformanceAnalystState,
            sessionIntelligence,
        } = createLiveAnalystRegistry();
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'advanced',
            current_request: 1,
            request: {
                type: 'frontend_request',
                subscriber: 'live_recorded_analysis',
                status: 'complete' as const,
                title: 'Request recorded-session classifier',
            },
        }));
        const planRegistry = createAiCommandRegistry({
            sessionMode: 'live',
            analysisContext,
            sessionIntelligence,
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            livePerformanceAnalystState,
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            setLivePerformanceAnalystEnabled: jest.fn((enabled) => {
                livePerformanceAnalystState.enabled = enabled;
            }),
            getOpportunityTelemetryRows: () => [],
            getLabelName: (labelId) => labelNames[labelId] || labelId,
            getCategoryLabels: (category) => categories[category] ?? [],
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Collect a baseline and use recorded-session analysis to choose a focus.',
                requests: [
                    {
                        type: 'driver_action',
                        subscriber: 'baseline_collection',
                        status: 'complete',
                        title: 'Collect a clean baseline lap',
                    },
                    {
                        type: 'frontend_request',
                        subscriber: 'live_recorded_analysis',
                        status: 'pending',
                        title: 'Request recorded-session classifier',
                        payload: { force: false },
                    },
                ],
                currentStep: 0,
                sourceEvent: 'live_analysis_plan_started',
            }),
        });
        const sendObservation = jest.fn();
        sessionIntelligence.onLiveAnalystObservation(sendObservation);
        const toolContextSendObservation = jest.fn();

        const startResult = await startAgentRuntime(
            'live_performance_analyst',
            context,
            {},
            { sendObservation: toolContextSendObservation },
        );
        expect(startResult).toMatchObject({
            status: 'started',
            initial: {
                status: 'checked',
                snapshot: {
                    baseline_ready: true,
                },
                focus: null,
            },
        });
        expect(sendObservation).not.toHaveBeenCalledWith(expect.objectContaining({
            event: 'baseline_classifier_request_ready',
        }));
        expect(analysisContext.runRecordedAiAnalysis).not.toHaveBeenCalled();

        const result = await planRegistry.advance_plan_step(
            { reason: 'baseline complete' },
            { sendObservation: toolContextSendObservation },
        );

        expect(result).toMatchObject({
            status: 'advanced',
            current_request: 1,
        });
        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('baseline complete');
        expect(analysisContext.runRecordedAiAnalysis).not.toHaveBeenCalled();
        expect(sendObservation).not.toHaveBeenCalledWith(expect.objectContaining({
            event: 'recorded_analysis_ready',
        }));
        expect(toolContextSendObservation).not.toHaveBeenCalledWith(expect.objectContaining({
            source: 'live_performance_analyst',
        }));

        if (livePerformanceAnalystState.intervalId) {
            clearInterval(livePerformanceAnalystState.intervalId);
        }
    });

    it('runs the classifier when the subscribed classifier request is already active', async () => {
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'complete',
            current_request: 1,
            request: {
                type: 'frontend_request',
                subscriber: 'live_recorded_analysis',
                status: 'complete' as const,
                title: 'Request recorded-session classifier',
            },
        }));
        const {
            analysisContext,
            registry,
            sessionIntelligence,
        } = createLiveAnalystRegistry({
            advanceProcedurePlanStep,
            getProcedurePlan: () => ({
                goal: 'Collect a baseline and use recorded-session analysis to choose a focus.',
                requests: [
                    {
                        type: 'driver_action',
                        subscriber: 'baseline_collection',
                        status: 'complete' as const,
                        title: 'Collect a clean baseline lap',
                    },
                    {
                        type: 'frontend_request',
                        subscriber: 'live_recorded_analysis',
                        status: 'pending',
                        title: 'Request recorded-session classifier',
                        payload: { force: false },
                    },
                ],
                currentStep: 1,
                sourceEvent: 'live_analysis_plan_started',
            }),
        });
        const sendObservation = jest.fn();
        sessionIntelligence.onLiveAnalystObservation(sendObservation);

        const result = await registry.advance_plan_step(
            { reason: 'run active classifier request' },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'complete',
            current_request: 1,
        });
        expect(analysisContext.runRecordedAiAnalysis).toHaveBeenCalledWith({ force: false });
        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('run active classifier request');
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'recorded_analysis_ready',
            analysis: expect.objectContaining({
                status: 'ready',
            }),
        }));
    });

    it('requires recorded analysis when the subscribed classifier request is advanced', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.startBaselineCollectionAtLapStart();
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.98,
        });
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.03,
        });
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 1,
            Graphics_normalized_car_position: 0,
        });

        const livePerformanceAnalystState: any = {
            intervalId: null,
            inFlight: false,
            enabled: false,
            lastObservationKey: null,
            lastObservationAt: 0,
            lastSpokenAt: 0,
        };
        const context: AiCommandRegistryContext = {
            sessionMode: 'live',
            sessionIntelligence,
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            livePerformanceAnalystState,
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            setLivePerformanceAnalystEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
            advanceProcedurePlanStep: jest.fn(),
            getProcedurePlan: () => ({
                goal: 'Collect a baseline and use recorded-session analysis to choose a focus.',
                requests: [
                    {
                        type: 'driver_action',
                        subscriber: 'baseline_collection',
                        status: 'complete',
                        title: 'Collect a clean baseline lap',
                    },
                    {
                        type: 'frontend_request',
                        subscriber: 'live_recorded_analysis',
                        status: 'pending',
                        title: 'Request recorded-session classifier',
                        payload: { force: false },
                    },
                ],
                currentStep: 1,
                sourceEvent: 'live_analysis_plan_started',
            }),
        };
        const registry = createAiCommandRegistry(context);
        const sendObservation = jest.fn();
        sessionIntelligence.onLiveAnalystObservation(sendObservation);
        const toolContextSendObservation = jest.fn();

        await startAgentRuntime(
            'live_performance_analyst',
            context,
            {},
            { sendObservation: toolContextSendObservation },
        );
        const result = await registry.advance_plan_step(
            { reason: 'baseline complete' },
            { sendObservation: toolContextSendObservation },
        );

        expect(result).toMatchObject({
            status: 'error',
            error: 'recorded_session_required',
            snapshot: {
                baseline_ready: true,
            },
        });
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'recorded_session_required',
            message: expect.stringContaining('recorded session'),
        }));
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'live_analysis_plan_started',
            snapshot: expect.objectContaining({
                baseline_ready: true,
            }),
        }));
        const startupObservation = sendObservation.mock.calls.find(([payload]) => (
            payload.event === 'live_analysis_plan_started'
        ))?.[0];
        expect(startupObservation).not.toHaveProperty('goal');
        expect(startupObservation).not.toHaveProperty('requests');
        expect(startupObservation).not.toHaveProperty('current_request');
        expect(toolContextSendObservation).not.toHaveBeenCalledWith(expect.objectContaining({
            source: 'live_performance_analyst',
        }));

        if (livePerformanceAnalystState.intervalId) {
            clearInterval(livePerformanceAnalystState.intervalId);
        }
    });

    it('keeps raw section telemetry behind an internal handler', async () => {
        const { registry } = createLiveAnalystRegistry();

        const result = await registry._get_live_section_telemetry(
            { section_name: 'T1 Paddock Hill Bend', lap: 0 },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            section: {
                name: 'T1 Paddock Hill Bend',
            },
            rows: expect.any(Array),
        });
    });

    it('returns show_map arguments for the selected analyst focus section', async () => {
        const { registry } = createLiveAnalystRegistry();

        await registry._record_live_section_classification(
            {
                section_name: 'T2 Druids',
                lap: 0,
                start_sample_idx: 10,
                end_sample_idx: 30,
                mistake_count: 3,
                expert_adherence_count: 0,
                severity: 2,
                confidence: 0.9,
                child_labels: ['late brake'],
            },
            { sendObservation: jest.fn() },
        );

        const result = await registry.get_live_focus_section(
            {},
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            focus: {
                section: {
                    name: 'T2 Druids',
                },
                show_map_arguments: {
                    section_start: 0.11,
                    section_end: 0.18,
                    section_label: 'T2 Druids',
                },
            },
        });
    });
});
