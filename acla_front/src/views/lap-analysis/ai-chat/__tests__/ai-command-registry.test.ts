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
} from '../ai-command-registry';
import { RecordedAiAnalysisState } from 'views/lap-analysis/recorded-session-analysis';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';

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

        return {
            analysisContext,
            sessionIntelligence,
            livePerformanceAnalystState,
            registry: createAiCommandRegistry({
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
                getLabelName: (labelId) => labelNames[labelId] || labelId,
                getCategoryLabels: (category) => categories[category] ?? [],
                ...overrides,
            }),
        };
    };

    it('advertises public live analyst tools only in live mode', () => {
        const liveToolNames = getFrontendToolSchemasForSessionMode('live').map((tool) => tool.name);
        const recordedToolNames = getFrontendToolSchemasForSessionMode('recorded').map((tool) => tool.name);

        expect(liveToolNames).toEqual(expect.arrayContaining([
            'start_live_performance_analysis',
            'stop_live_performance_analysis',
            'get_live_session_snapshot',
            'get_live_focus_section',
            'get_live_section_history',
            'advance_plan_step',
            'set_procedure_plan',
        ]));
        expect(liveToolNames).not.toEqual(expect.arrayContaining([
            '_get_live_section_telemetry',
            '_record_live_section_classification',
        ]));
        expect(recordedToolNames).not.toEqual(expect.arrayContaining([
            'start_live_performance_analysis',
            'get_live_session_snapshot',
            'advance_plan_step',
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
                    { type: 'request', title: 'Collect a complete baseline lap.' },
                    { type: 'request', title: 'Analyze the baseline.' },
                    { type: 'request', title: 'Select the focus section.' },
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
                focus_name: 'T2 Druids',
                requests: [
                    {
                        type: 'tool_call',
                        name: 'show_map',
                        title: 'Show the focus map',
                        payload: { section_name: 'T2 Druids' },
                    },
                    {
                        type: 'driver_action',
                        title: 'Brake earlier on the next approach',
                    },
                ],
            },
            { sendObservation: jest.fn() },
        );

        expect(setProcedurePlan).toHaveBeenCalledWith(expect.objectContaining({
            goal: 'Improve Druids entry.',
            focusName: 'T2 Druids',
            requests: [
                expect.objectContaining({
                    type: 'tool_call',
                    name: 'show_map',
                    title: 'Show the focus map',
                    payload: { section_name: 'T2 Druids' },
                }),
                expect.objectContaining({
                    type: 'driver_action',
                    title: 'Brake earlier on the next approach',
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
        const registry = createAiCommandRegistry({
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
                    { type: 'request', title: 'Collect a complete baseline lap.' },
                    { type: 'request', title: 'Analyze the baseline.' },
                    { type: 'request', title: 'Select the focus section.' },
                ],
                currentStep: 0,
                sourceEvent: 'live_analysis_plan_started',
            }),
        });

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

    it('blocks plan advancement to the focus step until a focus section exists', async () => {
        const { registry } = createLiveAnalystRegistry({
            advanceProcedurePlanStep: jest.fn(),
            getProcedurePlan: () => ({
                goal: 'Run live analysis from a clean baseline.',
                requests: [
                    { type: 'request', title: 'Collect a complete baseline lap.' },
                    { type: 'request', title: 'Analyze the baseline.' },
                    { type: 'request', title: 'Select the focus section.' },
                ],
                currentStep: 1,
                sourceEvent: 'live_baseline_ready_for_classification',
            }),
        });

        const result = await registry.advance_plan_step(
            { reason: 'skip focus analysis' },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'error',
            error: 'focus_section_not_ready',
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

        const registry = createAiCommandRegistry({
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
        });

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
        const registry = createAiCommandRegistry({
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
        });
        const sendObservation = jest.fn();

        const result = await registry.start_live_performance_analysis(
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

        if (livePerformanceAnalystState.intervalId) {
            clearInterval(livePerformanceAnalystState.intervalId);
        }
    });

    it('builds a focus plan from shared recorded analysis instead of classifying every section', async () => {
        const { analysisContext, registry, livePerformanceAnalystState } = createLiveAnalystRegistry();
        const sendObservation = jest.fn();

        const result = await registry.start_live_performance_analysis(
            {},
            { sendObservation },
        );

        expect(result).toMatchObject({
            status: 'started',
            initial: {
                status: 'checked',
                snapshot: {
                    baseline_ready: true,
                },
                focus: {
                    section: {
                        name: 'T2 Druids',
                    },
                },
                plan: {
                    goal: expect.stringContaining('T2 Druids'),
                },
            },
        });
        expect(analysisContext.runRecordedAiAnalysis).toHaveBeenCalledWith({ force: false });
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'recorded_analysis_plan_ready',
            goal: expect.stringContaining('T2 Druids'),
            focus: expect.objectContaining({
                section: expect.objectContaining({
                    name: 'T2 Druids',
                }),
            }),
        }));
        const recordedPlanObservation = sendObservation.mock.calls.find(([payload]) => (
            payload.event === 'recorded_analysis_plan_ready'
        ))?.[0];
        expect(recordedPlanObservation).not.toHaveProperty('plan');
        expect(recordedPlanObservation).not.toHaveProperty('internal_tool_hint');
        expect(recordedPlanObservation).not.toHaveProperty('sections');

        if (livePerformanceAnalystState.intervalId) {
            clearInterval(livePerformanceAnalystState.intervalId);
        }
    });

    it('asks the live classifier for a focus when no recorded analysis is available', async () => {
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
        const registry = createAiCommandRegistry({
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
        });
        const sendObservation = jest.fn();

        const result = await registry.start_live_performance_analysis(
            {},
            { sendObservation },
        );

        expect(result).toMatchObject({
            status: 'started',
            initial: {
                analysis_status: {
                    status: 'needs_live_section_classification',
                    recorded_analysis_error: 'recorded_session_required',
                },
                snapshot: {
                    baseline_ready: true,
                },
            },
        });
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'live_baseline_ready_for_classification',
            completed_lap: 0,
            candidate_sections: expect.arrayContaining([
                expect.objectContaining({
                    name: 'T1 Paddock Hill Bend',
                    lap: 0,
                }),
            ]),
        }));
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'live_analysis_plan_started',
            snapshot: expect.objectContaining({
                baseline_ready: true,
            }),
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
