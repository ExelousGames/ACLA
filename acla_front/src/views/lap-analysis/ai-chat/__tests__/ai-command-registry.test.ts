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
    frontendToolDefinitions,
    startAgentRuntime,
} from '../ai-command-registry';
import { RecordedAiAnalysisState } from 'views/lap-analysis/recorded-session-analysis';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import { buildBaselineCollectionTag, type BaselineLapRecord } from '../BaselineCollectionTracker';
import apiService from 'services/api.service';

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

beforeEach(() => {
    jest.clearAllMocks();
});

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
    it('registers frontend commands from AI tool definitions', () => {
        const registryNames = Object.keys(createRegistry()).sort();
        const definitionNames = frontendToolDefinitions.map((tool) => tool.name).sort();

        expect(registryNames).toEqual(definitionNames);
        expect(frontendToolDefinitions).toEqual(expect.arrayContaining([
            expect.objectContaining({
                name: 'collect_live_baseline',
                execute: expect.any(Function),
                schema: { properties: {}, required: [] },
                required: [],
            }),
        ]));
    });

    it('keeps frontend command registration separate from advertised tool grouping', () => {
        const registry = createRegistry();
        const definitionNames = new Set(frontendToolDefinitions.map((tool) => tool.name));

        expect(Object.keys(registry).every((name) => definitionNames.has(name))).toBe(true);
        expect(registry).toEqual(expect.objectContaining([
            'set_live_range_tracker',
            'update_live_range_tracker',
            'get_live_range_tracker',
            '_get_live_section_telemetry',
            'run_recorded_ai_analysis',
        ].reduce<Record<string, unknown>>((handlers, name) => {
            handlers[name] = expect.any(Function);
            return handlers;
        }, {})));
    });

    it('routes live range tracker tools to component-owned callbacks', async () => {
        const tracker = {
            status: 'open',
            ranges: [],
            created_at: 1,
            updated_at: 1,
        };
        const setLiveRangeTracker = jest.fn(() => ({ status: 'ready', tracker }));
        const updateLiveRangeTracker = jest.fn(() => ({ status: 'ready', tracker }));
        const getLiveRangeTracker = jest.fn(() => ({ status: 'ready', tracker }));
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
            setLiveRangeTracker,
            updateLiveRangeTracker,
            getLiveRangeTracker,
        });
        const toolContext = { sendObservation: jest.fn() };

        const setResult = await registry.set_live_range_tracker(
            { ranges: [{ id: 'r1', start_position: 0.1, end_position: 0.2 }] },
            toolContext,
        );
        const updateResult = await registry.update_live_range_tracker(
            { action: 'close' },
            toolContext,
        );
        const getResult = await registry.get_live_range_tracker({}, toolContext);

        expect(setLiveRangeTracker).toHaveBeenCalledWith({
            ranges: [{ id: 'r1', start_position: 0.1, end_position: 0.2 }],
        });
        expect(updateLiveRangeTracker).toHaveBeenCalledWith({ action: 'close' });
        expect(getLiveRangeTracker).toHaveBeenCalled();
        expect(setResult).toMatchObject({ ui_output: { status: 'ready', tracker } });
        expect(updateResult).toMatchObject({ ui_output: { status: 'ready', tracker } });
        expect(getResult).toMatchObject({ ui_output: { status: 'ready', tracker } });
        expect(setResult).toMatchObject({
            ai_output: {
                name: 'set_live_range_tracker',
                status: 'ready',
                range_count: 0,
            },
        });
        expect((setResult.ai_output as any)).not.toHaveProperty('tracker');
    });

    it('normalizes bracketed telemetry fields and common tire/fuel aliases', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.tick({
            Physics_fuel: 38,
            Physics_wheel_pressure_front_left: 26.1,
            Physics_wheel_pressure_front_right: 26.2,
            Physics_wheel_pressure_rear_left: 25.9,
            Physics_wheel_pressure_rear_right: 26,
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
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
        });

        const fuelResult = await registry.query_telemetry_metric(
            { fields: '[FuelLevel]', scope: { type: 'now' }, reduce: 'avg' },
            { sendObservation: jest.fn() },
        );
        const tireResult = await registry.query_telemetry_metric(
            {
                fields: "['TirePressureFL', 'TirePressureFR', 'TirePressureRL', 'TirePressureRR']",
                scope: { type: 'now' },
                reduce: 'avg',
            },
            { sendObservation: jest.fn() },
        );
        const bareTyreResult = await registry.query_telemetry_metric(
            {
                fields: "['tyre_pressure_front_left', 'tyre_pressure_front_right', 'tyre_pressure_rear_left', 'tyre_pressure_rear_right']",
                scope: { type: 'now' },
                reduce: 'avg',
            },
            { sendObservation: jest.fn() },
        );

        expect(fuelResult).toMatchObject({
            status: 'complete',
            ui_output: {
                Physics_fuel: 38,
            },
            ai_output: {
                name: 'query_telemetry_metric',
                status: 'complete',
                values: {
                    Physics_fuel: 38,
                },
            },
        });
        expect((fuelResult.ai_output as any)).not.toHaveProperty('ok');
        expect((fuelResult.ai_output as any).values).not.toHaveProperty('ok');
        expect(tireResult).toMatchObject({
            status: 'complete',
            ui_output: {
                Physics_wheel_pressure_front_left: 26.1,
                Physics_wheel_pressure_front_right: 26.2,
                Physics_wheel_pressure_rear_left: 25.9,
                Physics_wheel_pressure_rear_right: 26,
            },
        });
        expect(bareTyreResult).toMatchObject({
            status: 'complete',
            ui_output: {
                Physics_wheel_pressure_front_left: 26.1,
                Physics_wheel_pressure_front_right: 26.2,
                Physics_wheel_pressure_rear_left: 25.9,
                Physics_wheel_pressure_rear_right: 26,
            },
        });
    });

    it('registers the map-level user summary search tool', () => {
        expect(frontendToolDefinitions.some((tool) => tool.name === 'search_user_summary_map_level')).toBe(true);
    });

    it('registers the available user summary maps tool', () => {
        expect(frontendToolDefinitions.some((tool) => tool.name === 'get_available_user_summary_maps')).toBe(true);
    });

    it('registers the map display tool', () => {
        expect(frontendToolDefinitions.some((tool) => tool.name === 'show_map')).toBe(true);
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
                game: 'acc' as const,
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
        });
        expect(result.ui_output).toMatchObject({
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
        });
        expect(result.ui_output).toMatchObject({
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
        });
        expect(result.ui_output).toMatchObject({
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
        });
        expect(result.ui_output).toMatchObject({
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

        expect((result.ui_output as any).maps[0].sections).toEqual(expect.arrayContaining([
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
        expect((result.ui_output as any).maps[0].top_mistake_sections[0]).toEqual(expect.objectContaining({
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

    it('registers recorded session analysis tools', () => {
        expect(frontendToolDefinitions.some((tool) => tool.name === 'run_recorded_ai_analysis')).toBe(true);
        expect(frontendToolDefinitions.some((tool) => tool.name === 'get_recorded_session_analysis')).toBe(true);
        expect(frontendToolDefinitions.some((tool) => tool.name === 'get_recorded_session_context')).toBe(true);
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
        });
        expect(result.ui_output).toMatchObject({
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
        });
        expect(result.ui_output).toMatchObject({
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
        });
        expect(result.ui_output).toMatchObject({
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

        expect(result).toMatchObject({
            status: 'error',
            error: 'recorded_session_live_tools_unavailable',
            final: true,
            ui_output: { error: 'recorded_session_live_tools_unavailable' },
            tool_name: 'query_telemetry_metric',
        });
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
        });
        expect(result.ui_output).toMatchObject({
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
        const getBaselineLapRecord = () => {
            const snapshot = sessionIntelligence.getLiveSessionSnapshot() as unknown as Record<string, any>;
            const records = sessionIntelligence.getLastCompletedLapRows()
                .map((row) => ({ ...(row as Record<string, any>) }));

            return {
                id: `brands_hatch:Ferrari 296:${snapshot.baseline_lap}:${records.length}`,
                lap: Number(snapshot.baseline_lap ?? 0),
                captured_at: 1,
                track: 'brands_hatch',
                car: 'Ferrari 296',
                sample_count: records.length,
                snapshot,
                records,
            };
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
            getBaselineLapRecord,
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

    it('collects baseline through the subscribed tool result channel', async () => {
        const setBaselineCollectionEnabled = jest.fn();
        const restartBaselineCollection = jest.fn();
        const { registry, livePerformanceAnalystState } = createLiveAnalystRegistry({
            setBaselineCollectionEnabled,
            restartBaselineCollection,
        });

        const result = await registry.collect_live_baseline(
            {},
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'complete',
            progress_percent: 100,
            message: 'Baseline complete. Cached lap record is ready.',
            final: true,
            tool_name: 'collect_live_baseline',
            ui_output: {
                progress_percent: 100,
                status: 'complete',
                car: 'Ferrari 296',
                track: 'brands_hatch',
                message: 'Baseline complete. Cached lap record is ready.',
            },
        });
        expect(result).not.toHaveProperty('source');
        expect(result).not.toHaveProperty('agent_mode');
        expect(result).not.toHaveProperty('baseline');
        expect(livePerformanceAnalystState.enabled).toBe(true);
        expect(setBaselineCollectionEnabled).toHaveBeenCalledWith(true);
        expect(restartBaselineCollection).not.toHaveBeenCalled();
    });

    it('restarts live baseline collection through a separate AI command', async () => {
        const setBaselineCollectionEnabled = jest.fn();
        const restartBaselineCollection = jest.fn();
        const { registry } = createLiveAnalystRegistry({
            setBaselineCollectionEnabled,
            restartBaselineCollection,
        });

        const result = await registry.restart_live_baseline(
            {},
            { sendObservation: jest.fn() },
        );

        expect(restartBaselineCollection).toHaveBeenCalledTimes(1);
        expect(setBaselineCollectionEnabled).toHaveBeenCalledWith(true);
        expect(result).toMatchObject({
            status: 'restarted',
            progress_percent: 0,
            message: 'Baseline collection restarted.',
            final: true,
            tool_name: 'restart_live_baseline',
            ui_output: {
                status: 'restarted',
                progress_percent: 0,
                message: 'Baseline collection restarted.',
            },
        });
    });

    it('returns baseline timeout through the subscribed tool result channel', async () => {
        jest.useFakeTimers();
        try {
            const collectingTag = buildBaselineCollectionTag({
                status: 'ready',
                track: 'brands_hatch',
                car: 'Ferrari 296',
                current_lap: 0,
                completed_laps: 0,
                normalized_position: 0.35,
                sample_count: 5,
                live_session_type: 'solo_practice',
                baseline_ready: false,
                baseline_collection_started: true,
                baseline_progress_percent: 35,
                baseline_lap: 0,
                completed_lap_count: 0,
                section_count: 0,
            });
            const { registry } = createLiveAnalystRegistry({
                getBaselineCollectionTag: () => collectingTag,
                getBaselineLapRecord: () => null,
            });

            const resultPromise = registry.collect_live_baseline(
                { timeout_seconds: 30 },
                { sendObservation: jest.fn() },
            );

            jest.advanceTimersByTime(250);
            await Promise.resolve();

            jest.advanceTimersByTime(30000);
            await Promise.resolve();
            await expect(resultPromise).resolves.toMatchObject({
                status: 'error',
                error: 'baseline_collection_timeout',
                final: true,
                tool_name: 'collect_live_baseline',
                ui_output: {
                    status: 'error',
                    error: 'baseline_collection_timeout',
                    progress_percent: 35,
                    car: 'Ferrari 296',
                    track: 'brands_hatch',
                    message: 'Baseline collection did not complete before the tool timeout.',
                },
            });
        } finally {
            jest.useRealTimers();
        }
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
        });
        expect(result.ui_output).toMatchObject({
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
        });
        expect(result.ui_output).toMatchObject({
            status: 'started',
            conversation_role: 'agent',
            agent_mode: 'track_guide',
            agent_session_id: 'agent-track_guide',
        });
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
                    { type: 'request', status: 'complete', title: 'Collect a complete baseline lap.' },
                    { type: 'request', status: 'pending', title: 'Compare the next pass.' },
                    { type: 'request', status: 'pending', title: 'Select the focus section.' },
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
        expect(result).toMatchObject({
            status: 'advanced',
            final: true,
            ui_output: {
                status: 'advanced',
                current_step: 1,
                step: 'Compare the next pass.',
            },
            tool_name: 'advance_plan_step',
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
                        payload: { tool: 'show_map', section_name: 'T2 Druids' },
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
            requests: [
                expect.objectContaining({
                    type: 'tool_call',
                    name: 'show_map',
                    title: 'Show the focus map',
                    payload: { tool: 'show_map', section_name: 'T2 Druids' },
                    status: 'pending',
                }),
                expect.objectContaining({
                    type: 'driver_action',
                    title: 'Brake earlier on the next approach',
                    status: 'pending',
                }),
            ],
            currentStep: 0,
        }));
        expect(result).toMatchObject({
            status: 'ready',
        });
        expect(result.ui_output).toMatchObject({
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
        expect(result).toMatchObject({
            status: 'cleared',
            final: true,
            ui_output: {
                status: 'cleared',
                reason: 'plan is complete',
            },
            tool_name: 'clear_procedure_plan',
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
                    { type: 'request', status: 'complete', title: 'Complete the first task.' },
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
        expect(result).toMatchObject({
            status: 'advanced',
            final: true,
            ui_output: {
                status: 'advanced',
                current_step: 1,
                step: 'Run the worker.',
            },
            tool_name: 'advance_plan_step',
        });
    });

    it('advances the visible plan without executing active tool_call requests', async () => {
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
                        name: 'get_live_focus_section',
                        status: 'pending',
                        title: 'Read live session state.',
                    },
                    { type: 'driver_action', status: 'pending', title: 'Use the snapshot.' },
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
            final: true,
            ui_output: {
                status: 'advanced',
                current_step: 1,
                step: 'Use the snapshot.',
            },
            tool_name: 'advance_plan_step',
        });
    });

    it('advances tag-only plan requests without executing the tagged tool', async () => {
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
                        name: 'get_live_focus_section',
                        status: 'pending',
                        title: 'Show current state.',
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

        expect(result).toMatchObject({
            status: 'complete',
            final: true,
            ui_output: {
                status: 'complete',
                current_step: 0,
                step: 'Show current state.',
            },
            tool_name: 'advance_plan_step',
        });
    });

    it('does not use baseline prerequisites to block visible plan advancement', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.2,
        });
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'advanced',
            current_step: 2,
            step: 'Select the focus section.',
        }));
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
            getBaselineCollectionTag: () => buildBaselineCollectionTag({
                status: 'ready',
                track: 'brands_hatch',
                car: 'Ferrari 296',
                current_lap: 0,
                completed_laps: 0,
                normalized_position: 0.2,
                sample_count: 1,
                live_session_type: 'solo_practice',
                baseline_ready: false,
                baseline_collection_started: true,
                baseline_progress_percent: 20,
                baseline_lap: 0,
                completed_lap_count: 0,
                section_count: 0,
            }),
            getProcedurePlan: () => ({
                goal: 'Run live analysis from a clean baseline.',
                requests: [
                    { type: 'request', status: 'pending', title: 'Collect a complete baseline lap.' },
                    { type: 'request', status: 'pending', title: 'Analyze the baseline.' },
                    { type: 'request', status: 'pending', title: 'Select the focus section.' },
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

        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('skip ahead');
        expect(result).toMatchObject({
            status: 'advanced',
        });
        expect(result.ui_output).toMatchObject({
            status: 'advanced',
            current_step: 2,
        });
    });

    it('leaves baseline readiness to the baseline component when advancing the visible plan', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Static_car_model: 'Ferrari 296',
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.2,
        });
        sessionIntelligence.startBaselineCollectionAtLapStart();

        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'advanced',
            current_step: 1,
            step: 'Request recorded-session classifier',
        }));
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
                        type: 'tool_call',
                        status: 'pending',
                        title: 'Collect a clean baseline lap',
                        name: 'collect_live_baseline',
                        detail: 'Complete one full lap before requesting classifier analysis.',
                    },
                    {
                        type: 'tool_call',
                        status: 'pending',
                        title: 'Request recorded-session classifier',
                        name: 'analyze_live_recorded_analysis',
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
        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('baseline step completed');
        expect(result).toMatchObject({
            status: 'advanced',
        });
        expect(result.ui_output).toMatchObject({
            status: 'advanced',
            current_step: 1,
        });
    });

    it('returns immediately until a recorded baseline lap is cached, then runs live recorded analysis', async () => {
        (apiService.post as jest.Mock).mockResolvedValueOnce({
            data: {
                status: 'success',
                session_id: 'live-baseline',
                samples_analyzed: 2,
                segment_count: 1,
                segments: [
                    {
                        id: 'live-segment-1',
                        start_index: 0,
                        end_index: 1,
                        main_label_id: 'brands_hatch2',
                        labels: ['brands_hatch2', 'late_brake'],
                        child_segments: [],
                    },
                ],
            },
        });
        let cachedRecord: BaselineLapRecord | null = null;
        const currentTag = buildBaselineCollectionTag({
            status: 'ready',
            track: 'brands_hatch',
            car: 'Ferrari 296',
            current_lap: 0,
            completed_laps: 0,
            normalized_position: 0.42,
            sample_count: 2,
            live_session_type: 'solo_practice',
            baseline_ready: false,
            baseline_collection_started: true,
            baseline_progress_percent: 42,
            baseline_lap: 0,
            completed_lap_count: 0,
            section_count: 0,
        });
        const { registry } = createLiveAnalystRegistry({
            getBaselineCollectionTag: () => currentTag,
            getBaselineLapRecord: () => cachedRecord,
        });

        const missingResult = await registry.analyze_live_recorded_analysis(
            { limit: 5 },
            { sendObservation: jest.fn() },
        );

        expect(missingResult).toMatchObject({
            status: 'error',
            error: 'baseline_lap_record_required',
            message: expect.stringContaining('recorded baseline lap'),
        });
        expect(apiService.post).not.toHaveBeenCalled();

        cachedRecord = {
            id: 'brands_hatch:Ferrari 296:0:2',
            lap: 0,
            captured_at: 1,
            track: 'brands_hatch',
            car: 'Ferrari 296',
            sample_count: 2,
            snapshot: {
                baseline_ready: true,
                baseline_progress_percent: 100,
            },
            records: [
                { Graphics_completed_laps: 0, Graphics_normalized_car_position: 0.01 },
                { Graphics_completed_laps: 0, Graphics_normalized_car_position: 0.99 },
            ],
        };

        await expect(registry.analyze_live_recorded_analysis(
            { limit: 5 },
            { sendObservation: jest.fn() },
        )).resolves.toMatchObject({
            status: 'ready',
            ui_output: {
                source: 'baseline_lap_record',
                baseline: {
                    id: cachedRecord.id,
                    lap: 0,
                    sample_count: 2,
                },
                analysis: {
                    samples_analyzed: 2,
                    segment_count: 1,
                    returned_segment_count: 1,
                    segments: [
                        expect.objectContaining({
                            start_position: 0.01,
                            end_position: 0.99,
                        }),
                    ],
                },
            },
        });
        expect(apiService.post).toHaveBeenCalledWith('/racing-session/analyze-live-recorded-analysis', {
            track: cachedRecord.track,
            car: cachedRecord.car,
            baseline_lap: cachedRecord.lap,
            records: cachedRecord.records,
        }, { timeout: 120000 });
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
                    { type: 'request', status: 'complete', title: 'Complete the first task.' },
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
        expect(result).toMatchObject({
            status: 'advanced',
            final: true,
            ui_output: {
                status: 'advanced',
                current_step: 1,
                step: 'Run the worker.',
            },
            tool_name: 'advance_plan_step',
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
            ui_output: expect.objectContaining({
                snapshot: expect.objectContaining({
                    baseline_ready: false,
                }),
            }),
            ai_output: expect.objectContaining({
                name: 'get_live_focus_section',
                status: 'error',
                error: 'baseline_collection_incomplete',
            }),
        });
        expect((focusResult.ai_output as any)).not.toHaveProperty('snapshot');
        expect(telemetryResult).toMatchObject({
            status: 'error',
            error: 'baseline_collection_incomplete',
            ui_output: expect.objectContaining({
                snapshot: expect.objectContaining({
                    baseline_ready: false,
                }),
            }),
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
        const setBaselineCollectionEnabled = jest.fn();
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
            setBaselineCollectionEnabled,
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
        expect(setBaselineCollectionEnabled).not.toHaveBeenCalled();
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
                type: 'tool_call',
                status: 'complete' as const,
                title: 'Request recorded-session classifier',
                name: 'analyze_live_recorded_analysis',
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
                        type: 'tool_call',
                        status: 'complete',
                        title: 'Collect a clean baseline lap',
                        name: 'collect_live_baseline',
                    },
                    {
                        type: 'tool_call',
                        status: 'pending',
                        title: 'Request recorded-session classifier',
                        name: 'analyze_live_recorded_analysis',
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
            ui_output: expect.objectContaining({
                current_request: 1,
            }),
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
        (apiService.post as jest.Mock).mockResolvedValueOnce({
            data: {
                status: 'success',
                session_id: 'live-baseline',
                samples_analyzed: 2,
                segment_count: 1,
                expert_time_available: true,
                segments: [
                    {
                        id: 'live-segment-1',
                        start_index: 0,
                        end_index: 1,
                        main_label_id: 'brands_hatch2',
                        labels: ['brands_hatch2', 'late_brake'],
                        child_segments: [
                            {
                                start_index: 0,
                                end_index: 2,
                                labels: ['late_brake'],
                                time_gap: {
                                    start_ms: 0,
                                    end_ms: 125,
                                    delta_ms: 125,
                                },
                            },
                        ],
                    },
                ],
            },
        });
        const advanceProcedurePlanStep = jest.fn(() => ({
            status: 'complete',
            current_request: 1,
            request: {
                type: 'tool_call',
                status: 'complete' as const,
                title: 'Request recorded-session classifier',
                name: 'analyze_live_recorded_analysis',
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
                        type: 'tool_call',
                        status: 'complete' as const,
                        title: 'Collect a clean baseline lap',
                        name: 'collect_live_baseline',
                    },
                    {
                        type: 'tool_call',
                        status: 'pending',
                        title: 'Request recorded-session classifier',
                        name: 'analyze_live_recorded_analysis',
                        payload: { force: false },
                    },
                ],
                currentStep: 1,
                sourceEvent: 'live_analysis_plan_started',
            }),
        });
        const sendObservation = jest.fn();
        sessionIntelligence.onLiveAnalystObservation(sendObservation);

        const result = await registry.analyze_live_recorded_analysis(
            { limit: 8 },
            { sendObservation: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'ready',
            ui_output: expect.objectContaining({
                source: 'baseline_lap_record',
                analysis: expect.objectContaining({
                    expert_time_available: true,
                    segments: [
                        expect.objectContaining({
                            start_position: 0.98,
                            end_position: 0.03,
                            child_segments: [
                                expect.objectContaining({
                                    start_position: 0.98,
                                    end_position: 0.03,
                                    time_gap: {
                                        start_ms: 0,
                                        end_ms: 125,
                                        delta_ms: 125,
                                    },
                                }),
                            ],
                        }),
                    ],
                }),
            }),
        });
        expect((result.ai_output as any)).toMatchObject({
            segments: [
                expect.objectContaining({
                    start_position: 0.98,
                    end_position: 0.03,
                }),
            ],
        });
        expect(((result.ui_output as any).analysis.segments[0])).not.toHaveProperty('start_index');
        expect(((result.ui_output as any).analysis.segments[0])).not.toHaveProperty('end_index');
        expect(apiService.post).toHaveBeenCalledWith(
            '/racing-session/analyze-live-recorded-analysis',
            expect.objectContaining({
                track: 'brands_hatch',
                car: 'Ferrari 296',
                baseline_lap: 0,
                records: expect.any(Array),
            }),
            expect.objectContaining({ timeout: 120000 }),
        );
        expect(analysisContext.runRecordedAiAnalysis).not.toHaveBeenCalled();
        expect(advanceProcedurePlanStep).not.toHaveBeenCalled();
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'recorded_analysis_ready',
            analysis: expect.objectContaining({
                status: 'ready',
                source: 'baseline_lap_record',
                analysis: expect.objectContaining({
                    expert_time_available: true,
                    segments: expect.arrayContaining([
                        expect.objectContaining({
                            start_position: 0.98,
                            end_position: 0.03,
                            child_segments: [
                                expect.objectContaining({
                                    start_position: 0.98,
                                    end_position: 0.03,
                                    time_gap: {
                                        start_ms: 0,
                                        end_ms: 125,
                                        delta_ms: 125,
                                    },
                                }),
                            ],
                        }),
                    ]),
                }),
            }),
        }));
    });

    it('requires cached baseline records when the subscribed classifier request is advanced', async () => {
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
                        type: 'tool_call',
                        status: 'complete',
                        title: 'Collect a clean baseline lap',
                        name: 'collect_live_baseline',
                    },
                    {
                        type: 'tool_call',
                        status: 'pending',
                        title: 'Request recorded-session classifier',
                        name: 'analyze_live_recorded_analysis',
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
        const result = await registry.analyze_live_recorded_analysis(
            { limit: 8 },
            { sendObservation: toolContextSendObservation },
        );

        expect(result).toMatchObject({
            status: 'error',
            error: 'baseline_lap_record_required',
            ui_output: expect.objectContaining({
                snapshot: expect.objectContaining({
                    baseline_ready: false,
                }),
            }),
        });
        expect(sendObservation).toHaveBeenCalledWith(expect.objectContaining({
            event: 'baseline_lap_record_required',
            message: expect.stringContaining('recorded baseline lap'),
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

        expect(result).toMatchObject({ status: 'ready' });
        expect((result.ui_output as any).section).toMatchObject({
            name: 'T1 Paddock Hill Bend',
        });
        expect((result.ui_output as any).rows).toEqual(expect.any(Array));
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

        expect(result).toMatchObject({ status: 'ready' });
        expect((result.ui_output as any).focus).toMatchObject({
            section: {
                name: 'T2 Druids',
            },
            show_map_arguments: {
                section_start: 0.11,
                section_end: 0.18,
                section_label: 'T2 Druids',
            },
        });
    });
});

