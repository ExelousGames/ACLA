jest.mock('views/lap-analysis/visualization/VisualizationRegistry', () => ({
    visualizationController: {
        getVisualizationAssistantContext: jest.fn(),
        openVisualization: jest.fn(),
        closeVisualization: jest.fn(),
        invokeVisualizationControl: jest.fn(),
        getCurrentInstances: jest.fn(() => []),
        executeCommand: jest.fn(),
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
import { RecordingState } from 'views/lap-analysis/recording-state';
import { getToolEnvelopeUiOutput, type ToolOutputEnvelope } from '../ai-tool-base';
import { visualizationController } from 'views/lap-analysis/visualization/VisualizationRegistry';

const labelNames: Record<string, string> = {
    brands_hatch: 'Brands Hatch',
    brands_hatch1: 'Paddock Hill Bend',
    brands_hatch2: 'Druids',
    MSP: 'Mistake (Practice)',
    MSP1: 'Initiate brake too late',
    MSR: 'Mistake (Racing)',
    MSR1: 'Failed overtake attempt',
    monza: 'Monza',
    monza1: 'Rettifilo',
};

const getUiOutput = (result: ToolOutputEnvelope) => getToolEnvelopeUiOutput(result) as any;

const categories: Record<string, string[]> = {
    brands_hatch: ['brands_hatch1', 'brands_hatch2'],
    monza: ['monza1'],
    MSP: ['MSP1'],
    MSR: ['MSR1'],
};

beforeEach(() => {
    jest.clearAllMocks();
});

const createRegistry = () => createAiCommandRegistry({
    sessionMode: 'live',
    recordingState: RecordingState.RECORDING,
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
            ranges: [{
                id: 'r1',
                start_position: 0.1,
                end_position: 0.2,
                lifecycle_status: 'pending',
                child_segments: [{
                    labels: ['entry', 'apex'],
                    start_index: 10,
                    end_index: 20,
                }],
            }],
            created_at: 1,
            updated_at: 1,
        };
        const setLiveRangeTracker = jest.fn(() => ({ status: 'ready', tracker }));
        const updateLiveRangeTracker = jest.fn(() => ({ status: 'ready', tracker }));
        const getLiveRangeTracker = jest.fn(() => ({ status: 'ready', tracker }));
        const registry = createAiCommandRegistry({
            sessionMode: 'live',
            recordingState: RecordingState.RECORDING,
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
        const toolContext = { sendToolStatus: jest.fn() };

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
        expect(getUiOutput(setResult)).toMatchObject({ status: 'ready', tracker });
        expect(getUiOutput(updateResult)).toMatchObject({ status: 'ready', tracker });
        expect(getUiOutput(getResult)).toMatchObject({ status: 'ready', tracker });
        expect(setResult).toMatchObject({
            output: {
                name: 'set_live_range_tracker',
                status: 'ready',
                tracker_status: 'open',
            },
        });
        expect((setResult.output as any)).not.toHaveProperty('tracker');
        expect((setResult.output as any)).not.toHaveProperty('ranges');
        expect((setResult.output as any)).not.toHaveProperty('range_count');
        expect(updateResult).toMatchObject({
            output: {
                name: 'update_live_range_tracker',
                status: 'ready',
                range_count: 1,
            },
        });
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
            recordingState: RecordingState.RECORDING,
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
            { sendToolStatus: jest.fn() },
        );
        const tireResult = await registry.query_telemetry_metric(
            {
                fields: "['TirePressureFL', 'TirePressureFR', 'TirePressureRL', 'TirePressureRR']",
                scope: { type: 'now' },
                reduce: 'avg',
            },
            { sendToolStatus: jest.fn() },
        );
        const bareTyreResult = await registry.query_telemetry_metric(
            {
                fields: "['tyre_pressure_front_left', 'tyre_pressure_front_right', 'tyre_pressure_rear_left', 'tyre_pressure_rear_right']",
                scope: { type: 'now' },
                reduce: 'avg',
            },
            { sendToolStatus: jest.fn() },
        );

        expect(fuelResult).toMatchObject({
            status: 'complete',
            output: {
                name: 'query_telemetry_metric',
                status: 'complete',
                values: {
                    Physics_fuel: 38,
                },
            },
        });
        expect(getUiOutput(fuelResult)).toMatchObject({
            Physics_fuel: 38,
        });
        expect((fuelResult.output as any)).not.toHaveProperty('ok');
        expect((fuelResult.output as any).values).not.toHaveProperty('ok');
        expect(tireResult).toMatchObject({ status: 'complete' });
        expect(getUiOutput(tireResult)).toMatchObject({
            Physics_wheel_pressure_front_left: 26.1,
            Physics_wheel_pressure_front_right: 26.2,
            Physics_wheel_pressure_rear_left: 25.9,
            Physics_wheel_pressure_rear_right: 26,
        });
        expect(bareTyreResult).toMatchObject({ status: 'complete' });
        expect(getUiOutput(bareTyreResult)).toMatchObject({
            Physics_wheel_pressure_front_left: 26.1,
            Physics_wheel_pressure_front_right: 26.2,
            Physics_wheel_pressure_rear_left: 25.9,
            Physics_wheel_pressure_rear_right: 26,
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
            recordingState: RecordingState.RECORDING,
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
            { sendToolStatus: jest.fn() },
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
            recordingState: RecordingState.RECORDING,
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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

describe('ai command registry live recording gate', () => {
    it('rejects live-only tools when live mode is not actively recording', async () => {
        const sessionIntelligence = new SessionIntelligence();
        sessionIntelligence.tick({ Physics_fuel: 20 });
        const registry = createAiCommandRegistry({
            sessionMode: 'live',
            recordingState: RecordingState.READY,
            sessionIntelligence,
            opportunityAgentState: {
                intervalId: null,
                inFlight: false,
                lastAlertKey: null,
                lastAlertAt: 0,
            },
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            startAgentSession: jest.fn(),
            getOpportunityTelemetryRows: () => [],
            setLiveRangeTracker: jest.fn(),
        });

        await expect(registry.query_telemetry_metric(
            { fields: ['Physics_fuel'], scope: { type: 'now' } },
            { sendToolStatus: jest.fn() },
        )).resolves.toMatchObject({
            error: 'non_live_context_live_tools_unavailable',
        });
        await expect(registry.start_agent_session(
            { agent_mode: 'live_performance_analyst' },
            { sendToolStatus: jest.fn() },
        )).resolves.toMatchObject({
            status: 'error',
            error: 'non_live_context_live_tools_unavailable',
        });
        await expect(registry.set_live_range_tracker(
            { ranges: [] },
            { sendToolStatus: jest.fn() },
        )).resolves.toMatchObject({
            error: 'non_live_context_live_tools_unavailable',
        });
    });

    it('rejects live agent runtime startup unless shared state is recording', async () => {
        const startTrackGuide = jest.fn();
        const result = await startAgentRuntime(
            'track_guide',
            {
                sessionMode: 'live',
                recordingState: RecordingState.READY,
                opportunityAgentState: {
                    intervalId: null,
                    inFlight: false,
                    lastAlertKey: null,
                    lastAlertAt: 0,
                },
                startTrackGuide,
                setTrackGuideEnabled: jest.fn(),
                getOpportunityTelemetryRows: () => [],
            },
            {},
            { sendToolStatus: jest.fn() },
        );

        expect(result).toEqual({ error: 'non_live_context_live_tools_unavailable' });
        expect(startTrackGuide).not.toHaveBeenCalled();
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
            parent_segment_count: 1,
            expert_reference_data: [],
            segments: [
                {
                    id: 'segment-1',
                    start_index: 10,
                    end_index: 40,
                    labels: ['MSP', 'MSP1'],
                    track_section: 'brands_hatch1',
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
                    trackSection: 'Paddock Hill Bend',
                    labels: ['Mistake (Practice)', 'Initiate brake too late'],
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
            { sendToolStatus: jest.fn() },
        );

        expect(analysisContext.runRecordedAiAnalysis).toHaveBeenCalledWith({ force: true });
        expect(result).toMatchObject({
            status: 'ready',
        });
        expect(result.ui_output).toMatchObject({
            status: 'ready',
            session_id: 'session-1',
            analysis: {
                segments: [
                    {
                        id: 'segment-1',
                        track_section: 'Paddock Hill Bend',
                        labels: ['Mistake (Practice)', 'Initiate brake too late'],
                    },
                ],
            },
        });
    });

    it('returns cached shared recorded analysis without running analysis again', async () => {
        const { analysisContext, registry } = createRecordedRegistry();

        const result = await registry.get_recorded_session_analysis(
            { limit: 1 },
            { sendToolStatus: jest.fn() },
        );

        expect(analysisContext.runRecordedAiAnalysis).not.toHaveBeenCalled();
        expect(result).toMatchObject({
            status: 'ready',
        });
        expect(result.ui_output).toMatchObject({
            status: 'ready',
            analysis: {
                samples_analyzed: 120,
            },
        });
    });

    it('returns selected recorded session playback context', async () => {
        const { registry } = createRecordedRegistry();

        const result = await registry.get_recorded_session_context(
            {},
            { sendToolStatus: jest.fn() },
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
                    trackSection: 'Paddock Hill Bend',
                    labels: ['Mistake (Practice)', 'Initiate brake too late'],
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
                parent_segment_count: 1,
                expert_reference_data: [],
                segments: [
                    {
                        id: 'brands_hatch2:10-30',
                        labels: ['MSP', 'MSP1'],
                        track_section: 'brands_hatch2',
                        start_index: 10,
                        end_index: 30,
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
            lastToolStatusKey: null,
            lastToolStatusAt: 0,
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
            recordingState: RecordingState.RECORDING,
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

    it('returns cached baseline completion immediately', async () => {
        const setBaselineCollectionEnabled = jest.fn();
        const restartBaselineCollection = jest.fn();
        const { registry, livePerformanceAnalystState } = createLiveAnalystRegistry({
            setBaselineCollectionEnabled,
            restartBaselineCollection,
        });

        const result = await registry.collect_live_baseline(
            {},
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
        );

        expect(restartBaselineCollection).toHaveBeenCalledTimes(1);
        expect(setBaselineCollectionEnabled).toHaveBeenCalledWith(true);
        expect(result).toMatchObject({
            status: 'complete',
            progress_percent: 0,
            message: 'Baseline collection restart completed.',
            final: true,
            tool_name: 'restart_live_baseline',
        });
        expect(getUiOutput(result)).toMatchObject({
            status: 'complete',
            progress_percent: 0,
            message: 'Baseline collection restart completed.',
        });
    });

    it('keeps collect live baseline non-final while collection continues in the tracker', async () => {
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
        const setBaselineCollectionEnabled = jest.fn();
        const { registry } = createLiveAnalystRegistry({
            getBaselineCollectionTag: () => collectingTag,
            getBaselineLapRecord: () => null,
            setBaselineCollectionEnabled,
        });

        const result = await registry.collect_live_baseline(
            { timeout_seconds: 30 },
            { sendToolStatus: jest.fn() },
        );

        expect(result).toMatchObject({
            status: 'started',
            final: false,
            tool_name: 'collect_live_baseline',
            ui_output: {
                status: 'started',
                progress_percent: 35,
                car: 'Ferrari 296',
                track: 'brands_hatch',
                message: 'Baseline collection started.',
            },
            ai_output: {
                name: 'collect_live_baseline',
                status: 'started',
                message: 'Baseline collection started.',
            },
        });
        expect(setBaselineCollectionEnabled).toHaveBeenCalledWith(true);
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
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
            recordingState: RecordingState.RECORDING,
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
                lastToolStatusKey: null,
                lastToolStatusAt: 0,
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
            { sendToolStatus: jest.fn() },
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
            recordingState: RecordingState.RECORDING,
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
                lastToolStatusKey: null,
                lastToolStatusAt: 0,
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
            { sendToolStatus: jest.fn() },
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

    it('waits for a baseline, then opens filtered analysis results with fallback label order', async () => {
        (visualizationController.openVisualization as jest.Mock).mockReturnValue({
            success: true,
            chartId: 'analysis-results-chart',
            chartType: 'analysis-results',
        });
        (apiService.post as jest.Mock).mockResolvedValueOnce({
            data: {
                status: 'success',
                session_id: 'live-baseline',
                samples_analyzed: 2,
                parent_segment_count: 3,
                segments: [
                    {
                        id: 'parent-only-practice',
                        start_index: 0,
                        end_index: 1,
                        labels: ['MSP'],
                        track_section: 'brands_hatch2',
                    },
                    {
                        id: 'parent-only-racing',
                        start_index: 0,
                        end_index: 1,
                        labels: ['MSR'],
                        track_section: 'brands_hatch2',
                    },
                    {
                        id: 'live-segment-1',
                        start_index: 0,
                        end_index: 1,
                        labels: ['MSP', 'FALLBACK_CHILD'],
                        track_section: 'brands_hatch2',
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
            getCategoryLabels: undefined,
        });

        const missingResult = await registry.analyze_live_recorded_analysis(
            { limit: 5 },
            { sendToolStatus: jest.fn() },
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

        const readyResult = await registry.analyze_live_recorded_analysis(
            { limit: 5 },
            { sendToolStatus: jest.fn() },
        );

        expect(readyResult).toMatchObject({ status: 'ready' });
        expect(getUiOutput(readyResult as ToolOutputEnvelope)).toMatchObject({
            chartId: 'analysis-results-chart',
            totalResultCount: 1,
            source: 'baseline_lap_record',
            baseline: {
                id: cachedRecord.id,
                lap: 0,
                sample_count: 2,
            },
            analysis: {
                samples_analyzed: 2,
                segments: [
                    expect.objectContaining({
                        id: 'parent-only-practice',
                        labels: ['Mistake (Practice)'],
                    }),
                    expect.objectContaining({
                        id: 'parent-only-racing',
                        labels: ['Mistake (Racing)'],
                    }),
                    expect.objectContaining({
                        id: 'live-segment-1',
                        track_section: 'Druids',
                        labels: ['Mistake (Practice)', 'FALLBACK_CHILD'],
                        start_position: 0.01,
                        end_position: 0.99,
                    }),
                ],
            },
        });
        expect(apiService.post).toHaveBeenCalledWith('/racing-session/analyze-live-recorded-analysis', {
            track: cachedRecord.track,
            car: cachedRecord.car,
            baseline_lap: cachedRecord.lap,
            records: cachedRecord.records,
        }, { timeout: 120000 });
        expect(visualizationController.openVisualization).toHaveBeenCalledWith(
            'analysis-results',
            {
                elements: [expect.objectContaining({
                    id: 'live-segment-1',
                    labels: ['Mistake (Practice)', 'FALLBACK_CHILD'],
                    section: 'Druids',
                })],
            },
        );
    });

    it('filters parent-only mistakes with taxonomy when updating analysis results', async () => {
        (visualizationController.getCurrentInstances as jest.Mock).mockReturnValueOnce([{
            id: 'existing-results',
            type: 'analysis-results',
            data: { elements: [{ id: 'old', labels: ['Old'] }] },
        }]);
        (apiService.post as jest.Mock).mockResolvedValueOnce({
            data: {
                status: 'success',
                session_id: 'live-baseline',
                samples_analyzed: 3,
                segments: [
                    { id: 'parent-only-practice', start_index: 0, end_index: 1, labels: ['MSP'] },
                    { id: 'parent-only-racing', start_index: 0, end_index: 1, labels: ['MSR'] },
                    { id: 'practice', start_index: 0, end_index: 1, labels: ['MSP', 'MSP1'] },
                    { id: 'racing', start_index: 0, end_index: 1, labels: ['MSR', 'MSR1'] },
                    { id: 'wrong-child', start_index: 0, end_index: 1, labels: ['MSP', 'MSR1'] },
                    { id: 'adherence', start_index: 0, end_index: 1, labels: ['EXPERT_ADHERENCE'] },
                    { id: 'recovery', start_index: 1, end_index: 2, labels: ['RECOVERY'] },
                    { id: 'future', start_index: 1, end_index: 2, labels: ['FUTURE_LABEL'] },
                ],
            },
        });
        const { registry } = createLiveAnalystRegistry();

        const result = await registry.analyze_live_recorded_analysis(
            { limit: 1 },
            { sendToolStatus: jest.fn() },
        );
        const uiOutput = getUiOutput(result as ToolOutputEnvelope);

        expect(uiOutput).toMatchObject({
            chartId: 'existing-results',
            totalResultCount: 5,
            analysis: { segments: [expect.objectContaining({ id: 'parent-only-practice' })] },
        });
        expect(visualizationController.openVisualization).not.toHaveBeenCalled();
        expect(visualizationController.executeCommand).toHaveBeenCalledWith({
            action: 'update',
            id: 'existing-results',
            data: {
                elements: [
                    expect.objectContaining({
                        id: 'practice',
                        labels: ['Mistake (Practice)', 'Initiate brake too late'],
                    }),
                    expect.objectContaining({
                        id: 'racing',
                        labels: ['Mistake (Racing)', 'Failed overtake attempt'],
                    }),
                    expect.objectContaining({ id: 'adherence', labels: ['EXPERT_ADHERENCE'] }),
                    expect.objectContaining({ id: 'recovery', labels: ['RECOVERY'] }),
                    expect.objectContaining({ id: 'future', labels: ['FUTURE_LABEL'] }),
                ],
            },
        });
        expect((result.output as any)).toMatchObject({
            chart_id: 'existing-results',
            total_result_count: 5,
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
                    { type: 'request', status: 'complete', title: 'Complete the first task.' },
                    { type: 'request', status: 'pending', title: 'Run the worker.' },
                ],
                currentStep: 0,
                sourceEvent: 'procedure_plan_started',
            }),
        });

        const result = await registry.advance_plan_step(
            { reason: 'first task complete' },
            { sendToolStatus: jest.fn() },
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
            recordingState: RecordingState.RECORDING,
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
                lastToolStatusKey: null,
                lastToolStatusAt: 0,
                lastSpokenAt: 0,
            },
            startTrackGuide: jest.fn(),
            setTrackGuideEnabled: jest.fn(),
            getOpportunityTelemetryRows: () => [],
        };
        const registry = createAiCommandRegistry(context);

        const focusResult = await registry.get_live_focus_section(
            {},
            { sendToolStatus: jest.fn() },
        );
        const telemetryResult = await registry._get_live_section_telemetry(
            { section_name: 'T2 Druids', lap: 0 },
            { sendToolStatus: jest.fn() },
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
            lastToolStatusKey: null,
            lastToolStatusAt: 0,
            lastSpokenAt: 0,
        };
        const setBaselineCollectionEnabled = jest.fn();
        const context: AiCommandRegistryContext = {
            sessionMode: 'live',
            recordingState: RecordingState.RECORDING,
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
        const sendToolStatus = jest.fn();
        sessionIntelligence.onLiveAnalystToolStatus(sendToolStatus);

        const result = await startAgentRuntime(
            'live_performance_analyst',
            context,
            {},
            { sendToolStatus },
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
        expect(sendToolStatus).toHaveBeenCalledTimes(1);
        expect(sendToolStatus).toHaveBeenCalledWith(expect.objectContaining({
            event: 'live_analysis_plan_started',
            message: expect.stringContaining('Collect a baseline first'),
            snapshot: expect.objectContaining({
                baseline_ready: false,
            }),
        }));
        const startupPlanToolStatus = sendToolStatus.mock.calls.find(([payload]) => (
            payload.event === 'live_analysis_plan_started'
        ))?.[0];
        expect(startupPlanToolStatus).not.toHaveProperty('goal');
        expect(startupPlanToolStatus).not.toHaveProperty('requests');
        expect(startupPlanToolStatus).not.toHaveProperty('current_request');
        expect(startupPlanToolStatus).not.toHaveProperty('internal_tool_hint');
        expect(startupPlanToolStatus).not.toHaveProperty('sections');

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
            recordingState: RecordingState.RECORDING,
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
        const sendToolStatus = jest.fn();
        sessionIntelligence.onLiveAnalystToolStatus(sendToolStatus);
        const toolContextSendToolStatus = jest.fn();

        const startResult = await startAgentRuntime(
            'live_performance_analyst',
            context,
            {},
            { sendToolStatus: toolContextSendToolStatus },
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
        expect(sendToolStatus).not.toHaveBeenCalledWith(expect.objectContaining({
            event: 'baseline_classifier_request_ready',
        }));
        expect(analysisContext.runRecordedAiAnalysis).not.toHaveBeenCalled();

        const result = await planRegistry.advance_plan_step(
            { reason: 'baseline complete' },
            { sendToolStatus: toolContextSendToolStatus },
        );

        expect(result).toMatchObject({
            status: 'advanced',
            ui_output: expect.objectContaining({
                current_request: 1,
            }),
        });
        expect(advanceProcedurePlanStep).toHaveBeenCalledWith('baseline complete');
        expect(analysisContext.runRecordedAiAnalysis).not.toHaveBeenCalled();
        expect(sendToolStatus).not.toHaveBeenCalledWith(expect.objectContaining({
            event: 'recorded_analysis_ready',
        }));
        expect(toolContextSendToolStatus).not.toHaveBeenCalledWith(expect.objectContaining({
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
                parent_segment_count: 1,
                expert_time_available: true,
                segments: [
                    {
                        id: 'live-segment-1',
                        start_index: 0,
                        end_index: 1,
                        labels: ['MSP', 'MSP1'],
                        track_section: 'brands_hatch2',
                        time_gap: {
                            start_ms: 0,
                            end_ms: 125,
                            delta_ms: 125,
                        },
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
        const sendToolStatus = jest.fn();
        sessionIntelligence.onLiveAnalystToolStatus(sendToolStatus);

        const result = await registry.analyze_live_recorded_analysis(
            { limit: 8 },
            { sendToolStatus: jest.fn() },
        );
        const uiOutput = getUiOutput(result as ToolOutputEnvelope);

        expect(result).toMatchObject({
            status: 'ready',
        });
        expect(uiOutput).toMatchObject({
            source: 'baseline_lap_record',
            analysis: expect.objectContaining({
                expert_time_available: true,
                segments: [
                    expect.objectContaining({
                        track_section: 'Druids',
                        labels: ['Mistake (Practice)', 'Initiate brake too late'],
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
        });
        expect((result.output as any)).toMatchObject({
            segments: [
                expect.objectContaining({
                    track_section: 'Druids',
                    labels: ['Mistake (Practice)', 'Initiate brake too late'],
                    start_position: 0.98,
                    end_position: 0.03,
                }),
            ],
        });
        expect(uiOutput.analysis.segments[0]).not.toHaveProperty('start_index');
        expect(uiOutput.analysis.segments[0]).not.toHaveProperty('end_index');
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
        expect(sendToolStatus).not.toHaveBeenCalledWith(expect.objectContaining({
            event: 'recorded_analysis_ready',
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
            lastToolStatusKey: null,
            lastToolStatusAt: 0,
            lastSpokenAt: 0,
        };
        const context: AiCommandRegistryContext = {
            sessionMode: 'live',
            recordingState: RecordingState.RECORDING,
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
        const sendToolStatus = jest.fn();
        sessionIntelligence.onLiveAnalystToolStatus(sendToolStatus);
        const toolContextSendToolStatus = jest.fn();

        await startAgentRuntime(
            'live_performance_analyst',
            context,
            {},
            { sendToolStatus: toolContextSendToolStatus },
        );
        const result = await registry.analyze_live_recorded_analysis(
            { limit: 8 },
            { sendToolStatus: toolContextSendToolStatus },
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
        expect(sendToolStatus).toHaveBeenCalledWith(expect.objectContaining({
            event: 'baseline_lap_record_required',
            message: expect.stringContaining('recorded baseline lap'),
        }));
        expect(sendToolStatus).toHaveBeenCalledWith(expect.objectContaining({
            event: 'live_analysis_plan_started',
            snapshot: expect.objectContaining({
                baseline_ready: true,
            }),
        }));
        const startupToolStatus = sendToolStatus.mock.calls.find(([payload]) => (
            payload.event === 'live_analysis_plan_started'
        ))?.[0];
        expect(startupToolStatus).not.toHaveProperty('goal');
        expect(startupToolStatus).not.toHaveProperty('requests');
        expect(startupToolStatus).not.toHaveProperty('current_request');
        expect(toolContextSendToolStatus).not.toHaveBeenCalledWith(expect.objectContaining({
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
            { sendToolStatus: jest.fn() },
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
            { sendToolStatus: jest.fn() },
        );

        const result = await registry.get_live_focus_section(
            {},
            { sendToolStatus: jest.fn() },
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
