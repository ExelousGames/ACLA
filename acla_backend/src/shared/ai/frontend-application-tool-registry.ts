export const FRONTEND_APPLICATION_QUERY_SCOPE_SCHEMA = {
    type: 'object',
    properties: {
        type: {
            type: 'string',
            enum: ['now', 'last_seconds', 'event', 'lap', 'range'],
        },
        seconds: { type: 'number' },
        eventType: { type: 'string', enum: ['CORNER', 'STRAIGHT', 'CRASHED', 'OVERTAKE'] },
        which: { type: 'string', enum: ['last', 'current'] },
        lap: {
            oneOf: [
                { type: 'string', enum: ['current', 'last'] },
                { type: 'integer' },
            ],
        },
        start: { type: 'integer' },
        end: { type: 'integer' },
    },
    required: ['type'],
} as const;

const TELEMETRY_METRIC_PHYSICS_FIELDS = [
    'Physics_pad_life_front_left',
    'Physics_wheel_angular_s_front_left',
    'Physics_brake_pressure_rear_left',
    'Physics_starter_engine_on',
    'Physics_is_engine_running',
    'Physics_tyre_contact_point_rear_right_z',
    'Physics_tyre_contact_normal_rear_right_x',
    'Physics_slip_angle_rear_left',
    'Physics_tyre_core_temp_front_left',
    'Physics_suspension_damage_rear_left',
    'Physics_tyre_contact_heading_rear_left_x',
    'Physics_rear_brake_compound',
    'Physics_local_angular_vel_x',
    'Physics_final_ff',
    'Physics_disc_life_rear_right',
    'Physics_tyre_core_temp_front_right',
    'Physics_tyre_contact_normal_front_right_z',
    'Physics_g_vibration',
    'Physics_brake_bias',
    'Physics_tyre_contact_point_front_right_x',
    'Physics_pad_life_front_right',
    'Physics_local_velocity_x',
    'Physics_brake_temp_rear_left',
    'Physics_tyre_contact_point_rear_left_y',
    'Physics_heading',
    'Physics_tyre_contact_heading_rear_right_z',
    'Physics_fuel',
    'Physics_tyre_contact_heading_front_left_z',
    'Physics_slip_vibration',
    'Physics_disc_life_front_left',
    'Physics_suspension_travel_front_right',
    'Physics_disc_life_rear_left',
    'Physics_slip_angle_front_right',
    'Physics_g_force_x',
    'Physics_rpm',
    'Physics_g_force_z',
    'Physics_car_damage_rear',
    'Physics_slip_ratio_front_left',
    'Physics_tyre_contact_heading_front_left_y',
    'Physics_tyre_contact_point_rear_right_y',
    'Physics_velocity_x',
    'Physics_tc',
    'Physics_wheel_pressure_front_right',
    'Physics_suspension_travel_front_left',
    'Physics_tyre_contact_heading_rear_right_y',
    'Physics_clutch',
    'Physics_road_temp',
    'Physics_wheel_pressure_front_left',
    'Physics_local_velocity_z',
    'Physics_wheel_angular_s_rear_right',
    'Physics_brake_temp_front_right',
    'Physics_tyre_contact_point_rear_left_x',
    'Physics_tyre_contact_heading_front_left_x',
    'Physics_air_temp',
    'Physics_g_force_y',
    'Physics_autoshifter_on',
    'Physics_brake_temp_rear_right',
    'Physics_abs_vibration',
    'Physics_gear',
    'Physics_wheel_pressure_rear_right',
    'Physics_tyre_contact_point_rear_left_z',
    'Physics_tyre_contact_heading_front_right_y',
    'Physics_suspension_travel_rear_right',
    'Physics_local_angular_vel_z',
    'Physics_tyre_contact_point_front_left_z',
    'Physics_brake_pressure_rear_right',
    'Physics_kerb_vibration',
    'Physics_tyre_contact_heading_rear_right_x',
    'Physics_tyre_contact_heading_front_right_z',
    'Physics_tyre_contact_heading_rear_left_z',
    'Physics_wheel_slip_rear_left',
    'Physics_slip_ratio_front_right',
    'Physics_tyre_contact_point_front_right_y',
    'Physics_steer_angle',
    'Physics_is_ai_controlled',
    'Physics_car_damage_left',
    'Physics_wheel_pressure_rear_left',
    'Physics_wheel_angular_s_rear_left',
    'Physics_pad_life_rear_right',
    'Physics_ignition_on',
    'Physics_car_damage_right',
    'Physics_tyre_contact_normal_rear_right_z',
    'Physics_velocity_z',
    'Physics_wheel_slip_rear_right',
    'Physics_tyre_contact_point_front_left_y',
    'Physics_tyre_core_temp_rear_left',
    'Physics_tyre_contact_point_front_right_z',
    'Physics_brake',
    'Physics_gas',
    'Physics_speed_kmh',
    'Physics_slip_angle_front_left',
    'Physics_slip_ratio_rear_right',
    'Physics_brake_pressure_front_right',
    'Physics_abs',
    'Physics_pitch',
    'Physics_tyre_contact_normal_rear_left_z',
    'Physics_roll',
    'Physics_tyre_contact_normal_rear_left_x',
    'Physics_pad_life_rear_left',
    'Physics_tyre_contact_normal_front_right_y',
    'Physics_local_angular_vel_y',
    'Physics_tyre_contact_normal_front_left_x',
    'Physics_suspension_travel_rear_left',
    'Physics_brake_temp_front_left',
    'Physics_slip_angle_rear_right',
    'Physics_slip_ratio_rear_left',
    'Physics_wheel_slip_front_right',
    'Physics_tyre_contact_heading_front_right_x',
    'Physics_suspension_damage_rear_right',
    'Physics_tyre_core_temp_rear_right',
    'Physics_tyre_contact_normal_rear_right_y',
    'Physics_tyre_contact_heading_rear_left_y',
    'Physics_disc_life_front_right',
    'Physics_wheel_angular_s_front_right',
    'Physics_tyre_contact_point_front_left_x',
    'Physics_tyre_contact_normal_front_right_x',
    'Physics_car_damage_front',
    'Physics_turbo_boost',
    'Physics_local_velocity_y',
    'Physics_water_temp',
    'Physics_tyre_contact_normal_front_left_z',
    'Physics_car_damage_center',
    'Physics_suspension_damage_front_left',
    'Physics_velocity_y',
    'Physics_tyre_contact_normal_front_left_y',
    'Physics_packed_id',
    'Physics_wheel_slip_front_left',
    'Physics_front_brake_compound',
    'Physics_suspension_damage_front_right',
    'Physics_brake_pressure_front_left',
    'Physics_tyre_contact_point_rear_right_x',
    'Physics_tyre_contact_normal_rear_left_y',
] as const;

const TELEMETRY_METRIC_GRAPHICS_FIELDS = [
    'Graphics_ideal_line_on',
    'Graphics_is_valid_lap',
    'Graphics_packed_id',
    'Graphics_delta_lap_time_str',
    'Graphics_mfd_tyre_pressure_rear_left',
    'Graphics_mfd_tyre_pressure_front_right',
    'Graphics_rain_light',
    'Graphics_current_tyre_set',
    'Graphics_flashing_light',
    'Graphics_wiper_stage',
    'Graphics_mfd_tyre_pressure_rear_right',
    'Graphics_missing_mandatory_pits',
    'Graphics_best_time_str',
    'Graphics_player_car_id',
    'Graphics_is_delta_positive',
    'Graphics_mfd_fuel_to_add',
    'Graphics_driver_stint_total_time_left',
    'Graphics_tyre_compound',
    'Graphics_session_index',
    'Graphics_driver_stint_time_left',
    'Graphics_global_green',
    'Graphics_global_chequered',
    'Graphics_global_red',
    'Graphics_current_sector_index',
    'Graphics_direction_light_right',
    'Graphics_gap_ahead',
    'Graphics_global_white',
    'Graphics_last_time',
    'Graphics_clock',
    'Graphics_last_time_str',
    'Graphics_wind_direction',
    'Graphics_gap_behind',
    'Graphics_abs_level',
    'Graphics_delta_lap_time',
    'Graphics_used_fuel',
    'Graphics_global_yellow_s3',
    'Graphics_car_coordinates',
    'Graphics_mfd_tyre_set',
    'Graphics_normalized_car_position',
    'Graphics_wind_speed',
    'Graphics_current_time_str',
    'Graphics_last_sector_time_str',
    'Graphics_mfd_tyre_pressure_front_left',
    'Graphics_penalty_time',
    'Graphics_mandatory_pit_done',
    'Graphics_tc_level',
    'Graphics_strategy_tyre_set',
    'Graphics_last_sector_time',
    'Graphics_fuel_estimated_laps',
    'Graphics_direction_light_left',
    'Graphics_session_time_left',
    'Graphics_fuel_per_lap',
    'Graphics_track_status',
    'Graphics_number_of_laps',
    'Graphics_is_setup_menu_visible',
    'Graphics_position',
    'Graphics_rain_tyres',
    'Graphics_global_yellow_s2',
    'Graphics_car_id',
    'Graphics_best_time',
    'Graphics_is_in_pit',
    'Graphics_exhaust_temp',
    'Graphics_estimated_lap_time',
    'Graphics_secondary_display_index',
    'Graphics_global_yellow_s1',
    'Graphics_completed_lap',
    'Graphics_distance_traveled',
    'Graphics_main_display_index',
    'Graphics_light_stage',
    'Graphics_global_yellow',
    'Graphics_engine_map',
    'Graphics_active_cars',
    'Graphics_tc_cut_level',
    'Graphics_estimated_lap_time_str',
    'Graphics_current_time',
] as const;

const TELEMETRY_METRIC_STATIC_FIELDS = [
    'Static_sector_count',
    'Static_pit_window_start',
    'Static_max_rpm',
    'Static_pit_window_end',
    'Static_aid_auto_clutch',
    'Static_track',
    'Static_number_of_session',
    'Static_aid_stability',
    'Static_max_fuel',
    'Static_ac_version',
    'Static_num_cars',
    'Static_aid_tyre_rate',
    'Static_sm_version',
    'Static_player_name',
    'Static_penalty_enabled',
    'Static_dry_tyres_name',
    'Static_player_surname',
    'Static_is_online',
    'Static_car_model',
    'Static_aid_mechanical_damage',
    'Static_wet_tyres_name',
    'Static_aid_fuel_rate',
] as const;

const TELEMETRY_METRIC_FIELD_DESCRIPTION = [
    'Telemetry fields to summarize. Use exact field names from this catalog, including the Physics_, Graphics_, or Static_ prefix; do not invent unlisted names.',
    'Use Physics_* for live car dynamics, driver inputs, tyre/brake/suspension/damage/fuel data.',
    'Use Graphics_* for lap timing, sectors, position, gaps, flags, pit/MFD/session/weather state.',
    'Use Static_* for car, track, player, tyre names, max fuel/RPM, aids, pit window, and other session constants.',
    'When comparing balance across the car, query related wheel/corner fields together, e.g. *_front_left, *_front_right, *_rear_left, and *_rear_right.',
    `Physics fields (${TELEMETRY_METRIC_PHYSICS_FIELDS.length}): ${TELEMETRY_METRIC_PHYSICS_FIELDS.join(', ')}.`,
    `Graphics fields (${TELEMETRY_METRIC_GRAPHICS_FIELDS.length}): ${TELEMETRY_METRIC_GRAPHICS_FIELDS.join(', ')}.`,
    `Static fields (${TELEMETRY_METRIC_STATIC_FIELDS.length}): ${TELEMETRY_METRIC_STATIC_FIELDS.join(', ')}.`,
].join(' ');

export const FRONTEND_APPLICATION_TOOLS = [
    {
        name: 'start_agent_session',
        description: 'Start a separate child AI agent session. The user should interact with that child session while it is active.',
        properties: {
            agent_mode: {
                type: 'string',
                enum: ['track_guide', 'overtake', 'live_performance_analyst'],
                description: 'Agent profile to start. Use this for every live child agent instead of dedicated agent start tools.',
            },
        },
        required: ['agent_mode'],
    },
    {
        name: 'stop_agent_session',
        description: 'Stop the active child AI agent session and return focus to the main assistant. Use this for every live child agent instead of dedicated agent stop tools.',
        properties: {
            agent_session_id: {
                type: 'string',
                description: 'Optional frontend child session id. Defaults to the active agent session.',
            },
        },
        required: [],
    },
    {
        name: 'set_live_range_tracker',
        description: 'Create or replace the single live range tracker with normalized start/end ranges. The tracker monitors live telemetry and requests classification after each range end is crossed.',
        properties: {
            ranges: {
                type: 'array',
                description: 'Tracked normalized ranges. Each range needs start_position and end_position from 0 to 1.',
                items: {
                    type: 'object',
                    properties: {
                        id: { type: 'string' },
                        label: { type: 'string' },
                        start_position: { type: 'number' },
                        end_position: { type: 'number' },
                    },
                    required: ['start_position', 'end_position'],
                },
            },
        },
        required: ['ranges'],
    },
    {
        name: 'update_live_range_tracker',
        description: 'Update the live range tracker. Use action=record_classification after the classifier determines the tracked range status.',
        properties: {
            action: {
                type: 'string',
                enum: ['update_ranges', 'remove_ranges', 'record_classification', 'close'],
            },
            ranges: {
                type: 'array',
                description: 'Ranges for update_ranges.',
                items: {
                    type: 'object',
                    properties: {
                        id: { type: 'string' },
                        label: { type: 'string' },
                        start_position: { type: 'number' },
                        end_position: { type: 'number' },
                    },
                },
            },
            range_ids: {
                type: 'array',
                description: 'Range ids for remove_ranges.',
                items: { type: 'string' },
            },
            range_id: {
                type: 'string',
                description: 'Range id for record_classification.',
            },
            classifier_status: {
                type: 'string',
                description: 'Classifier-derived status for the tracked range.',
            },
            parent_segment: {
                type: 'object',
                description: 'Parent segment with its own labels and optional start/end indexes.',
            },
            child_segments: {
                type: 'array',
                description: 'Child segments with labels, start_index, and end_index.',
                items: {
                    type: 'object',
                    properties: {
                        labels: {
                            type: 'array',
                            items: { type: 'string' },
                        },
                        start_index: { type: 'integer' },
                        end_index: { type: 'integer' },
                    },
                    required: ['labels', 'start_index', 'end_index'],
                },
            },
        },
        required: ['action'],
    },
    {
        name: 'get_live_range_tracker',
        description: 'View the current live range tracker, including tracked ranges, lifecycle states, classifier status, parent labels, and child segment labels/indexes.',
        properties: {},
        required: [],
    },
    {
        name: 'collect_live_baseline',
        description: 'Collect one complete live baseline lap through the dedicated baseline UI component and return the cached baseline lap record when complete.',
        properties: {
            timeout_seconds: {
                type: 'integer',
                description: 'Maximum time to wait for the baseline lap to complete. Defaults to 600 seconds.',
            },
        },
        required: [],
    },
    {
        name: 'restart_live_baseline',
        description: 'Restart the dedicated live baseline collection buffer so the next collect_live_baseline call records a fresh baseline lap.',
        properties: {},
        required: [],
    },
    {
        name: 'analyze_live_recorded_analysis',
        description: 'Submit the already recorded baseline lap to live recorded analysis and return classified sections with time gaps when available. Returns an error until baseline collection has recorded a cached lap.',
        properties: {
            limit: {
                type: 'integer',
                description: 'Maximum number of classified segments to return.',
            },
        },
        required: [],
    },
    {
        name: 'advance_plan_step',
        description: 'Report that the current visible procedure plan request is complete so the UI can move to the next request.',
        properties: {
            reason: {
                type: 'string',
                description: 'Short reason the current plan request is complete.',
            },
        },
        required: [],
    },
    {
        name: 'clear_procedure_plan',
        description: 'Clear or terminate the visible procedure plan UI when the plan is no longer useful.',
        properties: {
            reason: {
                type: 'string',
                description: 'Optional short reason the visible plan should be cleared.',
            },
        },
        required: [],
    },
    {
        name: 'set_procedure_plan',
        description: 'Create or replace the visible procedure plan UI with an AI-authored list of requests. Requests with type "tool_call" and a name are executed through the active AI session subscription.',
        properties: {
            goal: {
                type: 'string',
                description: 'Short goal shown above the request list.',
            },
            current_request: {
                type: 'integer',
                description: 'Zero-based index of the active request.',
            },
            requests: {
                type: 'array',
                description: 'Ordered list of requests the assistant plans to perform or ask the UI/backend to perform.',
                items: {
                    type: 'object',
                    properties: {
                        type: { type: 'string' },
                        title: { type: 'string' },
                        name: {
                            type: 'string',
                            description: 'Tool name for tool_call requests. The active AI session subscribes to this tool run and receives the final result.',
                        },
                        status: {
                            type: 'string',
                            enum: ['pending', 'running', 'complete', 'blocked', 'failed', 'skipped'],
                        },
                        detail: { type: 'string' },
                        payload: {
                            type: 'object',
                            description: 'Tool arguments for tool_call requests, optionally wrapped in arguments, args, or parameters.',
                        },
                    },
                    required: ['type', 'title'],
                },
            },
        },
        required: ['goal', 'requests'],
    },
    {
        name: 'get_next_corner',
        properties: {},
        required: [],
    },
    {
        name: 'query_telemetry_metric',
        description: 'Ask for the current, average, minimum, or maximum telemetry value for selected fields over a live-session scope and return summarized numbers instead of raw telemetry rows.',
        properties: {
            fields: {
                type: 'array',
                description: TELEMETRY_METRIC_FIELD_DESCRIPTION,
                items: { type: 'string' },
            },
            scope: {
                ...FRONTEND_APPLICATION_QUERY_SCOPE_SCHEMA,
                description: 'Telemetry window to summarize. Use type="now" for current values; use last_seconds, event, lap, or range for time/windowed summaries.',
            },
            reduce: {
                type: 'string',
                enum: ['avg', 'min', 'max', 'stats'],
                description: 'Aggregation to return: avg, min, max, or stats. Prefer stats when the user asks generally or wants a complete summary.',
            },
        },
        required: ['fields', 'scope', 'reduce'],
    },
    {
        name: 'get_event_log',
        properties: {
            eventType: {
                type: 'string',
                enum: ['CORNER', 'STRAIGHT', 'CRASHED', 'OVERTAKE'],
            },
            scope: {
                type: 'string',
                enum: ['last', 'last_n', 'lap_current', 'lap_last', 'all'],
            },
            n: {
                type: 'integer',
            },
        },
        required: ['eventType', 'scope'],
    },
    {
        name: 'get_user_summary_map_level',
        description: 'Return map-level user summary data. With no map_id, returns all maps with aggregate stats and top sections; with map_id, returns that map with full section breakdowns, mistake counts, expert-adherence counts, percentages, and segment/category summaries.',
        properties: {
            map_id: {
                type: 'string',
            },
        },
        required: [],
    },
    {
        name: 'get_available_user_summary_maps',
        description: 'Return a compact list of maps that have user summary data, including map id, name, analyzed session count, total analyzed time count, section count, and a human-readable map_options list.',
        properties: {},
        required: [],
    },
    {
        name: 'search_user_summary_map_level',
        description: 'Search map-level user summary rows by map name, map id, top mistake section names or ids, top expert-adherence section names or ids, and aggregate words like mistake, weakness, expert, or strength. Returns scored matching maps with matched_fields.',
        properties: {
            query: {
                type: 'string',
            },
            limit: {
                type: 'integer',
            },
        },
        required: ['query'],
    },
    {
        name: 'show_map',
        description: 'Display a circuit map in the chat transcript, optionally highlighting a normalized lap section.',
        properties: {
            map_id: {
                type: 'string',
                description: 'Circuit map id to display. Prefer this when a map id is known.',
            },
            source_track_key: {
                type: 'string',
                description: 'ACC source track key such as brands_hatch, monza, or spa.',
            },
            map_name: {
                type: 'string',
                description: 'Human-readable map or circuit name when no id/key is known.',
            },
            section_start: {
                type: 'number',
                description: 'Start of the highlighted section as normalized lap position from 0 to 1.',
            },
            section_end: {
                type: 'number',
                description: 'End of the highlighted section as normalized lap position from 0 to 1. Values wrapping across start/finish are allowed.',
            },
            section_label: {
                type: 'string',
                description: 'Short label for the highlighted section.',
            },
            title: {
                type: 'string',
                description: 'Short title shown above the map.',
            },
            note: {
                type: 'string',
                description: 'Brief note shown below the map.',
            },
        },
        required: [],
    },
    {
        name: 'run_recorded_ai_analysis',
        description: 'Run or retrieve the AI segment analysis for the currently selected recorded session.',
        properties: {
            force: {
                type: 'boolean',
                description: 'When true, rerun analysis even if a cached result is available.',
            },
            limit: {
                type: 'integer',
                description: 'Maximum number of compact classified segments to return.',
            },
        },
        required: [],
    },
    {
        name: 'get_recorded_session_analysis',
        description: 'Return the shared AI segment analysis for the currently selected recorded session.',
        properties: {
            limit: {
                type: 'integer',
                description: 'Maximum number of compact classified segments to return.',
            },
        },
        required: [],
    },
    {
        name: 'get_recorded_session_context',
        description: 'Return compact selected recorded-session, playback, and AI-analysis context.',
        properties: {
            limit: {
                type: 'integer',
                description: 'Maximum number of compact classified segments to include.',
            },
        },
        required: [],
    },
    {
        name: 'analyze_telemetry',
        description: 'Classify driving actions over a telemetry scope and return engineer labels with definitions and optional solutions. Use only for live or recorded raw telemetry windows, such as "what just happened", "why did I lose time there on this lap", or "how was lap N".',
        properties: {
            scope: {
                ...FRONTEND_APPLICATION_QUERY_SCOPE_SCHEMA,
                description: 'Telemetry time window to classify.',
            },
        },
        required: ['scope'],
    },
    {
        name: 'classify_live_section',
        description: 'Classify the active Live Performance Analyst focus section after the driver passes through it again. This frontend tool brings the telemetry window to the AI service, records a compact comparison in section history, and returns only compact labels, stats, focus, and comparison data.',
        properties: {
            section_id: {
                type: 'string',
                description: 'Known track section id from the live analyst observation.',
            },
            section_name: {
                type: 'string',
                description: 'Optional section name if an id is not available.',
            },
            lap: {
                oneOf: [
                    { type: 'string', enum: ['last'] },
                    { type: 'integer' },
                ],
                description: 'Lap to classify for the active focus section. Use "last" for the most recent completed pass, or a specific lap number when supplied by the observation.',
            },
        },
        required: [],
    },
] as const;

type FrontendApplicationToolName = typeof FRONTEND_APPLICATION_TOOLS[number]['name'];
type FrontendApplicationSessionMode = 'live' | 'recorded' | 'user_summary';
type ToolPropertyMap = Record<string, unknown>;

type AiToolMetadata = {
    title: string;
    description: string;
    parameters: Record<string, { description: string }>;
};

const FRONTEND_APPLICATION_TOOL_TITLES: Record<FrontendApplicationToolName, string> = {
    start_agent_session: 'Starting agent mode',
    stop_agent_session: 'Stopping agent mode',
    set_live_range_tracker: 'Setting live range tracker',
    update_live_range_tracker: 'Updating live range tracker',
    get_live_range_tracker: 'Reading live range tracker',
    collect_live_baseline: 'Collecting baseline lap',
    restart_live_baseline: 'Restarting baseline lap',
    analyze_live_recorded_analysis: 'Analyzing baseline lap',
    advance_plan_step: 'Advancing plan',
    clear_procedure_plan: 'Clearing procedure plan',
    set_procedure_plan: 'Setting procedure plan',
    get_next_corner: 'Looking up next corner',
    query_telemetry_metric: 'Querying telemetry',
    get_event_log: 'Searching event log',
    get_user_summary_map_level: 'Reading user summary by map',
    get_available_user_summary_maps: 'Listing user summary maps',
    search_user_summary_map_level: 'Searching user summary maps',
    show_map: 'Displaying a circuit map',
    run_recorded_ai_analysis: 'Running recorded session AI analysis',
    get_recorded_session_analysis: 'Reading recorded AI analysis',
    get_recorded_session_context: 'Reading recorded session context',
    analyze_telemetry: 'Analyzing telemetry',
    classify_live_section: 'Classifying live section',
};

const FRONTEND_APPLICATION_TOOL_DESCRIPTION_OVERRIDES: Partial<Record<FrontendApplicationToolName, string>> = {
    get_next_corner: 'Return the name and normalized distance of the next corner ahead. Use for live questions about what corner is coming up.',
};

const COMMON_TOOL_NAMES = new Set<FrontendApplicationToolName>([
    'show_map',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    'stop_agent_session',
]);

const LIVE_TOOL_NAMES = new Set<FrontendApplicationToolName>([
    'start_agent_session',
    'analyze_telemetry',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
]);

const USER_SUMMARY_TOOL_NAMES = new Set<FrontendApplicationToolName>([
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
]);

const RECORDED_TOOL_NAMES = new Set<FrontendApplicationToolName>([
    'run_recorded_ai_analysis',
    'get_recorded_session_analysis',
    'get_recorded_session_context',
    'analyze_telemetry',
]);

const isFrontendApplicationSessionMode = (
    value: unknown,
): value is FrontendApplicationSessionMode => (
    value === 'live' || value === 'recorded' || value === 'user_summary'
);

const getAllowedToolNames = (
    sessionMode: FrontendApplicationSessionMode,
    conversationRole?: unknown,
) => {
    if (conversationRole === 'agent') {
        return new Set<FrontendApplicationToolName>([
            ...COMMON_TOOL_NAMES,
            ...Array.from(LIVE_TOOL_NAMES).filter((name) => name !== 'start_agent_session'),
            ...USER_SUMMARY_TOOL_NAMES,
        ]);
    }

    if (sessionMode === 'recorded') {
        return new Set<FrontendApplicationToolName>([
            ...COMMON_TOOL_NAMES,
            ...USER_SUMMARY_TOOL_NAMES,
            ...RECORDED_TOOL_NAMES,
        ]);
    }

    if (sessionMode === 'user_summary') {
        return new Set<FrontendApplicationToolName>([
            ...COMMON_TOOL_NAMES,
            ...USER_SUMMARY_TOOL_NAMES,
        ]);
    }

    return new Set<FrontendApplicationToolName>([
        ...COMMON_TOOL_NAMES,
        ...LIVE_TOOL_NAMES,
        ...USER_SUMMARY_TOOL_NAMES,
    ]);
};

export const getFrontendApplicationToolsForSessionContext = (
    sessionContext: Record<string, unknown> | null | undefined,
) => {
    const sessionMode = isFrontendApplicationSessionMode(sessionContext?.session_mode)
        ? sessionContext.session_mode
        : 'live';
    const allowedToolNames = getAllowedToolNames(
        sessionMode,
        sessionContext?.conversation_role,
    );

    return FRONTEND_APPLICATION_TOOLS.filter((tool) => allowedToolNames.has(tool.name));
};

const getDescriptionFromProperty = (property: unknown): string => {
    if (!property || typeof property !== 'object') return '';
    const description = (property as { description?: unknown }).description;
    return typeof description === 'string' ? description.trim() : '';
};

const getToolParameterMetadata = (
    properties: ToolPropertyMap | undefined,
): AiToolMetadata['parameters'] => {
    if (!properties) return {};

    return Object.entries(properties).reduce<AiToolMetadata['parameters']>(
        (out, [name, property]) => {
            const description = getDescriptionFromProperty(property);
            if (description) {
                out[name] = { description };
            }
            return out;
        },
        {},
    );
};

const getFrontendToolMetadata = (
    tool: typeof FRONTEND_APPLICATION_TOOLS[number],
): AiToolMetadata => {
    const description = 'description' in tool && typeof tool.description === 'string'
        ? tool.description
        : '';

    return {
        title: FRONTEND_APPLICATION_TOOL_TITLES[tool.name],
        description: FRONTEND_APPLICATION_TOOL_DESCRIPTION_OVERRIDES[tool.name]
            || description,
        parameters: getToolParameterMetadata(tool.properties),
    };
};

export const getAiToolMetadataForSessionContext = (
    sessionContext: Record<string, unknown> | null | undefined,
): Record<string, AiToolMetadata> => {
    const metadata: Record<string, AiToolMetadata> = {};

    getFrontendApplicationToolsForSessionContext(sessionContext).forEach((tool) => {
        metadata[tool.name] = getFrontendToolMetadata(tool);
    });

    return metadata;
};
