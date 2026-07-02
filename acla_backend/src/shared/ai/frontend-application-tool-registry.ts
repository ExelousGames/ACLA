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

export const FRONTEND_APPLICATION_TOOL_RESULT_HANDLING = [
    'Tools may return a status field such as running, complete, failed, blocked, or skipped.',
    'Treat complete or ok=true as a successful result and use the returned result/data payload.',
    'Treat running as not ready yet; wait for the final result instead of answering from partial data.',
    'Treat failed, blocked, or skipped as unavailable and explain the issue or choose another available tool.',
    'If no status is present, treat an error field as failed; otherwise treat the payload as a completed result.',
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
        name: 'get_live_focus_section',
        description: 'Return the current live analyst focus section, timing, and map-display arguments when available.',
        properties: {},
        required: [],
    },
    {
        name: 'get_live_section_history',
        description: 'Return compact live section classifications already recorded by the AI service.',
        properties: {
            limit: {
                type: 'integer',
                description: 'Maximum number of compact classifications to return.',
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
        properties: {
            fields: {
                type: 'array',
                items: { type: 'string' },
            },
            scope: FRONTEND_APPLICATION_QUERY_SCOPE_SCHEMA,
            reduce: {
                type: 'string',
                enum: ['avg', 'min', 'max', 'stats'],
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
        properties: {
            map_id: {
                type: 'string',
            },
        },
        required: [],
    },
    {
        name: 'get_available_user_summary_maps',
        properties: {},
        required: [],
    },
    {
        name: 'search_user_summary_map_level',
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
                description: 'Known track section id from the live analyst observation or get_live_focus_section result.',
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
    get_live_focus_section: 'Analyzing focus section',
    get_live_section_history: 'Reading section history',
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
    'get_live_focus_section',
    'get_live_section_history',
    'set_live_range_tracker',
    'update_live_range_tracker',
    'get_live_range_tracker',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
    'analyze_telemetry',
    'classify_live_section',
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
