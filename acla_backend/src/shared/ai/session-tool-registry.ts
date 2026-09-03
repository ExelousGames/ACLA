import { TELEMETRY_METRIC_FIELD_SCHEMA } from './telemetry-metric-fields';

/** Backend-owned schemas exposed to authenticated AI voice sessions. */
export const SESSION_TOOL_QUERY_SCOPE_SCHEMA = {
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

const BASELINE_TELEMETRY_CONDITION_SCHEMA = {
    type: 'object',
    properties: {
        field: TELEMETRY_METRIC_FIELD_SCHEMA,
        operator: { type: 'string', enum: ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'] },
        value: { type: 'number' },
    },
    required: ['field', 'operator', 'value'],
    additionalProperties: false,
} as const;

export const SESSION_TOOLS = [
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
                description: 'Optional browser child session id. Defaults to the active agent session.',
            },
        },
        required: [],
    },
    {
        name: 'add_event_to_live_range_todo_list',
        description: 'Atomically append executable events to the visible Live Range To-do List. AI Chat mounts the list when needed. Each nested tool runs only when telemetry makes its event due; this add call returns the updated list summary immediately after insertion.',
        properties: {
            events: {
                type: 'array',
                minItems: 1,
                description: 'Events to append without replacing the existing queue. Every event id must be unique in this batch and the active list.',
                items: {
                    type: 'object',
                    properties: {
                        event: {
                            type: 'object',
                            properties: {
                                id: { type: 'string', minLength: 1, pattern: '\\S', description: 'Unique event id.' },
                                normalized_position: { type: 'number', minimum: 0, maximum: 1 },
                                lead_time_seconds: { type: 'number', minimum: 0, description: 'How early to run the event. Defaults to 2 seconds.' },
                                content: {
                                    type: 'object',
                                    properties: {
                                        title: { type: 'string', minLength: 1, pattern: '\\S' },
                                        description: { type: 'string' },
                                    },
                                    required: ['title'],
                                    additionalProperties: false,
                                },
                            },
                            required: ['id', 'normalized_position', 'content'],
                            additionalProperties: false,
                        },
                        tool: {
                            type: 'object',
                            properties: {
                                name: { type: 'string', description: 'Available child live-session tool to execute when the event is due.' },
                                arguments: { type: 'object', description: 'JSON-safe arguments passed unchanged to the nested tool.' },
                            },
                            required: ['name', 'arguments'],
                            additionalProperties: false,
                        },
                    },
                    required: ['event', 'tool'],
                    additionalProperties: false,
                },
            },
        },
        required: ['events'],
    },
    {
        name: 'get_live_range_todo_list',
        description: 'Read the active Live Range To-do List summary, including event and lifecycle counts.',
        properties: {},
        required: [],
    },
    {
        name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
        description: 'Append Driver vs Expert comparison events for the active Analysis Results page\'s last successfully applied segment filter. Events keep the displayed segment order, retain existing to-do items, and publish only when live telemetry makes each event due.',
        properties: {},
        required: [],
    },
    {
        name: 'collect_live_baseline',
        description: 'Start live baseline recording through the dedicated baseline UI component and return the cached record when the selected stop condition is met. Choose either the full_lap preset or custom start_query and end_query conditions. If a previous baseline exists, this starts a new recording and replaces it; an existing completed baseline is not treated as an already-started error.',
        properties: {
            query: {
                description: 'Mutually exclusive baseline bounds. full_lap starts at normalized position 0 and ends at normalized position 1.',
                oneOf: [
                    {
                        type: 'object',
                        properties: {
                            preset: { type: 'string', enum: ['full_lap'] },
                        },
                        required: ['preset'],
                        additionalProperties: false,
                    },
                    {
                        type: 'object',
                        properties: {
                            start_query: BASELINE_TELEMETRY_CONDITION_SCHEMA,
                            end_query: BASELINE_TELEMETRY_CONDITION_SCHEMA,
                        },
                        required: ['start_query', 'end_query'],
                        additionalProperties: false,
                    },
                ],
            },
            timeout_seconds: {
                type: 'integer',
                minimum: 1,
                description: 'Maximum time to wait for the baseline stop condition. Defaults to 600 seconds.',
            },
        },
        required: ['query'],
    },
    {
        name: 'restart_live_baseline',
        description: 'Restart baseline recording with the same query only while it is waiting for its start condition or actively collecting. Returns an error when recording is not in progress; use collect_live_baseline to start a new recording after a completed baseline.',
        properties: {},
        required: [],
    },
    {
        name: 'analyze_live_recorded_analysis',
        description: 'Submit the already recorded baseline to live recorded analysis and return classified sections with time gaps when available. Returns an error until baseline collection has recorded a cached baseline.',
        properties: {
            limit: {
                type: 'integer',
                description: 'Maximum number of classified segments to return.',
            },
        },
        required: [],
    },
    {
        name: 'apply_query_to_analysis_result',
        description: [
            'Apply a final JSONata expression to the visible Analysis Results. The tool returns only its status and does not return the matched results.',
            'The expression receives only { "elements": [{ "id": "...", "labels": ["..."], "title": "...", "section": "...", "normalizedPositionRange": { "start": 0, "end": 1 }, "timeGap": {}, "comparison": {}, "metadata": {} }] } for the selected page; it does not receive the current View selection or hidden page data.',
            'The JSONata expression must evaluate to null, one element ID string, one object with a string id, or a flat array of element IDs or objects with string ids. Unknown IDs and nested arrays are rejected.',
            'Examples: elements; elements[labels[$ = "Lockup"]]; elements[labels[$ = "Mistake (Practice)"]].id.',
            'page_number uses the displayed 1-based retained-page array order: Page 1 is array index 0 and the highest page number is the most recent analysis.',
            'When page_number is omitted, below 1, or above the existing page count, the highest existing page is selected.',
            'The tool switches from Overall Trends to Lap Results, waits for the selected page to render, populates the editor, and uses the same commit path as manual Apply.',
        ].join(' '),
        properties: {
            query: {
                type: 'string',
                minLength: 1,
                pattern: '\\S',
                description: 'A non-blank final JSONata expression whose result identifies the Analysis Results elements to display.',
            },
            page_number: {
                type: 'integer',
                description: 'Optional 1-based displayed page number. Omitted or nonexistent page numbers select the highest existing page number.',
            },
        },
        required: ['query'],
    },
    {
        name: 'query_analysis_result',
        description: [
            'Evaluate a JSONata expression against all Analysis Results without rerunning analysis. The current View and active page do not change the query input.',
            'The expression receives exactly one root structure: { "analyses": [{ "id": "...", "createdAt": 0, "sourceIndex": 0, "baseline": { "lap": 1, "lapTimeMs": 0, "track": "...", "car": "..." }, "elements": [{ "id": "...", "labels": ["..."], "title": "...", "section": "...", "normalizedPositionRange": { "start": 0, "end": 1 }, "timeGap": {}, "comparison": {}, "metadata": {} }] }] }. analyses contains every retained lap analysis in displayed order. For a non-paginated recorded result it contains one analysis with null createdAt and baseline.',
            'The response is { "status": "ready", "data": ... }, where data is the actual JSON-safe JSONata value (scalar, object, array, or null), not a count unless the expression returns one.',
            'Examples: $count(analyses) counts analyses; $count(analyses.elements) counts segments across all analyses; analyses.elements[labels[$ = "Lockup"]].{ "id": id, "section": section }.',
        ].join(' '),
        properties: {
            query: {
                type: 'string',
                minLength: 1,
                pattern: '\\S',
                description: 'A non-blank JSONata expression evaluated against the normalized { analyses } root. Its actual JSON-safe value is returned in data.',
            },
        },
        required: ['query'],
    },
    {
        name: 'create_goal',
        description: 'Create one visible goal. The session tool calls will be executed sequentially, and these tool calls will be repeated until the goal is achieved or an error occurs. If the goal is missed, the goal will continue to be retried until the goal is achieved or the user cancels the goal. The stop-when tool call must return { "status": "ready", "data": finiteNumber }. When using query_analysis_result for stop_when, use { "query": "$count(analyses)" } to count all analysis results. The operator and target are evaluated against that number to decide when to stop.',
        properties: {
            name: {
                type: 'string',
                description: 'Short name displayed on the goal card.',
            },
            steps: {
                type: 'array',
                minItems: 1,
                description: 'Ordered session tool calls. Every id must be unique; create_goal and retry_goal_task cannot be nested.',
                items: {
                    type: 'object',
                    properties: {
                        id: { type: 'string', description: 'Unique stable step id.' },
                        title: { type: 'string', description: 'Short step label displayed to the user.' },
                        name: { type: 'string', description: 'Available session tool to execute.' },
                        arguments: { type: 'object', description: 'Arguments passed unchanged to the nested tool.' },
                    },
                    required: ['id', 'title', 'name'],
                },
            },
            stop_when: {
                type: 'object',
                description: 'Frontend query tool call that must return { "status": "ready", "data": finiteNumber }, plus the comparison evaluated after the ordered preparation steps. A query_analysis_result stop condition should normally use a $count(...) JSONata expression.',
                properties: {
                    tool: {
                        type: 'object',
                        description: 'Frontend query tool call that must return { "status": "ready", "data": finiteNumber } to determine whether the goal was achieved. For query_analysis_result, use { "query": "$count(analyses)" } to count all analysis results.',
                        properties: {
                            name: { type: 'string', description: 'Available session tool to execute.' },
                            arguments: { type: 'object', description: 'Arguments passed unchanged to the stop-when tool.' },
                        },
                        required: ['name'],
                    },
                    operator: { type: 'string', enum: ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'] },
                    target: { type: 'number' },
                },
                required: ['tool', 'operator', 'target'],
            },
        },
        required: ['name', 'steps', 'stop_when'],
    },
    {
        name: 'retry_goal_task',
        description: 'Retry the currently failed goal task once with its stored arguments, then continue the remaining goal workflow after success. Available only when the visible goal is in an error state with a failed task.',
        properties: {},
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
        description: 'Create or replace the visible procedure plan to execute tools in orders. Requests with a name are executed through the active AI session subscription. each tool call is executed sequentially, and end when the last request is complete. The plan can be cleared or terminated with clear_procedure_plan.',
        properties: {
            goal: {
                type: 'string',
                description: 'Short goal shown above the request list.',
            },
            requests: {
                type: 'array',
                description: 'Ordered list of requests the assistant plans to perform or ask the UI/backend to perform.',
                items: {
                    type: 'object',
                    properties: {
                        title: { type: 'string' },
                        name: {
                            type: 'string',
                            description: 'Tool name for executable requests. The active AI session subscribes to this tool run and receives the final result.',
                        },
                        payload: {
                            type: 'object',
                            description: 'Tool arguments for executable requests, optionally wrapped in arguments, args, or parameters.',
                        },
                    },
                    required: ['title', 'name', 'payload'],
                },
            },
        },
        required: ['goal', 'requests'],
    },
    {
        name: 'get_next_corner',
        description: 'Return the name and normalized distance of the next corner ahead. Use for live questions about what corner is coming up.',
        properties: {},
        required: [],
    },
    {
        name: 'query_telemetry_metric',
        description: 'Ask for the current, average, minimum, or maximum telemetry value for selected fields over a live-session scope and return summarized numbers instead of raw telemetry rows. Do not use `query_telemetry_metric` for performance checking, pace diagnosis, or track-improvement requests; use `live_performance_analyst` for those.',
        properties: {
            fields: {
                type: 'array',
                description: 'Telemetry fields to summarize. Use only the supported exact field names.',
                items: TELEMETRY_METRIC_FIELD_SCHEMA,
            },
            scope: {
                ...SESSION_TOOL_QUERY_SCOPE_SCHEMA,
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
        description: 'Classify driving actions over a telemetry scope and return engineer labels with definitions and optional solutions. Use this to do a quick analysis of a telemetry window without launching a dedicated ai analysis agent.',
        properties: {
            scope: {
                ...SESSION_TOOL_QUERY_SCOPE_SCHEMA,
                description: 'Telemetry time window to classify.',
            },
        },
        required: ['scope'],
    },
] as const;

type SessionToolName = typeof SESSION_TOOLS[number]['name'];
type SessionMode = 'front_desk' | 'live' | 'recorded' | 'user_summary';

const COMMON_TOOL_NAMES: SessionToolName[] = [
    'show_map',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    'stop_agent_session',
];

const LIVE_SESSION_TOOL_NAMES: SessionToolName[] = [
    'start_agent_session',
    'apply_query_to_analysis_result',
    'query_analysis_result',
    'analyze_telemetry',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
];

const LIVE_AGENT_SESSION_TOOL_NAMES: SessionToolName[] = [
    'analyze_telemetry',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
    'add_event_to_live_range_todo_list',
    'get_live_range_todo_list',
    'collect_live_baseline',
    'restart_live_baseline',
    'analyze_live_recorded_analysis',
];

const LIVE_PERFORMANCE_ANALYST_TOOL_NAMES: SessionToolName[] = [
    'create_goal',
    'retry_goal_task',
    'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
];

const LIVE_RANGE_TODO_NESTED_TOOL_EXCLUSIONS = new Set<SessionToolName>([
    'create_goal',
    'retry_goal_task',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    'add_event_to_live_range_todo_list',
    'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
]);

const USER_SUMMARY_SESSION_TOOL_NAMES: SessionToolName[] = [
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
];

const RECORDED_SESSION_TOOL_NAMES: SessionToolName[] = [
    'run_recorded_ai_analysis',
    'get_recorded_session_analysis',
    'get_recorded_session_context',
    'apply_query_to_analysis_result',
    'query_analysis_result',
    'analyze_telemetry',
];

const isSessionMode = (
    value: unknown,
): value is SessionMode => (
    value === 'front_desk' || value === 'live' || value === 'recorded' || value === 'user_summary'
);

const isSessionAgentMode = (
    value: unknown,
): value is 'track_guide' | 'overtake' | 'live_performance_analyst' => (
    value === 'track_guide' || value === 'overtake' || value === 'live_performance_analyst'
);

const getAllowedToolNames = (
    sessionMode: SessionMode,
    agentMode?: unknown,
) => {
    if (isSessionAgentMode(agentMode)) {
        return new Set<SessionToolName>([
            ...COMMON_TOOL_NAMES,
            ...LIVE_AGENT_SESSION_TOOL_NAMES,
            ...USER_SUMMARY_SESSION_TOOL_NAMES,
            ...(sessionMode === 'live' ? [
                'apply_query_to_analysis_result',
                'query_analysis_result',
            ] as const : []),
            ...(sessionMode === 'live' && agentMode === 'live_performance_analyst'
                ? LIVE_PERFORMANCE_ANALYST_TOOL_NAMES
                : []),
        ]);
    }

    if (sessionMode === 'recorded') {
        return new Set<SessionToolName>([
            ...COMMON_TOOL_NAMES,
            ...USER_SUMMARY_SESSION_TOOL_NAMES,
            ...RECORDED_SESSION_TOOL_NAMES,
        ]);
    }

    if (sessionMode === 'user_summary') {
        return new Set<SessionToolName>([
            ...COMMON_TOOL_NAMES,
            ...USER_SUMMARY_SESSION_TOOL_NAMES,
        ]);
    }

    if (sessionMode === 'front_desk') {
        return new Set<SessionToolName>([
            ...COMMON_TOOL_NAMES,
            ...USER_SUMMARY_SESSION_TOOL_NAMES,
        ]);
    }

    return new Set<SessionToolName>([
        ...COMMON_TOOL_NAMES,
        ...LIVE_SESSION_TOOL_NAMES,
        ...USER_SUMMARY_SESSION_TOOL_NAMES,
    ]);
};

export const getSessionToolsForSessionContext = (
    sessionContext: Record<string, unknown> | null | undefined,
) => {
    const sessionMode = isSessionMode(sessionContext?.session_mode)
        ? sessionContext.session_mode
        : 'live';
    const allowedToolNames = getAllowedToolNames(
        sessionMode,
        sessionContext?.agent_mode,
    );

    const tools = SESSION_TOOLS.filter((tool) => allowedToolNames.has(tool.name));
    const nestedToolNames = tools
        .map((tool) => tool.name)
        .filter((name) => name !== 'create_goal' && name !== 'retry_goal_task');
    const liveRangeTodoNestedToolNames = tools
        .map((tool) => tool.name)
        .filter((name) => !LIVE_RANGE_TODO_NESTED_TOOL_EXCLUSIONS.has(name));

    const expandedTools = tools.map((tool) => {
        if (tool.name === 'add_event_to_live_range_todo_list') {
            const events = tool.properties.events;
            const item = events.items;
            const nestedTool = item.properties.tool;
            return {
                ...tool,
                properties: {
                    ...tool.properties,
                    events: {
                        ...events,
                        items: {
                            ...item,
                            properties: {
                                ...item.properties,
                                tool: {
                                    ...nestedTool,
                                    properties: {
                                        ...nestedTool.properties,
                                        name: {
                                            ...nestedTool.properties.name,
                                            enum: liveRangeTodoNestedToolNames,
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            };
        }
        if (tool.name !== 'create_goal') return tool;
        const steps = tool.properties.steps;
        const stopWhen = tool.properties.stop_when;
        const stopWhenTool = stopWhen.properties.tool;
        return {
            ...tool,
            properties: {
                ...tool.properties,
                steps: {
                    ...steps,
                    items: {
                        ...steps.items,
                        properties: {
                            ...steps.items.properties,
                            name: {
                                ...steps.items.properties.name,
                                enum: nestedToolNames,
                            },
                        },
                    },
                },
                stop_when: {
                    ...stopWhen,
                    properties: {
                        ...stopWhen.properties,
                        tool: {
                            ...stopWhenTool,
                            properties: {
                                ...stopWhenTool.properties,
                                name: {
                                    ...stopWhenTool.properties.name,
                                    enum: nestedToolNames,
                                },
                            },
                        },
                    },
                },
            },
        };
    });

    return expandedTools.map((tool) => ({
        name: tool.name,
        description: 'description' in tool && typeof tool.description === 'string'
            ? tool.description
            : '',
        properties: tool.properties,
        required: [...tool.required],
    }));
};
