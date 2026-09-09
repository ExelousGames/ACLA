import { TELEMETRY_METRIC_FIELD_SCHEMA } from './telemetry-metric-fields';

/** Model Command Protocol: tools and workflows for model-to-client communication. */
export const MODEL_COMMAND_QUERY_SCOPE_SCHEMA = {
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

const MODEL_COMMAND_DEFINITIONS = [
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
        description: [
            'Stop the active child AI agent session and return focus to the main assistant. Use this for every live child agent instead of dedicated agent stop tools.',
            'Use when the driver asks to stop, close, exit, or return from a child session to the main assistant.',
        ].join(' '),
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
        description: [
            'Atomically append executable events to the visible Live Range To-do List. AI Chat mounts the list when needed. Each nested tool runs only when telemetry makes its event due; this add call returns the updated list summary immediately after insertion.',
            'Use the native tool call channel. The arguments object must use workflow as its single outer key, with name set to add_event_to_live_range_todo_list.',
            'Inside that wrapper, provide an explicit ordered tools list. Each child has a tool object with a name allowed by the current catalog child schema. All workflow categories are forbidden as children, including creation, control, and read commands. Do not include prose-only tasks or hidden steps.',
            'Preserve each child\'s tool-specific arguments object unchanged. Do not unwrap, flatten, rename, or reinterpret keys inside it.',
            'No legacy compatibility: do not use unwrapped creation bodies, requests/steps/events lists, tool-name keys, or payload/args/parameters aliases for child arguments. Result and state fields are not creation inputs.',
            'Keep metadata inside the tool object event: id, normalized_position, optional lead_time_seconds, and content with title and optional description. Keep the required arguments object alongside event, using {} for a tool with no inputs.',
            'The immediate insertion summary does not mean the nested tools have run. Wait for the later event results before describing their outcomes.',
            `Example native arguments:
\`\`\`json
{
  "workflow": {
    "name": "add_event_to_live_range_todo_list",
    "tools": [
      {
        "tool": {
          "name": "show_map",
          "event": {
            "id": "spa-opening-map",
            "normalized_position": 0.1,
            "lead_time_seconds": 2,
            "content": {
              "title": "Opening section",
              "description": "Show the selected Spa section"
            }
          },
          "arguments": {
            "source_track_key": "spa",
            "section_start": 0,
            "section_end": 0.2
          }
        }
      }
    ]
  }
}
\`\`\``,
        ].join(' '),
        properties: {
            tools: {
                type: 'array',
                minItems: 1,
                description: 'Explicit tool events to append without replacing the existing queue. Each entry contains tool with a name. Every event id must be unique in this batch and the active list. Workflow commands cannot be nested.',
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
                        arguments: { type: 'object', description: 'JSON-safe arguments passed unchanged to the nested tool.' },
                    },
                    required: ['event', 'arguments'],
                    additionalProperties: false,
                },
            },
        },
        required: ['tools'],
    },
    {
        name: 'get_live_range_todo_list',
        description: [
            'Read the active Live Range To-do List summary, including event and lifecycle counts.',
        ].join(' '),
        properties: {},
        required: [],
    },
    {
        name: 'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
        description: [
            'Visualize analysis results with Driver vs Expert comparisons in the overlay while driving. Use when the user asks to visualize analysis results while driving.',
            'Append Driver vs Expert comparison events for the active Analysis Results page\'s last successfully applied segment filter. Events keep the displayed segment order, retain existing to-do items, and publish only when live telemetry makes each event due.',
        ].join(' '),
        properties: {},
        required: [],
    },
    {
        name: 'collect_live_baseline',
        description: 'Start live baseline recording through the dedicated baseline UI component and return the cached record when the selected stop condition is met. Choose either the full_lap preset or custom start_query and end_query conditions. The baseline recorder holds only one recording at a time. If a previous baseline exists, this starts a new recording and replaces it; an existing completed baseline is not treated as an already-started error.',
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
        description: [
            'Submit the already recorded baseline to live recorded analysis and return classified sections with time gaps when available. Returns an error until baseline collection has recorded a cached baseline.',
            'The completed analysis opens the Analysis Results panel. Use apply_query_to_analysis_result when the driver asks to filter that view, and add_filtered_driver_expert_comparisons_to_live_range_todo_list to display its filtered comparisons in the overlay while driving when available.',
        ].join(' '),
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
            'Apply a final JSONata expression to the visible Analysis Results tab. The tool returns only its status and does not return the matched results.',
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
            'Each query independently enforces a maximum of 8,192 bytes of compact UTF-8 JSON for the entire { "status": "ready", "data": ... } payload, excluding the transport envelope, and a maximum of 50 items in every returned array, including nested arrays. Oversized results are rejected completely with QUERY_RESULT_LIMIT_EXCEEDED; no partial data is returned. Filter the results, select fewer fields, or aggregate to fit these limits. There are no caller-controlled overrides or pagination parameters.',
            'JSONata can calculate over all analysis data before these output limits are applied. Small complete datasets and repeated bounded queries are allowed. Normalized error details are limited to 1,024 serialized bytes; oversized diagnostics are replaced with a fixed error without the original message or cause.',
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
        name: 'create_repeatable_plan',
        description: [
            'Create one visible repeatable plan that executes ordered model command calls, checks a numeric stopping condition, and repeats the plan until the condition is met. Repetition continues until the target is reached, an error occurs, or the user cancels the plan. The stop-when tool call must return { "status": "ready", "data": finiteNumber }. The operator compares the returned data with the target, so both values must be finite numbers measured on the same scale.',
            'Use the native tool call channel. The arguments object must use workflow as its single outer key, with name set to create_repeatable_plan.',
            'Inside that wrapper, provide an explicit ordered tools list. Each child has a tool object with a name allowed by the current catalog child schema. All workflow categories are forbidden as children, including creation, control, and read commands. Do not include prose-only tasks or hidden steps.',
            'Preserve each child\'s tool-specific arguments object unchanged. Do not unwrap, flatten, rename, or reinterpret keys inside it.',
            'No legacy compatibility: do not use unwrapped creation bodies, requests/steps/events lists, tool-name keys, or payload/args/parameters aliases for child arguments. Result and state fields are not creation inputs.',
            'Each tool object contains a name, a unique id, title, and arguments. stop_when.tool also contains a name and arguments, and cannot contain any workflow creation, control, or read command. Child and stop-check arguments may be omitted only when the chosen tool needs no inputs.',
            'The application executes the ordered calls and stop check. Wait for its later results before reporting completion; do not run the subscribed children again yourself.',
            'For a five-lap analysis, collect a full_lap baseline, analyze it, and query $count(analyses) until at least five analyzed laps are retained. This target counts all retained analyses, including any that already exist.',
            `Example native arguments:
\`\`\`json
{
  "workflow": {
    "name": "create_repeatable_plan",
    "tools": [
      {
        "tool": {
          "name": "collect_live_baseline",
          "id": "collect",
          "title": "Record a full lap",
          "arguments": {
            "query": {
              "preset": "full_lap"
            }
          }
        }
      },
      {
        "tool": {
          "name": "analyze_live_recorded_analysis",
          "id": "analyze",
          "title": "Analyze the recorded lap"
        }
      }
    ],
    "stop_when": {
      "tool": {
        "name": "query_analysis_result",
        "arguments": {
          "query": "$count(analyses)"
        }
      },
      "operator": "gte",
      "target": 5
    },
    "goal": "Analyze five laps"
  }
}
\`\`\``,
        ].join(' '),
        properties: {
            goal: {
                type: 'string',
                description: 'Short name displayed on the repeatable plan card.',
            },
            tools: {
                type: 'array',
                minItems: 1,
                description: 'Ordered tool calls. Each entry contains tool with a name, and every id must be unique. Workflow commands cannot be nested.',
                items: {
                    type: 'object',
                    properties: {
                        id: { type: 'string', description: 'Unique stable step id.' },
                        title: { type: 'string', description: 'Short step label displayed to the user.' },
                        arguments: { type: 'object', default: {}, description: 'Arguments passed unchanged to the nested tool. Defaults to {}.' },
                    },
                    required: ['id', 'title'],
                    additionalProperties: false,
                },
            },
            stop_when: {
                type: 'object',
                description: 'Frontend tool call plus the comparison evaluated after the ordered preparation steps. The tool must return { "status": "ready", "data": finiteNumber }, and its data must be comparable with the target.',
                properties: {
                    tool: {
                        type: 'object',
                        description: 'Frontend tool call that must return { "status": "ready", "data": finiteNumber } to determine whether the repeatable plan reached its target.',
                        properties: {
                            arguments: { type: 'object', default: {}, description: 'Arguments passed unchanged to the stop-when tool. Defaults to {}.' },
                        },
                        required: [],
                        additionalProperties: false,
                    },
                    operator: { type: 'string', enum: ['eq', 'neq', 'lt', 'lte', 'gt', 'gte'] },
                    target: { type: 'number', description: 'Finite number comparable to the value returned by the stop-when tool.' },
                },
                required: ['tool', 'operator', 'target'],
                additionalProperties: false,
            },
        },
        required: ['goal', 'tools', 'stop_when'],
    },
    {
        name: 'retry_repeatable_plan_task',
        description: [
            'Retry the currently failed repeatable plan task once with its stored arguments, then continue the remaining plan after success. Available only when the visible repeatable plan is in an error state with a failed task.',
        ].join(' '),
        properties: {},
        required: [],
    },
    {
        name: 'advance_plan_step',
        description: [
            'Report that the current visible procedure plan request is complete so the UI can move to the next request. The application owns subscribed request execution; use the later tool result or user message to confirm completion before advancing. Do not skip an unfinished request unless the driver explicitly asks to skip it.',
        ].join(' '),
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
        description: [
            'Clear or terminate the visible procedure plan only when the driver explicitly asks to cancel, clear, stop, or opt out of the plan. Do not abandon an active plan merely because it no longer seems useful.',
        ].join(' '),
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
        description: [
            'Create or replace the visible procedure plan to execute ordered tools through the active AI session subscription. Each tool call executes sequentially, and the plan ends when the last call is complete. The plan can be cleared or terminated with clear_procedure_plan.',
            'Use the native tool call channel. The arguments object must use workflow as its single outer key, with name set to set_procedure_plan.',
            'Inside that wrapper, provide an explicit ordered tools list. Each child has a tool object with a name allowed by the current catalog child schema. All workflow categories are forbidden as children, including creation, control, and read commands. Do not include prose-only tasks or hidden steps.',
            'Preserve each child\'s tool-specific arguments object unchanged. Do not unwrap, flatten, rename, or reinterpret keys inside it.',
            'No legacy compatibility: do not use unwrapped creation bodies, requests/steps/events lists, tool-name keys, or payload/args/parameters aliases for child arguments. Result and state fields are not creation inputs.',
            'Each tool object contains a name, title and arguments; arguments is required, using {} for a tool with no inputs.',
            'A procedure plan is active when procedure_plan exists in session context or a tool result includes goal, requests, and current_request. The application owns visible plan state and subscribed request execution.',
            'Tool calls are fire-and-forget. Use the later tool result or user message before deciding what to say or whether another plan step should advance. Do not execute subscribed children again yourself.',
            'Do not skip, clear, replace, or abandon an active plan unless the driver explicitly asks to cancel, clear, stop, skip, or opt out of the plan.',
            `Example native arguments:
\`\`\`json
{
  "workflow": {
    "name": "set_procedure_plan",
    "goal": "Review the Spa opening section",
    "tools": [
      {
        "tool": {
          "name": "show_map",
          "title": "Show the opening section",
          "arguments": {
            "source_track_key": "spa",
            "section_start": 0,
            "section_end": 0.2
          }
        }
      }
    ]
  }
}
\`\`\``,
        ].join(' '),
        properties: {
            goal: {
                type: 'string',
                description: 'Short goal shown above the request list.',
            },
            tools: {
                type: 'array',
                minItems: 1,
                description: 'Ordered tool calls. Each entry contains tool with a name; repeated calls to the same tool are allowed. Workflow commands cannot be nested.',
                items: {
                    type: 'object',
                    properties: {
                        title: { type: 'string', minLength: 1, pattern: '\\S' },
                        arguments: {
                            type: 'object',
                            description: 'Arguments passed unchanged to the nested tool.',
                        },
                    },
                    required: ['title', 'arguments'],
                    additionalProperties: false,
                },
            },
        },
        required: ['goal', 'tools'],
    },
    {
        name: 'get_next_corner',
        description: 'Return the name and normalized distance of the next corner ahead. Use for live questions about what corner is coming up.',
        properties: {},
        required: [],
    },
    {
        name: 'query_telemetry_metric',
        description: [
            'Ask for the current, average, minimum, or maximum telemetry value for selected fields over a live-session scope and return summarized numbers instead of raw telemetry rows. Do not use `query_telemetry_metric` for performance checking, pace diagnosis, or track-improvement requests; use `live_performance_analyst` for those.',
            'Use summarized telemetry numbers when they naturally answer the driver\'s question. This is not a required step before or after analyze_telemetry.',
        ].join(' '),
        properties: {
            fields: {
                type: 'array',
                description: 'Telemetry fields to summarize. Use only the supported exact field names.',
                items: TELEMETRY_METRIC_FIELD_SCHEMA,
            },
            scope: {
                ...MODEL_COMMAND_QUERY_SCOPE_SCHEMA,
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
        description: [
            'Read session events when they may explain performance, such as incidents, traffic, or interruptions. Select a supported eventType and scope; use n with last_n.',
        ].join(' '),
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
        description: [
            'Return map-level user summary data. With no map_id, returns all maps with aggregate stats and top sections; with map_id, returns that map with full section breakdowns, mistake counts, expert-adherence counts, percentages, and segment/category summaries.',
            'During live performance analysis, use long-term driver history only when it improves the current analysis; keep live telemetry as the primary source of truth.',
        ].join(' '),
        properties: {
            map_id: {
                type: 'string',
            },
        },
        required: [],
    },
    {
        name: 'get_available_user_summary_maps',
        description: [
            'Return a compact list of maps that have user summary data, including map id, name, analyzed session count, total analyzed time count, section count, and a human-readable map_options list.',
            'During live performance analysis, use long-term driver history only when it improves the current analysis; keep live telemetry as the primary source of truth.',
        ].join(' '),
        properties: {},
        required: [],
    },
    {
        name: 'search_user_summary_map_level',
        description: [
            'Search map-level user summary rows by map name, map id, top mistake section names or ids, top expert-adherence section names or ids, and aggregate words like mistake, weakness, expert, or strength. Returns scored matching maps with matched_fields.',
            'During live performance analysis, use long-term driver history only when it improves the current analysis; keep live telemetry as the primary source of truth.',
        ].join(' '),
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
        description: [
            'Display a circuit map in the chat transcript, optionally highlighting a normalized lap section.',
            'Use when a map helps the driver locate an identified section; highlight the normalized lap section when available.',
        ].join(' '),
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
        description: [
            'Classify driving actions over a telemetry scope and return engineer labels with definitions and optional solutions. Use this to do a quick analysis of a telemetry window without launching a dedicated ai analysis agent.',
            'Use when an answer needs available telemetry or detected driving behaviours, rather than for simple conversation or common racing concepts. Use explain_label when a detected behaviour needs a clearer meaning or coaching explanation; it is not required for every telemetry result.',
            'If telemetry is unavailable or the tool errors, say so plainly. Never fabricate numbers, driving behaviours, or label names; translate technical label codes into natural driving descriptions.',
        ].join(' '),
        properties: {
            scope: {
                ...MODEL_COMMAND_QUERY_SCOPE_SCHEMA,
                description: 'Telemetry time window to classify.',
            },
        },
        required: ['scope'],
    },
] as const;

type ModelCommandDefinition = typeof MODEL_COMMAND_DEFINITIONS[number];
type ModelCommandName = ModelCommandDefinition['name'];
type SessionMode = 'front_desk' | 'live' | 'recorded' | 'user_summary';

const COMMON_COMMAND_NAMES: ModelCommandName[] = [
    'show_map',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    'stop_agent_session',
];

const LIVE_COMMAND_NAMES: ModelCommandName[] = [
    'start_agent_session',
    'apply_query_to_analysis_result',
    'query_analysis_result',
    'analyze_telemetry',
    'get_next_corner',
    'query_telemetry_metric',
    'get_event_log',
];

const LIVE_AGENT_COMMAND_NAMES: ModelCommandName[] = [
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

const LIVE_PERFORMANCE_ANALYST_COMMAND_NAMES: ModelCommandName[] = [
    'create_repeatable_plan',
    'retry_repeatable_plan_task',
    'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
];

// All AI calls use explicit workflow or tool envelopes.
const WORKFLOW_COMMAND_NAMES = new Set<ModelCommandName>([
    'create_repeatable_plan',
    'retry_repeatable_plan_task',
    'set_procedure_plan',
    'advance_plan_step',
    'clear_procedure_plan',
    'add_event_to_live_range_todo_list',
    'get_live_range_todo_list',
    'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
]);

const USER_SUMMARY_COMMAND_NAMES: ModelCommandName[] = [
    'get_user_summary_map_level',
    'get_available_user_summary_maps',
    'search_user_summary_map_level',
];

const RECORDED_COMMAND_NAMES: ModelCommandName[] = [
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
        return new Set<ModelCommandName>([
            ...COMMON_COMMAND_NAMES,
            ...LIVE_AGENT_COMMAND_NAMES,
            ...USER_SUMMARY_COMMAND_NAMES,
            ...(sessionMode === 'live' ? [
                'apply_query_to_analysis_result',
                'query_analysis_result',
            ] as const : []),
            ...(sessionMode === 'live' && agentMode === 'live_performance_analyst'
                ? LIVE_PERFORMANCE_ANALYST_COMMAND_NAMES
                : []),
        ]);
    }

    if (sessionMode === 'recorded') {
        return new Set<ModelCommandName>([
            ...COMMON_COMMAND_NAMES,
            ...USER_SUMMARY_COMMAND_NAMES,
            ...RECORDED_COMMAND_NAMES,
        ]);
    }

    if (sessionMode === 'user_summary') {
        return new Set<ModelCommandName>([
            ...COMMON_COMMAND_NAMES,
            ...USER_SUMMARY_COMMAND_NAMES,
        ]);
    }

    if (sessionMode === 'front_desk') {
        return new Set<ModelCommandName>([
            ...COMMON_COMMAND_NAMES,
            ...USER_SUMMARY_COMMAND_NAMES,
        ]);
    }

    return new Set<ModelCommandName>([
        ...COMMON_COMMAND_NAMES,
        ...LIVE_COMMAND_NAMES,
        ...USER_SUMMARY_COMMAND_NAMES,
    ]);
};

const createToolBodySchema = (
    toolNames: ModelCommandName[],
    metadata: { properties: object; required: readonly string[] },
) => ({
    type: 'object',
    oneOf: toolNames.map((name) => ({
        type: 'object',
        properties: { name: { type: 'string', enum: [name] }, ...metadata.properties },
        required: ['name', ...metadata.required],
        additionalProperties: false,
    })),
});

const createToolCallSchema = (
    toolNames: ModelCommandName[],
    metadata: { properties: object; required: readonly string[] },
) => ({
    type: 'object',
    properties: { tool: createToolBodySchema(toolNames, metadata) },
    required: ['tool'],
    additionalProperties: false,
});

const expandWorkflowSchemas = (commands: readonly ModelCommandDefinition[]) => {
    const nestedToolNames = commands
        .map(({ name }) => name)
        .filter((name) => !WORKFLOW_COMMAND_NAMES.has(name));

    return commands.map((command) => {
        if (!WORKFLOW_COMMAND_NAMES.has(command.name)) {
            return {
                ...command,
                description: `${command.description} Use native arguments { "tool": { "name": "${command.name}", "arguments": { ... } } }. Put this tool's inputs inside arguments; omit arguments only for a tool with no required inputs.`,
                properties: {
                    tool: {
                        type: 'object',
                        properties: {
                            name: { type: 'string', enum: [command.name] },
                            arguments: {
                                type: 'object',
                                properties: command.properties,
                                required: command.required,
                                additionalProperties: false,
                            },
                        },
                        required: command.required.length ? ['name', 'arguments'] : ['name'],
                        additionalProperties: false,
                    },
                },
                required: ['tool'],
            };
        }
        if (
            command.name !== 'set_procedure_plan'
            && command.name !== 'create_repeatable_plan'
            && command.name !== 'add_event_to_live_range_todo_list'
        ) return {
            ...command,
            description: `${command.description} Use native arguments { "workflow": { "name": "${command.name}", "tools": [] } }. Put any optional reason inside workflow. This command uses the existing workflow, so tools must be empty.`,
            properties: {
                workflow: {
                    type: 'object',
                    properties: {
                        name: { type: 'string', enum: [command.name] },
                        tools: { type: 'array', items: {}, maxItems: 0 },
                        ...command.properties,
                    },
                    required: ['name', 'tools', ...command.required],
                    additionalProperties: false,
                },
            },
            required: ['workflow'],
        };

        const properties = {
            name: { type: 'string', enum: [command.name] },
            ...command.properties,
            tools: {
                ...command.properties.tools,
                items: createToolCallSchema(nestedToolNames, command.properties.tools.items),
            },
            ...(command.name === 'create_repeatable_plan' ? {
                stop_when: {
                    ...command.properties.stop_when,
                    properties: {
                        ...command.properties.stop_when.properties,
                        tool: {
                            ...createToolBodySchema(
                                nestedToolNames,
                                command.properties.stop_when.properties.tool,
                            ),
                            description: command.properties.stop_when.properties.tool.description,
                        },
                    },
                },
            } : {}),
        };

        return {
            ...command,
            properties: {
                workflow: {
                    type: 'object',
                    properties,
                    required: ['name', ...command.required],
                    additionalProperties: false,
                },
            },
            required: ['workflow'],
        };
    });
};

export const MODEL_COMMAND_PROTOCOL = expandWorkflowSchemas(MODEL_COMMAND_DEFINITIONS);

export const getModelCommandsForSessionContext = (
    sessionContext: Record<string, unknown> | null | undefined,
) => {
    const sessionMode = isSessionMode(sessionContext?.session_mode)
        ? sessionContext.session_mode
        : 'live';
    const allowedToolNames = getAllowedToolNames(
        sessionMode,
        sessionContext?.agent_mode,
    );

    const commands = MODEL_COMMAND_DEFINITIONS.filter(({ name }) => allowedToolNames.has(name));
    return expandWorkflowSchemas(commands).map((command) => ({
        name: command.name,
        description: 'description' in command && typeof command.description === 'string'
            ? command.description
            : '',
        properties: command.properties,
        required: [...command.required],
    }));
};
