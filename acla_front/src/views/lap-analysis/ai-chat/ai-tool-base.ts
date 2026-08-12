export type ToolOutputStatus = string;
export type AiToolNormalOutput = object;
export type AiToolExecutionOutput = AiToolNormalOutput | Error;

export {
    AiToolError,
    AmbiguousComponentTargetError,
    CircuitMapLookupFailedError,
    CreateGoalToolUnavailableError,
    FocusSectionNotReadyError,
    InvalidProcedurePlanRequestsError,
    InvalidToolCallError,
    LivePerformanceAnalystToolUnavailableError,
    LiveSectionClassificationFailedError,
    LiveSectionTelemetryUnavailableError,
    NoCornerDataError,
    NoLiveSessionError,
    NoLiveTelemetryError,
    NoTelemetryForScopeError,
    NotRecordedModeError,
    RetryGoalTaskToolUnavailableError,
    SectionNotFoundError,
    TelemetryAnalysisFailedError,
    TelemetryFieldsRequiredError,
    ToolExecutionError,
    ToolNotRegisteredError,
    UnsupportedAgentModeError,
    VisualizationControlUnavailableError,
    normalizeAiToolError,
    serializeErrorCause,
} from 'errors/AiToolError';
export type {
    AiToolErrorOptions,
    SerializedError,
    SerializedErrorCause,
} from 'errors/AiToolError';

export type ToolOutputEnvelope = {
    tool_name: string;
    run_id: string;
    status: ToolOutputStatus;
    output: object;
    message?: string;
    final: true;
    progress_percent?: number;
    /** Chat-only UI value. Non-enumerable so it never enters a WS frame. */
    readonly uiOutput?: object;
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

export const getToolEnvelopeUiOutput = (
    envelope: ToolOutputEnvelope,
): AiToolNormalOutput | undefined => envelope.uiOutput;

export const isToolOutputEnvelope = (value: unknown): value is ToolOutputEnvelope => (
    isRecord(value)
    && typeof value.tool_name === 'string'
    && typeof value.run_id === 'string'
    && typeof value.status === 'string'
    && value.final === true
    && isRecord(value.output)
);
