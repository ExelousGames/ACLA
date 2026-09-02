export type AiToolNormalOutput = { [key: string]: unknown };
export type AiToolExecutionOutput = AiToolNormalOutput | string | Error;
export type AiToolStatusPayload = { [key: string]: unknown };

export {
    createAiToolDeferred,
    createControlledAiToolOperation,
    createAiToolOperation,
    createAiToolOperationFrom,
    mapAiToolOperation,
    resolvedAiToolOperation,
} from 'components/ai-engineering-tools/ai-tool-operation';
export type {
    AiToolDeferred,
    AiToolOperation,
    AiToolOperationResult,
    AiToolOperationStatus,
    AiToolOperationTerminationStatus,
    AiToolTermination,
    ControlledAiToolOperation,
} from 'components/ai-engineering-tools/ai-tool-operation';

export {
    AiToolError,
    AmbiguousComponentTargetError,
    CircuitMapLookupFailedError,
    CreateGoalToolUnavailableError,
    InvalidProcedurePlanRequestsError,
    InvalidToolCallError,
    NoCornerDataError,
    NoLiveSessionError,
    NoLiveTelemetryError,
    NoTelemetryForScopeError,
    NotRecordedModeError,
    RetryGoalTaskToolUnavailableError,
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
    SerializedErrorDetail,
} from 'errors/AiToolError';
