export type AiToolErrorOptions = {
    cause?: unknown;
};

export type SerializedError = {
    name: string;
    message: string;
    cause?: SerializedErrorCause;
};

export type SerializedErrorCause =
    | string
    | number
    | boolean
    | null
    | SerializedError;

const MAX_SERIALIZED_CAUSE_DEPTH = 5;
const CIRCULAR_CAUSE_MESSAGE = '[Circular cause]';
const TRUNCATED_CAUSE_MESSAGE = '[Cause chain truncated]';

const safeString = (value: unknown): string => {
    try {
        return String(value);
    } catch {
        return '[Unserializable cause]';
    }
};

const serializeCause = (
    value: unknown,
    depth: number,
    seen: Set<unknown>,
): SerializedErrorCause => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') {
        return value;
    }
    if (typeof value === 'number') {
        return Number.isFinite(value) ? value : safeString(value);
    }
    if (depth >= MAX_SERIALIZED_CAUSE_DEPTH) return TRUNCATED_CAUSE_MESSAGE;
    if (seen.has(value)) return CIRCULAR_CAUSE_MESSAGE;

    if (value instanceof Error) {
        seen.add(value);
        const serialized: SerializedError = {
            name: value.name || 'Error',
            message: value.message,
        };
        const cause = (value as Error & { cause?: unknown }).cause;
        if (cause !== undefined) {
            serialized.cause = serializeCause(cause, depth + 1, seen);
        }
        seen.delete(value);
        return serialized;
    }

    return safeString(value);
};

export const serializeErrorCause = (cause: unknown): SerializedErrorCause => (
    serializeCause(cause, 0, new Set())
);

export class AiToolError extends Error {
    override name = 'AiToolError';

    constructor(message: string, options: AiToolErrorOptions = {}) {
        super(message);
        if (Object.prototype.hasOwnProperty.call(options, 'cause')) {
            Object.defineProperty(this, 'cause', {
                configurable: true,
                enumerable: false,
                value: options.cause,
                writable: false,
            });
        }
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

export interface AiToolError {
    readonly cause?: unknown;
}

export class ToolExecutionError extends AiToolError {
    override name = 'ToolExecutionError';
}

export class ToolNotRegisteredError extends AiToolError {
    override name = 'ToolNotRegisteredError';
}

export class InvalidToolCallError extends AiToolError {
    override name = 'InvalidToolCallError';
}

export class NoLiveSessionError extends AiToolError {
    override name = 'NoLiveSessionError';
}

export class NoLiveTelemetryError extends AiToolError {
    override name = 'NoLiveTelemetryError';
}

export class TelemetryFieldsRequiredError extends AiToolError {
    override name = 'TelemetryFieldsRequiredError';
}

export class NoCornerDataError extends AiToolError {
    override name = 'NoCornerDataError';
}

export class CreateGoalToolUnavailableError extends AiToolError {
    override name = 'CreateGoalToolUnavailableError';
}

export class RetryGoalTaskToolUnavailableError extends AiToolError {
    override name = 'RetryGoalTaskToolUnavailableError';
}

export class InvalidProcedurePlanRequestsError extends AiToolError {
    override name = 'InvalidProcedurePlanRequestsError';
}

export class CircuitMapLookupFailedError extends AiToolError {
    override name = 'CircuitMapLookupFailedError';
}

export class NoTelemetryForScopeError extends AiToolError {
    override name = 'NoTelemetryForScopeError';
}

export class TelemetryAnalysisFailedError extends AiToolError {
    override name = 'TelemetryAnalysisFailedError';
}

export class AmbiguousComponentTargetError extends AiToolError {
    override name = 'AmbiguousComponentTargetError';
}

export class VisualizationControlUnavailableError extends AiToolError {
    override name = 'VisualizationControlUnavailableError';
}

export class UnsupportedAgentModeError extends AiToolError {
    override name = 'UnsupportedAgentModeError';
}

export class NotRecordedModeError extends AiToolError {
    override name = 'NotRecordedModeError';
}

export const normalizeAiToolError = (error: unknown): AiToolError => {
    if (error instanceof AiToolError) return error;
    const message = error instanceof Error && error.message.trim()
        ? error.message
        : typeof error === 'string' && error.trim()
            ? error
            : 'Tool execution failed.';
    return new ToolExecutionError(message, { cause: error });
};
