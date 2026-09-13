export type OperationErrorOptions = {
    cause?: unknown;
};

export type SerializedError = {
    name: string;
    message: string;
    detail?: SerializedErrorDetail;
    cause?: SerializedErrorCause;
};

export type SerializedErrorDetail = {
    code: string;
    position?: number;
    token?: string;
    message: string;
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

const readDataProperty = (value: object, key: string): unknown => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && 'value' in descriptor ? descriptor.value : undefined;
};

const serializeErrorDetail = (error: Error): SerializedErrorDetail | undefined => {
    const detail = readDataProperty(error, 'detail');
    if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return undefined;

    const code = readDataProperty(detail, 'code');
    const position = readDataProperty(detail, 'position');
    const token = readDataProperty(detail, 'token');
    const message = readDataProperty(detail, 'message');
    if (typeof code !== 'string' || typeof message !== 'string') return undefined;
    if (position !== undefined && (
        typeof position !== 'number'
        || !Number.isInteger(position)
        || position < 0
    )) return undefined;
    if (token !== undefined && typeof token !== 'string') return undefined;

    return {
        code,
        ...(position !== undefined ? { position } : {}),
        ...(token !== undefined ? { token } : {}),
        message,
    };
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
        const detail = serializeErrorDetail(value);
        if (detail) serialized.detail = detail;
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

export const serializeError = (error: Error): SerializedError => (
    serializeCause(error, 0, new Set()) as SerializedError
);

export class OperationError extends Error {
    override name = 'OperationError';

    constructor(message: string, options: OperationErrorOptions = {}) {
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

export interface OperationError {
    readonly cause?: unknown;
}

export class OperationExecutionError extends OperationError {
    override name = 'OperationExecutionError';
}

export class OperationNotRegisteredError extends OperationError {
    override name = 'OperationNotRegisteredError';
}

export class InvalidOperationCallError extends OperationError {
    override name = 'InvalidOperationCallError';
}

export class NoLiveSessionError extends OperationError {
    override name = 'NoLiveSessionError';
}

export class NoLiveTelemetryError extends OperationError {
    override name = 'NoLiveTelemetryError';
}

export class TelemetryFieldsRequiredError extends OperationError {
    override name = 'TelemetryFieldsRequiredError';
}

export class NoCornerDataError extends OperationError {
    override name = 'NoCornerDataError';
}

export class CreateGoalWorkflowUnavailableError extends OperationError {
    override name = 'CreateGoalWorkflowUnavailableError';
}

export class RetryGoalTaskWorkflowUnavailableError extends OperationError {
    override name = 'RetryGoalTaskWorkflowUnavailableError';
}

export class InvalidProcedurePlanRequestsError extends OperationError {
    override name = 'InvalidProcedurePlanRequestsError';
}

export class CircuitMapLookupFailedError extends OperationError {
    override name = 'CircuitMapLookupFailedError';
}

export class NoTelemetryForScopeError extends OperationError {
    override name = 'NoTelemetryForScopeError';
}

export class TelemetryAnalysisFailedError extends OperationError {
    override name = 'TelemetryAnalysisFailedError';
}

export class AmbiguousComponentTargetError extends OperationError {
    override name = 'AmbiguousComponentTargetError';
}

export class VisualizationControlUnavailableError extends OperationError {
    override name = 'VisualizationControlUnavailableError';
}

export class UnsupportedAgentModeError extends OperationError {
    override name = 'UnsupportedAgentModeError';
}

export class NotRecordedModeError extends OperationError {
    override name = 'NotRecordedModeError';
}

export const normalizeOperationError = (error: unknown): OperationError => {
    if (error instanceof OperationError) return error;
    const message = error instanceof Error && error.message.trim()
        ? error.message
        : typeof error === 'string' && error.trim()
            ? error
            : 'Operation execution failed.';
    return new OperationExecutionError(message, { cause: error });
};
