export type ToolOutputStatus = string;

export type ToolOutputEnvelope = {
    tool_name: string;
    run_id: string;
    status: ToolOutputStatus;
    payload: unknown;
    message?: string;
    error?: string;
    final: boolean;
    progress_percent?: number;
};

export type ToolOutputEmitOptions = {
    final?: boolean;
};

export type ToolOutputEmitter = (
    envelope: ToolOutputEnvelope,
    options?: ToolOutputEmitOptions,
) => void;

export type ToolOutputController = {
    progress: (payload: unknown, options?: { message?: string; progressPercent?: number }) => ToolOutputEnvelope;
    final: (payload: unknown, options?: { message?: string }) => ToolOutputEnvelope;
    error: (error: string, payload?: unknown, options?: { message?: string }) => ToolOutputEnvelope;
    getFinalOutput: () => ToolOutputEnvelope | null;
};

export type AiToolVisibility = 'public' | 'agent' | 'internal';

export type AiToolSchema = {
    properties: Record<string, unknown>;
    required: string[];
};

export type AiToolDefinition<TContext, THandlerContext> = {
    name: string;
    description?: string;
    schema: AiToolSchema;
    required: string[];
    sessionModes: Array<'live' | 'recorded' | 'user_summary'>;
    agentModes?: Array<'track_guide' | 'overtake' | 'live_performance_analyst'>;
    visibility: AiToolVisibility;
    execute: (
        args: Record<string, unknown>,
        context: TContext,
        output: ToolOutputController,
        handlerContext: THandlerContext,
    ) => Promise<unknown> | unknown;
    formatOutput?: (result: unknown) => unknown;
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const getPayloadStatus = (
    payload: unknown,
    fallback: ToolOutputStatus,
): ToolOutputStatus => {
    if (isRecord(payload) && typeof payload.status === 'string' && payload.status) {
        return payload.status;
    }
    return fallback;
};

const getPayloadMessage = (payload: unknown): string | undefined => (
    isRecord(payload) && typeof payload.message === 'string'
        ? payload.message
        : undefined
);

const getPayloadError = (payload: unknown): string | undefined => (
    isRecord(payload) && typeof payload.error === 'string'
        ? payload.error
        : undefined
);

const getPayloadProgressPercent = (payload: unknown): number | undefined => {
    if (!isRecord(payload)) return undefined;
    const raw = payload.progress_percent ?? payload.progressPercent;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : undefined;
};

export const isToolOutputEnvelope = (value: unknown): value is ToolOutputEnvelope => (
    isRecord(value)
    && typeof value.tool_name === 'string'
    && typeof value.run_id === 'string'
    && typeof value.status === 'string'
    && typeof value.final === 'boolean'
    && 'payload' in value
);

const createEnvelope = (
    toolName: string,
    runId: string,
    payload: unknown,
    final: boolean,
    fallbackStatus: ToolOutputStatus,
    options: {
        message?: string;
        error?: string;
        progressPercent?: number;
    } = {},
): ToolOutputEnvelope => {
    const payloadRecord = isRecord(payload) ? payload : {};
    const payloadError = getPayloadError(payload);
    const envelope: ToolOutputEnvelope = {
        ...payloadRecord,
        tool_name: toolName,
        run_id: runId,
        status: options.error || payloadError ? 'error' : getPayloadStatus(payload, fallbackStatus),
        payload,
        message: options.message ?? getPayloadMessage(payload),
        error: options.error ?? payloadError,
        final,
        progress_percent: options.progressPercent ?? getPayloadProgressPercent(payload),
    };

    if (envelope.message === undefined) {
        delete envelope.message;
    }
    if (envelope.error === undefined) {
        delete envelope.error;
    }
    if (envelope.progress_percent === undefined) {
        delete envelope.progress_percent;
    }

    return envelope;
};

export const createToolOutputController = (
    toolName: string,
    runId: string,
    emit?: ToolOutputEmitter,
): ToolOutputController => {
    let finalOutput: ToolOutputEnvelope | null = null;

    const emitEnvelope = (envelope: ToolOutputEnvelope) => {
        if (envelope.final) {
            finalOutput = envelope;
        }
        emit?.(envelope, { final: envelope.final });
        return envelope;
    };

    return {
        progress(payload, options = {}) {
            return emitEnvelope(createEnvelope(
                toolName,
                runId,
                payload,
                false,
                'progress',
                options,
            ));
        },
        final(payload, options = {}) {
            return emitEnvelope(createEnvelope(
                toolName,
                runId,
                payload,
                true,
                'complete',
                options,
            ));
        },
        error(error, payload = { status: 'error', error }, options = {}) {
            return emitEnvelope(createEnvelope(
                toolName,
                runId,
                payload,
                true,
                'error',
                {
                    ...options,
                    error,
                },
            ));
        },
        getFinalOutput() {
            return finalOutput;
        },
    };
};

export const getToolEnvelopeError = (envelope: ToolOutputEnvelope): string | null => {
    if (envelope.error) return envelope.error;
    return envelope.status === 'error' ? envelope.message || 'Tool failed.' : null;
};

export const toFrontendToolSchema = <TContext, THandlerContext>(
    definition: AiToolDefinition<TContext, THandlerContext>,
) => ({
    name: definition.name,
    description: definition.description,
    properties: definition.schema.properties,
    required: definition.required,
});

export const executeAiToolDefinition = async <TContext, THandlerContext extends {
    toolRunId?: string;
    toolName?: string;
    sendToolOutput?: ToolOutputEmitter;
}>(
    definition: AiToolDefinition<TContext, THandlerContext>,
    args: Record<string, unknown>,
    context: TContext,
    handlerContext: THandlerContext,
): Promise<ToolOutputEnvelope> => {
    const runId = handlerContext.toolRunId || `${definition.name}-${Date.now()}`;
    const output = createToolOutputController(
        definition.name,
        runId,
        handlerContext.sendToolOutput,
    );

    try {
        const rawResult = await definition.execute(args, context, output, {
            ...handlerContext,
            toolName: definition.name,
            toolRunId: runId,
        });
        if (isToolOutputEnvelope(rawResult)) {
            return rawResult;
        }
        const finalOutput = output.getFinalOutput();
        if (finalOutput) {
            return finalOutput;
        }
        const formatted = definition.formatOutput
            ? definition.formatOutput(rawResult)
            : rawResult;
        return output.final(formatted);
    } catch (error) {
        const message = (error as Error)?.message || String(error);
        return output.error(message, {
            status: 'error',
            error: message,
            message,
        });
    }
};
