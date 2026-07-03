export type ToolOutputStatus = string;

export type ToolOutputEnvelope = {
    tool_name: string;
    run_id: string;
    status: ToolOutputStatus;
    output: unknown;
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
    progress: (uiOutput: unknown, options?: {
        message?: string;
        progressPercent?: number;
        aiOutput?: unknown;
    }) => ToolOutputEnvelope;
    final: (uiOutput: unknown, options?: {
        message?: string;
        aiOutput?: unknown;
    }) => ToolOutputEnvelope;
    error: (error: string, uiOutput?: unknown, options?: {
        message?: string;
        aiOutput?: unknown;
    }) => ToolOutputEnvelope;
    getFinalOutput: () => ToolOutputEnvelope | null;
};

export type AiToolSchema = {
    properties: Record<string, unknown>;
    required: string[];
};

export type AiToolDefinition<TContext, THandlerContext> = {
    name: string;
    description?: string;
    schema: AiToolSchema;
    required: string[];
    execute: (
        args: Record<string, unknown>,
        context: TContext,
        output: ToolOutputController,
        handlerContext: THandlerContext,
    ) => Promise<unknown> | unknown;
    formatOutput?: (result: unknown) => unknown;
    formatAiOutput?: (uiOutput: unknown, result: unknown) => unknown;
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const getUiOutputStatus = (
    uiOutput: unknown,
    fallback: ToolOutputStatus,
): ToolOutputStatus => {
    if (isRecord(uiOutput) && typeof uiOutput.status === 'string' && uiOutput.status) {
        return uiOutput.status;
    }
    return fallback;
};

const getUiOutputMessage = (uiOutput: unknown): string | undefined => (
    isRecord(uiOutput) && typeof uiOutput.message === 'string'
        ? uiOutput.message
        : undefined
);

const getUiOutputError = (uiOutput: unknown): string | undefined => (
    isRecord(uiOutput) && typeof uiOutput.error === 'string'
        ? uiOutput.error
        : undefined
);

const getUiOutputProgressPercent = (uiOutput: unknown): number | undefined => {
    if (!isRecord(uiOutput)) return undefined;
    const raw = uiOutput.progress_percent ?? uiOutput.progressPercent;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : undefined;
};

const buildDefaultAiOutput = (
    toolName: string,
    status: ToolOutputStatus,
    message?: string,
    error?: string,
): Record<string, unknown> => {
    const output: Record<string, unknown> = {
        name: toolName,
        status,
    };
    if (message) {
        output.message = message;
    }
    if (error) {
        output.error = error;
    }
    return output;
};

const toolEnvelopeUiOutput = new WeakMap<ToolOutputEnvelope, unknown>();

export const getToolEnvelopeUiOutput = (envelope: ToolOutputEnvelope): unknown => (
    toolEnvelopeUiOutput.get(envelope)
);

export const isToolOutputEnvelope = (value: unknown): value is ToolOutputEnvelope => (
    isRecord(value)
    && typeof value.tool_name === 'string'
    && typeof value.run_id === 'string'
    && typeof value.status === 'string'
    && typeof value.final === 'boolean'
    && 'output' in value
);

const createEnvelope = (
    toolName: string,
    runId: string,
    uiOutput: unknown,
    final: boolean,
    fallbackStatus: ToolOutputStatus,
    options: {
        message?: string;
        error?: string;
        progressPercent?: number;
        aiOutput?: unknown;
    } = {},
): ToolOutputEnvelope => {
    const uiOutputError = getUiOutputError(uiOutput);
    const status = options.error || uiOutputError ? 'error' : getUiOutputStatus(uiOutput, fallbackStatus);
    const message = options.message ?? getUiOutputMessage(uiOutput);
    const error = options.error ?? uiOutputError;
    const envelope: ToolOutputEnvelope = {
        tool_name: toolName,
        run_id: runId,
        status,
        output: options.aiOutput ?? buildDefaultAiOutput(toolName, status, message, error),
        message,
        error,
        final,
        progress_percent: options.progressPercent ?? getUiOutputProgressPercent(uiOutput),
    };
    toolEnvelopeUiOutput.set(envelope, uiOutput);

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
        progress(uiOutput, options = {}) {
            return emitEnvelope(createEnvelope(
                toolName,
                runId,
                uiOutput,
                false,
                'progress',
                options,
            ));
        },
        final(uiOutput, options = {}) {
            return emitEnvelope(createEnvelope(
                toolName,
                runId,
                uiOutput,
                true,
                'complete',
                options,
            ));
        },
        error(error, uiOutput = { status: 'error', error }, options = {}) {
            return emitEnvelope(createEnvelope(
                toolName,
                runId,
                uiOutput,
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

export const executeAiToolDefinition = async <TContext, THandlerContext extends {
    toolRunId?: string;
    toolName?: string;
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
        const aiOutput = definition.formatAiOutput
            ? definition.formatAiOutput(formatted, rawResult)
            : undefined;
        return output.final(formatted, { aiOutput });
    } catch (error) {
        const message = (error as Error)?.message || String(error);
        return output.error(message, {
            status: 'error',
            error: message,
            message,
        });
    }
};

