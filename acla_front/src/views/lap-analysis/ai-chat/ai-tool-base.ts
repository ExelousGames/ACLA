export type ToolOutputStatus = string;

export type AiToolNormalOutput = object;
export type AiToolExecutionOutput = AiToolNormalOutput | Error;

export type AiToolErrorOptions = {
    details?: Record<string, unknown>;
    cause?: unknown;
};

export class AiToolError extends Error {
    readonly code: string;
    readonly details?: Record<string, unknown>;
    readonly cause?: unknown;

    constructor(code: string, message: string, options: AiToolErrorOptions = {}) {
        super(message);
        this.name = 'AiToolError';
        this.code = code;
        this.details = options.details;
        this.cause = options.cause;
    }
}

export const normalizeAiToolError = (
    error: unknown,
    fallbackCode = 'tool_execution_failed',
): AiToolError => {
    if (error instanceof AiToolError) return error;
    const message = error instanceof Error && error.message.trim()
        ? error.message
        : typeof error === 'string' && error.trim()
            ? error
            : 'Tool execution failed.';
    return new AiToolError(fallbackCode, message, { cause: error });
};

export type ToolOutputEnvelope = {
    tool_name: string;
    run_id: string;
    status: ToolOutputStatus;
    output: AiToolExecutionOutput;
    message?: string;
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
    progress: (uiOutput: AiToolNormalOutput, options?: {
        message?: string;
        progressPercent?: number;
        aiOutput?: AiToolNormalOutput;
    }) => ToolOutputEnvelope;
    final: (uiOutput: AiToolNormalOutput, options?: {
        message?: string;
        aiOutput?: AiToolNormalOutput;
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
    ) => Promise<AiToolExecutionOutput> | AiToolExecutionOutput;
    formatOutput?: (result: AiToolExecutionOutput) => AiToolNormalOutput;
    formatAiOutput?: (
        uiOutput: AiToolNormalOutput,
        result: AiToolExecutionOutput,
    ) => AiToolNormalOutput;
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const getUiOutputStatus = (
    uiOutput: AiToolNormalOutput,
    fallback: ToolOutputStatus,
): ToolOutputStatus => {
    if (isRecord(uiOutput) && typeof uiOutput.status === 'string' && uiOutput.status) {
        return uiOutput.status;
    }
    return fallback;
};

const getUiOutputMessage = (uiOutput: AiToolNormalOutput): string | undefined => (
    isRecord(uiOutput) && typeof uiOutput.message === 'string'
        ? uiOutput.message
        : undefined
);

const getUiOutputProgressPercent = (uiOutput: AiToolNormalOutput): number | undefined => {
    if (!isRecord(uiOutput)) return undefined;
    const raw = uiOutput.progress_percent ?? uiOutput.progressPercent;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : undefined;
};

const buildDefaultAiOutput = (
    toolName: string,
    status: ToolOutputStatus,
    message?: string,
): Record<string, unknown> => {
    const output: Record<string, unknown> = {
        name: toolName,
        status,
    };
    if (message) {
        output.message = message;
    }
    return output;
};

const toolEnvelopeUiOutput = new WeakMap<ToolOutputEnvelope, AiToolNormalOutput>();

export const getToolEnvelopeUiOutput = (envelope: ToolOutputEnvelope): AiToolNormalOutput | undefined => (
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
    uiOutput: AiToolNormalOutput,
    final: boolean,
    fallbackStatus: ToolOutputStatus,
    options: {
        message?: string;
        progressPercent?: number;
        aiOutput?: AiToolNormalOutput;
    } = {},
): ToolOutputEnvelope => {
    const status = getUiOutputStatus(uiOutput, fallbackStatus);
    const message = options.message ?? getUiOutputMessage(uiOutput);
    const envelope: ToolOutputEnvelope = {
        tool_name: toolName,
        run_id: runId,
        status,
        output: options.aiOutput ?? buildDefaultAiOutput(toolName, status, message),
        message,
        final,
        progress_percent: options.progressPercent ?? getUiOutputProgressPercent(uiOutput),
    };
    toolEnvelopeUiOutput.set(envelope, uiOutput);

    if (envelope.message === undefined) {
        delete envelope.message;
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
        getFinalOutput() {
            return finalOutput;
        },
    };
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
    if (formatted instanceof Error) throw formatted;
    const aiOutput = definition.formatAiOutput
        ? definition.formatAiOutput(formatted, rawResult)
        : undefined;
    return output.final(formatted, { aiOutput });
};

