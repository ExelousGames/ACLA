import type { Operation, OperationExecutionOutput, OperationStatusPayload } from './operation';
import type { FrontendToolName } from 'views/lap-analysis/ai-chat/ai-command-registry';

/** A single executable action, sharing the operation lifecycle. */
export interface Tool<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
> extends Operation<TResult, TStatus, TTerminationStatus> {
    readonly kind: 'tool';
}

/** A single named tool call; workflow names are excluded from executable input. */
export type ToolCall<TMetadata> = {
    tool: { name: FrontendToolName } & TMetadata;
};

/** Read the explicit tool envelope without interpreting its arguments. */
export const readToolCall = (value: unknown): Record<string, unknown> | null => {
    if (!value || typeof value !== 'object' || Array.isArray(value)
        || Reflect.ownKeys(value).length !== 1
        || !Object.prototype.hasOwnProperty.call(value, 'tool')) return null;
    const tool = (value as Record<string, unknown>).tool;
    if (!tool || typeof tool !== 'object' || Array.isArray(tool)
        || !Object.prototype.hasOwnProperty.call(tool, 'name')) return null;
    const call = tool as Record<string, unknown>;
    return typeof call.name === 'string' && call.name.trim() === call.name && call.name.length > 0
        ? call : null;
};

export type ToolDispatcher = ((
    name: FrontendToolName,
    args?: Record<string, unknown>,
    signal?: AbortSignal,
) => Tool<OperationExecutionOutput, OperationStatusPayload>) & {
    validate(name: string): void;
};

export function assertTool<TResult, TStatus extends object, TTerminationStatus extends string>(
    operation: Operation<TResult, TStatus, TTerminationStatus>,
): asserts operation is Tool<TResult, TStatus, TTerminationStatus> {
    if (!operation || (operation as Tool<TResult, TStatus, TTerminationStatus>).kind !== 'tool') {
        throw new Error('Workflow children must return a Tool.');
    }
}

export const asTool = <
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(operation: Operation<TResult, TStatus, TTerminationStatus>): Tool<TResult, TStatus, TTerminationStatus> => (
    Object.assign(operation, { kind: 'tool' as const })
);
