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
    [Name in FrontendToolName]: Record<Name, TMetadata>
        & Partial<Record<Exclude<FrontendToolName, Name>, never>>;
}[FrontendToolName];

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
