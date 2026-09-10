import type { Operation, OperationExecutionOutput, OperationStatusPayload } from './operation';
import type { FrontendOperationName, FrontendToolName } from 'views/lap-analysis/ai-chat/ai-command-registry';
import type { WorkflowComponentBase } from './WorkflowComponentBase';
import type { Workflow } from './workflow';

/** A single executable action, sharing the operation lifecycle. */
export interface Tool<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
> extends Operation<TResult, TStatus, TTerminationStatus> {
    readonly kind: 'tool';
}

/** Native tool call descriptor. Workflow children use OperationCall. */
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

export type WorkflowDispatcher = ((
    name: FrontendOperationName,
    args?: Record<string, unknown>,
    signal?: AbortSignal,
    caller?: WorkflowComponentBase<any>,
) => Tool<OperationExecutionOutput, OperationStatusPayload> | Workflow<OperationExecutionOutput, OperationStatusPayload>) & {
    validate(name: string): void;
    workflowCaller?: WorkflowComponentBase<any>;
};

export const bindWorkflowDispatcher = (dispatch: WorkflowDispatcher, owner: WorkflowComponentBase<any>): WorkflowDispatcher => (
    Object.assign((name: FrontendOperationName, args?: Record<string, unknown>, signal?: AbortSignal) => (
        dispatch(name, args, signal, owner)
    ), { validate: dispatch.validate, workflowCaller: owner })
);

/** Compatibility alias for callers migrating to the workflow dispatcher. */
export type ToolDispatcher = WorkflowDispatcher;

export const asTool = <
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(operation: Operation<TResult, TStatus, TTerminationStatus>): Tool<TResult, TStatus, TTerminationStatus> => (
    Object.assign(operation, { kind: 'tool' as const })
);
