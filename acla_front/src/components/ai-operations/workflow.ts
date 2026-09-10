import type { Operation } from './operation';
import type { FrontendWorkflowName } from 'views/lap-analysis/ai-chat/ai-command-registry';

export type WorkflowProgress<TStep = unknown> = {
    /** Successful child step executions, including repeatable plan retries. */
    completed_step_count: number;
    /** Workflow-owned description of the interrupted step, or null. */
    stopped_at_step: TStep | null;
};

export const readWorkflowProgress = (value: unknown): WorkflowProgress | undefined => {
    if (!value || typeof value !== 'object') return undefined;
    const { completed_step_count, stopped_at_step } = value as WorkflowProgress;
    return typeof completed_step_count === 'number'
        && stopped_at_step !== undefined
        ? { completed_step_count, stopped_at_step } : undefined;
};

export type WorkflowCall<TName extends FrontendWorkflowName, TInput> = {
    workflow: { name: TName } & TInput;
};

export const readWorkflowCall = (
    value: unknown,
    name: FrontendWorkflowName,
): Record<string, unknown> | null => {
    if (!value || typeof value !== 'object' || Array.isArray(value)
        || Reflect.ownKeys(value).length !== 1
        || !Object.prototype.hasOwnProperty.call(value, 'workflow')) return null;
    const workflow = (value as Record<string, unknown>).workflow;
    if (!workflow || typeof workflow !== 'object' || Array.isArray(workflow)
        || !Object.prototype.hasOwnProperty.call(workflow, 'name')
        || (workflow as Record<string, unknown>).name !== name) return null;
    return workflow as Record<string, unknown>;
};

/** An operation that coordinates the execution of other operations. */
export interface Workflow<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
> extends Operation<TResult, TStatus, TTerminationStatus> {
    readonly kind: 'workflow';
}

export const asWorkflow = <
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(
    operation: Operation<TResult, TStatus, TTerminationStatus>,
    progress?: WorkflowProgress,
): Workflow<TResult, TStatus, TTerminationStatus> => {
    // Runners define their output; retain their progress on errors after cleanup too.
    if (progress) operation.notifyTerminated(({ result }) => {
        if (result && typeof result === 'object') {
            Object.assign(result, progress);
        }
    });
    return Object.assign(operation, { kind: 'workflow' as const });
};
