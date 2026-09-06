import type { Operation } from './operation';

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
>(operation: Operation<TResult, TStatus, TTerminationStatus>): Workflow<TResult, TStatus, TTerminationStatus> => (
    Object.assign(operation, { kind: 'workflow' as const })
);
