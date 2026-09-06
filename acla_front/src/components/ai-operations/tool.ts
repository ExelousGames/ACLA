import type { Operation } from './operation';

/** A single executable action, sharing the operation lifecycle. */
export interface Tool<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
> extends Operation<TResult, TStatus, TTerminationStatus> {
    readonly kind: 'tool';
}

export const asTool = <
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(operation: Operation<TResult, TStatus, TTerminationStatus>): Tool<TResult, TStatus, TTerminationStatus> => (
    Object.assign(operation, { kind: 'tool' as const })
);
