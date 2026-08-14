/**
 * Promise-native contract shared by every frontend AI tool.
 *
 * Status promises represent independently observable progress. The result is
 * deliberately a terminal barrier: it cannot settle until every advertised
 * status has settled. A rejected status is progress-delivery failure only and
 * never changes the operation's final result.
 */
export interface AiToolOperation<TResult, TStatus extends object = never> {
    result: Promise<TResult | Error>;
    statuses: readonly Promise<TStatus>[];
}

export type AiToolQueryResult<TData> = {
    status: 'ready';
    data: TData;
};

export type AiToolOperationResult<TOperation> = TOperation extends AiToolOperation<infer TResult, object>
    ? TResult
    : never;

export type AiToolOperationStatus<TOperation> = TOperation extends AiToolOperation<unknown, infer TStatus>
    ? TStatus
    : never;

export interface AiToolDeferred<TValue> {
    promise: Promise<TValue>;
    resolve(value: TValue | PromiseLike<TValue>): void;
    reject(reason?: unknown): void;
    readonly settled: boolean;
}

export const createAiToolDeferred = <TValue,>(): AiToolDeferred<TValue> => {
    let resolvePromise!: (value: TValue | PromiseLike<TValue>) => void;
    let rejectPromise!: (reason?: unknown) => void;
    let settled = false;
    const promise = new Promise<TValue>((resolve, reject) => {
        resolvePromise = resolve;
        rejectPromise = reject;
    });
    return {
        promise,
        resolve: (value) => {
            if (settled) return;
            settled = true;
            resolvePromise(value);
        },
        reject: (reason) => {
            if (settled) return;
            settled = true;
            rejectPromise(reason);
        },
        get settled() {
            return settled;
        },
    };
};

export const createAiToolOperation = <TResult, TStatus extends object = never>(
    result: TResult | Error | PromiseLike<TResult | Error>,
    statuses: readonly Promise<TStatus>[] = [],
): AiToolOperation<TResult, TStatus> => {
    const statusBarrier = Promise.allSettled(statuses);
    const terminalResult = Promise.resolve(result).then(
        async (value) => {
            await statusBarrier;
            return value;
        },
        async (error) => {
            await statusBarrier;
            throw error;
        },
    );
    // Operations are often replaced by UI lifecycle events before their owner
    // awaits them. Mark the rejection observed without changing what callers
    // receive when they await `result`.
    void terminalResult.catch(() => undefined);
    return { result: terminalResult, statuses };
};

export const resolvedAiToolOperation = <TResult,>(
    result: TResult | Error,
): AiToolOperation<TResult> => createAiToolOperation<TResult>(result);

export const createAiToolOperationFrom = <TResult, TStatus extends object = never>(
    run: () => TResult | Error | PromiseLike<TResult | Error>,
    statuses: readonly Promise<TStatus>[] = [],
): AiToolOperation<TResult, TStatus> => createAiToolOperation(
    Promise.resolve().then(run),
    statuses,
);

export const mapAiToolOperation = <
    TSourceResult,
    TResult,
    TSourceStatus extends object,
    TStatus extends object = TSourceStatus,
>(
    operation: AiToolOperation<TSourceResult, TSourceStatus>,
    mapResult: (result: TSourceResult) => TResult | Error | PromiseLike<TResult | Error>,
    mapStatus: (status: TSourceStatus) => TStatus | PromiseLike<TStatus> = (
        (status: TSourceStatus) => status as unknown as TStatus
    ),
): AiToolOperation<TResult, TStatus> => createAiToolOperation(
    operation.result.then((result) => (
        result instanceof Error ? result : mapResult(result)
    )),
    operation.statuses.map((status) => status.then(mapStatus)),
);
