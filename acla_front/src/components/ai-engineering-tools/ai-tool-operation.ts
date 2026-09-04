/**
 * Promise-native contract shared by every frontend AI tool.
 *
 * Status promises represent independently observable progress. On ordinary
 * completion, the result is a terminal barrier that cannot settle until every
 * advertised status has settled. A rejected status is progress-delivery
 * failure only and never changes the operation's final result.
 *
 * Termination is a separate, one-shot lifecycle signal. Its status is supplied
 * explicitly by the operation producer and is never read from the result body.
 * Every operation can also be aborted. Aborting runs the producer's synchronous
 * cleanup first, then rejects unfinished public promises and emits termination.
 */
export const AI_TOOL_ABORTED_STATUS = 'aborted' as const;

export class AiToolOperationAbortedError extends Error {
    override name = 'AbortError';
    readonly cause?: unknown;

    constructor(cause?: unknown) {
        super('AI tool operation was aborted.');
        this.cause = cause;
        Object.setPrototypeOf(this, AiToolOperationAbortedError.prototype);
    }
}

export type AiToolAbortHandler = () => void;

export type AiToolTermination<TResult, TTerminationStatus extends string = string> = {
    status: TTerminationStatus;
    result: TResult | Error;
};

export interface AiToolOperation<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
> {
    result: Promise<TResult | Error>;
    statuses: readonly Promise<TStatus>[];
    abort(): void;
    notifyTerminated(
        listener: (
            termination: AiToolTermination<
                TResult,
                TTerminationStatus | typeof AI_TOOL_ABORTED_STATUS
            >
        ) => void,
    ): () => void;
}

export type AiToolQueryResult<TData> = {
    status: 'ready';
    data: TData;
};

export type AiToolOperationResult<TOperation> = TOperation extends AiToolOperation<infer TResult, object, string>
    ? TResult
    : never;

export type AiToolOperationStatus<TOperation> = TOperation extends AiToolOperation<unknown, infer TStatus, string>
    ? TStatus
    : never;

export type AiToolOperationTerminationStatus<TOperation> = (
    TOperation extends AiToolOperation<unknown, object, infer TTerminationStatus>
        ? TTerminationStatus | typeof AI_TOOL_ABORTED_STATUS
        : never
);

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

const toTerminationError = (error: unknown): Error => (
    error instanceof Error ? error : new Error(String(error))
);

type TerminationNotifier<TResult, TTerminationStatus extends string> = {
    notifyTerminated: AiToolOperation<TResult, never, TTerminationStatus>['notifyTerminated'];
    terminate(termination: AiToolTermination<TResult, TTerminationStatus>): boolean;
};

const createTerminationNotifier = <
    TResult,
    TTerminationStatus extends string,
>(): TerminationNotifier<TResult, TTerminationStatus> => {
    const listeners = new Set<(
        termination: AiToolTermination<TResult, TTerminationStatus>
    ) => void>();
    let termination: AiToolTermination<TResult, TTerminationStatus> | null = null;

    return {
        notifyTerminated: (listener) => {
            if (termination) {
                listener(termination);
                return () => undefined;
            }
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
        terminate: (nextTermination) => {
            if (termination) return false;
            termination = nextTermination;
            const currentListeners = Array.from(listeners);
            listeners.clear();
            currentListeners.forEach((listener) => listener(nextTermination));
            return true;
        },
    };
};

const createOperationWithTermination = <
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(
    result: TResult | Error | PromiseLike<TResult | Error>,
    statuses: readonly Promise<TStatus>[],
    termination: PromiseLike<AiToolTermination<TResult, TTerminationStatus>>,
    handleAbort: AiToolAbortHandler = () => undefined,
): AiToolOperation<TResult, TStatus, TTerminationStatus> => {
    type OperationState = 'running' | 'aborting' | 'aborted' | 'finished';
    let state: OperationState = 'running';
    const aborted = createAiToolDeferred<never>();
    const publicStatuses = statuses.map((status) => Promise.race([
        status,
        aborted.promise,
    ]));
    const statusBarrier = Promise.allSettled(publicStatuses);
    const completedResult = Promise.resolve(result).then(
        async (value) => {
            await statusBarrier;
            return value;
        },
        async (error) => {
            await statusBarrier;
            throw error;
        },
    );
    const terminalResult = Promise.race([completedResult, aborted.promise]).then(
        (value) => {
            if (state === 'running') state = 'finished';
            return value;
        },
        (error) => {
            if (state === 'running') state = 'finished';
            throw error;
        },
    );
    const notifier = createTerminationNotifier<
        TResult,
        TTerminationStatus | typeof AI_TOOL_ABORTED_STATUS
    >();
    void Promise.resolve(termination).then(async (value) => {
        await statusBarrier;
        if (state === 'aborting' || state === 'aborted') return;
        notifier.terminate(value);
    });
    // Operations are often replaced by UI lifecycle events before their owner
    // awaits them. Mark rejections observed without changing what callers
    // receive when they await `result`.
    void terminalResult.catch(() => undefined);
    return {
        result: terminalResult,
        statuses: publicStatuses,
        abort: () => {
            if (state !== 'running') return;
            state = 'aborting';
            let cleanupError: unknown;
            try {
                handleAbort();
            } catch (error) {
                cleanupError = error;
            }
            const error = new AiToolOperationAbortedError(cleanupError);
            state = 'aborted';
            aborted.reject(error);
            notifier.terminate({ status: AI_TOOL_ABORTED_STATUS, result: error });
        },
        notifyTerminated: notifier.notifyTerminated,
    };
};

export function createAiToolOperation<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
>(
    result: TResult | Error | PromiseLike<TResult | Error>,
    statuses: readonly Promise<TStatus>[],
    terminationStatus: TTerminationStatus,
    handleAbort?: AiToolAbortHandler,
): AiToolOperation<TResult, TStatus, TTerminationStatus | 'failed'>;
export function createAiToolOperation<
    TResult,
    TTerminationStatus extends string = string,
    TStatus extends object = never,
>(
    result: TResult | Error | PromiseLike<TResult | Error>,
    terminationStatus: TTerminationStatus,
    statuses?: readonly Promise<TStatus>[],
    handleAbort?: AiToolAbortHandler,
): AiToolOperation<TResult, TStatus, TTerminationStatus | 'failed'>;
export function createAiToolOperation<
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(
    result: TResult | Error | PromiseLike<TResult | Error>,
    statusesOrTerminationStatus: readonly Promise<TStatus>[] | TTerminationStatus,
    terminationStatusOrStatuses?: TTerminationStatus | readonly Promise<TStatus>[],
    handleAbort?: AiToolAbortHandler,
): AiToolOperation<TResult, TStatus, TTerminationStatus | 'failed'> {
    const statusWasSuppliedFirst = typeof statusesOrTerminationStatus === 'string';
    const statuses = statusWasSuppliedFirst
        ? (terminationStatusOrStatuses ?? []) as readonly Promise<TStatus>[]
        : statusesOrTerminationStatus as readonly Promise<TStatus>[];
    const terminationStatus = (statusWasSuppliedFirst
        ? statusesOrTerminationStatus
        : terminationStatusOrStatuses) as TTerminationStatus;
    const sourceResult = Promise.resolve(result);
    const termination: Promise<AiToolTermination<TResult, TTerminationStatus | 'failed'>> = (
        sourceResult.then((value): AiToolTermination<TResult, TTerminationStatus | 'failed'> => (
            value instanceof Error
            ? { status: 'failed', result: value }
            : { status: terminationStatus, result: value }
        )).catch((error) => ({ status: 'failed', result: toTerminationError(error) }))
    );
    return createOperationWithTermination(sourceResult, statuses, termination, handleAbort);
}

export const resolvedAiToolOperation = <
    TResult,
    TTerminationStatus extends string,
>(
    result: TResult | Error,
    terminationStatus: TTerminationStatus,
): AiToolOperation<TResult, never, TTerminationStatus | 'failed'> => (
    createAiToolOperation<TResult, TTerminationStatus>(result, terminationStatus)
);

export function createAiToolOperationFrom<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
>(
    run: (signal: AbortSignal) => TResult | Error | PromiseLike<TResult | Error>,
    statuses: readonly Promise<TStatus>[],
    terminationStatus: TTerminationStatus,
    handleAbort?: AiToolAbortHandler,
): AiToolOperation<TResult, TStatus, TTerminationStatus | 'failed'>;
export function createAiToolOperationFrom<
    TResult,
    TTerminationStatus extends string = string,
    TStatus extends object = never,
>(
    run: (signal: AbortSignal) => TResult | Error | PromiseLike<TResult | Error>,
    terminationStatus: TTerminationStatus,
    statuses?: readonly Promise<TStatus>[],
    handleAbort?: AiToolAbortHandler,
): AiToolOperation<TResult, TStatus, TTerminationStatus | 'failed'>;
export function createAiToolOperationFrom<
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(
    run: (signal: AbortSignal) => TResult | Error | PromiseLike<TResult | Error>,
    statusesOrTerminationStatus: readonly Promise<TStatus>[] | TTerminationStatus,
    terminationStatusOrStatuses?: TTerminationStatus | readonly Promise<TStatus>[],
    handleAbort?: AiToolAbortHandler,
): AiToolOperation<TResult, TStatus, TTerminationStatus | 'failed'> {
    const abortController = new AbortController();
    const sourceResult = Promise.resolve().then(() => {
        if (abortController.signal.aborted) throw new AiToolOperationAbortedError();
        return run(abortController.signal);
    });
    const safelyAbort = () => {
        abortController.abort();
        handleAbort?.();
    };
    return typeof statusesOrTerminationStatus !== 'string'
        ? createAiToolOperation<TResult, TStatus, TTerminationStatus>(
            sourceResult,
            statusesOrTerminationStatus as readonly Promise<TStatus>[],
            terminationStatusOrStatuses as TTerminationStatus,
            safelyAbort,
        )
        : createAiToolOperation<TResult, TTerminationStatus, TStatus>(
            sourceResult,
            statusesOrTerminationStatus,
            terminationStatusOrStatuses as readonly Promise<TStatus>[] | undefined,
            safelyAbort,
        );
}

export interface ControlledAiToolOperation<
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
> {
    operation: AiToolOperation<TResult, TStatus, TTerminationStatus>;
    readonly signal: AbortSignal;
    resolve(status: TTerminationStatus, result: TResult | Error): void;
    reject(status: TTerminationStatus, error: Error): void;
    readonly settled: boolean;
}

export const createControlledAiToolOperation = <
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
>(
    statuses: readonly Promise<TStatus>[] = [],
    handleAbort: AiToolAbortHandler = () => undefined,
): ControlledAiToolOperation<TResult, TStatus, TTerminationStatus> => {
    const result = createAiToolDeferred<TResult | Error>();
    const termination = createAiToolDeferred<AiToolTermination<TResult, TTerminationStatus>>();
    const abortController = new AbortController();
    let aborted = false;
    return {
        operation: createOperationWithTermination(
            result.promise,
            statuses,
            termination.promise,
            () => {
                abortController.abort();
                try {
                    handleAbort();
                } finally {
                    aborted = true;
                }
            },
        ),
        signal: abortController.signal,
        resolve: (status, value) => {
            if (aborted || termination.settled) return;
            termination.resolve({ status, result: value });
            result.resolve(value);
        },
        reject: (status, error) => {
            if (aborted || termination.settled) return;
            termination.resolve({ status, result: error });
            result.reject(error);
        },
        get settled() {
            return aborted || termination.settled;
        },
    };
};

export const mapAiToolOperation = <
    TSourceResult,
    TResult,
    TSourceStatus extends object,
    TStatus extends object = TSourceStatus,
    TTerminationStatus extends string = string,
>(
    operation: AiToolOperation<TSourceResult, TSourceStatus, TTerminationStatus>,
    mapResult: (result: TSourceResult) => TResult | Error | PromiseLike<TResult | Error>,
    mapStatus: (status: TSourceStatus) => TStatus | PromiseLike<TStatus> = (
        (status: TSourceStatus) => status as unknown as TStatus
    ),
): AiToolOperation<
    TResult,
    TStatus,
    TTerminationStatus | 'failed' | typeof AI_TOOL_ABORTED_STATUS
> => {
    const mappedResult = operation.result.then((result) => (
        result instanceof Error ? result : mapResult(result)
    ));
    const mappedTermination = new Promise<AiToolTermination<
        TResult,
        TTerminationStatus | 'failed' | typeof AI_TOOL_ABORTED_STATUS
    >>((resolve) => {
        operation.notifyTerminated((sourceTermination) => {
            if (sourceTermination.result instanceof Error) {
                resolve({
                    status: sourceTermination.status,
                    result: sourceTermination.result,
                });
                return;
            }
            void mappedResult.then(
                (result) => resolve(result instanceof Error
                    ? { status: 'failed', result }
                    : { status: sourceTermination.status, result }),
                (error) => resolve({ status: 'failed', result: toTerminationError(error) }),
            );
        });
    });
    return createOperationWithTermination(
        mappedResult,
        operation.statuses.map((status) => status.then(mapStatus)),
        mappedTermination,
        () => operation.abort(),
    );
};
