import type { FrontendOperationName } from 'views/lap-analysis/ai-chat/ai-command-registry';

/** A workflow child call. The name may identify a tool or a workflow. */
export type OperationCall<TMetadata> = {
    operation: { name: FrontendOperationName } & TMetadata;
};

/** Read the explicit operation envelope without interpreting its arguments. */
export const readOperationCall = (value: unknown): Record<string, unknown> | null => {
    if (!value || typeof value !== 'object' || Array.isArray(value)
        || Reflect.ownKeys(value).length !== 1
        || !Object.prototype.hasOwnProperty.call(value, 'operation')) return null;
    const operation = (value as Record<string, unknown>).operation;
    if (!operation || typeof operation !== 'object' || Array.isArray(operation)
        || !Object.prototype.hasOwnProperty.call(operation, 'name')) return null;
    const call = operation as Record<string, unknown>;
    return typeof call.name === 'string' && call.name.trim() === call.name && call.name.length > 0
        ? call : null;
};

/**
 * Promise-native contract shared by frontend tools and workflows.
 *
 * Status promises represent independently observable progress and never gate
 * completion. Producers must await all work, including nested operations and
 * cleanup, before settling their result.
 *
 * Termination is a one-shot lifecycle signal emitted when that work settles,
 * before the public result settles. Its status is producer-supplied metadata,
 * never a completion condition or a value read from the result body.
 * Every operation can also be aborted. Aborting runs the producer's synchronous
 * cleanup first, then rejects unfinished public promises and emits termination.
 */
export const OPERATION_ABORTED_STATUS = 'aborted' as const;

export class OperationAbortedError extends Error {
    override name = 'AbortError';
    readonly cause?: unknown;

    constructor(cause?: unknown) {
        super('Operation was aborted.');
        this.cause = cause;
        Object.setPrototypeOf(this, OperationAbortedError.prototype);
    }
}

export type OperationAbortHandler = () => void;

export type OperationTermination<TResult, TTerminationStatus extends string = string> = {
    status: TTerminationStatus;
    result: TResult | Error;
};

export interface Operation<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
> {
    result: Promise<TResult | Error>;
    statuses: readonly Promise<TStatus>[];
    abort(): void;
    notifyTerminated(
        listener: (
            termination: OperationTermination<
                TResult,
                TTerminationStatus | typeof OPERATION_ABORTED_STATUS
            >
        ) => void,
    ): () => void;
}

export type OperationQueryResult<TData> = {
    status: 'ready';
    data: TData;
};

export type OperationKind = 'tool' | 'workflow';
export type OperationNormalOutput = { [key: string]: unknown };
export type OperationExecutionOutput = OperationNormalOutput | string | Error;
export type OperationStatusPayload = { [key: string]: unknown };

export type OperationDispatcher = (
    name: string,
    args?: Record<string, unknown>,
    signal?: AbortSignal,
) => Operation<OperationNormalOutput | string, OperationStatusPayload>;

export type OperationResult<TOperation> = TOperation extends Operation<infer TResult, object, string>
    ? TResult
    : never;

export type OperationStatus<TOperation> = TOperation extends Operation<unknown, infer TStatus, string>
    ? TStatus
    : never;

export type OperationTerminationStatus<TOperation> = (
    TOperation extends Operation<unknown, object, infer TTerminationStatus>
        ? TTerminationStatus | typeof OPERATION_ABORTED_STATUS
        : never
);

export interface OperationDeferred<TValue> {
    promise: Promise<TValue>;
    resolve(value: TValue | PromiseLike<TValue>): void;
    reject(reason?: unknown): void;
    readonly settled: boolean;
}

export const createOperationDeferred = <TValue,>(): OperationDeferred<TValue> => {
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
    notifyTerminated: Operation<TResult, never, TTerminationStatus>['notifyTerminated'];
    terminate(termination: OperationTermination<TResult, TTerminationStatus>): boolean;
};

const createTerminationNotifier = <
    TResult,
    TTerminationStatus extends string,
>(): TerminationNotifier<TResult, TTerminationStatus> => {
    const listeners = new Set<(
        termination: OperationTermination<TResult, TTerminationStatus>
    ) => void>();
    let termination: OperationTermination<TResult, TTerminationStatus> | null = null;
    const notifyListener = (
        listener: (value: OperationTermination<TResult, TTerminationStatus>) => void,
        value: OperationTermination<TResult, TTerminationStatus>,
    ) => {
        try {
            listener(value);
        } catch (error) {
            console.error('Operation termination listener failed.', error);
        }
    };

    return {
        notifyTerminated: (listener) => {
            if (termination) {
                notifyListener(listener, termination);
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
            currentListeners.forEach((listener) => notifyListener(listener, nextTermination));
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
    termination: PromiseLike<OperationTermination<TResult, TTerminationStatus>>,
    handleAbort: OperationAbortHandler = () => undefined,
): Operation<TResult, TStatus, TTerminationStatus> => {
    type OperationState = 'running' | 'aborting' | 'aborted' | 'finished';
    let state: OperationState = 'running';
    const aborted = createOperationDeferred<never>();
    const publicStatuses = statuses.map((status) => Promise.race([
        status,
        aborted.promise,
    ]));
    publicStatuses.forEach((status) => { void status.catch(() => undefined); });
    const notifier = createTerminationNotifier<
        TResult,
        TTerminationStatus | typeof OPERATION_ABORTED_STATUS
    >();
    const notifyCompletion = async () => {
        const value = await termination;
        if (state !== 'running') return;
        state = 'finished';
        notifier.terminate(value);
    };
    const completedResult = Promise.resolve(result).then(
        async (value) => {
            await notifyCompletion();
            return value;
        },
        async (error) => {
            await notifyCompletion();
            throw error;
        },
    );
    const terminalResult = Promise.race([completedResult, aborted.promise]);
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
            const error = new OperationAbortedError(cleanupError);
            state = 'aborted';
            aborted.reject(error);
            notifier.terminate({ status: OPERATION_ABORTED_STATUS, result: error });
        },
        notifyTerminated: notifier.notifyTerminated,
    };
};

export function createOperation<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
>(
    result: TResult | Error | PromiseLike<TResult | Error>,
    statuses: readonly Promise<TStatus>[],
    terminationStatus: TTerminationStatus,
    handleAbort?: OperationAbortHandler,
): Operation<TResult, TStatus, TTerminationStatus | 'failed'>;
export function createOperation<
    TResult,
    TTerminationStatus extends string = string,
    TStatus extends object = never,
>(
    result: TResult | Error | PromiseLike<TResult | Error>,
    terminationStatus: TTerminationStatus,
    statuses?: readonly Promise<TStatus>[],
    handleAbort?: OperationAbortHandler,
): Operation<TResult, TStatus, TTerminationStatus | 'failed'>;
export function createOperation<
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(
    result: TResult | Error | PromiseLike<TResult | Error>,
    statusesOrTerminationStatus: readonly Promise<TStatus>[] | TTerminationStatus,
    terminationStatusOrStatuses?: TTerminationStatus | readonly Promise<TStatus>[],
    handleAbort?: OperationAbortHandler,
): Operation<TResult, TStatus, TTerminationStatus | 'failed'> {
    const statusWasSuppliedFirst = typeof statusesOrTerminationStatus === 'string';
    const statuses = statusWasSuppliedFirst
        ? (terminationStatusOrStatuses ?? []) as readonly Promise<TStatus>[]
        : statusesOrTerminationStatus as readonly Promise<TStatus>[];
    const terminationStatus = (statusWasSuppliedFirst
        ? statusesOrTerminationStatus
        : terminationStatusOrStatuses) as TTerminationStatus;
    const sourceResult = Promise.resolve(result);
    const termination: Promise<OperationTermination<TResult, TTerminationStatus | 'failed'>> = (
        sourceResult.then((value): OperationTermination<TResult, TTerminationStatus | 'failed'> => (
            value instanceof Error
            ? { status: 'failed', result: value }
            : { status: terminationStatus, result: value }
        )).catch((error) => ({ status: 'failed', result: toTerminationError(error) }))
    );
    return createOperationWithTermination(sourceResult, statuses, termination, handleAbort);
}

export const resolvedOperation = <
    TResult,
    TTerminationStatus extends string,
>(
    result: TResult | Error,
    terminationStatus: TTerminationStatus,
): Operation<TResult, never, TTerminationStatus | 'failed'> => (
    createOperation<TResult, TTerminationStatus>(result, terminationStatus)
);

export function createOperationFrom<
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
>(
    run: (signal: AbortSignal) => TResult | Error | PromiseLike<TResult | Error>,
    statuses: readonly Promise<TStatus>[],
    terminationStatus: TTerminationStatus,
    handleAbort?: OperationAbortHandler,
): Operation<TResult, TStatus, TTerminationStatus | 'failed'>;
export function createOperationFrom<
    TResult,
    TTerminationStatus extends string = string,
    TStatus extends object = never,
>(
    run: (signal: AbortSignal) => TResult | Error | PromiseLike<TResult | Error>,
    terminationStatus: TTerminationStatus,
    statuses?: readonly Promise<TStatus>[],
    handleAbort?: OperationAbortHandler,
): Operation<TResult, TStatus, TTerminationStatus | 'failed'>;
export function createOperationFrom<
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
>(
    run: (signal: AbortSignal) => TResult | Error | PromiseLike<TResult | Error>,
    statusesOrTerminationStatus: readonly Promise<TStatus>[] | TTerminationStatus,
    terminationStatusOrStatuses?: TTerminationStatus | readonly Promise<TStatus>[],
    handleAbort?: OperationAbortHandler,
): Operation<TResult, TStatus, TTerminationStatus | 'failed'> {
    const abortController = new AbortController();
    const sourceResult = Promise.resolve().then(() => {
        if (abortController.signal.aborted) throw new OperationAbortedError();
        return run(abortController.signal);
    });
    const safelyAbort = () => {
        abortController.abort();
        handleAbort?.();
    };
    return typeof statusesOrTerminationStatus !== 'string'
        ? createOperation<TResult, TStatus, TTerminationStatus>(
            sourceResult,
            statusesOrTerminationStatus as readonly Promise<TStatus>[],
            terminationStatusOrStatuses as TTerminationStatus,
            safelyAbort,
        )
        : createOperation<TResult, TTerminationStatus, TStatus>(
            sourceResult,
            statusesOrTerminationStatus,
            terminationStatusOrStatuses as readonly Promise<TStatus>[] | undefined,
            safelyAbort,
        );
}

export interface ControlledOperation<
    TResult,
    TStatus extends object,
    TTerminationStatus extends string,
> {
    operation: Operation<TResult, TStatus, TTerminationStatus>;
    readonly signal: AbortSignal;
    resolve(status: TTerminationStatus, result: TResult | Error): void;
    reject(status: TTerminationStatus, error: Error): void;
    readonly settled: boolean;
}

export const createControlledOperation = <
    TResult,
    TStatus extends object = never,
    TTerminationStatus extends string = string,
>(
    statuses: readonly Promise<TStatus>[] = [],
    handleAbort: OperationAbortHandler = () => undefined,
): ControlledOperation<TResult, TStatus, TTerminationStatus> => {
    const result = createOperationDeferred<TResult | Error>();
    const termination = createOperationDeferred<OperationTermination<TResult, TTerminationStatus>>();
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

export const mapOperation = <
    TSourceResult,
    TResult,
    TSourceStatus extends object,
    TStatus extends object = TSourceStatus,
    TTerminationStatus extends string = string,
>(
    operation: Operation<TSourceResult, TSourceStatus, TTerminationStatus>,
    mapResult: (result: TSourceResult) => TResult | Error | PromiseLike<TResult | Error>,
    mapStatus: (status: TSourceStatus) => TStatus | PromiseLike<TStatus> = (
        (status: TSourceStatus) => status as unknown as TStatus
    ),
): Operation<
    TResult,
    TStatus,
    TTerminationStatus | 'failed' | typeof OPERATION_ABORTED_STATUS
> => {
    const mappedResult = operation.result.then((result) => (
        result instanceof Error ? result : mapResult(result)
    ));
    const mappedTermination = new Promise<OperationTermination<
        TResult,
        TTerminationStatus | 'failed' | typeof OPERATION_ABORTED_STATUS
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
