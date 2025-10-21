import { PythonShellOptions } from './pythonService';

type Listener<T> = (event: PythonStreamEvent<T>) => void;

type Cleanup = () => void;

type Deferred<T> = {
    resolve: (value: T | PromiseLike<T>) => void;
    reject: (reason?: unknown) => void;
    promise: Promise<T>;
};

const DEFAULT_READY_TIMEOUT_MS = 10000;
const DEFAULT_TERMINATE_TIMEOUT_MS = 5000;
const DEFAULT_SCRIPT_PATH = 'src/py-scripts';

const createDeferred = <T>(): Deferred<T> => {
    let resolve!: (value: T | PromiseLike<T>) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { resolve, reject, promise };
};

export interface PythonStreamEvent<T = unknown> {
    status: string;
    source?: string;
    request_id?: string;
    data?: T;
    message?: string;
    traceback?: string;
    [key: string]: unknown;
}

export interface CreatePythonStreamOptions {
    scriptName: string;
    pythonOptions?: PythonShellOptions;
    scriptPath?: string;
    args?: string[];
    readyTimeoutMs?: number;
}

export interface PythonStreamSession<T = unknown> {
    readonly shellId: number;
    readonly isReady: boolean;
    waitUntilReady(): Promise<void>;
    onMessage(listener: Listener<T>): Cleanup;
    send(action: string, payload?: unknown, requestId?: string): Promise<void>;
    dispose(options?: { force?: boolean; terminateTimeoutMs?: number }): Promise<void>;
}

export const createPythonStreamSession = async <T = unknown>(
    options: CreatePythonStreamOptions
): Promise<PythonStreamSession<T>> => {
    if (!window?.electronAPI?.runPythonScript || !window.electronAPI.onPythonMessage) {
        throw new Error('Python streaming API is not available in this environment');
    }

    const baseOptions: PythonShellOptions = {
        mode: 'text',
        pythonOptions: ['-u'],
        scriptPath: DEFAULT_SCRIPT_PATH,
        args: [],
        ...(options.pythonOptions ?? {})
    };

    const mergedArgs: string[] = [
        ...(baseOptions.args ?? []),
        ...(options.args ?? [])
    ];

    if (!mergedArgs.includes('--stream')) {
        mergedArgs.push('--stream');
    }

    const pythonOptions: PythonShellOptions = {
        ...baseOptions,
        args: mergedArgs,
        scriptPath: options.scriptPath ?? baseOptions.scriptPath ?? DEFAULT_SCRIPT_PATH
    };

    const listeners = new Set<Listener<T>>();
    let removeMessageListener: Cleanup | null = null;
    let removeEndListener: Cleanup | null = null;
    let isReady = false;
    let isClosed = false;

    let shellId: number | null = null;
    const pendingMessages: Array<{ shellId: number; rawMessage: string }> = [];
    const pendingEndEvents: number[] = [];

    const readyDeferred = createDeferred<void>();
    const closedDeferred = createDeferred<void>();

    const readyTimeoutMs = options.readyTimeoutMs ?? DEFAULT_READY_TIMEOUT_MS;
    let readyTimeoutId: number | null = null;

    if (readyTimeoutMs > 0) {
        readyTimeoutId = window.setTimeout(() => {
            if (!isReady) {
                const error = new Error(
                    `Python stream ${options.scriptName} did not become ready within ${readyTimeoutMs}ms`
                );
                readyDeferred.reject(error);
                void dispose({ force: true });
            }
        }, readyTimeoutMs);
    }

    const notifyListeners = (event: PythonStreamEvent<T>) => {
        listeners.forEach((listener) => {
            try {
                listener(event);
            } catch (error) {
                console.error('Python stream listener threw an error', error);
            }
        });
    };

    const finalizeReady = (error?: Error) => {
        if (readyTimeoutId !== null) {
            window.clearTimeout(readyTimeoutId);
            readyTimeoutId = null;
        }

        if (error) {
            if (!isReady) {
                readyDeferred.reject(error);
            }
            return;
        }

        if (!isReady) {
            isReady = true;
            readyDeferred.resolve();
        }
    };

    const finalizeClosed = () => {
        if (isClosed) {
            return;
        }
        isClosed = true;
        removeMessageListener?.();
        removeEndListener?.();
        closedDeferred.resolve();
    };

    const handleMessage = (_shellId: number, rawMessage: string, fromQueue = false) => {
        if (shellId === null) {
            if (!fromQueue) {
                pendingMessages.push({ shellId: _shellId, rawMessage });
            }
            return;
        }

        if (_shellId !== shellId || isClosed) {
            return;
        }

        let parsed: PythonStreamEvent<T>;
        try {
            parsed = JSON.parse(rawMessage) as PythonStreamEvent<T>;
        } catch (error) {
            console.warn('Received non-JSON message from Python stream', rawMessage, error);
            return;
        }

        if (!isReady) {
            if (parsed.status === 'ready') {
                finalizeReady();
            } else if (parsed.status === 'error') {
                finalizeReady(new Error(parsed.message as string || 'Python stream reported error'));
            } else if (parsed.status === 'shutdown') {
                finalizeReady(new Error('Python stream shut down before becoming ready'));
            }
        }

        notifyListeners(parsed);

        if (parsed.status === 'shutdown') {
            finalizeClosed();
        }
    };

    const handleEnd = (_shellId: number, fromQueue = false) => {
        if (shellId === null) {
            if (!fromQueue) {
                pendingEndEvents.push(_shellId);
            }
            return;
        }

        if (_shellId !== shellId) {
            return;
        }

        if (!isReady) {
            finalizeReady(new Error('Python stream terminated before becoming ready'));
        }
        finalizeClosed();
    };

    removeMessageListener = window.electronAPI.onPythonMessage((incomingShellId, rawMessage) => {
        handleMessage(incomingShellId, rawMessage);
    });
    removeEndListener = window.electronAPI.onPythonEnd((incomingShellId) => {
        handleEnd(incomingShellId);
    });

    try {
        const runResult = await window.electronAPI.runPythonScript(
            options.scriptName,
            pythonOptions
        );
        shellId = runResult.shellId;

        if (shellId === null || typeof shellId !== 'number') {
            throw new Error('Failed to obtain Python shell id');
        }

        if (pendingMessages.length) {
            for (const queued of pendingMessages.splice(0, pendingMessages.length)) {
                handleMessage(queued.shellId, queued.rawMessage, true);
            }
        }

        if (pendingEndEvents.length) {
            for (const queuedShellId of pendingEndEvents.splice(0, pendingEndEvents.length)) {
                handleEnd(queuedShellId, true);
            }
        }
    } catch (error) {
        removeMessageListener?.();
        removeEndListener?.();
        throw error;
    }

    if (shellId === null) {
        removeMessageListener?.();
        removeEndListener?.();
        throw new Error('Python stream shell id not available after initialization');
    }

    const shellIdNumber = shellId;

    const send = async (action: string, payload?: unknown, requestId?: string) => {
        if (isClosed) {
            throw new Error('Cannot send message: Python stream has been disposed');
        }
        if (!action) {
            throw new Error('Action is required when sending a message to the Python stream');
        }

        const envelope: Record<string, unknown> = { action };
        if (payload !== undefined) {
            envelope.payload = payload;
        }
        if (requestId !== undefined) {
            envelope.request_id = requestId;
        }

        const response = await window.electronAPI.sendMessageToPython(shellIdNumber, JSON.stringify(envelope));
        if (!response?.success) {
            throw new Error(response?.error || 'Failed to send message to Python stream');
        }
    };

    const waitUntilReady = async () => {
        await readyDeferred.promise;
    };

    const onMessage = (listener: Listener<T>) => {
        listeners.add(listener);
        return () => {
            listeners.delete(listener);
        };
    };

    const dispose = async (
        { force = false, terminateTimeoutMs = DEFAULT_TERMINATE_TIMEOUT_MS } = {}
    ) => {
        if (isClosed) {
            return;
        }

        if (force) {
            if (typeof window.electronAPI.stopPythonScript === 'function') {
                await window.electronAPI.stopPythonScript(shellIdNumber);
            }
            finalizeClosed();
            return;
        }

        try {
            await send('shutdown');
        } catch (error) {
            console.warn('Failed to request graceful shutdown, forcing', error);
            await dispose({ force: true });
            return;
        }

        if (terminateTimeoutMs > 0) {
            await Promise.race([
                closedDeferred.promise,
                new Promise<void>((resolve) => window.setTimeout(resolve, terminateTimeoutMs))
            ]);
        } else {
            await closedDeferred.promise;
        }

        finalizeClosed();
    };

    return {
        shellId: shellIdNumber,
        get isReady() {
            return isReady;
        },
        waitUntilReady,
        onMessage,
        send,
        dispose
    };
};
