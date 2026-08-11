import React, {
    MutableRefObject,
    createContext,
    useContext,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from 'react';

export const AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS = 5000;

export const AI_TOOL_COMPONENT_NAMES = Object.freeze({
    DASHBOARD_ASSISTANT: 'dashboard-assistant',
    LIVE_SESSION: 'live-session',
    SESSION_ANALYSIS: 'session-analysis',
    USER_SUMMARY: 'user-summary',
    LIVE_VISUALIZATION_MANAGER: 'live-visualization-manager',
    RECORDED_VISUALIZATION_MANAGER: 'recorded-visualization-manager',
    LIVE_RANGE_TODO_LIST: 'live-range-todo-list',
    GOAL: 'goal',
    BASELINE_COLLECTION: 'baseline-collection',
});

export type AiToolComponentRefErrorCode =
    | 'component_ref_unavailable'
    | 'component_name_mismatch'
    | 'duplicate_component_name'
    | 'component_mount_timeout';

export class AiToolComponentRefError extends Error {
    readonly code: AiToolComponentRefErrorCode;
    readonly componentName: string;

    constructor(code: AiToolComponentRefErrorCode, componentName: string, message: string) {
        super(message);
        this.name = 'AiToolComponentRefError';
        this.code = code;
        this.componentName = componentName;
    }
}

export interface NamedAiToolComponentHandle {
    getComponentName(): string;
}

export type AiToolComponentRefOwner = symbol;

type DirectoryEntry = {
    owner: AiToolComponentRefOwner;
    ref: MutableRefObject<NamedAiToolComponentHandle | null>;
};

type ComponentWaiter = {
    resolve: (ref: MutableRefObject<NamedAiToolComponentHandle | null>) => void;
    reject: (error: AiToolComponentRefError) => void;
    timeoutId: ReturnType<typeof setTimeout>;
};

export interface AiToolComponentRefDirectory {
    findComponentRef<THandle extends NamedAiToolComponentHandle>(
        name: string,
    ): MutableRefObject<THandle | null> | null;
    reserveComponentRef<THandle extends NamedAiToolComponentHandle>(
        name: string,
        owner: AiToolComponentRefOwner,
        handle: THandle,
    ): MutableRefObject<THandle | null>;
    createComponentRef<THandle extends NamedAiToolComponentHandle>(
        name: string,
        owner: AiToolComponentRefOwner,
        handle: THandle,
    ): MutableRefObject<THandle | null>;
    awaitComponentRef<THandle extends NamedAiToolComponentHandle>(
        name: string,
        timeoutMs?: number,
    ): Promise<MutableRefObject<THandle | null>>;
    releaseComponentRef(name: string, owner: AiToolComponentRefOwner): boolean;
    getComponentNames(): string[];
}

export const createAiToolComponentRefDirectory = (
    onChange: () => void = () => undefined,
): AiToolComponentRefDirectory => {
    const entries = new Map<string, DirectoryEntry>();
    const waiters = new Map<string, Set<ComponentWaiter>>();

    const findComponentRef = <THandle extends NamedAiToolComponentHandle>(name: string) => (
        (entries.get(name)?.ref as MutableRefObject<THandle | null> | undefined) ?? null
    );

    const reserveComponentRef = <THandle extends NamedAiToolComponentHandle>(
        name: string,
        owner: AiToolComponentRefOwner,
        handle: THandle,
    ): MutableRefObject<THandle | null> => {
        const reportedName = handle.getComponentName();
        if (reportedName !== name) {
            throw new AiToolComponentRefError(
                'component_name_mismatch',
                name,
                `Component '${reportedName}' cannot register as '${name}'.`,
            );
        }

        const existing = entries.get(name);
        if (existing && existing.owner !== owner) {
            throw new AiToolComponentRefError(
                'duplicate_component_name',
                name,
                `A mounted component already owns '${name}'.`,
            );
        }

        if (existing) {
            existing.ref.current = handle;
            return existing.ref as MutableRefObject<THandle | null>;
        }

        const ref: MutableRefObject<NamedAiToolComponentHandle | null> = { current: handle };
        entries.set(name, { owner, ref });

        const pending = waiters.get(name);
        if (pending) {
            pending.forEach((waiter) => {
                clearTimeout(waiter.timeoutId);
                waiter.resolve(ref);
            });
            waiters.delete(name);
        }
        onChange();
        return ref as MutableRefObject<THandle | null>;
    };

    const awaitComponentRef = <THandle extends NamedAiToolComponentHandle>(
        name: string,
        timeoutMs = AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS,
    ): Promise<MutableRefObject<THandle | null>> => {
        const existing = findComponentRef<THandle>(name);
        if (existing?.current) return Promise.resolve(existing);

        return new Promise((resolve, reject) => {
            const waiter: ComponentWaiter = {
                resolve: (ref) => resolve(ref as MutableRefObject<THandle | null>),
                reject,
                timeoutId: setTimeout(() => {
                    const current = waiters.get(name);
                    current?.delete(waiter);
                    if (current?.size === 0) waiters.delete(name);
                    reject(new AiToolComponentRefError(
                        'component_mount_timeout',
                        name,
                        `Component '${name}' did not mount within ${timeoutMs}ms.`,
                    ));
                }, timeoutMs),
            };
            const current = waiters.get(name) ?? new Set<ComponentWaiter>();
            current.add(waiter);
            waiters.set(name, current);
        });
    };

    const releaseComponentRef = (name: string, owner: AiToolComponentRefOwner): boolean => {
        const existing = entries.get(name);
        if (!existing || existing.owner !== owner) return false;
        existing.ref.current = null;
        entries.delete(name);
        onChange();
        return true;
    };

    return {
        findComponentRef,
        reserveComponentRef,
        createComponentRef: reserveComponentRef,
        awaitComponentRef,
        releaseComponentRef,
        getComponentNames: () => Array.from(entries.keys()).sort(),
    };
};

interface AiToolComponentRefContextValue {
    directory: AiToolComponentRefDirectory | null;
    revision: number;
}

const AiToolComponentRefContext = createContext<AiToolComponentRefContextValue>({
    directory: null,
    revision: 0,
});

export const AiToolComponentRefProvider = ({ children }: { children: React.ReactNode }) => {
    const [revision, setRevision] = useState(0);
    const directoryRef = useRef<AiToolComponentRefDirectory | null>(null);
    if (directoryRef.current === null) {
        directoryRef.current = createAiToolComponentRefDirectory(() => {
            setRevision((current) => current + 1);
        });
    }
    const value = useMemo(() => ({ directory: directoryRef.current, revision }), [revision]);

    return (
        <AiToolComponentRefContext.Provider value={value}>
            {children}
        </AiToolComponentRefContext.Provider>
    );
};

export const useAiToolComponentRefs = () => {
    const value = useContext(AiToolComponentRefContext);
    const directory = value.directory;
    if (!directory) {
        throw new Error('AiToolComponentRefProvider is required.');
    }
    return { directory, revision: value.revision };
};

export const useOptionalAiToolComponentRefDirectory = (): AiToolComponentRefDirectory | null => (
    useContext(AiToolComponentRefContext).directory
);

export const useAiToolComponentRefDirectory = (): AiToolComponentRefDirectory => (
    useAiToolComponentRefs().directory
);

export const useRegisterAiToolComponentRef = <THandle extends NamedAiToolComponentHandle>(
    name: string,
    handle: THandle,
) => {
    const { directory } = useContext(AiToolComponentRefContext);
    const ownerRef = useRef<AiToolComponentRefOwner>(Symbol(`ai-tool-component:${name}`));
    const mountedNameRef = useRef(name);
    const handleRef = useRef(handle);
    handleRef.current = handle;

    if (mountedNameRef.current !== name) {
        throw new AiToolComponentRefError(
            'component_name_mismatch',
            mountedNameRef.current,
            `Mounted component '${mountedNameRef.current}' cannot be renamed to '${name}'. Remount it with key={name}.`,
        );
    }

    useLayoutEffect(() => {
        if (!directory) return undefined;
        const owner = ownerRef.current;
        directory.reserveComponentRef(name, owner, handleRef.current);
        return () => {
            directory.releaseComponentRef(name, owner);
        };
    }, [directory, name]);

    useLayoutEffect(() => {
        if (!directory) return;
        directory.reserveComponentRef(name, ownerRef.current, handle);
    }, [directory, handle, name]);
};

export const resolveNamedComponentHandle = <THandle extends NamedAiToolComponentHandle>(
    directory: AiToolComponentRefDirectory,
    name: string,
): THandle => {
    const handle = directory.findComponentRef<THandle>(name)?.current;
    if (!handle) {
        throw new AiToolComponentRefError(
            'component_ref_unavailable',
            name,
            `Component '${name}' is not mounted in the active dashboard.`,
        );
    }
    if (handle.getComponentName() !== name) {
        throw new AiToolComponentRefError(
            'component_name_mismatch',
            name,
            `Component '${name}' reported the name '${handle.getComponentName()}'.`,
        );
    }
    return handle;
};

export const awaitNamedComponentHandle = async <THandle extends NamedAiToolComponentHandle>(
    directory: AiToolComponentRefDirectory,
    name: string,
    timeoutMs = AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS,
): Promise<THandle> => {
    const ref = await directory.awaitComponentRef<THandle>(name, timeoutMs);
    const handle = ref.current;
    if (!handle) {
        throw new AiToolComponentRefError(
            'component_ref_unavailable',
            name,
            `Component '${name}' unmounted before it could receive the command.`,
        );
    }
    if (handle.getComponentName() !== name) {
        throw new AiToolComponentRefError(
            'component_name_mismatch',
            name,
            `Component '${name}' reported the name '${handle.getComponentName()}'.`,
        );
    }
    return handle;
};

export type AiChatAssistantMode = 'front_desk' | 'live' | 'recorded' | 'user_summary';
