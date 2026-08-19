import React, {
    MutableRefObject,
    createContext,
    useContext,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import AiOverlayManager from 'views/floating-chat/AiOverlayManager';
import {
    AiToolComponentRefError,
    ComponentMountTimeoutError,
    ComponentRefUnavailableError,
    DuplicateComponentNameError,
} from './AiToolComponentError';

export {
    AiToolComponentRefError,
    ComponentMountTimeoutError,
    ComponentRefUnavailableError,
    DuplicateComponentNameError,
} from './AiToolComponentError';

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
    PROCEDURE_PLAN: 'procedure-plan',
    BASELINE_COLLECTION: 'baseline-collection',
});

export interface NamedAiToolComponentHandle {
    getComponentName(): string;
}

export type AiToolComponentRef<THandle extends NamedAiToolComponentHandle = NamedAiToolComponentHandle> =
    MutableRefObject<THandle | null>;

type ComponentWaiter = {
    resolve: (ref: AiToolComponentRef) => void;
    reject: (error: AiToolComponentRefError) => void;
    timeoutId: ReturnType<typeof setTimeout>;
};

export interface AiToolComponentRefDirectory {
    findComponentRef<THandle extends NamedAiToolComponentHandle>(name: string): AiToolComponentRef<THandle> | null;
    registerComponentRef<THandle extends NamedAiToolComponentHandle>(ref: AiToolComponentRef<THandle>): void;
    unregisterComponentRef(ref: AiToolComponentRef): boolean;
    awaitComponentRef<THandle extends NamedAiToolComponentHandle>(
        name: string,
        timeoutMs?: number,
    ): Promise<AiToolComponentRef<THandle>>;
    getComponentNames(): string[];
    getComponentRefs(): AiToolComponentRef[];
}

export const createAiToolComponentRefDirectory = (
    onChange: () => void = () => undefined,
): AiToolComponentRefDirectory => {
    const entries = new Map<string, AiToolComponentRef>();
    const namesByRef = new Map<AiToolComponentRef, string>();
    const waiters = new Map<string, Set<ComponentWaiter>>();

    const findComponentRef = <THandle extends NamedAiToolComponentHandle>(name: string) => (
        (entries.get(name) as AiToolComponentRef<THandle> | undefined) ?? null
    );

    const registerComponentRef = <THandle extends NamedAiToolComponentHandle>(
        ref: AiToolComponentRef<THandle>,
    ): void => {
        const handle = ref.current;
        if (!handle) {
            throw new ComponentRefUnavailableError(
                '(unmounted)',
                'A component reference must be mounted before it can be registered.',
            );
        }
        const componentName = handle.getComponentName();
        if (!componentName.trim()) {
            throw new ComponentRefUnavailableError(
                componentName,
                'A component must return a non-empty runtime name.',
            );
        }

        const directoryRef = ref as AiToolComponentRef;
        const registeredName = namesByRef.get(directoryRef);
        if (registeredName === componentName && entries.get(componentName) === ref) return;
        if (registeredName) {
            entries.delete(registeredName);
            namesByRef.delete(directoryRef);
        }

        const existing = entries.get(componentName);
        if (existing && existing !== ref) {
            throw new DuplicateComponentNameError(
                componentName,
                `A mounted component already owns '${componentName}'.`,
            );
        }

        entries.set(componentName, directoryRef);
        namesByRef.set(directoryRef, componentName);
        const pending = waiters.get(componentName);
        if (pending) {
            pending.forEach((waiter) => {
                clearTimeout(waiter.timeoutId);
                waiter.resolve(directoryRef);
            });
            waiters.delete(componentName);
        }
        onChange();
    };

    const unregisterComponentRef = (ref: AiToolComponentRef): boolean => {
        const componentName = namesByRef.get(ref);
        if (!componentName || entries.get(componentName) !== ref) return false;
        entries.delete(componentName);
        namesByRef.delete(ref);
        onChange();
        return true;
    };

    const awaitComponentRef = <THandle extends NamedAiToolComponentHandle>(
        name: string,
        timeoutMs = AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS,
    ): Promise<AiToolComponentRef<THandle>> => {
        const existing = findComponentRef<THandle>(name);
        if (existing?.current) return Promise.resolve(existing);

        return new Promise((resolve, reject) => {
            const waiter: ComponentWaiter = {
                resolve: (ref) => resolve(ref as AiToolComponentRef<THandle>),
                reject,
                timeoutId: setTimeout(() => {
                    const current = waiters.get(name);
                    current?.delete(waiter);
                    if (current?.size === 0) waiters.delete(name);
                    reject(new ComponentMountTimeoutError(
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

    return {
        findComponentRef,
        registerComponentRef,
        unregisterComponentRef,
        awaitComponentRef,
        getComponentNames: () => Array.from(entries.keys()).sort(),
        getComponentRefs: () => Array.from(entries.values()),
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
            <AiOverlayManager directory={directoryRef.current} directoryRevision={revision} />
            {children}
        </AiToolComponentRefContext.Provider>
    );
};

export const useAiToolComponentRefs = () => {
    const value = useContext(AiToolComponentRefContext);
    const directory = value.directory;
    if (!directory) throw new Error('AiToolComponentRefProvider is required.');
    return { directory, revision: value.revision };
};

export const useOptionalAiToolComponentRefDirectory = (): AiToolComponentRefDirectory | null => (
    useContext(AiToolComponentRefContext).directory
);

export const useAiToolComponentRefDirectory = (): AiToolComponentRefDirectory => (
    useAiToolComponentRefs().directory
);

export const useRegisterAiToolComponentRef = <THandle extends NamedAiToolComponentHandle>(
    ref: AiToolComponentRef<THandle>,
) => {
    const { directory } = useContext(AiToolComponentRefContext);

    useLayoutEffect(() => {
        if (!directory) return undefined;
        directory.registerComponentRef(ref);
        return () => {
            directory.unregisterComponentRef(ref);
        };
    }, [directory, ref]);
};

export const resolveNamedComponentHandle = <THandle extends NamedAiToolComponentHandle>(
    directory: AiToolComponentRefDirectory,
    name: string,
): THandle => {
    const handle = directory.findComponentRef<THandle>(name)?.current;
    if (!handle) {
        throw new ComponentRefUnavailableError(
            name,
            `Component '${name}' is not mounted in the active dashboard.`,
        );
    }
    return handle;
};

export const awaitNamedComponentHandle = async <THandle extends NamedAiToolComponentHandle>(
    directory: AiToolComponentRefDirectory,
    name: string,
    timeoutMs = AI_TOOL_COMPONENT_MOUNT_TIMEOUT_MS,
): Promise<THandle> => {
    await directory.awaitComponentRef<THandle>(name, timeoutMs);
    const handle = directory.findComponentRef<THandle>(name)?.current;
    if (!handle) {
        throw new ComponentRefUnavailableError(
            name,
            `Component '${name}' unmounted before it could receive the command.`,
        );
    }
    return handle;
};

export type AiChatAssistantMode = 'front_desk' | 'live' | 'recorded' | 'user_summary';
