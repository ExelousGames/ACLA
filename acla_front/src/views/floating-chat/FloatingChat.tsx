import React from 'react';
import '../lap-analysis/ai-chat/ai-chat.css';
import './floating-chat.css';
import { getOverlayDisplayDefinition } from './overlay-display-registry';
import {
    advanceOverlayTimers,
    applyOverlayComponentEvent,
    applyOverlayDisplayRequest,
    beginOverlayPresentation,
    endOverlayPresentation,
    getNextOverlayDeadline,
    initialOverlayPresentationState,
    orderOverlayInstances,
    overlayPresentationReducer,
    setOverlayEnabled,
    type OverlayDisplayInstance,
    type OverlayTransitionResult,
} from './overlay-presentation-manager';
import type {
    OverlayComponentEvent,
    OverlayDisplayAcknowledgement,
    OverlayDisplayRequest,
    OverlayLifecycleEvent,
    OverlayPresentationChange,
    OverlayShellMetadata,
} from './overlay-display-types';

const EMOTION_GIFS_KEY = 'acla-emotion-gifs';
const IDLE_WIDTH = 300;
const SPEAKING_WIDTH = 420;

interface ElectronOverlayRendererApi {
    onOverlayDisplayCommand?: (listener: (request: OverlayDisplayRequest) => void) => (() => void);
    acknowledgeOverlayDisplayRequest?: (acknowledgement: OverlayDisplayAcknowledgement) => void;
    emitOverlayLifecycle?: (event: OverlayLifecycleEvent) => void;
    reportOverlayReady?: () => void;
    onOverlayEnabledChange?: (listener: (enabled: boolean) => void) => (() => void);
    onOverlayPresentationChange?: (listener: (change: OverlayPresentationChange) => void) => (() => void);
    resizeFloatingChat?: (width: number, height: number) => void;
}

const getElectronApi = (): ElectronOverlayRendererApi | undefined => (
    (window as unknown as { electronAPI?: ElectronOverlayRendererApi }).electronAPI
);

const readEmotionGifs = (): Record<string, string> => {
    try {
        const value = JSON.parse(localStorage.getItem(EMOTION_GIFS_KEY) || '{}');
        return value && typeof value === 'object' ? value : {};
    } catch {
        return {};
    }
};

const resolveIdentity = (
    base: OverlayShellMetadata | undefined,
    speaking: OverlayDisplayInstance | undefined,
): OverlayShellMetadata => ({
    name: speaking?.metadata.name ?? base?.name ?? 'Kestrel',
    emotion: speaking?.metadata.emotion ?? base?.emotion ?? 'idle',
    agentTags: speaking?.metadata.agentTags ?? base?.agentTags ?? [],
});

const OverlayIdentity: React.FC<{
    identity: OverlayShellMetadata;
    emotionGifs: Record<string, string>;
    idle: boolean;
}> = ({ identity, emotionGifs, idle }) => {
    const avatar = emotionGifs[identity.emotion || 'idle'] || emotionGifs.idle;
    return (
        <div className="overlay-shell__drag" title="Drag overlay">
            <div className="overlay-shell__avatar" aria-hidden="true">
                {avatar ? <img src={avatar} alt="" /> : 'AI'}
            </div>
            <div className="overlay-shell__identity">
                <span className="overlay-shell__name">{identity.name || 'Kestrel'}</span>
                {(identity.agentTags || []).map((tag) => (
                    <span className="overlay-shell__tag" key={tag}>{tag}</span>
                ))}
                {idle && <span className="overlay-shell__idle-label">Overlay ready</span>}
            </div>
        </div>
    );
};

const SpeakingContext: React.FC<{
    instance: OverlayDisplayInstance;
    onComponentEvent: (instanceId: string, event: OverlayComponentEvent) => void;
}> = ({ instance, onComponentEvent }) => {
    const definition = getOverlayDisplayDefinition('ai_message');
    const emitComponentEvent = React.useCallback((event: OverlayComponentEvent) => {
        onComponentEvent(instance.instanceId, event);
    }, [instance.instanceId, onComponentEvent]);
    return (
        <section
            className="overlay-shell__speaking"
            data-instance-id={instance.instanceId}
            data-display-type={instance.type}
            data-policy={instance.policy}
        >
            <div className="overlay-shell__response">
                {definition.renderExpanded({
                    snapshot: instance.snapshot as never,
                    revision: instance.revision,
                    emitComponentEvent,
                })}
            </div>
        </section>
    );
};

const GeneratedDisplayItem: React.FC<{
    instance: OverlayDisplayInstance;
    onComponentEvent: (instanceId: string, event: OverlayComponentEvent) => void;
}> = ({ instance, onComponentEvent }) => {
    const definition = getOverlayDisplayDefinition(instance.type) as ReturnType<typeof getOverlayDisplayDefinition>;
    const emitComponentEvent = React.useCallback((event: OverlayComponentEvent) => {
        onComponentEvent(instance.instanceId, event);
    }, [instance.instanceId, onComponentEvent]);
    return (
        <article
            className={`overlay-list-item${instance.folded ? ' overlay-list-item--folded' : ''}`}
            data-instance-id={instance.instanceId}
            data-display-type={instance.type}
            data-policy={instance.policy}
        >
            <header className="overlay-list-item__header">
                <div className="overlay-list-item__summary">
                    {definition.renderSummary(instance.snapshot as never)}
                </div>
            </header>
            {!instance.folded && (
                <div className="overlay-list-item__body">
                    {definition.renderExpanded({
                        snapshot: instance.snapshot as never,
                        revision: instance.revision,
                        emitComponentEvent,
                    })}
                </div>
            )}
        </article>
    );
};

const FloatingChat: React.FC = () => {
    const [state, dispatch] = React.useReducer(
        overlayPresentationReducer,
        initialOverlayPresentationState,
    );
    const stateRef = React.useRef(state);
    const shellRef = React.useRef<HTMLElement>(null);
    const emotionGifs = React.useMemo(readEmotionGifs, []);

    React.useEffect(() => {
        stateRef.current = state;
    }, [state]);

    const commit = React.useCallback((result: OverlayTransitionResult) => {
        stateRef.current = result.state;
        dispatch({ type: 'replace', state: result.state });
        const api = getElectronApi();
        result.events.forEach((event) => api?.emitOverlayLifecycle?.(event));
        return result;
    }, []);

    React.useEffect(() => {
        const api = getElectronApi();
        const removeCommand = api?.onOverlayDisplayCommand?.((request) => {
            const result = commit(applyOverlayDisplayRequest(stateRef.current, request));
            if (result.acknowledgement) api.acknowledgeOverlayDisplayRequest?.(result.acknowledgement);
        });
        const removeEnabled = api?.onOverlayEnabledChange?.((enabled) => {
            commit(setOverlayEnabled(stateRef.current, enabled));
        });
        const removePresentation = api?.onOverlayPresentationChange?.((change) => {
            if (change.kind === 'started') {
                commit(beginOverlayPresentation(stateRef.current, change.presentation));
                return;
            }
            commit(endOverlayPresentation(stateRef.current, change.presentationId));
        });
        api?.reportOverlayReady?.();
        return () => {
            removeCommand?.();
            removeEnabled?.();
            removePresentation?.();
        };
    }, [commit]);

    React.useEffect(() => {
        const deadline = getNextOverlayDeadline(state.instances);
        if (deadline === null) return undefined;
        const timer = window.setTimeout(() => {
            commit(advanceOverlayTimers(stateRef.current));
        }, Math.max(0, deadline - Date.now()));
        return () => window.clearTimeout(timer);
    }, [commit, state.instances]);

    const speaking = state.instances.find((instance) => instance.type === 'ai_message');
    const generatedDisplays = React.useMemo(
        () => orderOverlayInstances(state.instances.filter((instance) => instance.type !== 'ai_message')),
        [state.instances],
    );
    const idle = !speaking && generatedDisplays.length === 0;
    const identity = resolveIdentity(state.presentation?.displayIdentity, speaking);
    const shellWidth = Math.max(
        idle ? IDLE_WIDTH : SPEAKING_WIDTH,
        ...generatedDisplays.map((instance) => {
            const definition = getOverlayDisplayDefinition(instance.type);
            return (instance.folded ? definition.dimensions.folded : definition.dimensions.expanded).width;
        }),
    );

    React.useLayoutEffect(() => {
        const shell = shellRef.current;
        if (!shell) return undefined;
        const resize = () => {
            const bounds = shell.getBoundingClientRect();
            getElectronApi()?.resizeFloatingChat?.(
                Math.ceil(bounds.width),
                Math.ceil(bounds.height),
            );
        };
        resize();
        const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize);
        observer?.observe(shell);
        return () => observer?.disconnect();
    }, [generatedDisplays, idle, speaking, shellWidth]);

    const handleComponentEvent = React.useCallback((
        instanceId: string,
        event: OverlayComponentEvent,
    ) => {
        commit(applyOverlayComponentEvent(stateRef.current, instanceId, event));
    }, [commit]);

    return (
        <main className="floating-overlay-stage" aria-live="polite">
            <section
                className={`overlay-shell${idle ? ' overlay-shell--idle' : ''}`}
                ref={shellRef}
                style={{ width: shellWidth }}
                data-presentation-id={state.presentation?.presentationId || ''}
            >
                <header className="overlay-shell__header">
                    <OverlayIdentity identity={identity} emotionGifs={emotionGifs} idle={idle} />
                </header>
                {speaking && (
                    <SpeakingContext
                        instance={speaking}
                        onComponentEvent={handleComponentEvent}
                    />
                )}
                {generatedDisplays.length > 0 && (
                    <div className="overlay-display-list">
                        {generatedDisplays.map((instance) => (
                            <GeneratedDisplayItem
                                key={instance.instanceId}
                                instance={instance}
                                onComponentEvent={handleComponentEvent}
                            />
                        ))}
                    </div>
                )}
            </section>
        </main>
    );
};

export default FloatingChat;
