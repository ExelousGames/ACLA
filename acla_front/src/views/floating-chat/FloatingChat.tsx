import React from 'react';
import '../lap-analysis/ai-chat/ai-chat.css';
import './floating-chat.css';
import type {
    AiOverlayPresentationAcknowledgement,
    AiOverlayPresentationCard,
    AiOverlayPresentationSnapshot,
    AiOverlayRendererEvent,
    AiOverlayShellMetadata,
} from './ai-overlay-types';
import { isJsonSafe } from './ai-overlay-types';
import {
    getAiOverlayRenderer,
    registerBuiltInAiOverlayRenderers,
} from './overlay-renderer-modules';

const EMOTION_GIFS_KEY = 'acla-emotion-gifs';
const IDLE_WIDTH = 300;
const SPEAKING_WIDTH = 420;

interface ElectronOverlayRendererApi {
    onOverlayPresentation?: (
        listener: (presentation: AiOverlayPresentationSnapshot) => void,
    ) => (() => void);
    acknowledgeOverlayPresentation?: (
        acknowledgement: AiOverlayPresentationAcknowledgement,
    ) => void;
    emitOverlayRendererEvent?: (event: AiOverlayRendererEvent) => void;
    reportOverlayReady?: () => void;
    resizeFloatingChat?: (width: number, height: number) => void;
}

const getElectronApi = (): ElectronOverlayRendererApi | undefined => (
    (window as unknown as { electronAPI?: ElectronOverlayRendererApi }).electronAPI
);

let rendererRegistrationError: Error | null = null;
try {
    registerBuiltInAiOverlayRenderers();
} catch (error) {
    rendererRegistrationError = error instanceof Error ? error : new Error(String(error));
}

const readEmotionGifs = (): Record<string, string> => {
    try {
        const value = JSON.parse(localStorage.getItem(EMOTION_GIFS_KEY) || '{}');
        return value && typeof value === 'object' ? value : {};
    } catch {
        return {};
    }
};

const validatePresentation = (presentation: AiOverlayPresentationSnapshot): string | null => {
    if (rendererRegistrationError) return rendererRegistrationError.message;
    if (!presentation || typeof presentation !== 'object' || !isJsonSafe(presentation)) {
        return 'Malformed overlay presentation.';
    }
    if (!presentation.presentationId?.trim()
        || !Number.isInteger(presentation.presentationRevision)
        || presentation.presentationRevision < 1
        || presentation.session?.presentationId !== presentation.presentationId
        || !Array.isArray(presentation.cards)) {
        return 'Malformed overlay presentation identity.';
    }
    const componentNames = new Set<string>();
    for (const card of presentation.cards) {
        if (!card.componentName?.trim() || !card.componentType?.trim()) {
            return 'Overlay cards require componentName and componentType.';
        }
        if (componentNames.has(card.componentName)) {
            return `Duplicate live overlay componentName '${card.componentName}'.`;
        }
        componentNames.add(card.componentName);
        try {
            const renderer = getAiOverlayRenderer(card.componentType);
            if (!renderer.validateSnapshot(card.snapshot)) {
                return `Invalid snapshot for overlay componentType '${card.componentType}'.`;
            }
            if (!renderer.dimensions[card.status]) {
                return `Overlay componentType '${card.componentType}' does not support '${card.status}'.`;
            }
        } catch (error) {
            return error instanceof Error ? error.message : String(error);
        }
    }
    return null;
};

const resolveIdentity = (
    base: AiOverlayShellMetadata | undefined,
    speaking: AiOverlayPresentationCard | undefined,
): AiOverlayShellMetadata => ({
    name: speaking?.metadata.name ?? base?.name ?? 'Kestrel',
    emotion: speaking?.metadata.emotion ?? base?.emotion ?? 'idle',
    agentTags: speaking?.metadata.agentTags ?? base?.agentTags ?? [],
});

const OverlayIdentity: React.FC<{
    identity: AiOverlayShellMetadata;
    emotionGifs: Record<string, string>;
}> = ({ identity, emotionGifs }) => {
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
            </div>
        </div>
    );
};

const useRenderContext = (
    presentationId: string,
    card: AiOverlayPresentationCard,
) => React.useMemo(() => ({
    componentName: card.componentName,
    revision: card.revision,
    emitRendererEvent: (event: string) => getElectronApi()?.emitOverlayRendererEvent?.({
        presentationId,
        componentName: card.componentName,
        revision: card.revision,
        event,
    }),
}), [card.componentName, card.revision, presentationId]);

const SpeakingContext: React.FC<{
    presentationId: string;
    card: AiOverlayPresentationCard;
}> = ({ presentationId, card }) => {
    const renderer = getAiOverlayRenderer(card.componentType);
    const context = useRenderContext(presentationId, card);
    return (
        <section
            className="overlay-shell__speaking"
            data-component-name={card.componentName}
            data-display-type={card.componentType}
        >
            <div className="overlay-shell__response">
                {renderer.renderOverlay(card.snapshot, card.status, context)}
            </div>
        </section>
    );
};

const GeneratedDisplayItem: React.FC<{
    presentationId: string;
    card: AiOverlayPresentationCard;
    deckIndex: number;
}> = ({ presentationId, card, deckIndex }) => {
    const renderer = getAiOverlayRenderer(card.componentType);
    const context = useRenderContext(presentationId, card);
    const fullSizeActive = card.status === 'full_size';
    const dimensions = renderer.dimensions[card.status]!;
    return (
        <article
            className={[
                'overlay-list-item',
                'overlay-list-item--deck',
                card.status === 'folded' ? 'overlay-list-item--folded' : '',
                fullSizeActive ? 'overlay-list-item--full-size-active' : '',
            ].filter(Boolean).join(' ')}
            style={{
                zIndex: deckIndex + 1,
                ...(fullSizeActive ? {
                    width: dimensions.width,
                    height: dimensions.height,
                    flexBasis: dimensions.height,
                    alignSelf: 'center',
                } : {}),
            }}
            data-component-name={card.componentName}
            data-display-type={card.componentType}
            data-placement={card.placement}
            data-full-size-active={fullSizeActive ? 'true' : undefined}
            data-renderer-width={dimensions.width}
            data-renderer-height={dimensions.height}
        >
            {card.status === 'folded' ? (
                <header className="overlay-list-item__header">
                    <div className="overlay-list-item__summary">
                        {renderer.renderOverlay(card.snapshot, card.status, context)}
                    </div>
                </header>
            ) : (
                <div className="overlay-list-item__body">
                    {renderer.renderOverlay(card.snapshot, card.status, context)}
                </div>
            )}
        </article>
    );
};

const FloatingChat: React.FC = () => {
    const [presentation, setPresentation] = React.useState<AiOverlayPresentationSnapshot | null>(null);
    const presentationRef = React.useRef(presentation);
    const shellRef = React.useRef<HTMLElement>(null);
    const emotionGifs = React.useMemo(readEmotionGifs, []);

    React.useEffect(() => {
        presentationRef.current = presentation;
    }, [presentation]);

    React.useEffect(() => {
        const api = getElectronApi();
        const removePresentation = api?.onOverlayPresentation?.((next) => {
            const validationError = validatePresentation(next);
            const current = presentationRef.current;
            const stale = current?.presentationId === next.presentationId
                && next.presentationRevision <= current.presentationRevision;
            const error = stale ? 'Stale overlay presentation revision.' : validationError;
            api.acknowledgeOverlayPresentation?.({
                presentationId: next?.presentationId || 'unknown',
                presentationRevision: next?.presentationRevision || 0,
                accepted: !error,
                ...(error ? { error } : {}),
            });
            if (!error) {
                presentationRef.current = next;
                setPresentation(next);
            }
        });
        api?.reportOverlayReady?.();
        return () => removePresentation?.();
    }, []);

    const cards = React.useMemo(() => presentation?.cards ?? [], [presentation]);
    const speaking = cards.find((card) => card.shellSlot === 'speech');
    const generatedDisplays = cards.filter((card) => card.shellSlot !== 'speech');
    const idle = !speaking && generatedDisplays.length === 0;
    const identity = resolveIdentity(presentation?.session.displayIdentity, speaking);
    const widths = generatedDisplays.map((card) => (
        getAiOverlayRenderer(card.componentType).dimensions[card.status]?.width ?? SPEAKING_WIDTH
    ));
    const shellWidth = Math.max(idle ? IDLE_WIDTH : SPEAKING_WIDTH, ...widths);

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
    }, [cards, shellWidth]);

    return (
        <main className="floating-overlay-stage" aria-live="polite">
            <section
                className={[
                    'overlay-shell',
                    idle ? 'overlay-shell--idle' : '',
                ].filter(Boolean).join(' ')}
                ref={shellRef}
                style={{ width: shellWidth }}
                data-presentation-id={presentation?.presentationId || ''}
            >
                <header className="overlay-shell__header">
                    <OverlayIdentity identity={identity} emotionGifs={emotionGifs} />
                </header>
                {speaking && presentation && (
                    <SpeakingContext presentationId={presentation.presentationId} card={speaking} />
                )}
                {generatedDisplays.length > 0 && presentation && (
                    <div className="overlay-display-list">
                        {generatedDisplays.map((card, deckIndex) => (
                            <GeneratedDisplayItem
                                key={card.componentName}
                                presentationId={presentation.presentationId}
                                card={card}
                                deckIndex={deckIndex}
                            />
                        ))}
                    </div>
                )}
            </section>
        </main>
    );
};

export default FloatingChat;
